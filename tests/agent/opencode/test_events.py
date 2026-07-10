"""Tests for the shared per-daemon event consumer: it decodes the daemon-global /event stream once
and routes each event to the registered session's watcher, aborting a session when its watcher flags
a cap crossing. Routing is exercised via _dispatch directly (deterministic); the stream lifecycle is
exercised with an httpx MockTransport."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent.opencode.events import EventConsumer


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _data(event: dict) -> str:
    """One SSE ``data:`` line, as the consumer reads them off the stream."""
    return "data: " + json.dumps(event)


def _idle(session_id: str) -> dict:
    return {"type": "session.idle", "properties": {"sessionID": session_id}}


class _FakeWatcher:
    def __init__(self, cross: bool = False) -> None:
        self.seen: list[dict] = []
        self._cross = cross

    def process(self, event: dict) -> bool:
        self.seen.append(event)
        return self._cross


def _ok(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=True)


@pytest.mark.asyncio
async def test_dispatch_routes_events_to_the_registered_session_only():
    consumer = EventConsumer("http://d", _mock_client(_ok))
    a, b = _FakeWatcher(), _FakeWatcher()
    consumer.add_watcher("sess-a", a)
    consumer.add_watcher("sess-b", b)

    consumer._dispatch(_data(_idle("sess-a")))
    consumer._dispatch(_data(_idle("sess-b")))
    consumer._dispatch(_data(_idle("unknown")))  # no watcher -> dropped
    consumer._dispatch("event: ping")  # non-data line -> ignored
    consumer._dispatch("data: not json")  # undecodable -> ignored

    assert [e["properties"]["sessionID"] for e in a.seen] == ["sess-a"]
    assert [e["properties"]["sessionID"] for e in b.seen] == ["sess-b"]


@pytest.mark.asyncio
async def test_dispatch_aborts_session_when_watcher_flags_a_cap():
    posted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request.url.path)
        return httpx.Response(200, json=True)

    consumer = EventConsumer("http://d", _mock_client(handler))
    consumer.add_watcher("sess-a", _FakeWatcher(cross=True))

    consumer._dispatch(_data(_idle("sess-a")))  # crossing -> fires an abort task

    for _ in range(100):
        if "/session/sess-a/abort" in posted:
            break
        await asyncio.sleep(0.01)
    assert "/session/sess-a/abort" in posted


@pytest.mark.asyncio
async def test_dispatch_survives_a_non_object_json_payload():
    # Regression: a valid-JSON but non-object data: line (a bare string/number/null/array) has no
    # session to route to and would raise inside routing; uncaught, that kills this shared reader for
    # the daemon's whole life (every later run then believes caps are enforced while nothing fires).
    # It must be skipped, and the consumer must keep routing well-formed events afterwards.
    consumer = EventConsumer("http://d", _mock_client(_ok))
    w = _FakeWatcher()
    consumer.add_watcher("sess-a", w)

    for payload in ('"ping"', "123", "null", "[1, 2]"):
        consumer._dispatch("data: " + payload)  # must not raise

    consumer._dispatch(_data(_idle("sess-a")))  # consumer still alive and routing
    assert [e["properties"]["sessionID"] for e in w.seen] == ["sess-a"]


@pytest.mark.asyncio
async def test_start_times_out_when_stream_returns_non_2xx():
    # A non-2xx /event delivers no events, so the consumer must not report itself connected — else a
    # capped run would proceed believing its budget/turn caps are enforced. start() times out instead,
    # so the daemon start awaiting it fails fast and retries.
    client = _mock_client(lambda _r: httpx.Response(500, json={"error": "boom"}))
    consumer = EventConsumer("http://d", client)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await consumer.start(ready_timeout=0.1)
        assert consumer.connected is False
    finally:
        await consumer.stop()
        await client.aclose()


@pytest.mark.asyncio
async def test_removed_watcher_stops_receiving():
    consumer = EventConsumer("http://d", _mock_client(_ok))
    w = _FakeWatcher()
    consumer.add_watcher("sess-a", w)
    consumer.remove_watcher("sess-a")
    consumer._dispatch(_data(_idle("sess-a")))
    assert w.seen == []


@pytest.mark.asyncio
async def test_start_marks_connected_then_stops_cleanly():
    async def open_stream():
        yield b"data: {}\n\n"
        await asyncio.sleep(30)  # keep the stream open so `connected` stays set after start()

    client = _mock_client(lambda _r: httpx.Response(200, content=open_stream()))
    consumer = EventConsumer("http://d", client)
    try:
        await consumer.start(ready_timeout=1.0)
        assert consumer.connected is True
    finally:
        await consumer.stop()
        assert consumer.connected is False
        await client.aclose()


@pytest.mark.asyncio
async def test_start_times_out_when_stream_never_opens():
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("stream refused")

    client = _mock_client(refuse)
    consumer = EventConsumer("http://d", client)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await consumer.start(ready_timeout=0.1)
    finally:
        await consumer.stop()
        await client.aclose()
