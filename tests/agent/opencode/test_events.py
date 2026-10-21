"""Tests for the shared per-daemon event consumer: it decodes the daemon-global /event stream once
and routes each event to the registered session's watcher, aborting a session when its watcher flags
a cap crossing or sending it messages. Routing is exercised via _dispatch directly (deterministic); the stream lifecycle is
exercised with an httpx MockTransport."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent.opencode import events
from agent.opencode.events import NO_ACTION, EventConsumer, WatchAction


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _data(event: dict) -> str:
    """One SSE ``data:`` line, as the consumer reads them off the stream."""
    return "data: " + json.dumps(event)


def _idle(session_id: str) -> dict:
    return {"type": "session.idle", "properties": {"sessionID": session_id}}


def _created(session_id: str, parent_id: str | None) -> dict:
    info = {"id": session_id, "parentID": parent_id}
    return {"type": "session.created", "properties": {"info": info}}


class _FakeWatcher:
    def __init__(self, action: WatchAction = NO_ACTION) -> None:
        self.seen: list[dict] = []
        self._action = action

    def process(self, event: dict) -> WatchAction:
        self.seen.append(event)
        return self._action


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
    consumer.add_watcher("sess-a", _FakeWatcher(WatchAction(abort=True)))

    consumer._dispatch(_data(_idle("sess-a")))  # crossing -> fires an abort task

    for _ in range(100):
        if "/session/sess-a/abort" in posted:
            break
        await asyncio.sleep(0.01)
    assert "/session/sess-a/abort" in posted


@pytest.mark.asyncio
async def test_dispatch_posts_a_watchers_prompts_to_prompt_async():
    posted: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append((request.url.path, json.loads(request.content)))
        return httpx.Response(204)

    consumer = EventConsumer("http://d", _mock_client(handler))
    prompts = (("sess-a", {"parts": ["root"]}), ("child-1", {"parts": ["child"]}))
    consumer.add_watcher("sess-a", _FakeWatcher(WatchAction(prompts=prompts)))

    consumer._dispatch(_data(_idle("sess-a")))
    await asyncio.gather(*consumer._action_tasks)

    assert sorted(posted, key=lambda p: p[0]) == [
        ("/session/child-1/prompt_async", {"parts": ["child"]}),
        ("/session/sess-a/prompt_async", {"parts": ["root"]}),
    ]
    assert not any(path.endswith("/abort") for path, _ in posted)


@pytest.mark.asyncio
async def test_a_failed_prompt_post_is_swallowed_and_logged(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    warnings: list[tuple] = []
    monkeypatch.setattr(events.logger, "warning", lambda *args: warnings.append(args))
    consumer = EventConsumer("http://d", _mock_client(handler))
    action = WatchAction(prompts=(("sess-a", {}),))
    consumer.add_watcher("sess-a", _FakeWatcher(action))

    consumer._dispatch(_data(_idle("sess-a")))
    await asyncio.gather(*consumer._action_tasks)  # must not raise

    assert len(warnings) == 1 and warnings[0][1] == "sess-a"


@pytest.mark.asyncio
async def test_subagent_events_reach_the_runs_watcher_down_to_grandchildren():
    consumer = EventConsumer("http://d", _mock_client(_ok))
    w = _FakeWatcher()
    consumer.add_watcher("root", w)

    consumer._dispatch(_data(_created("child", "root")))
    consumer._dispatch(_data(_created("grandchild", "child")))
    consumer._dispatch(_data(_created("stranger", "unrelated")))
    for session_id in ("child", "grandchild", "stranger"):
        consumer._dispatch(_data(_idle(session_id)))

    assert [e["properties"]["sessionID"] for e in w.seen] == ["child", "grandchild"]


@pytest.mark.asyncio
async def test_a_subagent_event_can_abort_the_run_it_belongs_to():
    posted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request.url.path)
        return httpx.Response(200, json=True)

    consumer = EventConsumer("http://d", _mock_client(handler))
    consumer.add_watcher("root", _FakeWatcher(WatchAction(abort=True)))
    consumer._dispatch(_data(_created("child", "root")))

    consumer._dispatch(_data(_idle("child")))
    await asyncio.gather(*consumer._action_tasks)

    assert posted == ["/session/root/abort"]  # the root, which cascades, not the child


@pytest.mark.asyncio
async def test_removing_a_watcher_drops_its_subagent_aliases():
    consumer = EventConsumer("http://d", _mock_client(_ok))
    w, other = _FakeWatcher(), _FakeWatcher()
    consumer.add_watcher("root", w)
    consumer.add_watcher("other", other)
    consumer._dispatch(_data(_created("child", "root")))
    consumer._dispatch(_data(_created("other-child", "other")))

    consumer.remove_watcher("root")
    consumer._dispatch(_data(_idle("child")))
    consumer._dispatch(_data(_idle("other-child")))

    assert w.seen == []
    assert [e["properties"]["sessionID"] for e in other.seen] == ["other-child"]


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
