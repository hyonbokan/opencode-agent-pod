"""The HTTP surface: bearer auth, request validation, and the SSE run endpoint end to end."""

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from agent.models import OpencodeResult
from pod import service
from pod.app import create_app
from pod.settings import PodSettings

_BASE_SETTINGS = PodSettings(
    bearer_token="secret",
    max_budget_usd=None,
    default_max_budget_usd=None,
    session_timeout=900.0,
    max_turns=30,
    keepalive_seconds=0.02,
    host="127.0.0.1",
    port=8080,
    key_proxy_enabled=False,
    key_proxy_host="127.0.0.1",
    workspace_max_bytes=2 * 1024**3,
    workspace_fetch_timeout=60.0,
    workspace_host_allowlist=(),
    workspace_tls_verify=True,
)


def _settings(**overrides) -> PodSettings:
    return replace(_BASE_SETTINGS, **overrides)


def _client(settings: PodSettings) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(settings))
    return httpx.AsyncClient(transport=transport, base_url="http://pod")


_AUTH = {"Authorization": "Bearer secret"}


@pytest.mark.asyncio
async def test_health_needs_no_auth():
    async with _client(_settings()) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_run_rejects_missing_token():
    async with _client(_settings()) as client:
        resp = await client.post("/agent/run", json={"model": "m", "prompt": "p"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_run_rejects_wrong_token():
    async with _client(_settings()) as client:
        resp = await client.post(
            "/agent/run",
            json={"model": "m", "prompt": "p"},
            headers={"Authorization": "Bearer nope"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_run_fails_closed_when_no_token_configured():
    # A pod with no token must refuse every run rather than serve unauthenticated.
    async with _client(_settings(bearer_token=None)) as client:
        resp = await client.post("/agent/run", json={"model": "m", "prompt": "p"}, headers=_AUTH)
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_run_rejects_malformed_body():
    async with _client(_settings()) as client:
        resp = await client.post("/agent/run", json={"prompt": "p"}, headers=_AUTH)  # no model
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_run_rejects_unsupported_workspace_scheme():
    async with _client(_settings()) as client:
        resp = await client.post(
            "/agent/run",
            json={"model": "m", "prompt": "p", "workspace": {"source": "s3://bucket/key"}},
            headers=_AUTH,
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_run_streams_sse_and_returns_the_result(monkeypatch):
    class _FakeRunner:
        def __init__(self, **kwargs):
            pass

        async def run(self, **kwargs):
            return OpencodeResult(
                text="the answer", total_cost_usd=0.02, duration_ms=10, subtype="success"
            )

    async def _stop(cwd: str) -> None:
        return None

    monkeypatch.setattr(service, "OpencodeRunner", _FakeRunner)
    monkeypatch.setattr(service, "stop_opencode_daemon", _stop)

    async with _client(_settings()) as client:
        resp = await client.post("/agent/run", json={"model": "m", "prompt": "p"}, headers=_AUTH)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: cost" in body
    assert "event: done" in body
    done = json.loads(body.split("event: done\n")[1].split("data: ")[1].split("\n\n")[0])
    assert done["text"] == "the answer"
    assert done["is_error"] is False
    await service.drain_cleanups()
