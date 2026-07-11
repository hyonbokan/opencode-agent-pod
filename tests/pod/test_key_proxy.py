"""The key-injecting proxy: real keys reach the upstream, never the caller, and bodies stream through.

A fake upstream records the request it receives so each test can assert the dummy auth the daemon
would send was replaced by the real key in the right header format. No opencode, no network beyond
localhost."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from pod.key_proxy import KeyProxy, ProviderRoute, build_routes

# The last request the fake upstream saw: method, path, headers, and body.
_SEEN: dict[str, Any] = {}


class _FakeUpstream(BaseHTTPRequestHandler):
    """Records each request and answers with a small JSON body, or an SSE stream for /stream."""

    def log_message(self, *_args) -> None:
        pass

    def _record(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        _SEEN.clear()
        _SEEN.update(
            method=self.command,
            path=self.path,
            headers={k.lower(): v for k, v in self.headers.items()},
            body=self.rfile.read(length) if length else b"",
        )

    def do_GET(self) -> None:
        self._record()
        if self.path.endswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: chunk-{i}\n\n".encode())
                self.wfile.flush()
            return
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_GET


@pytest.fixture
def upstream():
    """A fake provider upstream on a background thread, yielding its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


async def _proxy(routes: dict[str, ProviderRoute]) -> KeyProxy:
    return await KeyProxy(routes).start()


async def test_replaces_bearer_auth_with_the_real_key(upstream):
    proxy = await _proxy({"openai": ProviderRoute(upstream, "real-secret", "bearer")})
    try:
        async with httpx.AsyncClient(base_url=proxy.base_url) as client:
            resp = await client.post("/openai/v1/chat", headers={"authorization": "Bearer dummy"})
        assert resp.status_code == 200
        assert _SEEN["headers"]["authorization"] == "Bearer real-secret"
    finally:
        await proxy.stop()


async def test_anthropic_key_goes_in_x_api_key_header(upstream):
    proxy = await _proxy({"anthropic": ProviderRoute(upstream, "sk-real", "anthropic")})
    try:
        async with httpx.AsyncClient(base_url=proxy.base_url) as client:
            await client.post("/anthropic/v1/messages", headers={"x-api-key": "dummy"})
        headers = _SEEN["headers"]
        assert headers["x-api-key"] == "sk-real"
        assert "anthropic-version" in headers
        assert "authorization" not in headers  # the daemon's dummy auth is not forwarded
    finally:
        await proxy.stop()


async def test_forwards_path_and_body_to_the_upstream(upstream):
    proxy = await _proxy({"openai": ProviderRoute(upstream, "k", "bearer")})
    try:
        async with httpx.AsyncClient(base_url=proxy.base_url) as client:
            await client.post("/openai/v1/responses", content=b'{"m":1}')
        assert _SEEN["path"] == "/v1/responses"
        assert _SEEN["body"] == b'{"m":1}'
    finally:
        await proxy.stop()


async def test_streams_the_response_body_through(upstream):
    proxy = await _proxy({"openai": ProviderRoute(upstream, "k", "bearer")})
    try:
        async with httpx.AsyncClient(base_url=proxy.base_url) as client:
            resp = await client.get("/openai/v1/stream")
        assert resp.status_code == 200
        assert resp.text == "data: chunk-0\n\ndata: chunk-1\n\ndata: chunk-2\n\n"
    finally:
        await proxy.stop()


async def test_unknown_provider_is_rejected():
    proxy = await _proxy({"openai": ProviderRoute("http://x", "k", "bearer")})
    try:
        async with httpx.AsyncClient(base_url=proxy.base_url) as client:
            resp = await client.get("/unknown/v1/models")
        assert resp.status_code == 404
    finally:
        await proxy.stop()


def test_build_routes_covers_catalog_and_custom_providers():
    routes = build_routes(
        {
            "ANTHROPIC_API_KEY": "a-key",
            "GEMINI_API_KEY": "g-key",  # the fallback name for google
            "AGENT_CUSTOM_PROVIDERS": json.dumps(
                [{"name": "vllm", "base_url": "http://h:8000/v1", "models": ["m"], "api_key": "v"}]
            ),
        }
    )
    assert routes["anthropic"] == ProviderRoute(
        "https://api.anthropic.com/v1", "a-key", "anthropic"
    )
    assert routes["google"].api_key == "g-key"
    assert routes["vllm"] == ProviderRoute("http://h:8000/v1", "v", "bearer")
    assert "openai" not in routes  # no key present, so no route invented
