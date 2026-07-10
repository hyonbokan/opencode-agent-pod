"""Lifecycle tests for the opencode serve daemon manager, driving a fake ``opencode`` on PATH that
really binds the requested port and serves ``/api/health`` — so start/health/stop are exercised end
to end without the real binary. A separate mode makes the fake exit early or hang so the failure
paths (retry, health timeout) are covered too."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import httpx
import pytest

from agent.opencode.server import OpencodeServer

# A fake ``opencode``: on ``serve`` it reads --port from argv and behaves per FAKE_MODE — bind and
# serve health plus a live ``/event`` stream (default), exit early, or hang without ever binding.
# The server is threaded so the long-lived ``/event`` request doesn't block health polls.
_FAKE = """
import os, sys, time
mode = os.environ.get("FAKE_MODE", "healthy")
argv = sys.argv
port = None
for i, a in enumerate(argv):
    if a == "--port":
        port = int(argv[i + 1])
if mode == "early_exit":
    sys.stderr.write("fake opencode: bind failed\\n")
    sys.exit(1)
if mode == "hang":
    time.sleep(60)
    sys.exit(0)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"healthy":true}')
        elif self.path == "/event":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b": keepalive\\n\\n")
                    self.wfile.flush()
                    time.sleep(0.5)
            except Exception:
                pass
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *a):
        pass
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _fake_env(dir_path: Path, mode: str) -> dict[str, str]:
    """Write an executable fake ``opencode`` and return an env whose PATH finds it, in the given
    mode."""
    script = dir_path / "opencode"
    script.write_text(f"#!{sys.executable}\n{_FAKE}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return {"PATH": f"{dir_path}:/usr/bin:/bin", "FAKE_MODE": mode}


@pytest.mark.asyncio
async def test_start_becomes_healthy_then_stops(tmp_path):
    server = OpencodeServer(str(tmp_path), env=_fake_env(tmp_path, "healthy"), startup_timeout=10.0)
    await server.start()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(server.base_url + "/api/health")
        assert resp.status_code == 200
    finally:
        await server.stop()
    # After stop the daemon is gone and the URL is no longer addressable.
    with pytest.raises(RuntimeError):
        _ = server.base_url


@pytest.mark.asyncio
async def test_context_manager_starts_and_stops(tmp_path):
    async with OpencodeServer(
        str(tmp_path), env=_fake_env(tmp_path, "healthy"), startup_timeout=10.0
    ) as server:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(server.base_url + "/api/health")
        assert resp.status_code == 200
    with pytest.raises(RuntimeError):
        _ = server.base_url


@pytest.mark.asyncio
async def test_start_raises_when_daemon_exits_early(tmp_path):
    server = OpencodeServer(
        str(tmp_path), env=_fake_env(tmp_path, "early_exit"), startup_timeout=5.0
    )
    with pytest.raises(RuntimeError, match="failed to start"):
        await server.start(attempts=2)


@pytest.mark.asyncio
async def test_start_times_out_when_never_healthy(tmp_path):
    server = OpencodeServer(str(tmp_path), env=_fake_env(tmp_path, "hang"), startup_timeout=1.0)
    with pytest.raises(RuntimeError, match="failed to start"):
        await server.start(attempts=1)
    await server.stop()  # reap the hung fake
