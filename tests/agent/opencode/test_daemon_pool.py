"""Tests for the serve daemon pool: one daemon per working directory, the MCP registered once
per daemon, and shutdown stopping them all. OpencodeServer and register_mcp are stubbed, so no real
daemon is launched."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

import agent.opencode.daemon_pool as pool
from core.tools.mcp import MCP_SERVER_NAME


class _FakeServer:
    def __init__(self, cwd, **_kwargs):
        self.cwd = cwd
        self.base_url = f"http://daemon{cwd}"
        self.client = None  # the pool passes daemon.client to register_mcp
        self.started = 0
        self.stopped = 0
        self.is_alive = True

    async def start(self):
        self.started += 1
        return self

    async def stop(self):
        self.stopped += 1


@pytest.fixture
def clean_pool(monkeypatch):
    monkeypatch.setattr(pool, "OpencodeServer", _FakeServer)
    monkeypatch.setattr(pool, "register_mcp", AsyncMock(return_value={}))
    pool._daemons.clear()
    yield
    pool._daemons.clear()


@pytest.mark.asyncio
@pytest.mark.usefixtures("clean_pool")
async def test_get_daemon_starts_once_per_cwd_and_registers_mcp():
    first: Any = await pool.get_opencode_daemon("/a")
    again = await pool.get_opencode_daemon("/a")  # cached, not restarted
    other: Any = await pool.get_opencode_daemon("/b")

    assert first is again
    assert first is not other
    assert first.started == 1  # the cached call did not start a second daemon
    assert other.started == 1

    # The MCP server is registered once per distinct daemon, against that daemon's URL.
    register_mcp = cast(AsyncMock, pool.register_mcp)
    assert register_mcp.await_count == 2
    base_url, name, _config = register_mcp.await_args_list[0].args
    assert base_url == first.base_url
    assert name == MCP_SERVER_NAME


@pytest.mark.asyncio
@pytest.mark.usefixtures("clean_pool")
async def test_dead_daemon_is_evicted_and_relaunched():
    # A pooled daemon whose process has died must not be handed out again; it is stopped, dropped, and
    # a fresh one started so a mid-run crash does not poison every later agent.
    first: Any = await pool.get_opencode_daemon("/a")
    first.is_alive = False

    replacement: Any = await pool.get_opencode_daemon("/a")

    assert replacement is not first
    assert first.stopped == 1  # the dead daemon was reaped
    assert replacement.is_alive
    assert pool._daemons["/a"] is replacement


@pytest.mark.asyncio
async def test_register_failure_stops_daemon_and_leaves_pool_empty(monkeypatch):
    # A daemon that starts but whose MCP registration fails must be stopped, not left as an untracked
    # process the pool can never reap, and the error must surface to the caller.
    started: list[_FakeServer] = []

    class _RecordingServer(_FakeServer):
        async def start(self):
            started.append(self)
            return await super().start()

    monkeypatch.setattr(pool, "OpencodeServer", _RecordingServer)
    monkeypatch.setattr(pool, "register_mcp", AsyncMock(side_effect=RuntimeError("mcp down")))
    pool._daemons.clear()

    with pytest.raises(RuntimeError, match="mcp down"):
        await pool.get_opencode_daemon("/a")

    assert len(started) == 1
    assert started[0].stopped == 1  # the orphan was reaped
    assert pool._daemons == {}  # nothing cached, so a retry starts clean
    pool._daemons.clear()


@pytest.mark.asyncio
@pytest.mark.usefixtures("clean_pool")
async def test_shutdown_all_stops_daemons_and_clears_pool():
    a: Any = await pool.get_opencode_daemon("/a")
    b: Any = await pool.get_opencode_daemon("/b")

    await pool.shutdown_opencode_daemons()

    assert a.stopped == 1
    assert b.stopped == 1
    assert pool._daemons == {}
    # Safe to call again on an empty pool.
    await pool.shutdown_opencode_daemons()
