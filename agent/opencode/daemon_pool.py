"""A process-wide pool of ``opencode serve`` daemons, one per project directory.

A daemon is pinned to the directory it launches in, so a run's agents — all in the same project
directory — share one daemon, started on first use with the MCP server registered once. Daemons
live for the process lifetime and are stopped together at shutdown; a pod runs one project, so in
practice this holds a single daemon.

The pool exists because the runner is built per agent while the daemon must be shared: keying on the
working directory lets any runner reach the right daemon without threading it through the runner.
"""

from __future__ import annotations

import asyncio
import contextlib

from agent.opencode.client import mcp_local_config, register_mcp
from agent.opencode.server import OpencodeServer
from core.tools.mcp import MCP_SERVER_NAME, is_mcp_configured
from core.utils.logger import logger

_daemons: dict[str, OpencodeServer] = {}
# asyncio.Lock binds to the running loop on first acquire (not at construction), so a module-level
# instance is safe here.
_lock = asyncio.Lock()


async def get_opencode_daemon(cwd: str) -> OpencodeServer:
    """Return the serve daemon for a project directory, starting it on first use.

    Creation is serialized so concurrent first-use calls share one daemon rather than racing to launch
    several. A pooled daemon whose process has since died is evicted and relaunched so a mid-run crash
    does not poison every later agent. An MCP server is registered at creation only when one is
    configured; whether an agent sees its tools is decided per request, so registering whenever a
    server exists is safe and keeps startup uniform. With none configured (the placeholder registry),
    registration is skipped rather than failing to connect a server that does not exist.
    """
    async with _lock:
        daemon = _daemons.get(cwd)
        if daemon is not None and not daemon.is_alive:
            logger.warning("opencode serve daemon for %s is dead; relaunching", cwd)
            with contextlib.suppress(Exception):
                await daemon.stop()
            del _daemons[cwd]
            daemon = None
        if daemon is None:
            daemon = OpencodeServer(cwd)
            await daemon.start()
            if is_mcp_configured():
                # The daemon is running but not yet pooled, so a registration failure would leave an
                # untracked process shutdown can't reap; stop it before re-raising.
                try:
                    await register_mcp(
                        daemon.base_url,
                        MCP_SERVER_NAME,
                        mcp_local_config(cwd),
                        client=daemon.client,
                    )
                except Exception:
                    with contextlib.suppress(Exception):
                        await daemon.stop()
                    raise
            _daemons[cwd] = daemon
            logger.info("opencode serve daemon ready for %s at %s", cwd, daemon.base_url)
        return daemon


async def shutdown_opencode_daemons() -> None:
    """Stop every pooled opencode serve daemon and clear the pool. Safe to call when it is empty."""
    async with _lock:
        for daemon in _daemons.values():
            with contextlib.suppress(Exception):
                await daemon.stop()
        _daemons.clear()
