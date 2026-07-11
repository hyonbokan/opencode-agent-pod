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
from core.tools.mcp import registered_servers
from core.utils.logger import logger

_daemons: dict[str, OpencodeServer] = {}
# asyncio.Lock binds to the running loop on first acquire (not at construction), so a module-level
# instance is safe here.
_lock = asyncio.Lock()


async def get_opencode_daemon(cwd: str) -> OpencodeServer:
    """Return the serve daemon for a project directory, starting it on first use.

    Creation is serialized so concurrent first-use calls share one daemon rather than racing to launch
    several. A pooled daemon whose process has since died is evicted and relaunched so a mid-run crash
    does not poison every later agent. Every deployment-configured MCP server is registered at
    creation; whether an agent sees a server's tools is decided per request, so registering them all
    is safe and keeps startup uniform. With none configured the registration loop is a no-op.
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
            # The daemon is running but not yet pooled, so a registration failure would leave an
            # untracked process shutdown can't reap; stop it before re-raising. No servers → no-op.
            try:
                for server in registered_servers():
                    await register_mcp(
                        daemon.base_url,
                        server.name,
                        mcp_local_config(server),
                        client=daemon.client,
                    )
            except Exception:
                with contextlib.suppress(Exception):
                    await daemon.stop()
                raise
            _daemons[cwd] = daemon
            logger.info("opencode serve daemon ready for %s at %s", cwd, daemon.base_url)
        return daemon


async def stop_opencode_daemon(cwd: str) -> None:
    """Stop and evict the daemon for one working directory, if pooled. Safe when none is pooled.

    The pool otherwise holds a daemon for the process lifetime, which fits a single long-lived
    project directory. A caller that gives each run its own ephemeral directory (the pod service)
    uses this to reap that run's daemon before its directory is torn down, so daemons don't
    accumulate one per request. The daemon is popped under the lock but stopped outside it, so
    process teardown doesn't serialize new daemon starts.
    """
    async with _lock:
        daemon = _daemons.pop(cwd, None)
    if daemon is not None:
        with contextlib.suppress(Exception):
            await daemon.stop()


async def shutdown_opencode_daemons() -> None:
    """Stop every pooled opencode serve daemon and clear the pool. Safe to call when it is empty."""
    async with _lock:
        for daemon in _daemons.values():
            with contextlib.suppress(Exception):
                await daemon.stop()
        _daemons.clear()
