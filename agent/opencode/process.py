"""Reap an ``opencode serve`` daemon's process tree and drain its output."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections import deque

import psutil

from agent.logger import logger

# The daemon's output is drained in fixed-size chunks (never by line, so a newline-free flood cannot
# overrun the reader) into a bounded ring buffer that keeps only the tail for diagnostics.
_OUTPUT_CHUNK = 8192


async def terminate_tree(
    proc: asyncio.subprocess.Process,
    grace: float = 5.0,
    *,
    reap_marker: tuple[str, str] | None = None,
) -> None:
    """Reap the daemon and every process it spawned, in layers that each cover a case the others miss.

    While the daemon is still alive its process group is SIGTERMed (giving opencode a grace period to
    reap its own children) then SIGKILLed, which also clears any child still sharing the group; the
    descendants captured by parent pid before killing catch children opencode detached into their own
    session. These pid-based kills run *only* while the daemon lives — once it has exited, its pid can
    be recycled by an unrelated process, and signalling a recycled pid or pgid could kill an innocent
    group. For the already-dead daemon (and as a backstop for detached children the group/ppid layers
    miss), the marker sweep SIGKILLs any process still carrying the daemon's inherited environment
    tag — the one reap signal that outlives both the crash and pid recycling.
    """
    if proc.returncode is None:
        pid = proc.pid
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            pgid = pid  # leader just exited; its pgid == pid and stays valid while members live
        descendants = await _descendants(pid)  # capture before killing — pids vanish as they die
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGTERM)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=grace)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()
        for cpid in descendants:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(cpid, signal.SIGKILL)
    if reap_marker is not None:
        marker, value = reap_marker
        try:
            killed = await asyncio.to_thread(
                _reap_by_env_marker, marker, value, exclude=os.getpid()
            )
        except Exception as e:  # a reap sweep failure must not abort daemon teardown
            logger.warning("env-marker reap sweep failed: %s", e)
            killed = []
        if killed:
            logger.warning(
                "reaped %d orphaned opencode tool process(es) by env marker: %s",
                len(killed),
                killed,
            )


async def drain_output(stream: asyncio.StreamReader | None, sink: deque[str]) -> None:
    """Consume the subprocess output as it streams so a chatty child can never fill the pipe and
    stall the read loop, keeping only the tail (the ring buffer's bound) for diagnostics. Reads
    fixed-size chunks rather than lines so a newline-free flood cannot overrun the reader. The daemon
    is launched with stderr merged into stdout, so this drains the one combined stream.
    """
    if stream is None:
        return
    with contextlib.suppress(Exception):
        while True:
            chunk = await stream.read(_OUTPUT_CHUNK)
            if not chunk:
                return
            sink.append(chunk.decode(errors="replace"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _descendants(root_pid: int) -> list[int]:
    """Live descendant pids of a process, via a ``ps`` ppid walk. ``ps`` runs as an async subprocess
    so the walk never blocks the event loop."""
    proc = await asyncio.create_subprocess_exec(
        "ps",
        "-A",
        "-o",
        "pid=,ppid=",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    out = stdout.decode(errors="replace")
    by_parent: dict[int, list[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            by_parent.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen, stack = [], [root_pid]
    while stack:
        for child in by_parent.get(stack.pop(), []):
            if child not in seen:
                seen.append(child)
                stack.append(child)
    return seen


def _reap_by_env_marker(marker: str, value: str, *, exclude: int) -> list[int]:
    """SIGKILL every live process whose environment carries ``marker=value``, skipping ``exclude``.

    The daemon tags its own environment with a value unique to it, and every process it spawns
    inherits that tag at exec time. The tag survives a child detaching into its own session,
    reparenting to init when the daemon dies, and changing directory — none of which a process-group
    or parent-pid walk can follow. Reading it back from each process's environment is therefore the
    one reap signal that still finds tool children orphaned by a crashed daemon.
    """
    killed: list[int] = []
    for proc in psutil.process_iter():
        if proc.pid == exclude:
            continue
        try:
            if proc.environ().get(marker) == value:
                proc.kill()
                killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return killed
