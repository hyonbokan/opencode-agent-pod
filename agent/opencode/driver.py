"""The result dataclasses a session fills, and the process-tree reaping used to shut a daemon down."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import psutil

from agent.logger import logger

# The daemon's output is drained in fixed-size chunks (never by line, so a newline-free flood cannot
# overrun the reader) into a bounded ring buffer that keeps only the tail for diagnostics.
_OUTPUT_CHUNK = 8192


@dataclass
class ParsedRun:
    """The information the runner needs, reconstructed from the event stream."""

    text: str = ""
    structured: Any | None = None  # structured output from the run (validated by the caller)
    cost_usd: float = 0.0
    num_turns: int = 0
    error: str | None = None
    finish: str | None = None  # the final message's finish reason; "stop" means the model concluded
    # Whether cost_usd is the session's whole spend rather than this run's own. Set only when the
    # live stream gave nothing and the session's aggregate was read instead — on a resumed session
    # that aggregate already covers the earlier attempts, so the caller must not add them again.
    cost_is_session_total: bool = False
    # The provider failed the call after opencode's own retries, so another model could still answer
    provider_failed: bool = False


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


async def _drain_output(stream: asyncio.StreamReader | None, sink: deque[str]) -> None:
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


class SessionEnd(StrEnum):
    """How one session stopped — the single fact a result carries about its ending."""

    REPLIED = "replied"  # the daemon returned a final message, however the model itself finished
    TIMED_OUT = "timed_out"
    # The daemon stopped answering health probes mid-run, distinct from a plain wall-clock timeout
    # where it is still responsive. Callers can recycle the daemon and retry on this signal alone.
    WEDGED = "wedged"
    OVER_BUDGET = "over_budget"
    TURNS_EXHAUSTED = "turns_exhausted"
    TRANSPORT_LOST = "transport_lost"  # the daemon went away under the session


class WrapUpReason(StrEnum):
    """Which limit made a run ask its sessions to wrap up."""

    BUDGET = "budget"
    TURNS = "turns"
    TIME = "time"


@dataclass
class DriverResult:
    parsed: ParsedRun
    returncode: int | None
    duration_ms: int
    end: SessionEnd = SessionEnd.REPLIED
    subagent_cost_usd: float = 0.0  # the part of the run's spend made by subagent steps
    # The opencode session this run used. A wedged run's session outlives the dead daemon (opencode's
    # store is daemon-independent), so the runner can continue it on a fresh daemon.
    session_id: str | None = None
    wrap_up_reason: WrapUpReason | None = None
