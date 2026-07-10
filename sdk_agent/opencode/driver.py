"""Parse the opencode event stream into a timeline and reap process trees.

Holds the pure event parsing that reconstructs a per-step timeline (assistant text, tokens, cost, tool
calls) for trace recreation, the result dataclasses the client fills, and the process-tree reaping the
daemon manager uses to shut a daemon down.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import psutil

from core.utils.logger import logger

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


@dataclass
class TimelineTool:
    """One tool call inside a step, with its arguments, result, and wall-clock span (epoch ms)."""

    name: str
    input: Any = None
    output: Any = None
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass
class TimelineStep:
    """One model generation — the events sharing a ``messageID`` between a step_start and its
    step_finish — carrying the step's timing, token breakdown, cost, assistant text, and tools."""

    message_id: str
    start_ms: int | None = None
    end_ms: int | None = None
    cost_usd: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)
    text: str = ""
    tools: list[TimelineTool] = field(default_factory=list)


def _normalize_tokens(tokens: dict[str, Any]) -> dict[str, int]:
    """Flatten opencode's token breakdown (nested ``cache.{read,write}``) into the flat keys a
    Langfuse usage detail expects, dropping anything non-numeric."""
    cache = tokens.get("cache")
    cache = cache if isinstance(cache, dict) else {}
    flat = {
        "input": tokens.get("input"),
        "output": tokens.get("output"),
        "reasoning": tokens.get("reasoning"),
        "cache_read": cache.get("read"),
        "cache_write": cache.get("write"),
        "total": tokens.get("total"),
    }
    return {k: int(v) for k, v in flat.items() if isinstance(v, (int, float))}


def build_timeline(events: list[dict[str, Any]]) -> list[TimelineStep]:
    """Reconstruct the per-step timeline from decoded event dicts, for trace recreation.

    Events are grouped into steps by their ``messageID`` (a step_start opens the step, its
    step_finish closes it with tokens and cost, and the text and tool_use events in between belong
    to it). Steps keep the order in which they first appear.
    """
    steps: dict[str, TimelineStep] = {}
    order: list[str] = []

    def _step(message_id: str) -> TimelineStep:
        if message_id not in steps:
            steps[message_id] = TimelineStep(message_id=message_id)
            order.append(message_id)
        return steps[message_id]

    for ev in events:
        if not isinstance(ev, dict):
            continue
        part = ev.get("part", {}) if isinstance(ev.get("part"), dict) else {}
        message_id = part.get("messageID")
        if not isinstance(message_id, str):
            continue
        etype = ev.get("type")
        ts = ev.get("timestamp")
        if etype == "step_start":
            step = _step(message_id)
            if isinstance(ts, (int, float)):
                step.start_ms = int(ts)
        elif etype == "step_finish":
            step = _step(message_id)
            if isinstance(ts, (int, float)):
                step.end_ms = int(ts)
            cost = part.get("cost")
            if isinstance(cost, (int, float)):
                step.cost_usd = float(cost)
            tokens = part.get("tokens")
            if isinstance(tokens, dict):
                step.tokens = _normalize_tokens(tokens)
        elif etype == "text":
            _step(message_id).text += part.get("text", "")
        elif etype == "tool_use":
            state = part.get("state")
            state = state if isinstance(state, dict) else {}
            timing = state.get("time")
            timing = timing if isinstance(timing, dict) else {}
            start_ms = timing.get("start")
            end_ms = timing.get("end")
            _step(message_id).tools.append(
                TimelineTool(
                    name=part.get("tool", "?"),
                    input=state.get("input"),
                    output=state.get("output"),
                    start_ms=int(start_ms) if isinstance(start_ms, (int, float)) else None,
                    end_ms=int(end_ms) if isinstance(end_ms, (int, float)) else None,
                )
            )

    return [steps[mid] for mid in order]


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


@dataclass
class DriverResult:
    parsed: ParsedRun
    returncode: int | None
    duration_ms: int
    timed_out: bool = False
    budget_exceeded: bool = False
    turns_exceeded: bool = False
    timeline: list[TimelineStep] = field(default_factory=list)
