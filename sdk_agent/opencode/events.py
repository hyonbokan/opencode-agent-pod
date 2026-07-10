"""One shared event consumer per ``opencode serve`` daemon.

opencode's ``/event`` stream is daemon-global, so the daemon reads it once and routes each decoded
event to the registered session's watcher — rather than every run opening its own stream and
re-decoding the whole firehose. It aborts a session when its watcher flags a cap crossing, and
reconnects with backoff if the stream drops.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, Protocol

import httpx

from core.utils.logger import logger

# The stream normally opens in well under a second; if it can't within this, the daemon start that
# awaits it fails and is retried/relaunched rather than serving runs whose caps can't be enforced.
STREAM_READY_TIMEOUT = 5.0
_RECONNECT_BACKOFF_START = 0.5
_RECONNECT_BACKOFF_MAX = 5.0


class Watcher(Protocol):
    """The slice of a session watcher the consumer drives: fold one event in, return True to abort."""

    def process(self, event: dict[str, Any]) -> bool: ...


def _session_id_of(event: dict[str, Any]) -> str | None:
    """The sessionID an event belongs to, used to route the daemon-global stream to one run."""
    props = event.get("properties") or {}
    if isinstance(props.get("sessionID"), str):
        return props["sessionID"]
    part = props.get("part")
    return part.get("sessionID") if isinstance(part, dict) else None


class EventConsumer:
    """The daemon's single ``/event`` reader, dispatching decoded events to per-session watchers.

    A run registers its watcher with :meth:`add_watcher` before sending its prompt and drops it with
    :meth:`remove_watcher` when done. Registration/removal never awaits, so under the single-threaded
    loop no lock is needed.
    """

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url
        self._client = client
        self._watchers: dict[str, Watcher] = {}
        self._task: asyncio.Task[None] | None = None
        # Hold abort tasks so the loop's weak task references don't let them be garbage-collected
        # mid-flight; the done callback drops each once it settles.
        self._abort_tasks: set[asyncio.Task[None]] = set()
        self._connected = asyncio.Event()
        self._stopping = False

    @property
    def connected(self) -> bool:
        """Whether the stream is currently open. Cleared during a reconnect gap so a capped run can
        decline to proceed while cap enforcement is momentarily unavailable."""
        return self._connected.is_set()

    def add_watcher(self, session_id: str, watcher: Watcher) -> None:
        self._watchers[session_id] = watcher

    def remove_watcher(self, session_id: str) -> None:
        self._watchers.pop(session_id, None)

    async def start(self, ready_timeout: float = STREAM_READY_TIMEOUT) -> None:
        """Open the stream and start dispatching, returning once it is connected. Raises if the stream
        does not open within ``ready_timeout`` so the daemon start awaiting this can fail fast."""
        self._task = asyncio.create_task(self._run())
        await asyncio.wait_for(self._connected.wait(), ready_timeout)

    async def stop(self) -> None:
        """Stop dispatching and tear down the stream task. Safe to call more than once."""
        self._stopping = True
        self._connected.clear()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None

    async def _run(self) -> None:
        backoff = _RECONNECT_BACKOFF_START
        while not self._stopping:
            try:
                async with self._client.stream(
                    "GET", f"{self._base_url}/event", timeout=None
                ) as resp:
                    # A non-2xx stream delivers no events; fail here so the daemon start retries
                    # rather than serving runs whose caps can't be enforced.
                    resp.raise_for_status()
                    self._connected.set()
                    backoff = _RECONNECT_BACKOFF_START
                    async for line in resp.aiter_lines():
                        self._dispatch(line)
            except (httpx.HTTPError, RuntimeError):
                pass  # fall through to reconnect
            finally:
                self._connected.clear()
            if self._stopping:
                return
            logger.warning("opencode event stream dropped; reconnecting in %.1fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)

    def _dispatch(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            return
        # A non-object payload (e.g. a bare "ping") has no session to route to and would crash the
        # shared reader; skip it.
        if not isinstance(event, dict):
            return
        session_id = _session_id_of(event)
        if session_id is None:
            return
        watcher = self._watchers.get(session_id)
        # process() flags a cap crossing at most once (the watcher latches), so at most one abort task
        # per session; fire it off the dispatch loop so a slow abort POST can't stall other sessions.
        if watcher is not None and watcher.process(event):
            task = asyncio.create_task(self._abort(session_id))
            self._abort_tasks.add(task)
            task.add_done_callback(self._abort_tasks.discard)

    async def _abort(self, session_id: str) -> None:
        with contextlib.suppress(Exception):  # best-effort; must never crash the consumer
            await self._client.post(f"{self._base_url}/session/{session_id}/abort", json={})
