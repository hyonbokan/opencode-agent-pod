"""One shared event consumer per ``opencode serve`` daemon.

opencode's ``/event`` stream is daemon-global, so the daemon reads it once and routes each decoded
event to the registered session's watcher — rather than every run opening its own stream and
re-decoding the whole firehose. Events of a subagent session are routed to the run that started it.
It carries out what a watcher asks for (an abort, or messages to running sessions), and reconnects
with backoff if the stream drops.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from agent.logger import logger

# The stream normally opens in well under a second; if it can't within this, the daemon start that
# awaits it fails and is retried/relaunched rather than serving runs whose caps can't be enforced.
STREAM_READY_TIMEOUT = 5.0
_RECONNECT_BACKOFF_START = 0.5
_RECONNECT_BACKOFF_MAX = 5.0
_CREATED_EVENT = "session.created"


@dataclass(frozen=True)
class WatchAction:
    """What a watcher asks the consumer to do after an event: abort its session, and/or post these
    ``(session_id, body)`` messages to ``prompt_async``."""

    abort: bool = False
    prompts: tuple[tuple[str, dict[str, Any]], ...] = ()


NO_ACTION = WatchAction()


class Watcher(Protocol):
    """The slice of a session watcher the consumer drives: fold one event in, return what to do."""

    def process(self, event: dict[str, Any]) -> WatchAction:
        raise NotImplementedError


def session_id_of(event: dict[str, Any]) -> str | None:
    """The sessionID an event belongs to, used to route the daemon-global stream to one run."""
    props = event.get("properties") or {}
    if isinstance(props.get("sessionID"), str):
        return props["sessionID"]
    part = props.get("part")
    return part.get("sessionID") if isinstance(part, dict) else None


async def abort_session(client: httpx.AsyncClient, base_url: str, session_id: str) -> None:
    """Ask the daemon to stop a session; best-effort, so a failure is logged and never raised."""
    try:
        resp = await client.post(f"{base_url}/session/{session_id}/abort", json={})
        resp.raise_for_status()
    except Exception as e:
        logger.warning("opencode abort of session %s failed: %s", session_id, e)


class EventConsumer:
    """The daemon's single ``/event`` reader, dispatching decoded events to per-session watchers.

    A run registers its watcher with :meth:`add_watcher` before sending its prompt and drops it with
    :meth:`remove_watcher` when done. A session created under a registered one (a subagent, at any
    depth) is aliased to that run's watcher. Registration/removal never awaits, so under the
    single-threaded loop no lock is needed.
    """

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url
        self._client = client
        self._watchers: dict[str, Watcher] = {}
        self._subagent_roots: dict[str, str] = {}  # subagent session id -> its run's session id
        self._task: asyncio.Task[None] | None = None
        # Hold abort and message tasks so the loop's weak task references don't let them be
        # garbage-collected mid-flight; the done callback drops each once it settles.
        self._action_tasks: set[asyncio.Task[None]] = set()
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
        self._subagent_roots = {k: v for k, v in self._subagent_roots.items() if v != session_id}

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
        # This reader enforces every run's caps, so an event that can't be routed or folded in is
        # dropped loudly rather than allowed to end the stream.
        try:
            self._route(event)
        except Exception as e:
            logger.warning("opencode event dropped: %s: %s", type(e).__name__, e)

    def _route(self, event: dict[str, Any]) -> None:
        if event.get("type") == _CREATED_EVENT:
            self._alias_subagent(event)
            return
        session_id = session_id_of(event)
        if session_id is None:
            return
        run_id = self._subagent_roots.get(session_id, session_id)
        watcher = self._watchers.get(run_id)
        if watcher is None:
            return
        self._apply_action(run_id, watcher.process(event))

    def _apply_action(self, run_id: str, action: WatchAction) -> None:
        # Fire off the dispatch loop so a slow POST can't stall other sessions. A watcher asks for
        # an abort at most once (it latches), so there is at most one abort task per session.
        if action.abort:
            self._spawn(abort_session(self._client, self._base_url, run_id))
        for target, body in action.prompts:
            self._spawn(self._prompt(target, body))

    def _alias_subagent(self, event: dict[str, Any]) -> None:
        info = (event.get("properties") or {}).get("info") or {}
        parent = info.get("parentID")
        if not isinstance(parent, str) or not isinstance(info.get("id"), str):
            return
        run_id = parent if parent in self._watchers else self._subagent_roots.get(parent)
        if run_id is not None:
            self._subagent_roots[info["id"]] = run_id

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._action_tasks.add(task)
        task.add_done_callback(self._action_tasks.discard)

    async def _prompt(self, session_id: str, body: dict[str, Any]) -> None:
        try:
            resp = await self._client.post(
                f"{self._base_url}/session/{session_id}/prompt_async", json=body
            )
            resp.raise_for_status()
        except Exception as e:  # best-effort, like an abort
            logger.warning("opencode message to session %s failed: %s", session_id, e)
