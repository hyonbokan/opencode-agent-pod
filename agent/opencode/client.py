"""Drive one ``opencode serve`` session over HTTP and map it onto a DriverResult.

Create a session on the daemon, register a watcher with the daemon's shared event consumer, and send
one prompt. The prompt response carries the structured output and final text; the watcher, fed the
session's events by the consumer, carries the per-step cost, tokens, and tool calls, which are summed
into the run's cost and turns and rebuilt into the trace timeline. Watching live lets a budget or turn
cap abort the session mid-flight. Permissions, tools, the system prompt, and the structured-output
schema are all set per session and per request.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel

from agent.errors import SessionStartError
from agent.opencode.driver import DriverResult, ParsedRun, build_timeline
from agent.opencode.permission_config import permission_config
from agent.opencode.providers import provider_of
from agent.opencode.server import OpencodeServer
from agent.permissions import PermissionSpec
from core.tools.mcp import McpServer, mcp_tool_ids
from core.utils.logger import logger

# Retries opencode makes internally to coax a schema-valid structured output before giving up.
_STRUCTURED_RETRY_COUNT = 2

# Short wait for the trailing step-finish plus idle marker to arrive after the prompt POST returns.
_IDLE_DRAIN_TIMEOUT = 3.0


def mcp_local_config(server: McpServer) -> dict[str, Any]:
    """Build the ``POST /mcp`` config that registers one local MCP server on the daemon.

    The command comes straight from deployment config; the pod injects nothing, so a server that
    needs the workspace directory must carry it in its own declared argv.
    """
    return {"type": "local", "command": list(server.command), "enabled": True}


async def register_mcp(
    base_url: str,
    name: str,
    config: dict[str, Any],
    *,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    """Register an MCP server on the daemon, failing unless it reports connected.

    Registration is daemon-wide, so it happens once after the daemon is healthy, not per request.
    A server process that never spawns, or spawns and dies, still gets a 200 response — only the
    carried status says so. Left unchecked, every agent on the daemon would silently run without
    the server's tools, so a non-connected status raises here instead.
    """
    resp = await client.post(f"{base_url}/mcp", json={"name": name, "config": config})
    resp.raise_for_status()
    payload = resp.json()
    entry = payload.get(name) if isinstance(payload, dict) else None
    entry = entry if isinstance(entry, dict) else {}
    if entry.get("status") != "connected":
        raise RuntimeError(
            f"MCP server {name!r} failed to connect: {entry.get('error') or entry or payload}"
        )
    return payload


def to_permission_ruleset(spec: PermissionSpec, tools: list[str], cwd: str) -> list[dict[str, str]]:
    """Flatten the shared permission config into serve's per-session ruleset — a flat list of
    permission/pattern/action, insertion-ordered so the last matching rule wins.

    MCP tools are gated here too: each one the allow-list omits gets a ``*``-pattern deny, which
    opencode treats as disabling it. This must live in the session ruleset, not the per-message tools
    map — opencode turns that map into permissions and replaces the whole ruleset with it.
    """
    rules: list[dict[str, str]] = []
    for key, value in permission_config(spec, tools, cwd).items():
        if isinstance(value, str):
            rules.append({"permission": key, "pattern": "*", "action": value})
        else:
            for pattern, action in value.items():
                rules.append({"permission": key, "pattern": pattern, "action": action})
    allowed = set(tools)
    for tool_id in mcp_tool_ids():
        if tool_id not in allowed:
            rules.append({"permission": tool_id, "pattern": "*", "action": "deny"})
    return rules


def build_message_body(
    *,
    opencode_model: str,
    system_prompt: str | None,
    user_message: str,
    variant: str | None,
    response_model: type[BaseModel] | None,
) -> dict[str, Any]:
    """Assemble the message body for one prompt.

    No ``tools`` map is sent — opencode would treat it as permissions and replace the session
    ruleset, wiping the confinement. Tool gating lives in that ruleset instead.
    """
    provider = provider_of(opencode_model)
    model_id = opencode_model.split("/", 1)[-1]
    body: dict[str, Any] = {
        "model": {"providerID": provider, "modelID": model_id},
        "parts": [{"type": "text", "text": user_message}],
    }
    if system_prompt:
        body["system"] = system_prompt
    if variant:
        body["variant"] = variant
    if response_model is not None:
        body["format"] = {
            "type": "json_schema",
            "schema": response_model.model_json_schema(),
            "retryCount": _STRUCTURED_RETRY_COUNT,
        }
    return body


# Event stream envelope. The daemon-global SSE stream carries one JSON object per ``data:`` line; a
# part update wraps a step/tool/text part, and an idle event marks the run complete.
_PART_EVENT = "message.part.updated"
_IDLE_EVENT = "session.idle"
# serve names part types with hyphens; the timeline uses underscores. A type absent here is dropped.
_PART_TYPE_MAP: dict[str, str] = {
    "step-start": "step_start",
    "step-finish": "step_finish",
    "tool": "tool_use",
    "text": "text",
}

# A sink for live per-event updates as a run works: assistant-text deltas (``{"kind": "token",
# "text": …}``) and tool-call state transitions (``{"kind": "tool", "id", "name", "status",
# "input"?, "output"?}``). Invoked synchronously as events arrive so a caller can stream progress.
# Best-effort — a sink that raises is logged and swallowed so it can never disturb cap enforcement.
EventSink = Callable[[dict[str, Any]], None]


class _SessionWatcher:
    """Track one session's cost, turns, and timeline from the event stream, flagging when a budget or
    turn cap is crossed.

    Each step-finish adds its cost and one turn; the first crossing flags the run and signals an abort.
    The watcher stays pure — the caller does the aborting — so the accounting is testable offline.
    """

    def __init__(
        self,
        session_id: str,
        max_budget_usd: float | None,
        max_turns: int | None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.session_id = session_id
        self._budget = max_budget_usd
        self._max_turns = max_turns
        self.cost = 0.0
        self.turns = 0
        self.budget_exceeded = False
        self.turns_exceeded = False
        # Timeline part snapshots keyed by opencode's stable part id, latest kept: opencode re-sends a
        # full snapshot on every update (a bash tool re-emits its whole output per stdout chunk), so
        # keying by id collapses the hundreds of snapshots per part to one. First-seen order is kept.
        self._parts: dict[str, dict[str, Any]] = {}
        self._anon_seq = 0  # synthetic keys for a part with no id, so those are never merged
        self.idle = asyncio.Event()
        # Live-update sink plus the per-part state that keeps its stream free of the snapshot spam:
        # how many chars of each text part were already emitted (so only the new tail is sent) and the
        # last tool status emitted per part (so each transition, not every republished snapshot, fires).
        self._event_sink = event_sink
        self._emitted_text: dict[str, int] = {}
        self._emitted_tool_status: dict[str, str] = {}
        # Message ids that opened a generation step. opencode also re-emits the user's own prompt as a
        # text part (with no step), so text is streamed only for messages seen here — the assistant's,
        # not the echoed prompt.
        self._generating_messages: set[str] = set()

    @property
    def part_snapshots(self) -> list[dict[str, Any]]:
        """The latest snapshot of each part, in first-seen order, for the timeline."""
        return list(self._parts.values())

    def process(self, event: dict[str, Any]) -> bool:
        """Fold one event into the running state; return True when the run should be aborted now."""
        etype = event.get("type")
        if etype == _IDLE_EVENT:
            self.idle.set()
            return False
        if etype != _PART_EVENT:
            return False
        part = (event.get("properties") or {}).get("part") or {}
        run_type = _PART_TYPE_MAP.get(part.get("type", ""))
        if run_type is not None:
            part_id = part.get("id")
            if not isinstance(part_id, str):
                part_id = f"_anon{self._anon_seq}"
                self._anon_seq += 1
            # opencode's step parts carry no timestamp, so stamp arrival time for the trace timeline.
            self._parts[part_id] = {
                "type": run_type,
                "part": part,
                "timestamp": int(time.time() * 1000),
            }
            if run_type in ("step_start", "step_finish"):
                message_id = part.get("messageID")
                if isinstance(message_id, str):
                    self._generating_messages.add(message_id)
            self._emit_live(part_id, run_type, part)
        if part.get("type") != "step-finish":
            return False
        self.turns += 1
        cost = part.get("cost")
        if isinstance(cost, (int, float)):
            self.cost += cost
        if self.budget_exceeded or self.turns_exceeded:
            return False
        if self._budget is not None and self.cost >= self._budget:
            self.budget_exceeded = True
            return True
        if self._max_turns is not None and self.turns >= self._max_turns:
            self.turns_exceeded = True
            return True
        return False

    def _emit_live(self, part_id: str, run_type: str, part: dict[str, Any]) -> None:
        """Push a live update — a text delta or a tool-call transition — to the sink, if one is set.
        Best-effort: a failing sink is logged and swallowed so it can't disturb cap enforcement."""
        if self._event_sink is None:
            return
        event = self._live_event(part_id, run_type, part)
        if event is None:
            return
        try:
            self._event_sink(event)
        except Exception as e:
            logger.debug("live event sink raised; dropping update: %s", e)

    def _live_event(
        self, part_id: str, run_type: str, part: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Normalize a part update into a live event, or None when it carries nothing new.

        A text part yields only its newly-appended tail — opencode resends the whole text on every
        update, so the delta against what was already emitted is sent — and only for an assistant
        message, so the prompt opencode echoes back as a text part is not restreamed as output. A
        tool part yields one event per status transition (pending → running → completed), carrying
        the call's input and, once it finishes, its output; the repeated same-status snapshots in
        between are dropped.
        """
        if run_type == "text":
            if part.get("messageID") not in self._generating_messages:
                return None
            full = part.get("text")
            if not isinstance(full, str):
                return None
            seen = self._emitted_text.get(part_id, 0)
            if len(full) <= seen:
                return None
            self._emitted_text[part_id] = len(full)
            return {"kind": "token", "text": full[seen:]}
        if run_type == "tool_use":
            state = part.get("state")
            state = state if isinstance(state, dict) else {}
            status = state.get("status")
            if not isinstance(status, str) or self._emitted_tool_status.get(part_id) == status:
                return None
            self._emitted_tool_status[part_id] = status
            event: dict[str, Any] = {
                "kind": "tool",
                "id": part_id,
                "name": part.get("tool", "?"),
                "status": status,
            }
            if state.get("input") is not None:
                event["input"] = state["input"]
            if state.get("output") is not None:
                event["output"] = state["output"]
            return event
        return None


async def _session_total_cost(
    client: httpx.AsyncClient, base_url: str, session_id: str
) -> float | None:
    """The session's aggregate cost, or None if it can't be read. Best-effort — a failure here must
    never turn a completed run into an error."""
    with contextlib.suppress(httpx.HTTPError, RuntimeError, ValueError):
        resp = await client.get(f"{base_url}/session/{session_id}")
        resp.raise_for_status()
        data = resp.json()
        cost = data.get("cost") if isinstance(data, dict) else None
        if isinstance(cost, (int, float)):
            return float(cost)
    return None


def _to_driver_result(
    final_message: dict[str, Any],
    watcher: _SessionWatcher,
    duration_ms: int,
    fallback_cost: float | None = None,
) -> DriverResult:
    """Map the prompt response and the watched stream onto a DriverResult.

    The watcher supplies cost, turns, and timeline whenever it saw any steps. If it saw none, the
    session's own aggregate is used when available, since the final message reports only its last
    step's cost.
    """
    info = final_message.get("info") or {}
    parts = final_message.get("parts") or []
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    # An intentional abort (budget or turn cap) leaves opencode's truncated response marked with an
    # error. That is expected, not a failure, so drop it and let the cap flags classify the result.
    aborted = watcher.budget_exceeded or watcher.turns_exceeded
    error = None if aborted else info.get("error")
    if watcher.turns:
        cost, turns = watcher.cost, watcher.turns
    else:
        cost = fallback_cost if fallback_cost is not None else float(info.get("cost") or 0.0)
        turns = sum(1 for p in parts if p.get("type") == "step-finish")
    parsed = ParsedRun(
        text=text,
        structured=info.get("structured"),
        cost_usd=cost,
        num_turns=turns,
        error=json.dumps(error)[:500] if error else None,
    )
    return DriverResult(
        parsed=parsed,
        returncode=0,
        duration_ms=duration_ms,
        budget_exceeded=watcher.budget_exceeded,
        turns_exceeded=watcher.turns_exceeded,
        timeline=build_timeline(watcher.part_snapshots),
    )


async def run_session(
    server: OpencodeServer,
    *,
    opencode_model: str,
    system_prompt: str | None,
    user_message: str,
    variant: str | None,
    spec: PermissionSpec,
    tools: list[str],
    cwd: str,
    response_model: type[BaseModel] | None,
    timeout: float,
    max_budget_usd: float | None = None,
    max_turns: int | None = None,
    event_sink: EventSink | None = None,
) -> DriverResult:
    """Run one prompt as a serve session and return the mapped result.

    Spend and steps are tracked live off the daemon's shared event stream, via a watcher registered
    for this session; crossing the budget or turn cap aborts the session. When ``event_sink`` is
    given it receives live text deltas and tool-call transitions as they arrive, for streaming a
    run's progress. Failures before the prompt is sent (caps unenforceable, session create) raise
    SessionStartError for the runner to retry; once the prompt is in flight, a timeout returns a
    timed-out result and a transport failure an error result — never a raise, so a charged run is
    not silently re-run.
    """
    base_url = server.base_url
    client = server.client
    ruleset = to_permission_ruleset(spec, tools, cwd)
    t0 = time.monotonic()
    session_id: str | None = None
    try:
        # Caps are enforced off the shared event stream, so a capped run must not proceed while it is
        # disconnected. It's pre-prompt, so raise for the runner to retry; uncapped runs proceed.
        if (max_budget_usd is not None or max_turns is not None) and not server.events_ready:
            raise SessionStartError("event stream unavailable; cap cannot be enforced")

        try:
            session = await client.post(f"{base_url}/session", json={"permission": ruleset})
            session.raise_for_status()
            session_id = str(session.json()["id"])
        except (httpx.HTTPError, RuntimeError) as e:
            # Pre-prompt and idempotent (RuntimeError covers the shared client being closed by a
            # concurrent daemon eviction): raise so the runner retries against a fresh daemon.
            raise SessionStartError(f"session create failed: {str(e)[:300]}") from e

        # Register before sending the prompt (events for a session only start once the prompt runs),
        # so the shared consumer routes this session's events to the watcher without a gap.
        watcher = _SessionWatcher(session_id, max_budget_usd, max_turns, event_sink=event_sink)
        server.add_watcher(session_id, watcher)

        body = build_message_body(
            opencode_model=opencode_model,
            system_prompt=system_prompt,
            user_message=user_message,
            variant=variant,
            response_model=response_model,
        )
        message_json: dict[str, Any] = {}
        try:
            # asyncio.wait_for caps the whole exchange in wall-clock time. httpx's own timeout only
            # bounds the gap between response bytes, and the prompt POST blocks silently until the run
            # finishes, so without this a run that keeps the connection alive could outlast the budget.
            message = await asyncio.wait_for(
                client.post(f"{base_url}/session/{session_id}/message", json=body, timeout=timeout),
                timeout,
            )
            message.raise_for_status()
            message_json = message.json()
        except (TimeoutError, httpx.TimeoutException):
            # Wall-clock cap hit: abort so the daemon stops spending, then report the timeout with the
            # partial cost and timeline the watcher gathered (an empty result would under-count overruns).
            with contextlib.suppress(httpx.HTTPError, RuntimeError):
                await client.post(f"{base_url}/session/{session_id}/abort", json={})
            return DriverResult(
                parsed=ParsedRun(cost_usd=watcher.cost, num_turns=watcher.turns),
                returncode=None,
                duration_ms=_elapsed_ms(t0),
                timed_out=True,
                timeline=build_timeline(watcher.part_snapshots),
            )
        except (httpx.HTTPError, RuntimeError, ValueError) as e:
            # In flight and possibly already charged, so terminal — never retried into a silent
            # re-run (ValueError covers a reply body that won't decode). But an intentional cap abort
            # can also surface here; if the watcher already classified the run, that stands, so fall
            # through to its result.
            if not (watcher.budget_exceeded or watcher.turns_exceeded):
                return DriverResult(
                    parsed=ParsedRun(error=str(e)[:500]), returncode=1, duration_ms=_elapsed_ms(t0)
                )

        # Let the watcher fold in the final step-finish (and the idle marker) before reading it.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(watcher.idle.wait(), _IDLE_DRAIN_TIMEOUT)
        # No steps seen means the stream delivered nothing, so the live cost sum is empty; read the
        # session's own aggregate instead of the final message's last-step-only cost.
        fallback_cost = None
        if watcher.turns == 0:
            fallback_cost = await _session_total_cost(client, base_url, session_id)
        return _to_driver_result(
            message_json, watcher, duration_ms=_elapsed_ms(t0), fallback_cost=fallback_cost
        )
    finally:
        # Deregister the watcher and delete the session so a long run's shared daemon doesn't
        # accumulate either. Best-effort: a failure here must never mask the run's result.
        if session_id is not None:
            server.remove_watcher(session_id)
            with contextlib.suppress(Exception):
                await client.delete(f"{base_url}/session/{session_id}")


def _elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)
