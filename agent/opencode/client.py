"""Drive one ``opencode serve`` session over HTTP and map it onto a DriverResult.

Create a session on the daemon, register a watcher with the daemon's shared event consumer, and send
one prompt. The prompt response carries the structured output and final text; the watcher, fed the
session's events by the consumer, carries the per-step cost and tool calls, which are summed into
the run's cost and turns and streamed to an optional live sink. Watching live lets a budget or turn
cap abort the session mid-flight, and lets an opted-in run ask every running session to wrap up
first. Subagents' spend counts toward the run's budget. Permissions, tools, the system prompt, and
the structured-output schema are all set per session and per request.
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
from agent.logger import logger
from agent.mcp import McpServer, mcp_tool_ids
from agent.opencode.driver import (
    DriverResult,
    ParsedRun,
    SessionEnd,
    WrapUpReason,
)
from agent.opencode.events import NO_ACTION, WatchAction, session_id_of
from agent.opencode.permission_config import permission_config
from agent.opencode.providers import format_schema, provider_of
from agent.opencode.server import OpencodeServer
from agent.permissions import PermissionSpec

# Retries opencode makes internally to coax a schema-valid structured output before giving up.
_STRUCTURED_RETRY_COUNT = 2

# Short wait for the trailing step-finish plus idle marker to arrive after the prompt POST returns.
_IDLE_DRAIN_TIMEOUT = 3.0

# Final-message error names that mean the provider, not the run, failed: an API error opencode
# already retried, or a rejected key. A context overflow is left out — it is the prompt's size.
_PROVIDER_ERROR_NAMES = frozenset({"APIError", "ProviderAuthError"})

# How often to probe the daemon's health while a message is in flight, and how many consecutive
# misses to treat as wedged rather than a transient blip (e.g. the daemon briefly busy on a heavy
# tool call). A wedge is otherwise invisible until the full session timeout elapses, since the
# prompt POST blocks silently until the run finishes or that timeout fires.
_WEDGE_POLL_INTERVAL = 20.0
_WEDGE_POLL_MISSES = 2

# A wrap-up goes out when this share of the budget is spent, or this many turns or seconds before
# the turn cap or the session timeout. The step after the message usually still has tools, and a
# rejected structured answer takes up to two retries, so the margins leave room for both.
_WRAP_UP_BUDGET_SHARE = 0.9
_WRAP_UP_TURN_MARGIN = 4
_WRAP_UP_TIME_MARGIN_S = 180.0

# Sent to a session to make it answer with what it has; tools are switched off by the same message.
_ROOT_WRAP_UP_TEXT = (
    "This run has nearly reached its limits on spending, steps or time, so all your tools are now "
    "switched off. Do not try to call any tool. Submit your final answer now, based on the work you "
    "have done so far.\n\n"
    "Reports from any subagents you started are in their task results above. You can no longer "
    "check them yourself, so rely on what a report backs with specific evidence and leave out "
    "anything you doubt."
)
_CHILD_WRAP_UP_TEXT = (
    "This task has nearly reached its limits on spending, steps or time, so all your tools are now "
    "switched off. Do not try to call any tool. Reply now with your final report on the task you "
    "were given: what you found, the evidence for each point, and whether you verified it or only "
    "suspect it. Then list what you did not get to check."
)
# The tool a structured run answers through. opencode adds it before applying the per-message tools
# map, so it stays available only because the wrap-up tools map leaves it out. Recheck this list
# when bumping opencode: a built-in tool missing from it stays on during a wrap-up.
_ANSWER_TOOL = "StructuredOutput"
_BUILTIN_TOOLS: tuple[str, ...] = (
    "question",
    "bash",
    "read",
    "glob",
    "grep",
    "list",
    "edit",
    "write",
    "apply_patch",
    "task",
    "webfetch",
    "todowrite",
    "websearch",
    "skill",
)


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
            "schema": format_schema(response_model, provider),
            "retryCount": _STRUCTURED_RETRY_COUNT,
        }
    return body


def wrap_up_tools() -> dict[str, bool]:
    """The per-message tools map that switches off every built-in and registered MCP tool."""
    return dict.fromkeys((*_BUILTIN_TOOLS, *mcp_tool_ids()), False)


# Event stream envelope. The daemon-global SSE stream carries one JSON object per ``data:`` line; a
# part update wraps a step/tool/text part, a message update carries a message's role, and an idle
# event marks the run complete.
_PART_EVENT = "message.part.updated"
_MESSAGE_EVENT = "message.updated"
_IDLE_EVENT = "session.idle"

# A sink for live per-event updates as a run works: assistant-text deltas (``{"kind": "token",
# "text": …}``) and tool-call state transitions (``{"kind": "tool", "id", "name", "status",
# "input"?, "output"?}``). Invoked synchronously as events arrive so a caller can stream progress.
# Best-effort — a sink that raises is logged and swallowed so it can never disturb cap enforcement.
EventSink = Callable[[dict[str, Any]], None]


class _SessionWatcher:
    """Track one run's cost and turns from the event stream, flagging when a budget or
    turn cap is crossed and, if given a root wrap-up message, when to ask sessions to wrap up.
    ``wrap_up_at`` is checked on every event, not only at step ends, since a step can run long.

    The consumer feeds it the root session's events and its subagents'. Every step-finish adds its
    cost; only the root's count as turns, since subagents have their own step limit. Only the root's
    parts reach the live sink. The watcher stays pure — it returns what to do and
    the consumer does it — so the accounting is testable offline.
    """

    def __init__(
        self,
        session_id: str,
        max_budget_usd: float | None,
        max_turns: int | None,
        wrap_up_body: dict[str, Any] | None = None,
        wrap_up_at: float | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.session_id = session_id
        self._budget = max_budget_usd
        self._max_turns = max_turns
        self._wrap_up = wrap_up_body is not None
        self._wrap_up_at = wrap_up_at  # a time.monotonic() reading
        self.cost = 0.0
        self.subagent_cost = 0.0  # the part of cost spent by subagent steps
        self.turns = 0
        # Any step, a subagent's included: proof the stream delivered, even with no root turn yet.
        self.saw_steps = False
        self.budget_exceeded = False
        self.turns_exceeded = False
        self.wrap_up_reason: WrapUpReason | None = None
        self.cost_at_wrap_up = 0.0
        # Wrap-up message per session (a subagent's from its first user message), the sessions that
        # went idle, and those already told. An idle one must not get a message: it would restart its
        # loop and spend after its result was already taken. A session whose latest step ended its
        # loop is held back too, since its idle comes only after that step; if it does go on (a
        # rejected answer), it is told at its next step.
        self._bodies: dict[str, dict[str, Any]] = {session_id: wrap_up_body} if wrap_up_body else {}
        self._idle: set[str] = set()
        self._told: set[str] = set()
        self._finishing: set[str] = set()
        self._answer_steps: set[str] = set()  # message ids of steps that called the answer tool
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

    def process(self, event: dict[str, Any]) -> WatchAction:
        """Fold one event into the running state and say what the consumer should do about it."""
        etype = event.get("type")
        props = event.get("properties") or {}
        session_id = session_id_of(event)
        assert session_id is not None  # EventConsumer drops events without one
        is_root = session_id == self.session_id
        if etype == _MESSAGE_EVENT:
            info = props.get("info") or {}
            if is_root or info.get("role") != "user":
                return NO_ACTION
            return self._see_child(session_id, info)
        if etype == _IDLE_EVENT:
            self._idle.add(session_id)
            if is_root:
                self.idle.set()
            return NO_ACTION
        if etype != _PART_EVENT:
            return NO_ACTION
        part = props.get("part") or {}
        if is_root:
            self._stream_part(part)
        is_answer_call = part.get("type") == "tool" and part.get("tool") == _ANSWER_TOOL
        if is_answer_call and isinstance(part.get("messageID"), str):
            self._answer_steps.add(part["messageID"])
        if part.get("type") != "step-finish":
            return self._check_limits()
        self.saw_steps = True
        if is_root:
            self.turns += 1
        cost = part.get("cost")
        if isinstance(cost, (int, float)):
            self.cost += cost
            if not is_root:
                self.subagent_cost += cost
        if part.get("reason") != "tool-calls" or part.get("messageID") in self._answer_steps:
            self._finishing.add(session_id)
        else:
            self._finishing.discard(session_id)
        return self._check_limits()

    def _stream_part(self, part: dict[str, Any]) -> None:
        if part.get("type") in ("step-start", "step-finish") and isinstance(
            part.get("messageID"), str
        ):
            self._generating_messages.add(part["messageID"])
        # opencode re-sends a full snapshot of a part on every update; its stable id is what lets
        # the live stream send only what changed.
        part_id = part.get("id")
        if isinstance(part_id, str):
            self._emit_live(part_id, part)

    def _check_limits(self) -> WatchAction:
        if self.budget_exceeded or self.turns_exceeded:
            return NO_ACTION
        if self._budget is not None and self.cost >= self._budget:
            self.budget_exceeded = True
            return WatchAction(abort=True)
        if self._max_turns is not None and self.turns >= self._max_turns:
            self.turns_exceeded = True
            return WatchAction(abort=True)
        return self._wrap_up_if_due()

    def _wrap_up_if_due(self) -> WatchAction:
        if not self._wrap_up:
            return NO_ACTION
        if self.wrap_up_reason is None:
            if self._budget is not None and self.cost >= _WRAP_UP_BUDGET_SHARE * self._budget:
                self.wrap_up_reason = WrapUpReason.BUDGET
            elif (
                self._max_turns is not None and self.turns >= self._max_turns - _WRAP_UP_TURN_MARGIN
            ):
                self.wrap_up_reason = WrapUpReason.TURNS
            elif self._wrap_up_at is not None and time.monotonic() >= self._wrap_up_at:
                self.wrap_up_reason = WrapUpReason.TIME
            else:
                return NO_ACTION
            self.cost_at_wrap_up = self.cost
            logger.info(
                "wrap-up due for session %s (%s): spend $%.4f, %d turns, %d subagents running",
                self.session_id,
                self.wrap_up_reason,
                self.cost,
                self.turns,
                sum(1 for sid in self._bodies if sid != self.session_id and sid not in self._idle),
            )
        return self._wrap_up_prompts()

    def _wrap_up_prompts(self) -> WatchAction:
        due = [
            (sid, body)
            for sid, body in self._bodies.items()
            if sid not in self._idle and sid not in self._told and sid not in self._finishing
        ]
        self._told.update(sid for sid, _ in due)
        return WatchAction(prompts=tuple(due)) if due else NO_ACTION

    def _see_child(self, child_id: str, info: dict[str, Any]) -> WatchAction:
        """Record how to address a subagent from its first user message. A user message in one that
        went idle means the task tool continued it, so it runs again and is owed a fresh wrap-up.
        Either kind that appears after the wrap-up went out gets its message now. Our own wrap-up
        never goes to an idle session, so it cannot cause a restart that sends another."""
        if child_id not in self._bodies:
            body = _child_wrap_up_body(info)
            if body is None:
                return NO_ACTION
            self._bodies[child_id] = body
        elif child_id in self._idle:
            self._idle.discard(child_id)
            self._told.discard(child_id)
            self._finishing.discard(child_id)
        else:
            return NO_ACTION
        return NO_ACTION if self.wrap_up_reason is None else self._wrap_up_prompts()

    def _emit_live(self, part_id: str, part: dict[str, Any]) -> None:
        """Push a live update — a text delta or a tool-call transition — to the sink, if one is set.
        Best-effort: a failing sink is logged and swallowed so it can't disturb cap enforcement."""
        if self._event_sink is None:
            return
        event = self._live_event(part_id, part)
        if event is None:
            return
        try:
            self._event_sink(event)
        except Exception as e:
            logger.debug("live event sink raised; dropping update: %s", e)

    def _live_event(self, part_id: str, part: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize a part update into a live event, or None when it carries nothing new.

        A text part yields only its newly-appended tail — opencode resends the whole text on every
        update, so the delta against what was already emitted is sent — and only for an assistant
        message, so the prompt opencode echoes back as a text part is not restreamed as output. A
        tool part yields one event per status transition (pending → running → completed), carrying
        the call's input and, once it finishes, its output; the repeated same-status snapshots in
        between are dropped.
        """
        if part.get("type") == "text":
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
        if part.get("type") == "tool":
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


def _child_wrap_up_body(info: dict[str, Any]) -> dict[str, Any] | None:
    """The wrap-up message for a subagent, addressed like its own user message; None if that
    message does not say which agent and model it runs."""
    model = info.get("model") or {}
    agent, provider, model_id = info.get("agent"), model.get("providerID"), model.get("modelID")
    if not (isinstance(agent, str) and isinstance(provider, str) and isinstance(model_id, str)):
        return None
    body: dict[str, Any] = {
        "agent": agent,
        "model": {"providerID": provider, "modelID": model_id},
        "parts": [{"type": "text", "text": _CHILD_WRAP_UP_TEXT}],
        "tools": wrap_up_tools(),
    }
    if model.get("variant"):
        body["variant"] = model["variant"]
    return body


async def _wait_for_wedge(server: OpencodeServer) -> None:
    """Return once the daemon has failed health probes ``_WEDGE_POLL_MISSES`` times in a row."""
    misses = 0
    while True:
        await asyncio.sleep(_WEDGE_POLL_INTERVAL)
        if await server.is_responsive():
            misses = 0
        else:
            misses += 1
            if misses >= _WEDGE_POLL_MISSES:
                return


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
    session_id: str,
    fallback_cost: float | None = None,
) -> DriverResult:
    """Map the prompt response and the watched stream onto a DriverResult.

    The watcher supplies cost and turns whenever it saw any steps. If it saw none, the
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
    if watcher.saw_steps:
        cost, turns = watcher.cost, watcher.turns
        session_total = False
    else:
        session_total = fallback_cost is not None
        cost = fallback_cost if fallback_cost is not None else float(info.get("cost") or 0.0)
        turns = sum(1 for p in parts if p.get("type") == "step-finish")
    finish = info.get("finish")
    parsed = ParsedRun(
        text=text,
        structured=info.get("structured"),
        cost_usd=cost,
        num_turns=turns,
        error=json.dumps(error)[:500] if error else None,
        finish=finish if isinstance(finish, str) else None,
        cost_is_session_total=session_total,
        provider_failed=isinstance(error, dict) and error.get("name") in _PROVIDER_ERROR_NAMES,
    )
    return DriverResult(
        parsed=parsed,
        returncode=0,
        duration_ms=duration_ms,
        end=_session_end(watcher),
        subagent_cost_usd=watcher.subagent_cost,
        session_id=session_id,
        wrap_up_reason=watcher.wrap_up_reason,
    )


def _session_end(watcher: _SessionWatcher) -> SessionEnd:
    """Which cap cut a replying session short — budget first, since it is the one that spends."""
    if watcher.budget_exceeded:
        return SessionEnd.OVER_BUDGET
    if watcher.turns_exceeded:
        return SessionEnd.TURNS_EXHAUSTED
    return SessionEnd.REPLIED


def _cut_short_result(
    watcher: _SessionWatcher,
    duration_ms: int,
    session_id: str,
    *,
    end: SessionEnd,
    returncode: int | None,
    error: str | None = None,
) -> DriverResult:
    """A session that ended before its reply, reported with the spend and steps it did make."""
    return DriverResult(
        parsed=ParsedRun(error=error, cost_usd=watcher.cost, num_turns=watcher.turns),
        returncode=returncode,
        duration_ms=duration_ms,
        end=end,
        subagent_cost_usd=watcher.subagent_cost,
        session_id=session_id,
        wrap_up_reason=watcher.wrap_up_reason,
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
    wrap_up: bool = False,
    resume_session_id: str | None = None,
    event_sink: EventSink | None = None,
) -> DriverResult:
    """Run one prompt as a serve session and return the mapped result.

    Spend (subagents' included) and steps are tracked live off the daemon's shared event stream, via
    a watcher registered for this session; crossing the budget or turn cap aborts the session. With
    ``wrap_up``, nearing either cap or the timeout first sends every running session a message that
    disables its tools and asks for its answer, so the run can still deliver one. When
    ``event_sink`` is given it receives live text deltas and tool-call transitions as they arrive.
    Failures before the prompt is sent (caps unenforceable, session create) raise SessionStartError
    for the runner to retry; once the prompt is in flight, a timeout returns a timed-out or wedged
    result and a transport failure an error result — never a raise, so a charged run is not
    silently re-run.

    ``resume_session_id`` sends the prompt into an existing session instead of creating a new one —
    used to nudge a wedged run's session back to life on a fresh daemon. opencode's session store
    lives outside the daemon process, so the session (and its permission ruleset) survives a crash;
    only the daemon serving it needs replacing.
    """
    base_url = server.base_url
    client = server.client
    t0 = time.monotonic()
    # Stays None until the session is actually in play, so the cleanup below only ever touches a
    # session this call genuinely used.
    session_id: str | None = None
    skip_cleanup = False
    try:
        # Caps are enforced off the shared event stream, so a capped run must not proceed while it is
        # disconnected. It's pre-prompt, so raise for the runner to retry; uncapped runs proceed.
        if (max_budget_usd is not None or max_turns is not None) and not server.events_ready:
            raise SessionStartError("event stream unavailable; cap cannot be enforced")

        if resume_session_id is not None:
            session_id = resume_session_id
        else:
            ruleset = to_permission_ruleset(spec, tools, cwd)
            try:
                session = await client.post(f"{base_url}/session", json={"permission": ruleset})
                session.raise_for_status()
                session_id = str(session.json()["id"])
            except (httpx.HTTPError, RuntimeError) as e:
                # Pre-prompt and idempotent (RuntimeError covers the shared client being closed by a
                # concurrent daemon eviction): raise so the runner retries against a fresh daemon.
                raise SessionStartError(
                    f"session create failed: {type(e).__name__}: {str(e)[:300]}"
                ) from e

        body = build_message_body(
            opencode_model=opencode_model,
            system_prompt=system_prompt,
            user_message=user_message,
            variant=variant,
            response_model=response_model,
        )
        wrap_up_body = None
        if wrap_up:
            # Same model and format as the original, so the structured answer stays available.
            wrap_up_body = {
                **build_message_body(
                    opencode_model=opencode_model,
                    system_prompt=system_prompt,
                    user_message=_ROOT_WRAP_UP_TEXT,
                    variant=variant,
                    response_model=response_model,
                ),
                "tools": wrap_up_tools(),
            }

        # Register before sending the prompt (events for a session only start once the prompt runs),
        # so the shared consumer routes this session's events to the watcher without a gap.
        watcher = _SessionWatcher(
            session_id,
            max_budget_usd,
            max_turns,
            wrap_up_body,
            wrap_up_at=t0 + timeout - _WRAP_UP_TIME_MARGIN_S if wrap_up else None,
            event_sink=event_sink,
        )
        server.add_watcher(session_id, watcher)

        message_json: dict[str, Any] = {}
        post_task = asyncio.ensure_future(
            client.post(f"{base_url}/session/{session_id}/message", json=body, timeout=timeout)
        )
        wedge_task = asyncio.ensure_future(_wait_for_wedge(server))
        try:
            # Race the prompt against both the wall-clock cap (asyncio.wait's own timeout — httpx's
            # timeout only bounds the gap between response bytes, and the prompt POST blocks silently
            # until the run finishes) and a health-probe watchdog, so a daemon that stops responding
            # mid-run is caught within a few poll intervals instead of only once the full timeout fires.
            done, _ = await asyncio.wait(
                {post_task, wedge_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )

            # A 2xx reply means the run finished, so it is never discarded as a wedge — but an
            # exception is not a reply (a read timeout is a wedge symptom).
            post_ok = (
                post_task in done
                and post_task.exception() is None
                and post_task.result().is_success
            )
            wedged = not post_ok and (wedge_task in done or not (await server.is_responsive()))
            if wedged:
                # The daemon is gone, so cleaning the session up on it would only time out too.
                skip_cleanup = True
                return _cut_short_result(
                    watcher, _elapsed_ms(t0), session_id, end=SessionEnd.WEDGED, returncode=None
                )

            timed_out = post_task not in done or isinstance(
                post_task.exception(), httpx.TimeoutException
            )
            if timed_out:
                # Abort so the daemon stops spending, then report the timeout with the partial cost
                # the watcher gathered (an empty result would under-count overruns).
                with contextlib.suppress(httpx.HTTPError, RuntimeError):
                    await client.post(f"{base_url}/session/{session_id}/abort", json={})
                return _cut_short_result(
                    watcher, _elapsed_ms(t0), session_id, end=SessionEnd.TIMED_OUT, returncode=None
                )
            message = post_task.result()
            message.raise_for_status()
            message_json = message.json()
        except (httpx.HTTPError, RuntimeError, ValueError) as e:
            # In flight and possibly already charged, so terminal — never retried into a silent
            # re-run (ValueError covers a reply body that won't decode). But an intentional cap abort
            # can also surface here; if the watcher already classified the run, that stands, so fall
            # through to its result.
            if not (watcher.budget_exceeded or watcher.turns_exceeded):
                return _cut_short_result(
                    watcher,
                    _elapsed_ms(t0),
                    session_id,
                    end=SessionEnd.TRANSPORT_LOST,
                    returncode=1,
                    error=str(e)[:500],
                )
        finally:
            # Whichever of the two finished first, the other is now stale and must not hold the
            # message response open.
            for task in (post_task, wedge_task):
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

        # Let the watcher fold in the final step-finish (and the idle marker) before reading it.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(watcher.idle.wait(), _IDLE_DRAIN_TIMEOUT)
        if watcher.wrap_up_reason is not None:
            logger.info(
                "session %s ended after a wrap-up: spend $%.4f at wrap-up, $%.4f at the end",
                session_id,
                watcher.cost_at_wrap_up,
                watcher.cost,
            )
        # No steps seen means the stream delivered nothing, so the live cost sum is empty; read the
        # session's own aggregate instead of the final message's last-step-only cost.
        fallback_cost = None
        if not watcher.saw_steps:
            fallback_cost = await _session_total_cost(client, base_url, session_id)
        return _to_driver_result(
            message_json,
            watcher,
            duration_ms=_elapsed_ms(t0),
            session_id=session_id,
            fallback_cost=fallback_cost,
        )
    finally:
        # Deregister the watcher and delete the session so a long run's shared daemon doesn't
        # accumulate either. Best-effort: a failure here must never mask the run's result.
        if session_id is not None and not skip_cleanup:
            server.remove_watcher(session_id)
            with contextlib.suppress(Exception):
                await client.delete(f"{base_url}/session/{session_id}")


def _elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)
