"""Fold one run's share of the daemon's event stream into its spend, turns, and what to do next.

The watcher sees the root session's events and its subagents', sums their cost against the budget,
counts the root's steps against the turn cap, and decides when a session must be aborted or asked to
wrap up. It stays pure — it returns what to do and the event consumer does it — so the accounting is
testable offline. Only the root's text and tool parts are streamed to the optional live sink.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from agent.logger import logger
from agent.mcp import mcp_tool_ids
from agent.opencode.events import NO_ACTION, WatchAction, session_id_of
from agent.opencode.permission_config import BUILTIN_TOOL_PERMISSIONS
from agent.opencode.results import WrapUpReason

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

# A wrap-up goes out when this share of the budget is spent, or this many turns or seconds before
# the turn cap or the session timeout. The step after the message usually still has tools, and a
# rejected structured answer takes up to two retries, so the margins leave room for both.
_WRAP_UP_BUDGET_SHARE = 0.9
_WRAP_UP_TURN_MARGIN = 4
WRAP_UP_TIME_MARGIN_S = 180.0

# Sent to a session to make it answer with what it has; tools are switched off by the same message.
ROOT_WRAP_UP_TEXT = (
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
# map, so it stays available only because the wrap-up tools map leaves it out.
_ANSWER_TOOL = "StructuredOutput"


class SessionWatcher:
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


def wrap_up_tools() -> dict[str, bool]:
    """The per-message tools map that switches off every built-in and registered MCP tool."""
    return dict.fromkeys((*BUILTIN_TOOL_PERMISSIONS, *mcp_tool_ids()), False)


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
