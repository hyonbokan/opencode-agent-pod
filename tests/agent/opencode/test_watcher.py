"""Tests for the session watcher: spend and turn accounting, cap aborts, subagent routing, wrap-up
messages, and the live event sink. The watcher is pure, so events are fed to it directly."""

from __future__ import annotations

import json
import time

import pytest

from agent.opencode.watcher import (
    _CHILD_WRAP_UP_TEXT,
    ROOT_WRAP_UP_TEXT,
    SessionWatcher,
    wrap_up_tools,
)

_DEMO_SERVERS_JSON = json.dumps(
    [{"name": "demo", "command": ["python", "srv.py"], "tools": ["lookup", "fetch"]}]
)


def _part_event(session_id: str, part: dict) -> dict:
    return {"type": "message.part.updated", "properties": {"sessionID": session_id, "part": part}}


def _idle_event(session_id: str) -> dict:
    return {"type": "session.idle", "properties": {"sessionID": session_id}}


def _pev(part: dict) -> dict:
    return {"type": "message.part.updated", "properties": {"sessionID": "s", "part": part}}


def test_session_watcher_accumulates_cost_and_caps_budget():
    w = SessionWatcher("s", max_budget_usd=0.004, max_turns=None)
    assert not w.process(_pev({"type": "step-finish", "cost": 0.003, "messageID": "m1"})).abort
    assert w.cost == pytest.approx(0.003)
    assert w.turns == 1
    # The second step crosses the budget: it signals an abort and flags the run over-budget.
    assert w.process(_pev({"type": "step-finish", "cost": 0.002, "messageID": "m2"})).abort
    assert w.budget_exceeded is True
    assert w.turns == 2


def test_session_watcher_caps_turns_and_records_idle_and_tools():
    w = SessionWatcher("s", max_budget_usd=None, max_turns=2)
    assert not w.process(_pev({"type": "step-finish", "cost": 0.0, "messageID": "m1"})).abort
    assert w.process(_pev({"type": "step-finish", "cost": 0.0, "messageID": "m2"})).abort
    assert w.turns_exceeded is True

    # A tool part is not a turn; idle sets the completion flag.
    w2 = SessionWatcher("s", max_budget_usd=None, max_turns=None)
    w2.process(_pev({"type": "tool", "tool": "read", "messageID": "m1"}))
    assert w2.turns == 0
    w2.process({"type": "session.idle", "properties": {"sessionID": "s"}})
    assert w2.idle.is_set()


def _step(session_id: str, cost: float, reason: str = "tool-calls", message: str = "m") -> dict:
    part = {"type": "step-finish", "cost": cost, "reason": reason, "messageID": message}
    return {"type": "message.part.updated", "properties": {"sessionID": session_id, "part": part}}


def _answer_call(message: str) -> dict:
    part = {"type": "tool", "tool": "StructuredOutput", "id": f"t-{message}", "messageID": message}
    return {"type": "message.part.updated", "properties": {"sessionID": "s", "part": part}}


def _user_message(session_id: str, agent: str = "sweeper", variant: str | None = "high") -> dict:
    model = {"providerID": "xai", "modelID": "grok", **({"variant": variant} if variant else {})}
    info = {"id": f"u-{session_id}", "role": "user", "agent": agent, "model": model}
    return {"type": "message.updated", "properties": {"sessionID": session_id, "info": info}}


_ROOT_BODY = {"parts": [{"type": "text", "text": ROOT_WRAP_UP_TEXT}], "tools": wrap_up_tools()}


def _wrap_up_watcher(budget: float | None = 1.0, turns: int | None = None) -> SessionWatcher:
    return SessionWatcher("s", budget, turns, _ROOT_BODY)


def test_session_watcher_counts_subagent_spend_but_not_its_turns():
    w = SessionWatcher("s", max_budget_usd=1.0, max_turns=3)
    w.process(_step("s", 0.3))
    assert not w.process(_step("child", 0.5)).abort
    assert w.cost == pytest.approx(0.8)
    assert w.subagent_cost == pytest.approx(0.5)  # reported apart from the root's spend
    assert w.turns == 1
    # The subagent's spend alone can cross the budget and abort the run.
    action = w.process(_step("child", 0.3))
    assert action.abort and w.budget_exceeded
    assert w.turns == 1 and not w.turns_exceeded


def test_a_subagent_going_idle_does_not_end_the_run():
    w = SessionWatcher("s", max_budget_usd=None, max_turns=None)
    w.process(_user_message("child"))
    w.process(
        {
            "type": "message.part.updated",
            "properties": {"sessionID": "child", "part": {"type": "text", "text": "x", "id": "p"}},
        }
    )
    w.process(_idle_event("child"))

    assert not w.idle.is_set()
    w.process(_idle_event("s"))
    assert w.idle.is_set()


def test_wrap_up_fires_once_when_the_budget_is_ninety_percent_spent():
    w = _wrap_up_watcher(budget=1.0)
    assert w.process(_step("s", 0.8)).prompts == ()

    action = w.process(_step("s", 0.15))  # 0.95
    assert not action.abort
    assert [sid for sid, _ in action.prompts] == ["s"]
    assert action.prompts[0][1] is _ROOT_BODY
    assert w.wrap_up_reason == "budget"
    assert w.cost_at_wrap_up == pytest.approx(0.95)

    assert w.process(_step("s", 0.01)).prompts == ()  # at most one wrap-up
    assert not w.process(_step("s", 0.03)).abort  # 0.99, still under the budget
    assert not w.budget_exceeded


def test_a_step_that_crosses_the_budget_aborts():
    w = _wrap_up_watcher(budget=1.0, turns=10)
    w.process(_step("s", 0.6))
    action = w.process(_step("child", 0.45))  # 0.6 to 1.05
    assert action.abort and w.budget_exceeded
    assert action.prompts == ()


def test_spend_well_past_the_budget_still_aborts_after_a_wrap_up():
    # Covers a session whose wrap-up message never landed and keeps spending.
    w = _wrap_up_watcher(budget=1.0)
    assert w.process(_step("s", 0.95)).prompts
    assert not w.process(_step("s", 0.04)).abort  # 0.99
    action = w.process(_step("s", 0.02))  # 1.01
    assert action.abort and w.budget_exceeded


def test_a_subagent_can_trigger_the_budget_wrap_up():
    w = _wrap_up_watcher(budget=1.0)
    w.process(_step("s", 0.1))
    assert [sid for sid, _ in w.process(_step("child", 0.85)).prompts] == ["s"]
    assert w.wrap_up_reason == "budget"


def test_wrap_up_fires_four_turns_before_the_turn_cap():
    w = _wrap_up_watcher(budget=None, turns=10)
    for _ in range(5):
        assert w.process(_step("s", 0.0)).prompts == ()
    assert [sid for sid, _ in w.process(_step("s", 0.0)).prompts] == ["s"]  # 6th of 10
    assert w.wrap_up_reason == "turns"
    assert w.process(_step("s", 0.0)).prompts == ()
    for _ in range(3):
        w.process(_step("s", 0.0))
    assert w.turns_exceeded  # still aborts at the cap


def test_wrap_up_tells_running_subagents_and_skips_finished_ones():
    w = _wrap_up_watcher(budget=1.0)
    for child in ("running", "finished"):
        w.process(_user_message(child))
    w.process(_idle_event("finished"))

    action = w.process(_step("s", 0.95))

    assert [sid for sid, _ in action.prompts] == ["s", "running"]
    assert action.prompts[1][1] == {
        "agent": "sweeper",
        "model": {"providerID": "xai", "modelID": "grok"},
        "variant": "high",
        "parts": [{"type": "text", "text": _CHILD_WRAP_UP_TEXT}],
        "tools": wrap_up_tools(),
    }


def test_a_session_whose_step_ended_its_loop_is_not_told():
    # The step that crosses the threshold is often the session's last; a message then would restart
    # a session whose result was already taken.
    w = _wrap_up_watcher(budget=1.0)
    w.process(_user_message("done-child"))
    w.process(_user_message("busy-child"))
    w.process(_step("done-child", 0.0, reason="stop"))
    w.process(_step("busy-child", 0.1))
    w.process(_answer_call("m-answer"))
    action = w.process(_step("s", 0.85, message="m-answer"))
    assert [sid for sid, _ in action.prompts] == ["busy-child"]
    # A subagent's final report, crossing it, holds back only that subagent.
    w2 = _wrap_up_watcher(budget=1.0)
    w2.process(_user_message("child"))
    action = w2.process(_step("child", 0.95, reason="stop"))
    assert [sid for sid, _ in action.prompts] == ["s"]


def test_a_held_back_session_that_goes_on_is_told_at_its_next_step():
    w = _wrap_up_watcher(budget=1.0)
    w.process(_answer_call("m-answer"))
    assert w.process(_step("s", 0.9, message="m-answer")).prompts == ()
    # The answer was rejected, so the root takes another working step.
    action = w.process(_step("s", 0.01, message="m-next"))
    assert [sid for sid, _ in action.prompts] == ["s"]
    assert w.process(_step("s", 0.01, message="m-last")).prompts == ()


def test_a_subagent_that_starts_after_the_wrap_up_is_told_once():
    w = _wrap_up_watcher(budget=1.0)
    w.process(_step("s", 0.95))

    action = w.process(_user_message("late", variant=None))
    assert [sid for sid, _ in action.prompts] == ["late"]
    assert "variant" not in action.prompts[0][1]
    # Its own wrap-up message comes back as a user message; that must not trigger another.
    assert w.process(_user_message("late")).prompts == ()


@pytest.mark.parametrize("told_before_it_finished", [True, False])
def test_a_subagent_continued_after_it_finished_is_told_again(told_before_it_finished):
    w = _wrap_up_watcher(budget=1.0)
    w.process(_user_message("child"))
    if told_before_it_finished:
        assert [sid for sid, _ in w.process(_step("s", 0.95)).prompts] == ["s", "child"]
    w.process(_idle_event("child"))
    if not told_before_it_finished:
        assert [sid for sid, _ in w.process(_step("s", 0.95)).prompts] == ["s"]

    # The task tool continued the finished subagent, so it runs again under the wrap-up.
    action = w.process(_user_message("child"))
    assert [sid for sid, _ in action.prompts] == ["child"]
    assert w.process(_user_message("child")).prompts == ()


def test_wrap_up_fires_once_its_time_has_come_even_mid_step():
    late = SessionWatcher("s", None, None, _ROOT_BODY, wrap_up_at=time.monotonic() - 1)
    text = _part_event("s", {"id": "p1", "type": "text", "messageID": "m", "text": "..."})
    assert [sid for sid, _ in late.process(text).prompts] == ["s"]
    assert late.wrap_up_reason == "time"

    early = SessionWatcher("s", None, None, _ROOT_BODY, wrap_up_at=time.monotonic() + 60)
    assert early.process(_step("s", 0.0)).prompts == ()
    assert early.wrap_up_reason is None


def test_nothing_is_sent_without_a_wrap_up_body():
    w = SessionWatcher("s", max_budget_usd=1.0, max_turns=5)
    w.process(_user_message("child"))
    assert w.process(_step("s", 0.95)).prompts == ()
    assert w.wrap_up_reason is None
    assert w.process(_user_message("late")).prompts == ()


def test_wrap_up_tools_switch_off_mcp_and_builtins_but_not_structured_output(monkeypatch):
    monkeypatch.setenv("AGENT_MCP_SERVERS", _DEMO_SERVERS_JSON)
    tools = wrap_up_tools()
    assert {"demo_lookup", "demo_fetch"} <= set(tools)
    assert {"bash", "read", "task", "grep"} <= set(tools)
    assert "StructuredOutput" not in tools
    assert not any(tools.values())


def test_session_watcher_streams_token_deltas_and_tool_transitions():
    # With a sink set, the watcher pushes a live update per real change: a text part yields only its
    # newly-appended tail (opencode resends the whole text each update), and a tool part yields one
    # event per status transition carrying input and, once done, output.
    events: list[dict] = []
    w = SessionWatcher("s", max_budget_usd=None, max_turns=None, event_sink=events.append)

    # A generation step opens the assistant message; only then is its text streamed as output.
    w.process(_pev({"type": "step-start", "id": "s1", "messageID": "m1"}))
    w.process(_pev({"type": "text", "id": "t1", "text": "Hel", "messageID": "m1"}))
    w.process(_pev({"type": "text", "id": "t1", "text": "Hello wor", "messageID": "m1"}))
    w.process(
        _pev({"type": "text", "id": "t1", "text": "Hello wor", "messageID": "m1"})
    )  # no new text
    w.process(_pev({"type": "text", "id": "t1", "text": "Hello world", "messageID": "m1"}))

    tool_running = {"status": "running", "input": {"command": "ls"}}
    tool_done = {"status": "completed", "input": {"command": "ls"}, "output": "a\nb\n"}
    w.process(
        _pev({"type": "tool", "id": "b1", "tool": "bash", "messageID": "m1", "state": tool_running})
    )
    w.process(
        _pev({"type": "tool", "id": "b1", "tool": "bash", "messageID": "m1", "state": tool_running})
    )  # same status
    w.process(
        _pev({"type": "tool", "id": "b1", "tool": "bash", "messageID": "m1", "state": tool_done})
    )

    # A step-finish is a turn, not a live update, so it never reaches the sink.
    w.process(_pev({"type": "step-finish", "cost": 0.001, "messageID": "m1"}))

    assert events == [
        {"kind": "token", "text": "Hel"},
        {"kind": "token", "text": "lo wor"},
        {"kind": "token", "text": "ld"},
        {
            "kind": "tool",
            "id": "b1",
            "name": "bash",
            "status": "running",
            "input": {"command": "ls"},
        },
        {
            "kind": "tool",
            "id": "b1",
            "name": "bash",
            "status": "completed",
            "input": {"command": "ls"},
            "output": "a\nb\n",
        },
    ]


def test_session_watcher_does_not_stream_the_echoed_prompt():
    # opencode re-emits the user's own prompt as a text part with no generation step. That must not
    # be restreamed as assistant output, or a caller renders its own prompt back as the answer.
    events: list[dict] = []
    w = SessionWatcher("s", max_budget_usd=None, max_turns=None, event_sink=events.append)

    # The echoed prompt arrives first, on a message with no step — dropped.
    w.process(_pev({"type": "text", "id": "u1", "text": "analyze prefix X", "messageID": "user-m"}))
    # The assistant then generates: its message opens a step, so its text streams.
    w.process(_pev({"type": "step-start", "id": "s1", "messageID": "asst-m"}))
    w.process(_pev({"type": "text", "id": "a1", "text": "Result: ", "messageID": "asst-m"}))

    assert events == [{"kind": "token", "text": "Result: "}]  # only the assistant's text


def test_session_watcher_without_a_sink_emits_nothing_extra():
    # The sink is opt-in: the default watcher still tracks cost and turns and simply skips the
    # live-event path, so nothing changes for callers that don't want a live trace.
    w = SessionWatcher("s", max_budget_usd=None, max_turns=None)
    w.process(_pev({"type": "text", "id": "t1", "text": "hi", "messageID": "m1"}))
    assert w.turns == 0


def test_session_watcher_swallows_a_raising_sink():
    # A live update is best-effort UX: a sink that raises must never break cap accounting.
    def boom(_event: dict) -> None:
        raise RuntimeError("consumer went away")

    w = SessionWatcher("s", max_budget_usd=0.004, max_turns=None, event_sink=boom)
    w.process(_pev({"type": "step-start", "id": "s1", "messageID": "m1"}))
    w.process(
        _pev({"type": "text", "id": "t1", "text": "hi", "messageID": "m1"})
    )  # sink raises here
    assert w.process(_pev({"type": "step-finish", "cost": 0.005, "messageID": "m1"})).abort
    assert w.budget_exceeded is True  # accounting unaffected by the failing sink


def test_subagent_parts_never_reach_the_live_sink():
    # A subagent's text and tool calls are its own work, not the run's answer; streaming them as
    # tokens would interleave them into the caller's output.
    events: list[dict] = []
    w = SessionWatcher("s", max_budget_usd=None, max_turns=None, event_sink=events.append)
    child_step = {"type": "step-start", "id": "cs", "messageID": "cm"}
    child_text = {"type": "text", "id": "ct", "text": "child notes", "messageID": "cm"}
    for part in (child_step, child_text):
        w.process(
            {"type": "message.part.updated", "properties": {"sessionID": "child", "part": part}}
        )
    w.process(_pev({"type": "step-start", "id": "s1", "messageID": "m1"}))
    w.process(_pev({"type": "text", "id": "t1", "text": "answer", "messageID": "m1"}))

    assert events == [{"kind": "token", "text": "answer"}]
