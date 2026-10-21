"""Tests for the serve session client: the PermissionSpec->ruleset translation, the message body,
and the response->DriverResult mapping. The daemon is faked with an httpx MockTransport, so the request
bodies sent and the result parsing are both asserted without a live server."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, cast

import httpx
import pytest
from pydantic import BaseModel

from agent.errors import SessionStartError
from agent.mcp import McpServer, mcp_tool_ids
from agent.opencode import client as client_mod
from agent.opencode.client import (
    _to_driver_result,
    build_message_body,
    run_session,
    to_permission_ruleset,
)
from agent.opencode.providers import format_schema
from agent.opencode.results import DriverResult, ParsedRun, SessionEnd
from agent.opencode.server import OpencodeServer
from agent.opencode.watcher import ROOT_WRAP_UP_TEXT, SessionWatcher, wrap_up_tools
from agent.permissions import PermissionSpec
from agent.runner import OpencodeRunner

_DEMO_SERVER = McpServer(name="demo", command=("python", "srv.py"), tools=("lookup", "fetch"))
_DEMO_SERVERS_JSON = json.dumps(
    [{"name": "demo", "command": ["python", "srv.py"], "tools": ["lookup", "fetch"]}]
)


class _Findings(BaseModel):
    findings: list[str]


def _tuples(rules: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    return [(r["permission"], r["pattern"], r["action"]) for r in rules]


def test_to_permission_ruleset_flattens_config_preserving_precedence():
    rules = to_permission_ruleset(PermissionSpec(), ["read", "bash"], "/work")
    tuples = _tuples(rules)

    # Scalar tool rules become a "*" pattern; the external-directory guard and unallowed tools deny.
    assert ("external_directory", "*", "deny") in tuples
    assert ("edit", "*", "deny") in tuples
    # An allowed tool with no refinements gets one explicit "*" allow.
    assert [t for t in tuples if t[0] == "read"] == [("read", "*", "allow")]
    # bash is allowed: the broad allow must come before the destructive-command denies (last wins).
    bash = [i for i, (perm, _, _) in enumerate(tuples) if perm == "bash"]
    assert tuples[bash[0]] == ("bash", "*", "allow")
    assert ("bash", "*mkfs*", "deny") in tuples
    assert bash[0] < min(i for i in bash if tuples[i][2] == "deny")


def test_build_message_body_carries_model_system_variant_and_format():
    body = build_message_body(
        opencode_model="openai/gpt-5.4-mini",
        system_prompt="You are an auditor.",
        user_message="find bugs",
        variant="high",
        response_model=_Findings,
    )
    assert body["model"] == {"providerID": "openai", "modelID": "gpt-5.4-mini"}
    assert body["system"] == "You are an auditor."
    assert body["variant"] == "high"
    assert body["parts"] == [{"type": "text", "text": "find bugs"}]
    assert body["format"]["type"] == "json_schema"
    assert body["format"]["schema"] == format_schema(_Findings, "openai")


def test_build_message_body_omits_optional_fields_when_absent():
    body = build_message_body(
        opencode_model="anthropic/claude-haiku-4-5",
        system_prompt=None,
        user_message="hi",
        variant=None,
        response_model=None,
    )
    assert "system" not in body
    assert "variant" not in body
    assert "format" not in body
    assert "tools" not in body


def test_build_message_body_never_sends_tools_map():
    # A tools map would make opencode replace the session ruleset, wiping the confinement.
    body = build_message_body(
        opencode_model="anthropic/claude-haiku-4-5",
        system_prompt="sys",
        user_message="hi",
        variant=None,
        response_model=_Findings,
    )
    assert "tools" not in body


def test_to_permission_ruleset_gates_mcp_tools_by_allow_list(monkeypatch):
    # A deployment declares an MCP server via config; its tools are gated by the per-run allow-list.
    monkeypatch.setenv("AGENT_MCP_SERVERS", _DEMO_SERVERS_JSON)
    tool_ids = mcp_tool_ids()
    assert tool_ids  # the configured server exposes tools to gate
    one = tool_ids[0]

    # An allow-list naming one MCP tool denies every other MCP tool with a "*"-pattern deny rule
    # (which is what makes opencode hide the tool), and leaves the named one unruled (defaults on).
    tuples = _tuples(to_permission_ruleset(PermissionSpec(), ["read", one], "/work"))
    for tool_id in tool_ids:
        if tool_id == one:
            assert not any(perm == tool_id for perm, _, _ in tuples)
        else:
            assert (tool_id, "*", "deny") in tuples

    # An allow-list naming none of them denies them all.
    none_tuples = _tuples(to_permission_ruleset(PermissionSpec(), ["read"], "/work"))
    assert all((tool_id, "*", "deny") in none_tuples for tool_id in tool_ids)


def test_to_permission_ruleset_gates_nothing_when_no_servers_configured(monkeypatch):
    # With no MCP servers declared, the ruleset carries no MCP deny rules — only the built-in gating.
    monkeypatch.delenv("AGENT_MCP_SERVERS", raising=False)
    tuples = _tuples(to_permission_ruleset(PermissionSpec(), ["read"], "/work"))
    assert not any(perm.startswith("demo_") for perm, _, _ in tuples)


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _part_event(session_id: str, part: dict) -> dict:
    return {"type": "message.part.updated", "properties": {"sessionID": session_id, "part": part}}


def _idle_event(session_id: str) -> dict:
    return {"type": "session.idle", "properties": {"sessionID": session_id}}


def _serve_handler(final_message: dict, seen: dict):
    """A MockTransport handler for one serve run: session create, prompt, abort, delete. The event
    stream is the daemon's shared consumer, simulated by _FakeServer, not this HTTP handler."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/session":
            seen["session"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "sess-1"})
        if path.endswith("/abort"):
            seen["aborted"] = True
            return httpx.Response(200, json=True)
        if path.endswith("/message"):
            seen["message"] = json.loads(request.content)
            return httpx.Response(200, json=final_message)
        return httpx.Response(200, json=True)  # session delete, etc.

    return handler


class _FakeServer:
    """Stands in for OpencodeServer in run_session tests: holds the mock client and replays a
    session's events into its watcher the way the real shared consumer would."""

    base_url = "http://d"

    def __init__(self, client, *, events=None, events_ready=True, responsive=True, actions=None):
        self.client = client
        self.events_ready = events_ready
        self._events = events or []
        self._actions = actions
        self.removed: list[str] = []
        self._responsive = responsive

    def add_watcher(self, _session_id: str, watcher) -> None:
        for event in self._events:  # simulate the shared consumer streaming this session's events
            action = watcher.process(event)
            if self._actions is not None:
                self._actions.append(action)

    def remove_watcher(self, session_id: str) -> None:
        self.removed.append(session_id)

    async def is_responsive(self) -> bool:
        return self._responsive


_FINAL = {
    "info": {"finish": "tool-calls", "error": None, "structured": {"findings": ["reentrancy"]}},
    "parts": [{"type": "text", "text": "done"}, {"type": "step-finish", "cost": 0.0017}],
}


async def _run(
    handler, *, events=None, events_ready=True, responsive=True, actions=None, **overrides
):
    kwargs: dict[str, Any] = {
        "opencode_model": "openai/gpt-5.4-mini",
        "system_prompt": "sys",
        "user_message": "go",
        "variant": "high",
        "spec": PermissionSpec(),
        "tools": ["read"],
        "cwd": "/work",
        "response_model": _Findings,
        "timeout": 30.0,
    }
    kwargs.update(overrides)
    client = _mock_client(handler)
    server = _FakeServer(
        client, events=events, events_ready=events_ready, responsive=responsive, actions=actions
    )
    try:
        return await run_session(cast(OpencodeServer, server), **kwargs)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_run_session_success_aggregates_from_stream_and_sends_expected_bodies():
    seen: dict = {}
    events = [
        _part_event(
            "sess-1",
            {
                "type": "step-finish",
                "cost": 0.003,
                "tokens": {"input": 1, "output": 2},
                "messageID": "m1",
            },
        ),
        _part_event(
            "sess-1",
            {
                "type": "tool",
                "tool": "read",
                "state": {"input": {"f": "x"}, "output": "ok", "time": {"start": 1, "end": 2}},
                "messageID": "m1",
            },
        ),
        _part_event(
            "sess-1",
            {
                "type": "step-finish",
                "cost": 0.002,
                "tokens": {"input": 1, "output": 2},
                "messageID": "m2",
            },
        ),
        _idle_event("sess-1"),
    ]
    result = await _run(_serve_handler(_FINAL, seen), events=events)

    assert result.parsed.structured == {"findings": ["reentrancy"]}
    # Cost and turns are summed from this session's step-finish events.
    assert result.parsed.cost_usd == pytest.approx(0.005)
    assert result.parsed.num_turns == 2
    assert result.parsed.text == "done"
    # The finish reason travels through so the runner can tell a concluded run from a truncated one.
    assert result.parsed.finish == "tool-calls"
    assert result.returncode == 0

    assert isinstance(seen["session"]["permission"], list)
    assert seen["message"]["model"]["providerID"] == "openai"
    assert seen["message"]["format"]["type"] == "json_schema"
    # Gating lives in the session ruleset, not a per-message tools map (which would wipe the ruleset).
    assert "tools" not in seen["message"]


@pytest.mark.asyncio
async def test_run_session_reports_budget_exceeded_from_watcher():
    # The watcher (fed by the shared consumer) latches budget_exceeded; run_session maps it onto the
    # result. Issuing the abort itself is the consumer's job, covered in test_events.py.
    seen: dict = {}
    events = [
        _part_event("sess-1", {"type": "step-finish", "cost": 0.003, "messageID": "m1"}),
        _part_event("sess-1", {"type": "step-finish", "cost": 0.002, "messageID": "m2"}),
        _idle_event("sess-1"),
    ]
    result = await _run(_serve_handler(_FINAL, seen), events=events, max_budget_usd=0.004)

    assert result.end is SessionEnd.OVER_BUDGET
    assert result.parsed.cost_usd == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_run_session_reports_turn_cap_from_watcher():
    seen: dict = {}
    events = [
        _part_event("sess-1", {"type": "step-finish", "cost": 0.001, "messageID": "m1"}),
        _idle_event("sess-1"),
    ]
    result = await _run(_serve_handler(_FINAL, seen), events=events, max_turns=1)

    assert result.end is SessionEnd.TURNS_EXHAUSTED


@pytest.mark.asyncio
async def test_run_session_wrap_up_body_repeats_the_original_with_tools_off():
    seen: dict = {}
    actions: list = []
    events = [_step("sess-1", 0.95)]
    await _run(
        _serve_handler(_FINAL, seen),
        events=events,
        actions=actions,
        max_budget_usd=1.0,
        wrap_up=True,
    )

    ((session_id, body),) = actions[0].prompts
    assert session_id == "sess-1"
    assert body["parts"] == [{"type": "text", "text": ROOT_WRAP_UP_TEXT}]
    assert body["tools"] == wrap_up_tools()
    # Everything else matches the original message, so the structured answer stays available.
    assert {k: v for k, v in body.items() if k not in ("parts", "tools")} == {
        k: v for k, v in seen["message"].items() if k != "parts"
    }


@pytest.mark.asyncio
async def test_run_session_sends_no_wrap_up_unless_asked():
    actions: list = []
    events = [_step("sess-1", 0.95)]
    await _run(_serve_handler(_FINAL, {}), events=events, actions=actions, max_budget_usd=1.0)

    assert actions[0].prompts == ()


@pytest.mark.asyncio
async def test_run_session_abort_error_does_not_shadow_cap_classification():
    # opencode marks the truncated response with an error after an abort; that expected error must
    # not reclassify a turn-capped run as a generic failure (it would discard the agent's output).
    seen: dict = {}
    final = {
        "info": {"error": "aborted", "structured": None},
        "parts": [{"type": "text", "text": "partial"}],
    }
    events = [
        _part_event("sess-1", {"type": "step-finish", "cost": 0.001, "messageID": "m1"}),
        _idle_event("sess-1"),
    ]
    result = await _run(_serve_handler(final, seen), events=events, max_turns=1)

    assert result.end is SessionEnd.TURNS_EXHAUSTED
    assert result.parsed.error is None  # the abort's error is dropped
    mapped = OpencodeRunner._to_result(result, expects_structured=False)
    assert mapped.subtype == "max_turns"  # a designed ceiling, not an error
    assert mapped.is_error is False


@pytest.mark.asyncio
async def test_run_session_prefers_cap_classification_when_prompt_post_errors():
    # Aborting the in-flight prompt to enforce a cap can make the prompt POST come back non-2xx on
    # some providers. That transport error is the abort's own echo, so a run the watcher already
    # flagged over-budget (from the events it saw) must stay over-budget, not become a generic error.
    events = [
        _part_event("sess-1", {"type": "step-finish", "cost": 0.003, "messageID": "m1"}),
        _part_event("sess-1", {"type": "step-finish", "cost": 0.002, "messageID": "m2"}),
        _idle_event("sess-1"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "sess-1"})
        if request.url.path.endswith("/message"):
            return httpx.Response(500, json={"error": "aborted"})
        return httpx.Response(200, json=True)

    result = await _run(handler, events=events, max_budget_usd=0.004)

    assert result.end is SessionEnd.OVER_BUDGET
    assert result.parsed.error is None  # not reclassified as a generic error
    assert result.parsed.cost_usd == pytest.approx(0.005)
    assert (
        OpencodeRunner._to_result(result, expects_structured=False).subtype
        == "error_max_budget_usd"
    )


@pytest.mark.asyncio
async def test_run_session_falls_back_to_final_message_when_stream_is_empty():
    # No steps on the stream and no session aggregate available (the mock returns none), so cost and
    # turns fall back to the final message's own accounting.
    seen: dict = {}
    final = {
        "info": {"error": None, "cost": 0.0017, "structured": {"findings": []}},
        "parts": [{"type": "text", "text": "ok"}, {"type": "step-finish", "cost": 0.0017}],
    }
    result = await _run(_serve_handler(final, seen), events=[_idle_event("sess-1")])

    assert result.parsed.cost_usd == pytest.approx(0.0017)
    assert result.parsed.num_turns == 1
    assert result.parsed.structured == {"findings": []}
    # The final message's own cost covers this run only, so it is safe to add to earlier attempts.
    assert result.parsed.cost_is_session_total is False


@pytest.mark.asyncio
async def test_run_session_reconciles_cost_from_session_aggregate_when_stream_is_empty():
    # No steps seen, so the live sum is empty; the run reads the session's own aggregate cost rather
    # than the final message's last-step-only cost, which would undercount a multi-step run.
    final = {
        "info": {"error": None, "cost": 0.0017, "structured": {"findings": []}},
        "parts": [{"type": "text", "text": "ok"}, {"type": "step-finish", "cost": 0.0017}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/session":
            return httpx.Response(200, json={"id": "sess-1"})
        if path.endswith("/message"):
            return httpx.Response(200, json=final)
        if request.method == "GET" and path == "/session/sess-1":
            return httpx.Response(200, json={"id": "sess-1", "cost": 0.0091})
        return httpx.Response(200, json=True)

    result = await _run(handler, events=[_idle_event("sess-1")], response_model=None)

    assert result.parsed.cost_usd == pytest.approx(0.0091)
    # Flagged, because on a resumed session that aggregate spans the earlier attempts too and the
    # runner must not add their cost on top of it.
    assert result.parsed.cost_is_session_total is True


@pytest.mark.asyncio
async def test_run_session_timeout_is_terminal():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "sess-1"})
        raise httpx.TimeoutException("read timeout")  # the prompt POST (and best-effort abort)

    result = await _run(handler, response_model=None, timeout=1.0)
    assert result.end is SessionEnd.TIMED_OUT
    assert result.parsed.structured is None


@pytest.mark.asyncio
async def test_wedged_daemon_is_caught_before_the_full_timeout(monkeypatch):
    # The daemon accepts the connection but never answers — a hang the prompt POST alone would
    # only surface once the (here, generously long) session timeout elapses.
    monkeypatch.setattr(client_mod, "_WEDGE_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(client_mod, "_WEDGE_POLL_MISSES", 1)
    seen: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "sess-1"})
        if request.url.path.endswith("/abort"):
            seen["aborted"] = True
            return httpx.Response(200, json=True)
        if request.url.path.endswith("/message"):
            await asyncio.Event().wait()  # never resolves; the watchdog must win the race
            raise AssertionError("unreachable")
        if request.method == "DELETE":
            seen["deleted"] = True
        return httpx.Response(200, json=True)  # session delete, etc.

    start = time.monotonic()
    result = await _run(handler, response_model=None, timeout=300.0, responsive=False)
    elapsed = time.monotonic() - start

    assert result.end is SessionEnd.WEDGED
    assert "aborted" not in seen  # a proven wedge skips the pointless abort call
    # The session is what the retry resumes on a fresh daemon, so cleanup must leave it alone.
    assert "deleted" not in seen
    assert result.session_id == "sess-1"
    assert elapsed < 5.0  # nowhere near the 300s session timeout
    assert result.parsed.structured is None


@pytest.mark.asyncio
async def test_completed_run_is_kept_when_the_health_probe_blips():
    # The probe only decides runs that did not come back. A reply that arrived is paid for, so a
    # single missed health check on a busy daemon must not discard it as a wedge and re-run it.
    events = [
        _part_event("sess-1", {"type": "text", "text": "done", "messageID": "m1"}),
        _part_event("sess-1", {"type": "step-finish", "cost": 0.002, "messageID": "m1"}),
        _idle_event("sess-1"),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "sess-1"})
        if request.url.path.endswith("/message"):
            return httpx.Response(200, json=_FINAL)
        return httpx.Response(200, json=True)

    result = await _run(handler, events=events, responsive=False)

    assert result.end is SessionEnd.REPLIED
    assert result.parsed.text == "done"


@pytest.mark.asyncio
async def test_run_session_reports_spend_when_the_daemon_goes_away_mid_prompt():
    # Recycling a wedged daemon closes the HTTP client every sibling session is posting on. The
    # spend those siblings already made is real, so it has to come back with the failure.
    events = [
        _part_event("sess-1", {"type": "step-start", "messageID": "m1"}),
        _part_event("sess-1", {"type": "text", "text": "half an answer", "messageID": "m1"}),
        _part_event("sess-1", {"type": "step-finish", "cost": 0.003, "messageID": "m1"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "sess-1"})
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    result = await _run(handler, events=events, response_model=None)

    assert result.parsed.error is not None
    assert result.parsed.cost_usd == pytest.approx(0.003)
    assert result.parsed.num_turns == 1
    assert result.end is SessionEnd.TRANSPORT_LOST


@pytest.mark.asyncio
async def test_daemon_that_went_away_is_not_reported_as_the_agent_failing():
    lost = DriverResult(
        parsed=ParsedRun(error="client closed", cost_usd=0.003),
        returncode=1,
        duration_ms=1000,
        end=SessionEnd.TRANSPORT_LOST,
    )
    result = OpencodeRunner._to_result(lost, expects_structured=False)

    assert result.subtype == "error_transport"
    assert result.is_error is True
    assert result.total_cost_usd == pytest.approx(0.003)


@pytest.mark.asyncio
async def test_run_session_raises_when_events_not_ready_and_caps_set():
    # Caps are enforced off the daemon's shared event stream; if it isn't connected, a capped run
    # must raise (pre-prompt, so the runner retries) rather than proceed uncapped to the wall clock.
    posted = {"message": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "sess-1"})
        if request.url.path.endswith("/message"):
            posted["message"] = True
            return httpx.Response(200, json=_FINAL)
        return httpx.Response(200, json=True)

    with pytest.raises(SessionStartError):
        await _run(handler, events_ready=False, response_model=None, max_budget_usd=0.01)
    assert posted["message"] is False  # never even created the session


@pytest.mark.asyncio
async def test_run_session_proceeds_when_events_not_ready_but_no_caps():
    # With no caps, a disconnected stream only degrades accounting; the run should still proceed.
    seen: dict = {}
    result = await _run(
        _serve_handler(_FINAL, seen), events_ready=False, response_model=None, max_turns=None
    )
    assert result.returncode == 0
    assert seen.get("message") is not None  # the prompt was sent despite events not being ready


@pytest.mark.asyncio
async def test_run_session_session_create_error_raises_for_retry():
    # Session creation fails before any prompt is sent -> raise so the runner retries (and can evict
    # a dead daemon), rather than burning the attempt on a terminal result.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    with pytest.raises(SessionStartError):
        await _run(handler, response_model=None)


@pytest.mark.asyncio
async def test_run_session_terminal_when_client_closed_mid_prompt():
    # A concurrent daemon eviction can aclose the shared client while the prompt POST is in flight;
    # httpx raises RuntimeError, not an httpx error. It must be terminal (the prompt may already be
    # charged) rather than escape into a silent full-prompt re-run.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "sess-1"})
        if request.url.path.endswith("/message"):
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return httpx.Response(200, json=True)

    result = await _run(handler, response_model=None)
    assert result.returncode == 1
    assert result.parsed.error is not None


@pytest.mark.asyncio
async def test_run_session_terminal_when_response_body_is_unparseable():
    # The prompt returns 200 but a body that won't decode (e.g. a proxy error page). The run is in
    # flight and possibly charged, so this must be a terminal error result, not a raise that would
    # send the runner into a silent full-prompt re-run.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "sess-1"})
        if request.url.path.endswith("/message"):
            return httpx.Response(200, content=b"<html>502 Bad Gateway</html>")
        return httpx.Response(200, json=True)

    result = await _run(handler, response_model=None)
    assert result.returncode == 1
    assert result.parsed.error is not None


def _step(session_id: str, cost: float, reason: str = "tool-calls", message: str = "m") -> dict:
    part = {"type": "step-finish", "cost": cost, "reason": reason, "messageID": message}
    return {"type": "message.part.updated", "properties": {"sessionID": session_id, "part": part}}


def test_spend_by_subagents_alone_is_reported_from_the_stream():
    # Subagents can cross the budget before the root's first step finishes; their spend must still
    # be the run's cost, not the root session's own near-empty aggregate.
    w = SessionWatcher("s", max_budget_usd=0.03, max_turns=None)
    w.process(_step("child", 0.02))
    w.process(_step("child", 0.015))
    parsed = _to_driver_result({}, w, duration_ms=1, session_id="s", fallback_cost=None).parsed
    assert parsed.cost_usd == pytest.approx(0.035)
    assert parsed.num_turns == 0


def test_driver_result_feeds_existing_result_mapping():
    # The serve DriverResult must classify through the same _to_result the run driver uses.
    success = DriverResult(
        parsed=ParsedRun(
            text="ok", structured={"findings": []}, cost_usd=0.02, num_turns=3, finish="tool-calls"
        ),
        returncode=0,
        duration_ms=1200,
    )
    mapped = OpencodeRunner._to_result(success, expects_structured=True)
    assert mapped.is_error is False
    assert mapped.subtype == "success"
    assert mapped.total_cost_usd == 0.02
    assert mapped.num_turns == 3
    assert mapped.structured_output == {"findings": []}

    timed_out = DriverResult(
        parsed=ParsedRun(), returncode=None, duration_ms=1000, end=SessionEnd.TIMED_OUT
    )
    assert OpencodeRunner._to_result(timed_out, expects_structured=False).subtype == "error_timeout"


@pytest.mark.parametrize(
    ("error", "provider_failed"),
    [
        ({"name": "APIError", "data": {"message": "Overloaded", "statusCode": 529}}, True),
        ({"name": "ProviderAuthError", "data": {"providerID": "anthropic", "message": "x"}}, True),
        ({"name": "MessageOutputLengthError", "data": {}}, False),
        ({"name": "ContextOverflowError", "data": {"message": "too long"}}, False),
        ("aborted", False),
        (None, False),
    ],
)
def test_only_a_provider_error_in_the_final_message_flags_the_provider(error, provider_failed):
    w = SessionWatcher("s", max_budget_usd=None, max_turns=None)
    final = {"info": {"error": error}, "parts": []}
    parsed = _to_driver_result(final, w, duration_ms=1, session_id="s").parsed
    assert parsed.provider_failed is provider_failed
    assert (parsed.error is not None) is (error is not None)
