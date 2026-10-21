"""Tests for the serve HTTP driver: the PermissionSpec->ruleset translation, the message body, and
the response->DriverResult mapping. The daemon is faked with an httpx MockTransport, so the request
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
from agent.opencode import client as client_mod
from agent.opencode.client import (
    _CHILD_WRAP_UP_TEXT,
    _ROOT_WRAP_UP_TEXT,
    _SessionWatcher,
    _to_driver_result,
    build_message_body,
    mcp_local_config,
    register_mcp,
    run_session,
    to_permission_ruleset,
    wrap_up_tools,
)
from agent.opencode.driver import DriverResult, ParsedRun, SessionEnd, build_timeline
from agent.opencode.providers import format_schema
from agent.opencode.server import OpencodeServer
from agent.permissions import PermissionSpec
from agent.runner import OpencodeRunner
from core.tools.mcp import McpServer, mcp_tool_ids

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
    # An allowed tool with no refinements gets no rule (defaults to allow), so read never appears.
    assert not any(perm == "read" for perm, _, _ in tuples)
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


def test_mcp_local_config_builds_local_config_from_server_command():
    config = mcp_local_config(_DEMO_SERVER)
    assert config["type"] == "local"
    assert config["enabled"] is True
    # The command is the server's declared argv, verbatim — the pod injects nothing.
    assert config["command"] == ["python", "srv.py"]


@pytest.mark.asyncio
async def test_register_mcp_posts_config_and_returns_status():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={_DEMO_SERVER.name: {"status": "connected"}})

    client = _mock_client(handler)
    try:
        status = await register_mcp(
            "http://d", _DEMO_SERVER.name, mcp_local_config(_DEMO_SERVER), client=client
        )
    finally:
        await client.aclose()

    assert seen["path"] == "/mcp"
    assert seen["body"]["name"] == _DEMO_SERVER.name
    assert seen["body"]["config"]["type"] == "local"
    assert status[_DEMO_SERVER.name]["status"] == "connected"


@pytest.mark.asyncio
async def test_register_mcp_raises_when_server_fails_to_connect():
    # The daemon answers 200 even when the server process never comes up; only the carried status
    # says so. Registration must fail loudly, or every agent would silently run without the tools.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={_DEMO_SERVER.name: {"status": "failed", "error": "MCP error -32000"}},
        )

    client = _mock_client(handler)
    try:
        with pytest.raises(RuntimeError, match="failed to connect.*-32000"):
            await register_mcp(
                "http://d", _DEMO_SERVER.name, mcp_local_config(_DEMO_SERVER), client=client
            )
    finally:
        await client.aclose()


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
    assert len(result.timeline) == 2  # two messageID-grouped steps rebuilt for the trace
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
    assert body["parts"] == [{"type": "text", "text": _ROOT_WRAP_UP_TEXT}]
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
    assert result.timeline  # the partial work is traceable, not thrown away with the connection


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


def _pev(part: dict) -> dict:
    return {"type": "message.part.updated", "properties": {"sessionID": "s", "part": part}}


def test_session_watcher_accumulates_cost_and_caps_budget():
    w = _SessionWatcher("s", max_budget_usd=0.004, max_turns=None)
    assert not w.process(_pev({"type": "step-finish", "cost": 0.003, "messageID": "m1"})).abort
    assert w.cost == pytest.approx(0.003)
    assert w.turns == 1
    # The second step crosses the budget: it signals an abort and flags the run over-budget.
    assert w.process(_pev({"type": "step-finish", "cost": 0.002, "messageID": "m2"})).abort
    assert w.budget_exceeded is True
    assert w.turns == 2
    assert len(w.part_snapshots) == 2  # both steps captured for the timeline


def test_session_watcher_caps_turns_and_records_idle_and_tools():
    w = _SessionWatcher("s", max_budget_usd=None, max_turns=2)
    assert not w.process(_pev({"type": "step-finish", "cost": 0.0, "messageID": "m1"})).abort
    assert w.process(_pev({"type": "step-finish", "cost": 0.0, "messageID": "m2"})).abort
    assert w.turns_exceeded is True

    # A tool part is captured for the timeline but is not a turn; idle sets the completion flag.
    w2 = _SessionWatcher("s", max_budget_usd=None, max_turns=None)
    w2.process(_pev({"type": "tool", "tool": "read", "messageID": "m1"}))
    assert w2.turns == 0
    assert len(w2.part_snapshots) == 1
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


_ROOT_BODY = {"parts": [{"type": "text", "text": _ROOT_WRAP_UP_TEXT}], "tools": wrap_up_tools()}


def _wrap_up_watcher(budget: float | None = 1.0, turns: int | None = None) -> _SessionWatcher:
    return _SessionWatcher("s", budget, turns, _ROOT_BODY)


def test_session_watcher_counts_subagent_spend_but_not_its_turns():
    w = _SessionWatcher("s", max_budget_usd=1.0, max_turns=3)
    w.process(_step("s", 0.3))
    assert not w.process(_step("child", 0.5)).abort
    assert w.cost == pytest.approx(0.8)
    assert w.subagent_cost == pytest.approx(0.5)  # the trace adds this, as the timeline has none
    assert w.turns == 1
    # The subagent's spend alone can cross the budget and abort the run.
    action = w.process(_step("child", 0.3))
    assert action.abort and w.budget_exceeded
    assert w.turns == 1 and not w.turns_exceeded


def test_spend_by_subagents_alone_is_reported_from_the_stream():
    # Subagents can cross the budget before the root's first step finishes; their spend must still
    # be the run's cost, not the root session's own near-empty aggregate.
    w = _SessionWatcher("s", max_budget_usd=0.03, max_turns=None)
    w.process(_step("child", 0.02))
    w.process(_step("child", 0.015))
    parsed = _to_driver_result({}, w, duration_ms=1, session_id="s", fallback_cost=None).parsed
    assert parsed.cost_usd == pytest.approx(0.035)
    assert parsed.num_turns == 0


def test_a_subagent_going_idle_does_not_end_the_run_or_enter_its_timeline():
    w = _SessionWatcher("s", max_budget_usd=None, max_turns=None)
    w.process(_user_message("child"))
    w.process(
        {
            "type": "message.part.updated",
            "properties": {"sessionID": "child", "part": {"type": "text", "text": "x", "id": "p"}},
        }
    )
    w.process(_idle_event("child"))

    assert not w.idle.is_set()
    assert w.part_snapshots == []
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
    late = _SessionWatcher("s", None, None, _ROOT_BODY, wrap_up_at=time.monotonic() - 1)
    text = _part_event("s", {"id": "p1", "type": "text", "messageID": "m", "text": "..."})
    assert [sid for sid, _ in late.process(text).prompts] == ["s"]
    assert late.wrap_up_reason == "time"

    early = _SessionWatcher("s", None, None, _ROOT_BODY, wrap_up_at=time.monotonic() + 60)
    assert early.process(_step("s", 0.0)).prompts == ()
    assert early.wrap_up_reason is None


def test_nothing_is_sent_without_a_wrap_up_body():
    w = _SessionWatcher("s", max_budget_usd=1.0, max_turns=5)
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


def test_session_watcher_dedups_republished_part_snapshots_by_id():
    # opencode re-emits a full snapshot of a part on every update (a bash tool re-sends its whole
    # output per stdout chunk). Keying by the stable part id keeps one entry per part, so a chatty
    # tool costs one timeline entry with the latest output — not hundreds retained in memory.
    w = _SessionWatcher("s", max_budget_usd=None, max_turns=None)
    for chunk in ("a", "ab", "abc"):
        w.process(
            _pev(
                {
                    "type": "tool",
                    "id": "prt_1",
                    "tool": "bash",
                    "messageID": "m1",
                    "state": {"status": "running", "output": chunk},
                }
            )
        )
    # A second, distinct part is its own entry.
    w.process(_pev({"type": "text", "id": "prt_2", "text": "done", "messageID": "m1"}))

    assert len(w.part_snapshots) == 2  # three bash snapshots collapsed to one, plus the text part
    bash_line = next(line for line in w.part_snapshots if line["part"]["id"] == "prt_1")
    assert bash_line["part"]["state"]["output"] == "abc"  # the latest snapshot is kept

    # Parts without an id (e.g. the test fixtures) are never merged.
    w.process(_pev({"type": "step-finish", "cost": 0.0, "messageID": "m2"}))
    w.process(_pev({"type": "step-finish", "cost": 0.0, "messageID": "m3"}))
    assert len(w.part_snapshots) == 4


def test_watched_steps_carry_timing_into_the_timeline():
    # Regression: opencode's serve step parts carry no timestamp of their own, so the watcher must
    # stamp arrival time — without it every rebuilt step's start/end is None and the trace loses all
    # step timing/ordering. Feed a real step-start/step-finish pair (the serve part shape) and assert
    # the step comes out timed and ordered, not null.
    w = _SessionWatcher("s", max_budget_usd=None, max_turns=None)
    w.process(_pev({"type": "step-start", "id": "prt_1", "messageID": "m1"}))
    time.sleep(0.005)
    w.process(_pev({"type": "step-finish", "id": "prt_2", "cost": 0.001, "messageID": "m1"}))

    timeline = build_timeline(w.part_snapshots)
    assert len(timeline) == 1
    step = timeline[0]
    assert step.start_ms is not None
    assert step.end_ms is not None
    assert step.end_ms >= step.start_ms


def test_session_watcher_streams_token_deltas_and_tool_transitions():
    # With a sink set, the watcher pushes a live update per real change: a text part yields only its
    # newly-appended tail (opencode resends the whole text each update), and a tool part yields one
    # event per status transition carrying input and, once done, output.
    events: list[dict] = []
    w = _SessionWatcher("s", max_budget_usd=None, max_turns=None, event_sink=events.append)

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
    w = _SessionWatcher("s", max_budget_usd=None, max_turns=None, event_sink=events.append)

    # The echoed prompt arrives first, on a message with no step — dropped.
    w.process(_pev({"type": "text", "id": "u1", "text": "analyze prefix X", "messageID": "user-m"}))
    # The assistant then generates: its message opens a step, so its text streams.
    w.process(_pev({"type": "step-start", "id": "s1", "messageID": "asst-m"}))
    w.process(_pev({"type": "text", "id": "a1", "text": "Result: ", "messageID": "asst-m"}))

    assert events == [{"kind": "token", "text": "Result: "}]  # only the assistant's text


def test_session_watcher_without_a_sink_emits_nothing_extra():
    # The sink is opt-in: the default watcher still tracks cost/turns/timeline and simply skips the
    # live-event path, so nothing changes for callers that don't want a live trace.
    w = _SessionWatcher("s", max_budget_usd=None, max_turns=None)
    w.process(_pev({"type": "text", "id": "t1", "text": "hi", "messageID": "m1"}))
    assert w.turns == 0
    assert len(w.part_snapshots) == 1  # timeline still recorded


def test_session_watcher_swallows_a_raising_sink():
    # A live update is best-effort UX: a sink that raises must never break cap accounting.
    def boom(_event: dict) -> None:
        raise RuntimeError("consumer went away")

    w = _SessionWatcher("s", max_budget_usd=0.004, max_turns=None, event_sink=boom)
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
    w = _SessionWatcher("s", max_budget_usd=None, max_turns=None, event_sink=events.append)
    child_step = {"type": "step-start", "id": "cs", "messageID": "cm"}
    child_text = {"type": "text", "id": "ct", "text": "child notes", "messageID": "cm"}
    for part in (child_step, child_text):
        w.process(
            {"type": "message.part.updated", "properties": {"sessionID": "child", "part": part}}
        )
    w.process(_pev({"type": "step-start", "id": "s1", "messageID": "m1"}))
    w.process(_pev({"type": "text", "id": "t1", "text": "answer", "messageID": "m1"}))

    assert events == [{"kind": "token", "text": "answer"}]


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
    w = _SessionWatcher("s", max_budget_usd=None, max_turns=None)
    final = {"info": {"error": error}, "parts": []}
    parsed = _to_driver_result(final, w, duration_ms=1, session_id="s").parsed
    assert parsed.provider_failed is provider_failed
    assert (parsed.error is not None) is (error is not None)
