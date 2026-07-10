"""Tests for the serve HTTP driver: the PermissionSpec->ruleset translation, the message body, and
the response->DriverResult mapping. The daemon is faked with an httpx MockTransport, so the request
bodies sent and the result parsing are both asserted without a live server."""

from __future__ import annotations

import json
import time
from typing import Any, cast

import httpx
import pytest
from pydantic import BaseModel

from agent.errors import SessionStartError
from agent.opencode.client import (
    _MCP_TOOL_IDS,
    _SessionWatcher,
    build_message_body,
    mcp_local_config,
    register_mcp,
    run_session,
    to_permission_ruleset,
)
from agent.opencode.driver import build_timeline
from agent.opencode.server import OpencodeServer
from agent.permissions import PermissionSpec
from agent.runner import OpencodeRunner
from core.tools.mcp import MCP_SERVER_NAME


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
    assert body["format"]["schema"] == _Findings.model_json_schema()


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


def test_to_permission_ruleset_gates_mcp_tools_by_allow_list():
    assert _MCP_TOOL_IDS  # the MCP server exposes tools to gate
    one = _MCP_TOOL_IDS[0]

    # An allow-list naming one MCP tool denies every other MCP tool with a "*"-pattern deny rule
    # (which is what makes opencode hide the tool), and leaves the named one unruled (defaults on).
    tuples = _tuples(to_permission_ruleset(PermissionSpec(), ["read", one], "/work"))
    for tool_id in _MCP_TOOL_IDS:
        if tool_id == one:
            assert not any(perm == tool_id for perm, _, _ in tuples)
        else:
            assert (tool_id, "*", "deny") in tuples

    # An allow-list naming none of them denies them all.
    none_tuples = _tuples(to_permission_ruleset(PermissionSpec(), ["read"], "/work"))
    assert all((tool_id, "*", "deny") in none_tuples for tool_id in _MCP_TOOL_IDS)


def test_mcp_local_config_omits_emit_and_carries_project_dir():
    config = mcp_local_config("/work/project")
    assert config["type"] == "local"
    assert config["enabled"] is True
    command = config["command"]
    assert command[1].endswith("mcp_server.py")
    assert command[2:] == ["--project-dir", "/work/project", "--with-tools"]
    # serve produces structured output via `format`, so the emit half is absent.
    assert "--emit-model" not in command


@pytest.mark.asyncio
async def test_register_mcp_posts_config_and_returns_status():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={MCP_SERVER_NAME: {"status": "connected"}})

    client = _mock_client(handler)
    try:
        status = await register_mcp(
            "http://d", MCP_SERVER_NAME, mcp_local_config("/work"), client=client
        )
    finally:
        await client.aclose()

    assert seen["path"] == "/mcp"
    assert seen["body"]["name"] == MCP_SERVER_NAME
    assert seen["body"]["config"]["type"] == "local"
    assert status[MCP_SERVER_NAME]["status"] == "connected"


@pytest.mark.asyncio
async def test_register_mcp_raises_when_server_fails_to_connect():
    # The daemon answers 200 even when the server process never comes up; only the carried status
    # says so. Registration must fail loudly, or every agent would silently run without the tools.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={MCP_SERVER_NAME: {"status": "failed", "error": "MCP error -32000"}},
        )

    client = _mock_client(handler)
    try:
        with pytest.raises(RuntimeError, match="failed to connect.*-32000"):
            await register_mcp(
                "http://d", MCP_SERVER_NAME, mcp_local_config("/work"), client=client
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

    def __init__(self, client, *, events=None, events_ready=True):
        self.client = client
        self.events_ready = events_ready
        self._events = events or []
        self.removed: list[str] = []

    def add_watcher(self, _session_id: str, watcher) -> None:
        for event in self._events:  # simulate the shared consumer streaming this session's events
            watcher.process(event)

    def remove_watcher(self, session_id: str) -> None:
        self.removed.append(session_id)


_FINAL = {
    "info": {"finish": "tool-calls", "error": None, "structured": {"findings": ["reentrancy"]}},
    "parts": [{"type": "text", "text": "done"}, {"type": "step-finish", "cost": 0.0017}],
}


async def _run(handler, *, events=None, events_ready=True, **overrides):
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
    server = _FakeServer(client, events=events, events_ready=events_ready)
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

    assert result.budget_exceeded is True
    assert result.parsed.cost_usd == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_run_session_reports_turn_cap_from_watcher():
    seen: dict = {}
    events = [
        _part_event("sess-1", {"type": "step-finish", "cost": 0.001, "messageID": "m1"}),
        _idle_event("sess-1"),
    ]
    result = await _run(_serve_handler(_FINAL, seen), events=events, max_turns=1)

    assert result.turns_exceeded is True


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

    assert result.turns_exceeded is True
    assert result.parsed.error is None  # the abort's error is dropped
    mapped = OpencodeRunner._to_result(result)
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

    assert result.budget_exceeded is True
    assert result.parsed.error is None  # not reclassified as a generic error
    assert result.parsed.cost_usd == pytest.approx(0.005)
    assert OpencodeRunner._to_result(result).subtype == "error_max_budget_usd"


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


@pytest.mark.asyncio
async def test_run_session_timeout_is_terminal():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(200, json={"id": "sess-1"})
        raise httpx.TimeoutException("read timeout")  # the prompt POST (and best-effort abort)

    result = await _run(handler, response_model=None, timeout=1.0)
    assert result.timed_out is True
    assert result.parsed.structured is None


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
    assert w.process(_pev({"type": "step-finish", "cost": 0.003, "messageID": "m1"})) is False
    assert w.cost == pytest.approx(0.003)
    assert w.turns == 1
    # The second step crosses the budget: it signals an abort and flags the run over-budget.
    assert w.process(_pev({"type": "step-finish", "cost": 0.002, "messageID": "m2"})) is True
    assert w.budget_exceeded is True
    assert w.turns == 2
    assert len(w.part_snapshots) == 2  # both steps captured for the timeline


def test_session_watcher_caps_turns_and_records_idle_and_tools():
    w = _SessionWatcher("s", max_budget_usd=None, max_turns=2)
    assert w.process(_pev({"type": "step-finish", "cost": 0.0, "messageID": "m1"})) is False
    assert w.process(_pev({"type": "step-finish", "cost": 0.0, "messageID": "m2"})) is True
    assert w.turns_exceeded is True

    # A tool part is captured for the timeline but is not a turn; idle sets the completion flag.
    w2 = _SessionWatcher("s", max_budget_usd=None, max_turns=None)
    w2.process(_pev({"type": "tool", "tool": "read", "messageID": "m1"}))
    assert w2.turns == 0
    assert len(w2.part_snapshots) == 1
    w2.process({"type": "session.idle", "properties": {"sessionID": "s"}})
    assert w2.idle.is_set()


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


def test_driver_result_feeds_existing_result_mapping():
    # The serve DriverResult must classify through the same _to_result the run driver uses.
    from agent.opencode.driver import DriverResult, ParsedRun

    success = DriverResult(
        parsed=ParsedRun(text="ok", structured={"findings": []}, cost_usd=0.02, num_turns=3),
        returncode=0,
        duration_ms=1200,
    )
    mapped = OpencodeRunner._to_result(success)
    assert mapped.is_error is False
    assert mapped.subtype == "success"
    assert mapped.total_cost_usd == 0.02
    assert mapped.num_turns == 3
    assert mapped.structured_output == {"findings": []}

    timed_out = DriverResult(parsed=ParsedRun(), returncode=None, duration_ms=1000, timed_out=True)
    assert OpencodeRunner._to_result(timed_out).subtype == "error_timeout"
