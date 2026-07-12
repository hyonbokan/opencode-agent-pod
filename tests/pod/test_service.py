"""The run→SSE orchestration: budget resolution, runner assembly, event streaming, and reaping."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent.models import OpencodeResult
from llm import ReasoningEffort
from pod import service
from pod.schema import RunRequest, Workspace, response_model_from_schema
from pod.settings import PodSettings

_BASE_SETTINGS = PodSettings(
    bearer_token="secret",
    max_budget_usd=None,
    default_max_budget_usd=None,
    session_timeout=900.0,
    max_turns=30,
    keepalive_seconds=0.02,
    host="127.0.0.1",
    port=8080,
    key_proxy_enabled=False,
    key_proxy_host="127.0.0.1",
)


def _settings(**overrides) -> PodSettings:
    return replace(_BASE_SETTINGS, **overrides)


def _events(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs, ignoring keep-alive comments."""
    out: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        lines = block.splitlines()
        event = next((ln[7:] for ln in lines if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in lines if ln.startswith("data: ")), None)
        if event and data is not None:
            out.append((event, json.loads(data)))
    return out


class _FakeRunner:
    """Stands in for OpencodeRunner: records how it was built and run, returns a canned result."""

    instances: list[_FakeRunner] = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.run_kwargs: dict = {}
        self.result = OpencodeResult(
            text="answer", total_cost_usd=0.012, duration_ms=42, num_turns=2, subtype="success"
        )
        _FakeRunner.instances.append(self)

    async def run(self, **kwargs):
        self.run_kwargs = kwargs
        return self.result


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakeRunner.instances = []


@pytest.fixture
def stops(monkeypatch) -> list[str]:
    """Capture the cwds whose daemons get reaped, without touching the real pool."""
    seen: list[str] = []

    async def _stop(cwd: str) -> None:
        seen.append(cwd)

    monkeypatch.setattr(service, "stop_opencode_daemon", _stop)
    return seen


# --- resolve_budget ---------------------------------------------------------


def test_resolve_budget_prefers_request_then_default():
    assert service.resolve_budget(0.5, _settings()) == 0.5
    assert service.resolve_budget(None, _settings(default_max_budget_usd=0.2)) == 0.2
    assert service.resolve_budget(None, _settings()) is None


def test_resolve_budget_clamps_to_the_ceiling():
    s = _settings(max_budget_usd=1.0)
    assert service.resolve_budget(5.0, s) == 1.0  # a request can't exceed the pod ceiling
    assert service.resolve_budget(0.3, s) == 0.3
    assert service.resolve_budget(None, s) == 1.0  # ceiling also floors an unset request


# --- build_runner -----------------------------------------------------------


def test_build_runner_keeps_an_empty_tool_list_toolless():
    runner = service.build_runner(RunRequest(model="m", prompt="p"), _settings())
    assert runner._tools == []  # not the engine's DEFAULT_TOOLS
    assert runner._response_model is None
    assert runner._reasoning_effort == ReasoningEffort.MEDIUM  # omitted → sensible default


def test_build_runner_wires_schema_and_caps():
    req = RunRequest(
        model="m",
        prompt="p",
        tools=["Read", "Bash"],
        response_schema={"type": "object"},
        reasoning_effort=ReasoningEffort.HIGH,
        max_budget_usd=0.4,
    )
    runner = service.build_runner(req, _settings(max_budget_usd=1.0, max_turns=12))
    assert runner._tools == ["Read", "Bash"]
    assert runner._response_model is not None
    assert runner._reasoning_effort == ReasoningEffort.HIGH
    assert runner._max_budget_usd == 0.4
    assert runner._max_turns == 12


# --- run_events -------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_events_streams_cost_then_done_and_reaps(monkeypatch, stops):
    monkeypatch.setattr(service, "OpencodeRunner", _FakeRunner)
    req = RunRequest(model="anthropic/claude-haiku-4-5", prompt="hi")

    body = "".join([chunk async for chunk in service.run_events(req, _settings())])
    events = _events(body)

    assert [e for e, _ in events] == ["cost", "done"]
    cost = dict(events)["cost"]
    assert cost["total_cost_usd"] == 0.012
    done = dict(events)["done"]
    assert done["text"] == "answer"
    assert done["subtype"] == "success"
    assert done["num_turns"] == 2

    # The run got a real ephemeral cwd, and that same cwd's daemon was reaped exactly once. Reaping
    # is detached (survives a client disconnect), so drain it before asserting.
    await service.drain_cleanups()
    runner = _FakeRunner.instances[0]
    cwd = runner.run_kwargs["cwd"]
    assert Path(cwd).name.startswith("agent-pod-")  # temp dir handed to the run
    assert runner.run_kwargs["user_message"] == "hi"
    assert stops == [cwd]


@pytest.mark.asyncio
async def test_run_events_flattens_structured_output_to_plain_json(monkeypatch, stops):
    model = response_model_from_schema({"type": "object"})

    class _SORunner(_FakeRunner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.result = OpencodeResult(
                text="done",
                structured_output=model.model_validate({"findings": ["x"]}),
                subtype="success",
            )

    monkeypatch.setattr(service, "OpencodeRunner", _SORunner)
    req = RunRequest(model="m", prompt="p", response_schema={"type": "object"})

    body = "".join([chunk async for chunk in service.run_events(req, _settings())])
    done = dict(_events(body))["done"]
    assert done["structured_output"] == {"findings": ["x"]}  # a dict, not a model wrapper
    await service.drain_cleanups()


@pytest.mark.asyncio
async def test_run_events_streams_live_token_and_tool_events(monkeypatch, stops):
    # The runner receives an event_sink; the engine would call it with live updates. Simulate that:
    # the fake run pushes a token and a tool event through the sink, and they must reach the SSE
    # stream as `token`/`tool` events, in order, before the terminal `cost`/`done`.
    class _StreamingRunner(_FakeRunner):
        async def run(self, **kwargs):
            self.run_kwargs = kwargs
            sink = kwargs["event_sink"]
            sink({"kind": "token", "text": "Analyz"})
            sink({"kind": "token", "text": "ing…"})
            sink(
                {
                    "kind": "tool",
                    "id": "b1",
                    "name": "bash",
                    "status": "running",
                    "input": {"cmd": "ls"},
                }
            )
            sink(
                {"kind": "tool", "id": "b1", "name": "bash", "status": "completed", "output": "ok"}
            )
            return self.result

    monkeypatch.setattr(service, "OpencodeRunner", _StreamingRunner)
    req = RunRequest(model="m", prompt="p", tools=["Bash"])

    body = "".join([chunk async for chunk in service.run_events(req, _settings())])
    events = _events(body)

    assert [e for e, _ in events] == ["token", "token", "tool", "tool", "cost", "done"]
    payloads = [d for _, d in events]
    assert payloads[0] == {"text": "Analyz"}
    assert payloads[2] == {"id": "b1", "name": "bash", "status": "running", "input": {"cmd": "ls"}}
    assert payloads[3] == {"id": "b1", "name": "bash", "status": "completed", "output": "ok"}
    assert payloads[-1]["text"] == "answer"  # done still carries the result
    await service.drain_cleanups()


@pytest.mark.asyncio
async def test_run_events_emits_keepalive_while_the_run_is_in_flight(monkeypatch, stops):
    class _SlowRunner(_FakeRunner):
        async def run(self, **kwargs):
            self.run_kwargs = kwargs
            await asyncio.sleep(0.08)  # a few keep-alive ticks at 0.02s
            return self.result

    monkeypatch.setattr(service, "OpencodeRunner", _SlowRunner)
    req = RunRequest(model="m", prompt="p")

    chunks = [chunk async for chunk in service.run_events(req, _settings(keepalive_seconds=0.02))]
    body = "".join(chunks)
    assert ": keep-alive" in body  # the connection was kept warm during the run
    assert [e for e, _ in _events(body)] == ["cost", "done"]
    await service.drain_cleanups()


@pytest.mark.asyncio
async def test_run_events_reaps_daemon_and_workspace_on_client_disconnect(monkeypatch, stops):
    # Regression for a live-found leak: a client disconnect mid-run cancels the SSE generator under
    # Starlette/anyio level-triggered cancellation. The reap must still happen (it runs on a detached
    # task), or the daemon and its workspace leak.
    class _HangingRunner(_FakeRunner):
        async def run(self, **kwargs):
            self.run_kwargs = kwargs
            await asyncio.sleep(30)  # still running when the client goes away
            return self.result

    monkeypatch.setattr(service, "OpencodeRunner", _HangingRunner)
    gen = service.run_events(RunRequest(model="m", prompt="p"), _settings(keepalive_seconds=0.01))

    first = await gen.__anext__()  # enter the keep-alive loop while the run hangs
    assert ": keep-alive" in first
    await gen.aclose()  # the client disconnects
    await service.drain_cleanups()  # the detached cleanup runs to completion

    cwd = _FakeRunner.instances[0].run_kwargs["cwd"]
    assert stops == [cwd]  # daemon reaped despite the disconnect
    assert not Path(cwd).exists()  # workspace reaped too


@pytest.mark.asyncio
async def test_run_events_reports_workspace_error_as_a_terminal_done(monkeypatch, stops, tmp_path):
    monkeypatch.setattr(service, "OpencodeRunner", _FakeRunner)
    req = RunRequest(
        model="m", prompt="p", workspace=Workspace(source=(tmp_path / "missing").as_uri())
    )

    body = "".join([chunk async for chunk in service.run_events(req, _settings())])
    events = dict(_events(body))
    assert events["done"]["is_error"] is True
    assert events["done"]["subtype"] == "error"
    assert not _FakeRunner.instances  # never got as far as building a runner
    assert stops == []  # no daemon was started, so none to reap
