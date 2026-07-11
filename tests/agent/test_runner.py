"""Runner tests with the serve session driver mocked out. They assert the DriverResult→OpencodeResult
mapping, the args threaded into the session (model, system prompt, variant, budget, turn cap), and
the timeout/budget/turn/error classifications — no daemon is started and no opencode binary is run."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from agent.opencode.driver import DriverResult, ParsedRun
from agent.runner import OpencodeRunner


def _dr(
    text="done",
    *,
    structured=None,
    cost_usd=0.01,
    num_turns=2,
    returncode=0,
    duration_ms=100,
    timed_out=False,
    budget_exceeded=False,
    turns_exceeded=False,
    error=None,
) -> DriverResult:
    parsed = ParsedRun(
        text=text, structured=structured, cost_usd=cost_usd, num_turns=num_turns, error=error
    )
    return DriverResult(
        parsed=parsed,
        returncode=returncode,
        duration_ms=duration_ms,
        timed_out=timed_out,
        budget_exceeded=budget_exceeded,
        turns_exceeded=turns_exceeded,
    )


class _FakeDaemon:
    base_url = "http://daemon"
    client = None  # the runner passes daemon.client to run_session, which is stubbed here


def _patch_serve(result=None, *, side_effect=None, capture=None):
    """Stub the daemon pool and session driver the runner drives. The stub records run_session's
    kwargs, raises side_effect, or returns a canned DriverResult."""

    async def fake_get_opencode_daemon(cwd):
        if capture is not None:
            capture["cwd"] = cwd
        return _FakeDaemon()

    async def fake_run_session(server, **kwargs):
        if capture is not None:
            capture.update(kwargs)
            capture["server"] = server
        if side_effect is not None:
            raise side_effect
        return result if result is not None else _dr()

    return patch.multiple(
        "agent.runner",
        get_opencode_daemon=fake_get_opencode_daemon,
        run_session=fake_run_session,
    )


@pytest.mark.asyncio
async def test_maps_result_and_passes_session_args(tmp_path):
    capture: dict = {}
    with _patch_serve(_dr(text="Found 5 contracts.", cost_usd=0.02, num_turns=3), capture=capture):
        runner = OpencodeRunner(model="claude-sonnet-4-6", max_retries=0)
        r = await runner.run(system_prompt="sys", user_message="hi", cwd=str(tmp_path))

    assert r.text == "Found 5 contracts."
    assert r.total_cost_usd == 0.02
    assert r.num_turns == 3
    assert r.is_error is False
    assert r.subtype == "success"
    # A daemon keyed on the cwd drives the session; the system prompt travels in its own field.
    assert capture["cwd"] == str(tmp_path)
    assert capture["server"].base_url == "http://daemon"  # the pooled daemon drives the session
    assert capture["system_prompt"] == "sys"
    assert capture["user_message"] == "hi"
    assert capture["opencode_model"] == "anthropic/claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_system_prompt_optional(tmp_path):
    capture: dict = {}
    with _patch_serve(capture=capture):
        await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="just this", cwd=str(tmp_path)
        )
    assert capture["system_prompt"] is None
    assert capture["user_message"] == "just this"


@pytest.mark.asyncio
async def test_threads_budget_and_turn_cap_into_session(tmp_path):
    capture: dict = {}
    with _patch_serve(capture=capture):
        await OpencodeRunner(
            model="claude-sonnet-4-6", max_turns=7, max_budget_usd=1.25, max_retries=0
        ).run(user_message="x", cwd=str(tmp_path))
    assert capture["max_turns"] == 7
    assert capture["max_budget_usd"] == 1.25


@pytest.mark.asyncio
async def test_timeout_is_terminal_error(tmp_path):
    with _patch_serve(_dr(text="", timed_out=True)):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.is_error is True
    assert r.subtype == "error_timeout"


@pytest.mark.asyncio
async def test_budget_exceeded_is_terminal_error(tmp_path):
    with _patch_serve(_dr(cost_usd=2.0, budget_exceeded=True)):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.is_error is True
    assert r.subtype == "error_max_budget_usd"


@pytest.mark.asyncio
async def test_turn_cap_is_not_an_error(tmp_path):
    # A turn cap is a designed ceiling, so the captured output stands and is_error stays False.
    with _patch_serve(_dr(text="partial", turns_exceeded=True)):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_turns=1, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.is_error is False
    assert r.subtype == "max_turns"
    assert r.text == "partial"


@pytest.mark.asyncio
async def test_reasoning_effort_maps_to_variant(tmp_path):
    from llm import ReasoningEffort

    capture: dict = {}
    with _patch_serve(capture=capture):
        await OpencodeRunner(
            model="claude-sonnet-4-6", reasoning_effort=ReasoningEffort.HIGH, max_retries=0
        ).run(user_message="x", cwd=str(tmp_path))
    assert capture["variant"] == "high"


@pytest.mark.asyncio
async def test_no_variant_without_reasoning_effort(tmp_path):
    # Default: no reasoning variant is passed, so the model runs at its default (behavior-neutral).
    capture: dict = {}
    with _patch_serve(capture=capture):
        await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert capture["variant"] is None


@pytest.mark.asyncio
async def test_daemon_or_transport_failure_returns_error_result(tmp_path):
    with _patch_serve(side_effect=RuntimeError("daemon down")):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.is_error is True
    assert r.duration_ms is not None


def _patch_serve_sequence(behaviors):
    """Stub the pool + session driver so successive run_session calls each raise or return per the
    ``behaviors`` list (an exception to raise, or a DriverResult to return). Records the call count."""
    calls = {"n": 0}

    async def fake_get_opencode_daemon(_cwd):
        return _FakeDaemon()

    async def fake_run_session(_server, **_kwargs):
        behavior = behaviors[min(calls["n"], len(behaviors) - 1)]
        calls["n"] += 1
        if isinstance(behavior, Exception):
            raise behavior
        return behavior

    patcher = patch.multiple(
        "agent.runner",
        get_opencode_daemon=fake_get_opencode_daemon,
        run_session=fake_run_session,
    )
    return patcher, calls


@pytest.mark.asyncio
async def test_retries_a_raising_run_then_succeeds(tmp_path):
    # A failure to acquire or reach the daemon is retried; a later attempt that succeeds is the result.
    patcher, calls = _patch_serve_sequence(
        [RuntimeError("daemon not ready"), _dr(text="ok on retry")]
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=2).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert calls["n"] == 2
    assert r.is_error is False
    assert r.text == "ok on retry"


@pytest.mark.asyncio
async def test_exhausts_retries_and_returns_error(tmp_path):
    # Every attempt fails to reach the daemon: after max_retries + 1 tries the runner gives up with a
    # terminal error result rather than raising or looping forever.
    patcher, calls = _patch_serve_sequence([RuntimeError("daemon down")])
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=2).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert calls["n"] == 3  # initial attempt + 2 retries
    assert r.is_error is True


@pytest.mark.asyncio
async def test_returned_error_result_is_not_retried(tmp_path):
    # run_session turns an in-flight (possibly charged) failure into a terminal error result rather
    # than raising. The runner must return it as-is: retrying would re-run — and re-bill — the prompt.
    patcher, calls = _patch_serve_sequence([_dr(text="", returncode=1, error="in-flight failure")])
    with patcher:
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=3).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert calls["n"] == 1  # a returned error result is terminal, not retried
    assert r.is_error is True


@pytest.mark.asyncio
async def test_structured_output_validated_against_response_model(tmp_path):
    class Out(BaseModel):
        answer: str

    with _patch_serve(_dr(structured={"answer": "42"})):
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert isinstance(r.structured_output, Out)
    assert r.structured_output.answer == "42"


@pytest.mark.asyncio
async def test_invalid_structured_output_dropped(tmp_path):
    class Out(BaseModel):
        answer: str

    with _patch_serve(_dr(structured={"wrong": "shape"})):
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.structured_output is None
