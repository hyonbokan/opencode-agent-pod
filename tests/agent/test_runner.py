"""Runner tests with the serve session driver mocked out. They assert the DriverResult→OpencodeResult
mapping, the args threaded into the session (model, system prompt, variant, budget, turn cap), and
the timeout/budget/turn/error classifications — no daemon is started and no opencode binary is run."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from agent.errors import SessionStartError
from agent.opencode.driver import DriverResult, ParsedRun, SessionEnd, WrapUpReason
from agent.runner import _WEDGE_CONTINUATION_NUDGE, _WEDGE_SPEND_CHARGE, OpencodeRunner


def _dr(
    text="done",
    *,
    structured=None,
    cost_usd=0.01,
    num_turns=2,
    returncode=0,
    duration_ms=100,
    end=SessionEnd.REPLIED,
    error=None,
    finish="stop",  # a healthy completed run ends with the model stopping on its own
    session_id=None,
    cost_is_session_total=False,
    wrap_up_reason=None,
    provider_failed=False,
    subagent_cost_usd=0.0,
) -> DriverResult:
    parsed = ParsedRun(
        text=text,
        structured=structured,
        cost_usd=cost_usd,
        num_turns=num_turns,
        error=error,
        finish=finish,
        cost_is_session_total=cost_is_session_total,
        provider_failed=provider_failed,
    )
    return DriverResult(
        parsed=parsed,
        returncode=returncode,
        duration_ms=duration_ms,
        end=end,
        session_id=session_id,
        wrap_up_reason=wrap_up_reason,
        subagent_cost_usd=subagent_cost_usd,
    )


class _FakeDaemon:
    base_url = "http://daemon"
    client = None  # the runner passes daemon.client to run_session, which is stubbed here

    def __init__(self, responsive: bool = True) -> None:
        self._responsive = responsive
        self.health_probes = 0

    async def is_responsive(self) -> bool:
        self.health_probes += 1
        return self._responsive


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
async def test_threads_event_sink_into_session(tmp_path):
    # The live-event sink a caller passes to run() must reach run_session, which hands it to the
    # watcher; the runner only threads it through.
    capture: dict = {}
    sink = lambda _event: None  # noqa: E731
    with _patch_serve(capture=capture):
        await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path), event_sink=sink
        )
    assert capture["event_sink"] is sink


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
async def test_wrap_up_is_opt_in(tmp_path):
    default: dict = {}
    with _patch_serve(capture=default):
        await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    opted_in: dict = {}
    with _patch_serve(capture=opted_in):
        await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0, wrap_up=True).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert default["wrap_up"] is False
    assert opted_in["wrap_up"] is True


@pytest.mark.asyncio
async def test_timeout_is_terminal_error(tmp_path):
    with _patch_serve(_dr(text="", end=SessionEnd.TIMED_OUT)):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.is_error is True
    assert r.subtype == "error_timeout"


@pytest.mark.asyncio
async def test_budget_exceeded_is_terminal_error(tmp_path):
    with _patch_serve(_dr(cost_usd=2.0, end=SessionEnd.OVER_BUDGET)):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.is_error is True
    assert r.subtype == "error_max_budget_usd"


@pytest.mark.asyncio
async def test_turn_cap_is_not_an_error(tmp_path):
    # A turn cap is a designed ceiling, so the captured output stands and is_error stays False.
    # The aborted session never reaches a "stop" finish; the cap must classify first regardless.
    with _patch_serve(_dr(text="partial", end=SessionEnd.TURNS_EXHAUSTED, finish=None)):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_turns=1, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.is_error is False
    assert r.subtype == "max_turns"
    assert r.text == "partial"


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["length", "unknown", "tool-calls", None])
async def test_prose_run_without_stop_finish_is_error_incomplete(tmp_path, finish):
    # A session can end cleanly (returncode 0, no error, no cap) without the model concluding —
    # e.g. the provider cut the stream at a token limit with the final write still pending, or the
    # last tool call was denied. Such a run must not read downstream as "the agent found nothing".
    with _patch_serve(_dr(text="recording the findings now", finish=finish)):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.is_error is True
    assert r.subtype == "error_incomplete"


@pytest.mark.asyncio
async def test_structured_run_ending_on_tool_call_finish_is_success(tmp_path):
    # A structured-output run legitimately ends on the StructuredOutput tool call, so its finish is
    # "tool-calls", never "stop" — its completion contract is the validated payload instead.
    class Out(BaseModel):
        answer: str

    with _patch_serve(_dr(structured={"answer": "42"}, finish="tool-calls")):
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.is_error is False
    assert r.subtype == "success"


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
    assert r.subtype == "error_session_start"


def _patch_serve_failing(*, responsive: bool, error: Exception):
    """Stub the pool + driver so every attempt fails with ``error`` against a daemon whose health
    check answers ``responsive``. Returns the patcher, the daemon, and the recycle mock."""
    daemon = _FakeDaemon(responsive=responsive)
    recycle = AsyncMock()
    patcher = patch.multiple(
        "agent.runner",
        get_opencode_daemon=AsyncMock(return_value=daemon),
        run_session=AsyncMock(side_effect=error),
        recycle_opencode_daemon=recycle,
    )
    return patcher, daemon, recycle


@pytest.mark.asyncio
async def test_daemon_that_stopped_answering_health_is_recycled(tmp_path):
    # The pool only evicts a dead process, so a daemon that is up but no longer serving would be
    # handed to every later agent on this project; the runner drops it instead.
    patcher, daemon, recycle = _patch_serve_failing(
        responsive=False, error=SessionStartError("session create failed")
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path)
        )

    assert recycle.await_count == 2  # both attempts saw a daemon that was up but not serving
    assert recycle.await_args_list[0].args == (str(tmp_path), daemon)
    assert r.is_error is True


@pytest.mark.asyncio
async def test_loaded_daemon_that_still_answers_health_is_kept(tmp_path):
    # Session create can fail on a daemon that is merely loaded. Recycling stops the process, killing
    # every sibling session on it, so a daemon that still answers health is left alone.
    patcher, _daemon, recycle = _patch_serve_failing(
        responsive=True, error=SessionStartError("session create failed")
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path)
        )

    recycle.assert_not_awaited()


@pytest.mark.asyncio
async def test_failure_after_the_prompt_is_sent_does_not_recycle(tmp_path):
    # Only a pre-prompt session-start failure implicates the daemon; anything else must not tear it
    # down, and must not even spend the health probe.
    patcher, daemon, recycle = _patch_serve_failing(
        responsive=False, error=RuntimeError("transport blew up mid-run")
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path)
        )

    recycle.assert_not_awaited()
    assert daemon.health_probes == 0


def _patch_serve_wedging(behaviors):
    """Stub the pool + driver so successive run_session calls return per the ``behaviors`` list.
    Returns the patcher, the daemon, the recycle mock, and the call counter."""
    daemon = _FakeDaemon()
    recycle = AsyncMock()
    calls = {"n": 0}

    async def fake_run_session(_server, **_kwargs):
        behavior = behaviors[min(calls["n"], len(behaviors) - 1)]
        calls["n"] += 1
        return behavior

    patcher = patch.multiple(
        "agent.runner",
        get_opencode_daemon=AsyncMock(return_value=daemon),
        run_session=fake_run_session,
        recycle_opencode_daemon=recycle,
    )
    return patcher, daemon, recycle, calls


@pytest.mark.asyncio
async def test_wedged_mid_session_timeout_retries_on_a_fresh_daemon(tmp_path):
    # A wedge found after the prompt is sent is a failure to reach the daemon, not a real timeout.
    # Charging the caller for it abandons work the replacement daemon can still do.
    patcher, _daemon, recycle, calls = _patch_serve_wedging(
        [
            _dr(text="", end=SessionEnd.WEDGED, session_id="sess-1"),
            _dr(text="ok on the replacement"),
        ]
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path)
        )

    assert calls["n"] == 2
    assert recycle.await_count == 1
    assert r.is_error is False
    assert r.text == "ok on the replacement"


@pytest.mark.asyncio
async def test_timeout_on_a_daemon_that_still_answers_health_is_terminal(tmp_path):
    # A genuine timeout must stay futile: the work itself overran, so a retry only spends it twice.
    # client.py already told us this wasn't a wedge (a plain TIMED_OUT, not WEDGED), so the runner
    # trusts that and never re-probes the daemon's health itself.
    patcher, _daemon, recycle, calls = _patch_serve_wedging(
        [_dr(text="", end=SessionEnd.TIMED_OUT), _dr(text="must not run")]
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path)
        )

    assert calls["n"] == 1
    recycle.assert_not_awaited()
    assert r.subtype == "error_timeout"


@pytest.mark.asyncio
async def test_wedged_attempt_is_charged_to_the_run_it_retried(tmp_path):
    # The wedged attempt's dollars are spent whatever its result does; reporting only the retry's
    # cost puts them outside the run budget that reads this number.
    patcher, _daemon, _recycle, _calls = _patch_serve_wedging(
        [
            _dr(
                text="", cost_usd=3.0, duration_ms=1800, end=SessionEnd.WEDGED, session_id="sess-1"
            ),
            _dr(text="ok on the replacement", cost_usd=4.0, duration_ms=600),
        ]
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path)
        )

    assert r.total_cost_usd == pytest.approx(7.0)
    assert r.duration_ms == 2400
    assert r.text == "ok on the replacement"


@pytest.mark.asyncio
async def test_resumed_attempt_reporting_the_session_total_is_not_charged_twice(tmp_path):
    # A resumed attempt whose event stream stayed empty reports the session's aggregate, which
    # already counts what the wedged attempt spent on that same session. Adding the wedge on top
    # would bill the caller twice for one set of tokens.
    patcher, _daemon, _recycle, _calls = _patch_serve_wedging(
        [
            _dr(
                text="", cost_usd=2.0, duration_ms=1800, end=SessionEnd.WEDGED, session_id="sess-1"
            ),
            _dr(text="ok", cost_usd=2.9, duration_ms=600, cost_is_session_total=True),
        ]
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path)
        )

    assert r.total_cost_usd == pytest.approx(2.9)
    # Wall-clock is still per-attempt, so the wedged attempt's time is added either way.
    assert r.duration_ms == 2400


@pytest.mark.asyncio
async def test_wedged_spend_survives_a_retry_that_never_starts(tmp_path):
    # The retry can die before it has a result of its own; the wedged attempt was still charged.
    daemon = _FakeDaemon()
    calls = {"n": 0}

    async def fake_run_session(_server, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _dr(text="", cost_usd=2.5, end=SessionEnd.WEDGED, session_id="sess-1")
        raise SessionStartError("session create failed")

    patcher = patch.multiple(
        "agent.runner",
        get_opencode_daemon=AsyncMock(return_value=daemon),
        run_session=fake_run_session,
        recycle_opencode_daemon=AsyncMock(),
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path)
        )

    assert r.subtype == "error_session_start"
    assert r.total_cost_usd == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_wedge_with_no_retries_left_is_terminal_after_one_recycle(tmp_path):
    # A wedge spends a retry like any other failure — no separate free budget — so with zero
    # retries left it is charged and returned after exactly one recycle.
    patcher, _daemon, recycle, calls = _patch_serve_wedging(
        [_dr(text="", end=SessionEnd.WEDGED, session_id="sess-1")]
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )

    assert calls["n"] == 1
    assert recycle.await_count == 1
    # Not a timeout: the agent never overran, the daemon under it died — and it has been recycled,
    # so a caller's own retry gets a healthy one.
    assert r.subtype == "error_transport"


@pytest.mark.asyncio
async def test_repeated_wedges_keep_resuming_the_same_session(tmp_path):
    # Every wedge resumes the same session with the nudge and spends one of max_retries — there is
    # no separate nudge budget and no falling back to a fresh session with the original prompt.
    calls: list[dict] = []

    async def fake_run_session(_server, **kwargs):
        calls.append(kwargs)
        if len(calls) <= 2:
            return _dr(text="", end=SessionEnd.WEDGED, session_id="sess-1")
        return _dr(text="ok on the resumed session")

    patcher = patch.multiple(
        "agent.runner",
        get_opencode_daemon=AsyncMock(return_value=_FakeDaemon()),
        run_session=fake_run_session,
        recycle_opencode_daemon=AsyncMock(),
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=2).run(
            user_message="original prompt", cwd=str(tmp_path)
        )

    assert len(calls) == 3
    assert calls[0]["resume_session_id"] is None
    assert calls[0]["user_message"] == "original prompt"
    assert calls[1]["resume_session_id"] == "sess-1"
    assert calls[1]["user_message"] == _WEDGE_CONTINUATION_NUDGE
    assert calls[2]["resume_session_id"] == "sess-1"
    assert calls[2]["user_message"] == _WEDGE_CONTINUATION_NUDGE
    assert r.is_error is False
    assert r.text == "ok on the resumed session"


@pytest.mark.asyncio
async def test_wedge_retry_runs_on_what_the_wedged_attempt_left(tmp_path):
    # The wedged attempt already spent real money, so the retry cannot start on the full cap again.
    # It is charged only _WEDGE_SPEND_CHARGE of it: the resumed session pays again to work out where
    # the interrupted one stopped and to redo whatever it cut in half.
    calls: list[dict] = []

    async def fake_run_session(_server, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _dr(text="", cost_usd=4.0, end=SessionEnd.WEDGED, session_id="sess-1")
        return _dr(text="ok on the replacement")

    patcher = patch.multiple(
        "agent.runner",
        get_opencode_daemon=AsyncMock(return_value=_FakeDaemon()),
        run_session=fake_run_session,
        recycle_opencode_daemon=AsyncMock(),
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1, max_budget_usd=10.0).run(
            user_message="x", cwd=str(tmp_path)
        )

    assert len(calls) == 2
    assert calls[0]["max_budget_usd"] == pytest.approx(10.0)
    assert calls[1]["max_budget_usd"] == pytest.approx(10.0 - 4.0 * _WEDGE_SPEND_CHARGE)
    assert r.is_error is False


@pytest.mark.asyncio
async def test_wedge_that_leaves_no_budget_is_not_resumed(tmp_path):
    # Resuming on an exhausted budget only buys a session that aborts on its first step, so the
    # wedge is returned as-is even though a retry was still available.
    patcher, _daemon, recycle, calls = _patch_serve_wedging(
        [
            _dr(text="", cost_usd=4.0, end=SessionEnd.WEDGED, session_id="sess-1"),
            _dr(text="must not run"),
        ]
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1, max_budget_usd=3.0).run(
            user_message="x", cwd=str(tmp_path)
        )

    assert calls["n"] == 1
    assert recycle.await_count == 1
    assert r.subtype == "error_transport"
    assert r.total_cost_usd == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_uncapped_run_keeps_its_wedge_retries_uncapped(tmp_path):
    # With no cap there is nothing to subtract from; a wedge must not turn its own spend into one.
    calls: list[dict] = []

    async def fake_run_session(_server, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _dr(text="", cost_usd=4.0, end=SessionEnd.WEDGED, session_id="sess-1")
        return _dr(text="ok on the replacement")

    patcher = patch.multiple(
        "agent.runner",
        get_opencode_daemon=AsyncMock(return_value=_FakeDaemon()),
        run_session=fake_run_session,
        recycle_opencode_daemon=AsyncMock(),
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path)
        )

    assert len(calls) == 2
    assert calls[1]["max_budget_usd"] is None
    assert r.is_error is False


@pytest.mark.asyncio
async def test_wedge_resume_still_validates_structured_output(tmp_path):
    # A resumed session goes through the exact same _to_result/validation path as a first-try
    # success, so a schema-constrained run wedging once must still come back validated.
    class Out(BaseModel):
        answer: str

    patcher, _daemon, recycle, calls = _patch_serve_wedging(
        [
            _dr(text="", end=SessionEnd.WEDGED, session_id="sess-1"),
            _dr(structured={"answer": "42"}, finish="tool-calls"),
        ]
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=1).run(
            user_message="x", cwd=str(tmp_path)
        )

    assert calls["n"] == 2
    assert recycle.await_count == 1
    assert r.is_error is False
    assert r.subtype == "success"
    assert isinstance(r.structured_output, Out)
    assert r.structured_output.answer == "42"


def _patch_serve_sequence(behaviors):
    """Stub the pool + session driver so successive run_session calls each raise or return per the
    ``behaviors`` list (an exception to raise, or a DriverResult to return). Records the call count
    and each call's kwargs."""
    calls: dict[str, Any] = {"n": 0, "kwargs": []}

    async def fake_get_opencode_daemon(_cwd):
        return _FakeDaemon()

    async def fake_run_session(_server, **kwargs):
        behavior = behaviors[min(calls["n"], len(behaviors) - 1)]
        calls["n"] += 1
        calls["kwargs"].append(kwargs)
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
async def test_incomplete_run_is_retried_once_and_recovers(tmp_path):
    # A clean-but-incomplete end (stream cut before the model concluded) gets one fresh attempt;
    # the recovered result stands, carrying the spend and time of both attempts.
    patcher, calls = _patch_serve_sequence(
        [
            _dr(text="cut mid-write", finish="unknown", cost_usd=0.30, duration_ms=100),
            _dr(text="done", cost_usd=0.50, duration_ms=200),
        ]
    )
    with patcher:
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert calls["n"] == 2
    assert r.is_error is False
    assert r.subtype == "success"
    assert r.text == "done"
    assert r.total_cost_usd == pytest.approx(0.80)
    assert r.duration_ms == 300


@pytest.mark.asyncio
async def test_incomplete_run_is_retried_only_once(tmp_path):
    # A second cut is returned as-is: the caller sees error_incomplete, never a retry loop.
    patcher, calls = _patch_serve_sequence([_dr(text="cut", finish="length")])
    with patcher:
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert calls["n"] == 2
    assert r.is_error is True
    assert r.subtype == "error_incomplete"


@pytest.mark.asyncio
async def test_incomplete_retry_runs_on_the_remaining_budget(tmp_path):
    patcher, calls = _patch_serve_sequence(
        [_dr(text="cut", finish="unknown", cost_usd=0.4), _dr(text="done")]
    )
    with patcher:
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0, max_budget_usd=1.0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert calls["n"] == 2
    assert calls["kwargs"][1]["max_budget_usd"] == pytest.approx(0.6)
    assert r.subtype == "success"


@pytest.mark.asyncio
async def test_incomplete_run_not_retried_when_budget_spent(tmp_path):
    patcher, calls = _patch_serve_sequence([_dr(text="cut", finish="unknown", cost_usd=1.0)])
    with patcher:
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0, max_budget_usd=1.0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert calls["n"] == 1
    assert r.is_error is True
    assert r.subtype == "error_incomplete"


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
async def test_invalid_structured_output_is_error_not_silent_empty(tmp_path):
    # Output that fails validation must not launder into a successful, empty result — the run is
    # flagged so a null can't read downstream as "the model found nothing".
    class Out(BaseModel):
        answer: str

    with _patch_serve(_dr(structured={"wrong": "shape"})):
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.structured_output is None
    assert r.is_error is True
    assert r.subtype == "error_structured_output"


@pytest.mark.asyncio
async def test_structured_output_recovered_from_single_key_wrapper(tmp_path):
    # opencode's Anthropic path sometimes wraps the whole object one level too deep under a single
    # key; recover it (after direct validation fails) rather than flagging a usable result as error.
    class Out(BaseModel):
        answer: str

    with _patch_serve(_dr(structured={"answer": {"answer": "42"}})):  # self-nested wrapper
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert isinstance(r.structured_output, Out)
    assert r.structured_output.answer == "42"
    assert r.is_error is False


@pytest.mark.asyncio
async def test_missing_structured_output_on_success_is_error(tmp_path):
    # A response_model run that returns no structured output at all is likewise a failure, not an
    # empty result.
    class Out(BaseModel):
        answer: str

    with _patch_serve(_dr(structured=None)):
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.structured_output is None
    assert r.is_error is True
    assert r.subtype == "error_structured_output"


@pytest.mark.asyncio
async def test_empty_but_valid_structured_output_is_success(tmp_path):
    # An empty object that validates against an all-optional schema is a legitimate "nothing to
    # report" result, not a failure — it must stay success, not be flagged as a null.
    class AllOptional(BaseModel):
        note: str | None = None
        items: list[str] = []

    with _patch_serve(_dr(structured={})):
        r = await OpencodeRunner(
            model="claude-sonnet-4-6", response_model=AllOptional, max_retries=0
        ).run(user_message="x", cwd=str(tmp_path))
    assert isinstance(r.structured_output, AllOptional)
    assert r.is_error is False
    assert r.subtype == "success"


@pytest.mark.asyncio
async def test_turn_capped_run_without_structured_output_is_error_but_not_rerun(tmp_path):
    # The turn cap is a designed ceiling for text runs, but a structured run's deliverable is the
    # structured output — with none produced, the run must not pass the is_error gate as an empty
    # result. It keeps its max_turns subtype, so a retrying caller sees a re-run as futile instead
    # of paying for the same cap again.
    class Out(BaseModel):
        answer: str

    with _patch_serve(_dr(structured=None, end=SessionEnd.TURNS_EXHAUSTED)):
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.structured_output is None
    assert r.is_error is True
    assert r.subtype == "max_turns"


@pytest.mark.asyncio
async def test_turn_capped_run_with_valid_structured_output_keeps_max_turns(tmp_path):
    # With a valid structured output the turn cap stays a non-error ceiling.
    class Out(BaseModel):
        answer: str

    with _patch_serve(_dr(structured={"answer": "42"}, end=SessionEnd.TURNS_EXHAUSTED)):
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert isinstance(r.structured_output, Out)
    assert r.is_error is False
    assert r.subtype == "max_turns"


@pytest.mark.asyncio
async def test_failed_run_keeps_its_own_subtype(tmp_path):
    # A run that failed outright (here: timeout) is not relabelled as a structured-output error.
    class Out(BaseModel):
        answer: str

    with _patch_serve(_dr(text="", structured=None, end=SessionEnd.TIMED_OUT)):
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.subtype == "error_timeout"
    assert r.structured_output is None


@pytest.mark.parametrize(
    ("reason", "subtype"),
    [
        (WrapUpReason.BUDGET, "error_max_budget_usd"),
        (WrapUpReason.TURNS, "max_turns"),
        (WrapUpReason.TIME, "error_timeout"),
    ],
)
@pytest.mark.asyncio
async def test_a_wrapped_up_run_without_an_answer_takes_the_subtype_of_its_limit(
    tmp_path, reason, subtype
):
    # Running again would hit the same limit, so the run reports that limit.
    class Out(BaseModel):
        answer: str

    no_answer = _dr(structured=None, error="StructuredOutputError", wrap_up_reason=reason)
    with _patch_serve(no_answer):
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.is_error is True
    assert r.subtype == subtype


@pytest.mark.asyncio
async def test_a_run_that_did_not_wrap_up_keeps_its_structured_output_error(tmp_path):
    class Out(BaseModel):
        answer: str

    with _patch_serve(_dr(structured={"wrong": "shape"})):
        r = await OpencodeRunner(model="claude-sonnet-4-6", response_model=Out, max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.subtype == "error_structured_output"


@pytest.mark.asyncio
async def test_a_provider_failure_is_its_own_subtype(tmp_path):
    # The provider failed after opencode's own retries, so another model could still answer; the
    # caller needs that distinguished from the agent failing.
    failed = _dr(text="", error='{"name": "APIError"}', cost_usd=0.02, provider_failed=True)
    with _patch_serve(failed):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.is_error is True
    assert r.subtype == "error_provider"
    assert r.total_cost_usd == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_subagent_spend_is_reported_and_summed_across_a_wedge(tmp_path):
    patcher, _daemon, _recycle, _calls = _patch_serve_wedging(
        [
            _dr(
                text="",
                cost_usd=3.0,
                subagent_cost_usd=1.0,
                end=SessionEnd.WEDGED,
                session_id="sess-1",
            ),
            _dr(text="ok", cost_usd=2.0, subagent_cost_usd=0.5),
        ]
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        r = await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path)
        )
    assert r.total_cost_usd == pytest.approx(5.0)
    assert r.subagent_cost_usd == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_a_wedge_resume_tells_the_live_sink_before_continuing(tmp_path):
    # The resumed session streams on after a pause and may redo interrupted work, so a caller
    # rendering the live trace is told a re-attempt started.
    events: list[dict] = []
    patcher, _daemon, _recycle, _calls = _patch_serve_wedging(
        [_dr(text="", end=SessionEnd.WEDGED, session_id="sess-1"), _dr(text="ok")]
    )
    with patcher, patch("agent.runner.asyncio.sleep", new_callable=AsyncMock):
        await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path), event_sink=events.append
        )
    assert events == [{"kind": "retry", "reason": "wedged"}]


@pytest.mark.asyncio
async def test_an_incomplete_retry_tells_the_live_sink_before_starting_over(tmp_path):
    # The fresh attempt answers from scratch, so without the notice a caller would show the cut-off
    # text and the new answer as one.
    events: list[dict] = []
    patcher, _calls = _patch_serve_sequence([_dr(text="cut", finish="length"), _dr(text="done")])
    with patcher:
        await OpencodeRunner(model="claude-sonnet-4-6", max_retries=0).run(
            user_message="x", cwd=str(tmp_path), event_sink=events.append
        )
    assert events == [{"kind": "retry", "reason": "incomplete"}]


@pytest.mark.asyncio
async def test_a_first_try_success_sends_no_retry_notice(tmp_path):
    events: list[dict] = []
    with _patch_serve(_dr(text="ok")):
        await OpencodeRunner(model="claude-sonnet-4-6", max_retries=1).run(
            user_message="x", cwd=str(tmp_path), event_sink=events.append
        )
    assert events == []
