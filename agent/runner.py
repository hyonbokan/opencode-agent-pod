import asyncio
import contextlib
import json
import os
import random
import time

from pydantic import BaseModel

from agent.errors import SessionStartError
from agent.logger import logger
from agent.models import DEFAULT_TOOLS, OpencodeResult, ReasoningEffort, RunSubtype
from agent.opencode.client import run_session
from agent.opencode.daemon_pool import get_opencode_daemon, recycle_opencode_daemon
from agent.opencode.providers import to_opencode_model, to_opencode_variant
from agent.opencode.results import DriverResult, SessionEnd, WrapUpReason
from agent.opencode.server import OpencodeServer
from agent.opencode.watcher import EventSink
from agent.permissions import PermissionSpec

# Sent in place of the original prompt when retrying a wedged session on a fresh daemon: opencode's
# session store outlives the dead daemon, so the retry can continue the same session instead of
# starting over, provided the model is told a turn was cut short.
_WEDGE_CONTINUATION_NUDGE = (
    "You got interrupted, please continue working and if anything else you were trying got "
    "interrupted as well feel free to rerun it."
)

# Share of a wedged attempt's spend charged against the budget its retry is handed. Not all of it:
# the resumed session re-reads what the interrupted turn left behind and redoes whatever was cut
# short, so pricing that work at the full spend would leave a late wedge unable to finish.
_WEDGE_SPEND_CHARGE = 0.85

# The limit a run that was asked to wrap up still hit when it ended without an answer. A re-run is
# handed the same limit and would only wrap up again, so it keeps that limit's subtype.
_WRAP_UP_SUBTYPES = {
    WrapUpReason.BUDGET: RunSubtype.ERROR_MAX_BUDGET_USD,
    WrapUpReason.TURNS: RunSubtype.MAX_TURNS,
    WrapUpReason.TIME: RunSubtype.ERROR_TIMEOUT,
}


# ---------------------------------------------------------------------------
# Process-wide concurrency limiter
# ---------------------------------------------------------------------------
# Constructed lazily so the Semaphore binds to the running event loop rather
# than the import-time loop (which may not exist yet under asyncio.run).
_GLOBAL_CONCURRENCY = int(os.getenv("AGENT_GLOBAL_CONCURRENCY", "7"))
_GLOBAL_AGENT_SEMAPHORE: asyncio.Semaphore | None = None


def _get_global_agent_semaphore() -> asyncio.Semaphore:
    global _GLOBAL_AGENT_SEMAPHORE
    if _GLOBAL_AGENT_SEMAPHORE is None:
        _GLOBAL_AGENT_SEMAPHORE = asyncio.Semaphore(_GLOBAL_CONCURRENCY)
    return _GLOBAL_AGENT_SEMAPHORE


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class OpencodeRunner:
    """Run one agent query against an ``opencode serve`` daemon, with timeout, budget/turn caps,
    permissions, and concurrency.

    Each query runs as one HTTP session on a daemon started per project directory and shared across
    the run's agents. Spend and step count are watched on the daemon's event stream, the session is
    aborted when a cap is crossed, and the result is mapped to an OpencodeResult.
    """

    def __init__(
        self,
        *,
        model: str,
        tools: list[str] | None = None,
        response_model: type[BaseModel] | None = None,
        max_turns: int = 30,
        session_timeout: float = 900.0,
        max_retries: int = 1,
        concurrency: int = 7,
        max_budget_usd: float | None = None,
        reasoning_effort: ReasoningEffort | None = ReasoningEffort.MEDIUM,
        wrap_up: bool = False,
    ) -> None:
        self._model = model
        self._tools = list(tools) if tools is not None else list(DEFAULT_TOOLS)
        self._response_model = response_model
        self._max_turns = max_turns
        self._reasoning_effort = reasoning_effort
        self._session_timeout = session_timeout
        self._max_retries = max_retries
        self._max_budget_usd = max_budget_usd
        self._wrap_up = wrap_up
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run(
        self,
        *,
        system_prompt: str | None = None,
        user_message: str,
        cwd: str,
        max_budget_usd: float | None = None,
        permission: PermissionSpec | None = None,
        event_sink: EventSink | None = None,
    ) -> OpencodeResult:
        """Run one query on a shared serve daemon under the concurrency limiter, retrying once if the
        session ends incomplete, then validate any structured output against the response model.
        When ``event_sink`` is given it receives the run's live text deltas and tool-call
        transitions as they arrive, plus a ``retry`` event before any internal re-attempt."""
        spec = permission or PermissionSpec()
        budget = max_budget_usd if max_budget_usd is not None else self._max_budget_usd
        opencode_model = to_opencode_model(self._model)
        variant = to_opencode_variant(
            self._reasoning_effort,
            opencode_model,
            structured_output=self._response_model is not None,
        )

        async with _get_global_agent_semaphore(), self._semaphore:
            logger.debug(
                "opencode agent start cwd=%s model=%s timeout=%.0fs retries=%d",
                cwd,
                opencode_model,
                self._session_timeout,
                self._max_retries,
            )

            async def serve(attempt_budget: float | None) -> OpencodeResult:
                return await self._run_serve(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    cwd=cwd,
                    spec=spec,
                    opencode_model=opencode_model,
                    variant=variant,
                    budget=attempt_budget,
                    event_sink=event_sink,
                )

            result = await serve(budget)
            # An incomplete end is a transient stream cut, not a judged failure, so one fresh
            # attempt is worth its cost; a second cut stays error_incomplete for the caller.
            if result.subtype is RunSubtype.ERROR_INCOMPLETE:
                remaining = None if budget is None else budget - (result.total_cost_usd or 0.0)
                if remaining is not None and remaining <= 0:
                    logger.warning(
                        "session ended incomplete (finish=%s) with budget spent; not retrying",
                        result.finish,
                    )
                else:
                    logger.warning(
                        "session ended incomplete (finish=%s); retrying once", result.finish
                    )
                    _emit_retry(event_sink, "incomplete")
                    first = result
                    result = await serve(remaining)
                    result.add_attempt(first)

        if self._response_model is not None:
            self._validate_structured(result)
        return result

    def _validate_structured(self, result: OpencodeResult) -> None:
        """Replace the raw structured output with the validated model, flagging a run that
        should have answered but did not as an error."""
        assert self._response_model is not None
        validated = None
        if result.structured_output is not None:
            try:
                validated = self._response_model.model_validate(result.structured_output)
            except Exception as e:
                validated = _recover_wrapped(self._response_model, result.structured_output)
                if validated is None:
                    logger.warning("structured output failed validation: %s", e)
                else:
                    logger.debug("recovered structured output from opencode single-key wrapper")
        result.structured_output = validated
        if validated is None and result.subtype in (RunSubtype.SUCCESS, RunSubtype.MAX_TURNS):
            logger.warning(
                "structured output missing (subtype=%s, finish=%s)", result.subtype, result.finish
            )
            result.is_error = True
            # A turn-capped run keeps its subtype so callers see a re-run would hit the cap again.
            if result.subtype is RunSubtype.SUCCESS:
                result.subtype = RunSubtype.ERROR_STRUCTURED_OUTPUT
        # opencode marks a run that stopped without the answer with an error, so a wrapped-up run
        # that never answered arrives here as a plain error.
        if (
            validated is None
            and result.wrap_up_reason is not None
            and result.subtype in (RunSubtype.ERROR, RunSubtype.ERROR_STRUCTURED_OUTPUT)
        ):
            result.subtype = _WRAP_UP_SUBTYPES[result.wrap_up_reason]

    async def _recycle_if_wedged(self, cwd: str, daemon: OpencodeServer) -> bool:
        """Evict a daemon that no longer answers health; whether it was wedged."""
        if await daemon.is_responsive():
            return False
        await recycle_opencode_daemon(cwd, daemon)
        return True

    async def _run_serve(
        self,
        *,
        system_prompt: str | None,
        user_message: str,
        cwd: str,
        spec: PermissionSpec,
        opencode_model: str,
        variant: str | None,
        budget: float | None,
        event_sink: EventSink | None = None,
    ) -> OpencodeResult:
        """Run one query on a serve daemon and return its result.

        A daemon for the project directory comes from the pool and one HTTP session sends the
        prompt. Timeout, budget, and turn cap are enforced inside the session as terminal results;
        the retry here only covers a failure to acquire or reach the daemon. A wedge is such a
        failure — the daemon stopped serving mid-session — so it spends a retry too, but resumes the
        same session with a continuation nudge instead of starting over, since opencode's session
        store outlives the dead daemon.
        """
        attempts = self._max_retries + 1
        last_error: Exception | None = None
        start_time = time.monotonic()
        discarded: OpencodeResult | None = None  # attempts a wedge threw away, still charged
        resume_session_id: str | None = None
        next_message = user_message
        wedge_spend = 0.0  # what those discarded attempts cost, charged against the retry's budget
        attempt_budget = budget

        for attempt in range(attempts):
            daemon = None
            try:
                daemon = await get_opencode_daemon(cwd)
                driver_result = await run_session(
                    daemon,
                    opencode_model=opencode_model,
                    system_prompt=system_prompt,
                    user_message=next_message,
                    variant=variant,
                    spec=spec,
                    tools=self._tools,
                    cwd=cwd,
                    response_model=self._response_model,
                    timeout=self._session_timeout,
                    max_budget_usd=attempt_budget,
                    max_turns=self._max_turns,
                    wrap_up=self._wrap_up,
                    resume_session_id=resume_session_id,
                    event_sink=event_sink,
                )
                # A mid-session wedge returns instead of raising, so the except below never sees it.
                # The watchdog inside run_session already confirmed it, so no second health probe.
                wedged = driver_result.end is SessionEnd.WEDGED
                result = self._to_result(
                    driver_result, expects_structured=self._response_model is not None
                )
                if wedged:
                    await recycle_opencode_daemon(cwd, daemon)
                    # Always this attempt's own spend: a wedge is reported off the live watcher,
                    # since the dead daemon has no aggregate left to read.
                    wedge_spend += result.total_cost_usd or 0.0
                    attempt_budget = (
                        None if budget is None else budget - wedge_spend * _WEDGE_SPEND_CHARGE
                    )
                # Resuming on an empty budget only buys a session that aborts on its first step.
                budget_spent = attempt_budget is not None and attempt_budget <= 0
                if wedged and budget_spent:
                    logger.warning(
                        "Attempt %d/%d wedged with $%.4f of the run's budget spent; not resuming",
                        attempt + 1,
                        attempts,
                        wedge_spend,
                    )
                if not wedged or attempt == self._max_retries or budget_spent:
                    # Every attempt here shares one session, so a cost read off that session's
                    # aggregate already covers them.
                    if discarded is not None:
                        result.add_attempt(
                            discarded,
                            cost_already_counted=driver_result.parsed.cost_is_session_total,
                        )
                    return result
                if discarded is None:
                    discarded = result
                else:
                    discarded.add_attempt(result)
                resume_session_id = driver_result.session_id
                next_message = _WEDGE_CONTINUATION_NUDGE
                logger.warning(
                    "Attempt %d/%d wedged mid-session ($%.4f charged); continuing session %s on a "
                    "fresh daemon (budget left: %s)",
                    attempt + 1,
                    attempts,
                    result.total_cost_usd or 0.0,
                    resume_session_id,
                    "uncapped" if attempt_budget is None else f"${attempt_budget:.4f}",
                )
                _emit_retry(event_sink, "wedged")
            except Exception as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d failed to start serve session: %s: %s",
                    attempt + 1,
                    attempts,
                    type(e).__name__,
                    e,
                )
                # A daemon that stopped serving stays alive, so nothing else evicts it.
                if daemon is not None and isinstance(e, SessionStartError):
                    await self._recycle_if_wedged(cwd, daemon)
            if attempt < self._max_retries:
                await asyncio.sleep(_retry_delay(attempt))

        elapsed = int((time.monotonic() - start_time) * 1000)
        logger.error("All %d attempts exhausted. Last error: %s", attempts, last_error)
        return OpencodeResult(
            text=str(last_error) or "opencode serve execution failed",
            is_error=True,
            total_cost_usd=discarded.total_cost_usd if discarded is not None else None,
            subagent_cost_usd=discarded.subagent_cost_usd if discarded is not None else None,
            duration_ms=elapsed,  # measured from the first attempt, so it already spans them all
            subtype=RunSubtype.ERROR_SESSION_START,
        )

    @staticmethod
    def _to_result(driver_result: DriverResult, *, expects_structured: bool) -> OpencodeResult:
        """Map the driver's raw result onto the public OpencodeResult, classifying how the run
        ended."""
        parsed = driver_result.parsed
        end = driver_result.end
        # A turn cap aborts the session by design, so neither that ending nor the exit code it
        # leaves is a failure signal; every other ending short of a reply is.
        turn_capped = end is SessionEnd.TURNS_EXHAUSTED
        is_error = (
            end not in (SessionEnd.REPLIED, SessionEnd.TURNS_EXHAUSTED)
            or parsed.error is not None
            or (driver_result.returncode not in (0, None) and not turn_capped)
        )
        if end is SessionEnd.TIMED_OUT:
            subtype = RunSubtype.ERROR_TIMEOUT
        elif end is SessionEnd.OVER_BUDGET:
            subtype = RunSubtype.ERROR_MAX_BUDGET_USD
        elif end in (SessionEnd.TRANSPORT_LOST, SessionEnd.WEDGED):
            # The daemon went away mid-session; a fresh one can still do the work.
            subtype = RunSubtype.ERROR_TRANSPORT
        elif is_error:
            subtype = RunSubtype.ERROR_PROVIDER if parsed.provider_failed else RunSubtype.ERROR
        elif turn_capped:
            # A designed ceiling, not a failure — the agent's output up to the cap stands, so this
            # stays a non-error result (callers gate discards on is_error).
            subtype = RunSubtype.MAX_TURNS
        elif not expects_structured and parsed.finish != "stop":
            # A session can end without error yet without the model concluding (e.g. cut at its
            # output-length limit), which would otherwise pass truncated text off as an answer.
            is_error = True
            subtype = RunSubtype.ERROR_INCOMPLETE
        else:
            subtype = RunSubtype.SUCCESS

        text = parsed.text or (parsed.error or subtype if is_error else parsed.text)
        return OpencodeResult(
            text=text,
            is_error=is_error,
            total_cost_usd=parsed.cost_usd,
            # Subagent spend is part of the total, so it is unknown exactly when the total is.
            subagent_cost_usd=None if parsed.cost_usd is None else driver_result.subagent_cost_usd,
            duration_ms=driver_result.duration_ms,
            num_turns=parsed.num_turns or None,
            structured_output=parsed.structured,
            subtype=subtype,
            finish=parsed.finish,
            wrap_up_reason=driver_result.wrap_up_reason,
        )


def _retry_delay(attempt: int) -> float:
    """Backoff before the next attempt: doubling from 2s, capped at 10s, with ±25% jitter."""
    return min(2.0 * 2**attempt, 10.0) * random.uniform(0.75, 1.25)


def _emit_retry(event_sink: EventSink | None, reason: str) -> None:
    """Tell the live sink an internal re-attempt is starting, so streamed output can be read in
    context. Best-effort, like every live update."""
    if event_sink is None:
        return
    try:
        event_sink({"kind": "retry", "reason": reason})
    except Exception as e:
        logger.debug("live event sink raised; dropping retry notice: %s", e)


def _recover_wrapped(model: type[BaseModel], raw: object) -> BaseModel | None:
    """Recover a structured payload opencode wrapped under a single key, or None."""
    if not (isinstance(raw, dict) and len(raw) == 1):
        return None
    inner = next(iter(raw.values()))
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except (ValueError, TypeError):
            return None
    with contextlib.suppress(Exception):
        return model.model_validate(inner)
    return None
