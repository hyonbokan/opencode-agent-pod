import asyncio
import contextlib
import time

from pydantic import BaseModel

from llm_router import ThinkingEffort
from llm_router.utils.retry import RetryConfig

from sdk_agent.models import DEFAULT_TOOLS, SDKAgentResult
from sdk_agent.opencode.client import run_session
from sdk_agent.opencode.daemon_pool import get_opencode_daemon
from sdk_agent.opencode.driver import DriverResult, TimelineStep
from sdk_agent.opencode.providers import to_opencode_model, to_opencode_variant
from sdk_agent.permissions import PermissionSpec

from config import config
from core.integrations.langfuse_opencode import record_opencode_trace
from core.utils.logger import logger

# Background Langfuse trace-ingestion tasks. Each run fires trace recording off its critical path so an
# agent call never waits on the ingestion roundtrip; the tasks are tracked so flush_pending_traces()
# can drain them before the process shuts Langfuse down, otherwise a still-in-flight tail trace on the
# one-shot scan pod would be lost.
_pending_trace_tasks: set[asyncio.Task[None]] = set()


async def flush_pending_traces(timeout: float = 30.0) -> None:
    """Wait for any in-flight trace-ingestion tasks to land, so they aren't dropped when the pod
    shuts the Langfuse client down. Best-effort: give up after ``timeout`` rather than hang exit."""
    if not _pending_trace_tasks:
        return
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(
            asyncio.gather(*_pending_trace_tasks, return_exceptions=True), timeout
        )


# ---------------------------------------------------------------------------
# Process-wide concurrency limiter
# ---------------------------------------------------------------------------
# Constructed lazily so the Semaphore binds to the running event loop rather
# than the import-time loop (which may not exist yet under asyncio.run).
_GLOBAL_SDK_SEMAPHORE: asyncio.Semaphore | None = None


def _get_global_sdk_semaphore() -> asyncio.Semaphore:
    global _GLOBAL_SDK_SEMAPHORE
    if _GLOBAL_SDK_SEMAPHORE is None:
        _GLOBAL_SDK_SEMAPHORE = asyncio.Semaphore(config.scan.SDK_GLOBAL_CONCURRENCY)
    return _GLOBAL_SDK_SEMAPHORE


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class SDKAgentRunner:
    """Run one agent query against an ``opencode serve`` daemon, with timeout, budget/turn caps,
    permissions, and concurrency.

    Each query runs as one HTTP session on a daemon started per project directory and shared across
    the scan's agents. Spend and step count are watched on the daemon's event stream, the session is
    aborted when a cap is crossed, and the result is mapped to an SDKAgentResult.
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
        thinking_effort: ThinkingEffort | None = ThinkingEffort.MEDIUM,
    ) -> None:
        self._model = model
        self._tools = list(tools) if tools is not None else list(DEFAULT_TOOLS)
        self._response_model = response_model
        self._max_turns = max_turns
        self._thinking_effort = thinking_effort
        self._session_timeout = session_timeout
        self._max_retries = max_retries
        self._max_budget_usd = max_budget_usd
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run(
        self,
        *,
        system_prompt: str | None = None,
        user_message: str,
        cwd: str,
        max_budget_usd: float | None = None,
        permission: PermissionSpec | None = None,
    ) -> SDKAgentResult:
        """Run one query on a shared serve daemon under the concurrency limiter, then validate any
        structured output against the response model."""
        spec = permission or PermissionSpec()
        budget = max_budget_usd if max_budget_usd is not None else self._max_budget_usd
        opencode_model = to_opencode_model(self._model)
        variant = to_opencode_variant(self._thinking_effort, opencode_model)

        async with _get_global_sdk_semaphore(), self._semaphore:
            logger.debug(
                "opencode agent start cwd=%s model=%s timeout=%.0fs retries=%d",
                cwd,
                opencode_model,
                self._session_timeout,
                self._max_retries,
            )
            result, timeline = await self._run_serve(
                system_prompt=system_prompt,
                user_message=user_message,
                cwd=cwd,
                spec=spec,
                opencode_model=opencode_model,
                variant=variant,
                budget=budget,
            )

        # Record the trace off the critical path: the agent call shouldn't wait on Langfuse ingestion
        # (a blocking HTTP roundtrip) to return its result. Track the task so it can be drained at pod
        # shutdown before the Langfuse client closes. An empty timeline has nothing to record.
        if timeline:
            task = asyncio.create_task(
                asyncio.to_thread(
                    record_opencode_trace,
                    timeline,
                    model=self._model,
                    input_message=user_message,
                )
            )
            _pending_trace_tasks.add(task)
            task.add_done_callback(_pending_trace_tasks.discard)

        if self._response_model is not None and result.structured_output is not None:
            try:
                result.structured_output = self._response_model.model_validate(
                    result.structured_output
                )
            except Exception as e:
                logger.warning("Failed to validate structured output: %s", e)
                result.structured_output = None

        return result

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
    ) -> tuple[SDKAgentResult, list[TimelineStep] | None]:
        """Run one query on a serve daemon and return the result plus its event timeline.

        A daemon for the project directory comes from the pool (started and audit-MCP-registered on
        first use, then shared) and one HTTP session sends the prompt. Timeout, budget, and turn cap
        are enforced inside the session as terminal results; the retry here only covers a failure to
        acquire or reach the daemon. A run that still cannot start returns an error result.
        """
        retry_cfg = RetryConfig(max_retries=self._max_retries, base_delay=2.0, max_delay=10.0)
        last_error: Exception | None = None
        start_time = time.monotonic()

        for attempt in range(retry_cfg.max_retries + 1):
            try:
                daemon = await get_opencode_daemon(cwd)
                driver_result = await run_session(
                    daemon,
                    opencode_model=opencode_model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    variant=variant,
                    spec=spec,
                    tools=self._tools,
                    cwd=cwd,
                    response_model=self._response_model,
                    timeout=self._session_timeout,
                    max_budget_usd=budget,
                    max_turns=self._max_turns,
                )
                return self._to_result(driver_result), driver_result.timeline
            except Exception as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d failed to start serve session: %s: %s",
                    attempt + 1,
                    retry_cfg.max_retries + 1,
                    type(e).__name__,
                    e,
                )
            if attempt < retry_cfg.max_retries:
                await asyncio.sleep(retry_cfg.get_delay(attempt))

        elapsed = int((time.monotonic() - start_time) * 1000)
        logger.error(
            "All %d attempts exhausted. Last error: %s", retry_cfg.max_retries + 1, last_error
        )
        return SDKAgentResult(
            text=str(last_error) or "opencode serve execution failed",
            is_error=True,
            duration_ms=elapsed,
        ), None

    @staticmethod
    def _to_result(driver_result: DriverResult) -> SDKAgentResult:
        """Map the driver's raw result onto the public SDKAgentResult, classifying the run as
        success, timeout, over-budget, or error."""
        parsed = driver_result.parsed
        is_error = (
            driver_result.timed_out
            or driver_result.budget_exceeded
            or parsed.error is not None
            # A turn-cap stop aborts the session, so its state is not a crash signal.
            or (driver_result.returncode not in (0, None) and not driver_result.turns_exceeded)
        )
        if driver_result.timed_out:
            subtype = "error_timeout"
        elif driver_result.budget_exceeded:
            subtype = "error_max_budget_usd"
        elif is_error:
            subtype = "error"
        elif driver_result.turns_exceeded:
            # A designed ceiling, not a failure — the agent's output up to the cap stands, so this
            # stays a non-error result (callers gate discards on is_error).
            subtype = "max_turns"
        else:
            subtype = "success"

        text = parsed.text or (parsed.error or subtype if is_error else parsed.text)
        return SDKAgentResult(
            text=text,
            is_error=is_error,
            total_cost_usd=parsed.cost_usd or None,
            duration_ms=driver_result.duration_ms,
            num_turns=parsed.num_turns or None,
            structured_output=parsed.structured,
            subtype=subtype,
        )
