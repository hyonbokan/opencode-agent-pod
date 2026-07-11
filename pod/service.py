"""Turn one request into an autonomous run and stream it as Server-Sent Events.

The heavy lifting is the engine's ``OpencodeRunner``; this layer only resolves per-request caps,
stages the workspace, drives the run to completion while keeping the connection alive, and reaps
both the run's daemon and its workspace when done — on success, error, or client disconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from agent.models import OpencodeResult
from agent.opencode.daemon_pool import stop_opencode_daemon
from agent.runner import OpencodeRunner
from core.utils.logger import logger
from llm import ReasoningEffort
from pod.schema import RunRequest, response_model_from_schema
from pod.settings import PodSettings
from pod.workspace import WorkspaceError, reap_workspace, stage_workspace

# In-flight run cleanups. A run's daemon + workspace are reaped on a detached task, not inline in the
# SSE generator's finally: a client disconnect cancels that generator under Starlette/anyio
# level-triggered cancellation, which aborts an inline ``await`` mid-teardown and leaks the daemon. A
# task created here sits outside the request's cancel scope, so it runs to completion regardless.
# Tracked so the pod can drain them at shutdown before the event loop stops.
_pending_cleanups: set[asyncio.Task[None]] = set()


async def drain_cleanups(timeout: float = 30.0) -> None:
    """Wait for in-flight run cleanups to finish, so a reap isn't dropped at shutdown."""
    if not _pending_cleanups:
        return
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(asyncio.gather(*_pending_cleanups, return_exceptions=True), timeout)


async def _cleanup_run(cwd: str, task: asyncio.Task) -> None:
    """Cancel the run if still going, then reap its daemon and workspace. Never raises."""
    if not task.done():
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
    with contextlib.suppress(Exception):
        await stop_opencode_daemon(cwd)
    await reap_workspace(cwd)


def _spawn_cleanup(cwd: str, task: asyncio.Task) -> None:
    """Reap a run's daemon + workspace on a detached task, immune to the request's cancel scope."""
    cleanup = asyncio.create_task(_cleanup_run(cwd, task))
    _pending_cleanups.add(cleanup)
    cleanup.add_done_callback(_pending_cleanups.discard)


def resolve_budget(requested: float | None, settings: PodSettings) -> float | None:
    """The budget a run actually gets: the request's, else the pod default, clamped to the ceiling."""
    chosen = requested if requested is not None else settings.default_max_budget_usd
    ceiling = settings.max_budget_usd
    if ceiling is None:
        return chosen
    return ceiling if chosen is None else min(chosen, ceiling)


def build_runner(request: RunRequest, settings: PodSettings) -> OpencodeRunner:
    """Assemble the engine runner for one request. An empty tool list stays empty — a tool-less run,
    not the engine's default tool set."""
    response_model = (
        response_model_from_schema(request.response_schema)
        if request.response_schema is not None
        else None
    )
    return OpencodeRunner(
        model=request.model,
        tools=request.tools,
        response_model=response_model,
        max_turns=settings.max_turns,
        session_timeout=settings.session_timeout,
        max_budget_usd=resolve_budget(request.max_budget_usd, settings),
        reasoning_effort=request.reasoning_effort or ReasoningEffort.MEDIUM,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _sse_comment(text: str) -> str:
    return f": {text}\n\n"


def _result_payload(result: OpencodeResult) -> dict:
    """Serialize the result for the ``done`` event, flattening a pass-through structured-output model
    into plain JSON so the caller gets its schema's object back, not a model wrapper."""
    payload = result.model_dump(mode="json")
    if isinstance(result.structured_output, BaseModel):
        payload["structured_output"] = result.structured_output.model_dump(mode="json")
    return payload


async def run_events(request: RunRequest, settings: PodSettings) -> AsyncGenerator[str, None]:
    """Stream one run as SSE: keep-alive comments while it works, then a ``cost`` and a ``done`` event.

    The final ``done`` event carries the OpencodeResult JSON — the response contract. The run's
    daemon and its ephemeral workspace are always reaped, via a detached cleanup task spawned in the
    finally, so a client disconnect mid-run can't leak either (see ``_spawn_cleanup``).
    """
    try:
        cwd = await stage_workspace(request.workspace)
    except WorkspaceError as e:
        logger.warning("workspace staging failed: %s", e)
        result = OpencodeResult(text=str(e), is_error=True, subtype="error")
        yield _sse("cost", {"total_cost_usd": None, "duration_ms": None})
        yield _sse("done", _result_payload(result))
        return

    runner = build_runner(request, settings)
    task = asyncio.create_task(
        runner.run(
            system_prompt=request.system_prompt,
            user_message=request.prompt,
            cwd=cwd,
        )
    )
    try:
        while True:
            finished, _ = await asyncio.wait({task}, timeout=settings.keepalive_seconds)
            if finished:
                break
            yield _sse_comment("keep-alive")
        result = task.result()
    finally:
        _spawn_cleanup(cwd, task)

    yield _sse("cost", {"total_cost_usd": result.total_cost_usd, "duration_ms": result.duration_ms})
    yield _sse("done", _result_payload(result))
