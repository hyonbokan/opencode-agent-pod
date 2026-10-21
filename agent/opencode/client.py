"""Drive one ``opencode serve`` session over HTTP and map it onto a DriverResult.

Create a session on the daemon, register a watcher with the daemon's shared event consumer, and send
one prompt. The prompt response carries the structured output and final text; the watcher carries
the run's spend and turns, enforces its caps, and streams progress to an optional live sink. A
health-probe watchdog runs alongside the prompt so a daemon that stops serving is caught well before
the session timeout. Permissions, tools, the system prompt, and the structured-output schema are all
set per session and per request.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import httpx
from pydantic import BaseModel

from agent.errors import SessionStartError
from agent.logger import logger
from agent.mcp import mcp_tool_ids
from agent.opencode.permission_config import permission_config
from agent.opencode.providers import format_schema, provider_of
from agent.opencode.results import DriverResult, ParsedRun, SessionEnd
from agent.opencode.server import OpencodeServer
from agent.opencode.watcher import (
    ROOT_WRAP_UP_TEXT,
    WRAP_UP_TIME_MARGIN_S,
    EventSink,
    SessionWatcher,
    wrap_up_tools,
)
from agent.permissions import PermissionSpec

# Retries opencode makes internally to coax a schema-valid structured output before giving up.
_STRUCTURED_RETRY_COUNT = 2

# Short wait for the trailing step-finish plus idle marker to arrive after the prompt POST returns.
_IDLE_DRAIN_TIMEOUT = 3.0

# Final-message error names that mean the provider, not the run, failed: an API error opencode
# already retried, or a rejected key. A context overflow is left out — it is the prompt's size.
_PROVIDER_ERROR_NAMES = frozenset({"APIError", "ProviderAuthError"})

# How often to probe the daemon's health while a message is in flight, and how many consecutive
# misses to treat as wedged rather than a transient blip (e.g. the daemon briefly busy on a heavy
# tool call). A wedge is otherwise invisible until the full session timeout elapses, since the
# prompt POST blocks silently until the run finishes or that timeout fires.
_WEDGE_POLL_INTERVAL = 20.0
_WEDGE_POLL_MISSES = 2


async def run_session(
    server: OpencodeServer,
    *,
    opencode_model: str,
    system_prompt: str | None,
    user_message: str,
    variant: str | None,
    spec: PermissionSpec,
    tools: list[str],
    cwd: str,
    response_model: type[BaseModel] | None,
    timeout: float,
    max_budget_usd: float | None = None,
    max_turns: int | None = None,
    wrap_up: bool = False,
    resume_session_id: str | None = None,
    event_sink: EventSink | None = None,
) -> DriverResult:
    """Run one prompt as a serve session and return the mapped result.

    Spend (subagents' included) and steps are tracked live off the daemon's shared event stream, via
    a watcher registered for this session; crossing the budget or turn cap aborts the session. With
    ``wrap_up``, nearing either cap or the timeout first sends every running session a message that
    disables its tools and asks for its answer, so the run can still deliver one. When
    ``event_sink`` is given it receives live text deltas and tool-call transitions as they arrive.
    Failures before the prompt is sent (caps unenforceable, session create) raise SessionStartError
    for the runner to retry; once the prompt is in flight, a timeout returns a timed-out or wedged
    result and a transport failure an error result — never a raise, so a charged run is not
    silently re-run.

    ``resume_session_id`` sends the prompt into an existing session instead of creating a new one —
    used to nudge a wedged run's session back to life on a fresh daemon. opencode's session store
    lives outside the daemon process, so the session (and its permission ruleset) survives a crash;
    only the daemon serving it needs replacing.
    """
    base_url = server.base_url
    client = server.client
    t0 = time.monotonic()
    # Stays None until the session is actually in play, so the cleanup below only ever touches a
    # session this call genuinely used.
    session_id: str | None = None
    skip_cleanup = False
    try:
        # Caps are enforced off the shared event stream, so a capped run must not proceed while it is
        # disconnected. It's pre-prompt, so raise for the runner to retry; uncapped runs proceed.
        if (max_budget_usd is not None or max_turns is not None) and not server.events_ready:
            raise SessionStartError("event stream unavailable; cap cannot be enforced")

        if resume_session_id is not None:
            session_id = resume_session_id
        else:
            ruleset = to_permission_ruleset(spec, tools, cwd)
            try:
                session = await client.post(f"{base_url}/session", json={"permission": ruleset})
                session.raise_for_status()
                session_id = str(session.json()["id"])
            except (httpx.HTTPError, RuntimeError) as e:
                # Pre-prompt and idempotent (RuntimeError covers the shared client being closed by a
                # concurrent daemon eviction): raise so the runner retries against a fresh daemon.
                raise SessionStartError(
                    f"session create failed: {type(e).__name__}: {str(e)[:300]}"
                ) from e

        body = build_message_body(
            opencode_model=opencode_model,
            system_prompt=system_prompt,
            user_message=user_message,
            variant=variant,
            response_model=response_model,
        )
        wrap_up_body = None
        if wrap_up:
            # Same model and format as the original, so the structured answer stays available.
            wrap_up_body = {
                **build_message_body(
                    opencode_model=opencode_model,
                    system_prompt=system_prompt,
                    user_message=ROOT_WRAP_UP_TEXT,
                    variant=variant,
                    response_model=response_model,
                ),
                "tools": wrap_up_tools(),
            }

        # Register before sending the prompt (events for a session only start once the prompt runs),
        # so the shared consumer routes this session's events to the watcher without a gap.
        watcher = SessionWatcher(
            session_id,
            max_budget_usd,
            max_turns,
            wrap_up_body,
            wrap_up_at=t0 + timeout - WRAP_UP_TIME_MARGIN_S if wrap_up else None,
            event_sink=event_sink,
        )
        server.add_watcher(session_id, watcher)

        message_json: dict[str, Any] = {}
        post_task = asyncio.ensure_future(
            client.post(f"{base_url}/session/{session_id}/message", json=body, timeout=timeout)
        )
        wedge_task = asyncio.ensure_future(_wait_for_wedge(server))
        try:
            # Race the prompt against both the wall-clock cap (asyncio.wait's own timeout — httpx's
            # timeout only bounds the gap between response bytes, and the prompt POST blocks silently
            # until the run finishes) and a health-probe watchdog, so a daemon that stops responding
            # mid-run is caught within a few poll intervals instead of only once the full timeout fires.
            done, _ = await asyncio.wait(
                {post_task, wedge_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )

            # A 2xx reply means the run finished, so it is never discarded as a wedge — but an
            # exception is not a reply (a read timeout is a wedge symptom).
            post_ok = (
                post_task in done
                and post_task.exception() is None
                and post_task.result().is_success
            )
            wedged = not post_ok and (wedge_task in done or not (await server.is_responsive()))
            if wedged:
                # The daemon is gone, so cleaning the session up on it would only time out too.
                skip_cleanup = True
                return _cut_short_result(
                    watcher, _elapsed_ms(t0), session_id, end=SessionEnd.WEDGED, returncode=None
                )

            timed_out = post_task not in done or isinstance(
                post_task.exception(), httpx.TimeoutException
            )
            if timed_out:
                # Abort so the daemon stops spending, then report the timeout with the partial cost
                # the watcher gathered (an empty result would under-count overruns).
                with contextlib.suppress(httpx.HTTPError, RuntimeError):
                    await client.post(f"{base_url}/session/{session_id}/abort", json={})
                return _cut_short_result(
                    watcher, _elapsed_ms(t0), session_id, end=SessionEnd.TIMED_OUT, returncode=None
                )
            message = post_task.result()
            message.raise_for_status()
            message_json = message.json()
        except (httpx.HTTPError, RuntimeError, ValueError) as e:
            # In flight and possibly already charged, so terminal — never retried into a silent
            # re-run (ValueError covers a reply body that won't decode). But an intentional cap abort
            # can also surface here; if the watcher already classified the run, that stands, so fall
            # through to its result.
            if not (watcher.budget_exceeded or watcher.turns_exceeded):
                return _cut_short_result(
                    watcher,
                    _elapsed_ms(t0),
                    session_id,
                    end=SessionEnd.TRANSPORT_LOST,
                    returncode=1,
                    error=str(e)[:500],
                )
        finally:
            # Whichever of the two finished first, the other is now stale and must not hold the
            # message response open.
            for task in (post_task, wedge_task):
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

        # Let the watcher fold in the final step-finish (and the idle marker) before reading it.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(watcher.idle.wait(), _IDLE_DRAIN_TIMEOUT)
        if watcher.wrap_up_reason is not None:
            logger.info(
                "session %s ended after a wrap-up: spend $%.4f at wrap-up, $%.4f at the end",
                session_id,
                watcher.cost_at_wrap_up,
                watcher.cost,
            )
        # No steps seen means the stream delivered nothing, so the live cost sum is empty; read the
        # session's own aggregate instead of the final message's last-step-only cost.
        fallback_cost = None
        if not watcher.saw_steps:
            fallback_cost = await _session_total_cost(client, base_url, session_id)
        return _to_driver_result(
            message_json,
            watcher,
            duration_ms=_elapsed_ms(t0),
            session_id=session_id,
            fallback_cost=fallback_cost,
        )
    finally:
        # Deregister the watcher and delete the session so a long run's shared daemon doesn't
        # accumulate either. Best-effort: a failure here must never mask the run's result.
        if session_id is not None and not skip_cleanup:
            server.remove_watcher(session_id)
            with contextlib.suppress(Exception):
                await client.delete(f"{base_url}/session/{session_id}")


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def to_permission_ruleset(spec: PermissionSpec, tools: list[str], cwd: str) -> list[dict[str, str]]:
    """Flatten the shared permission config into serve's per-session ruleset — a flat list of
    permission/pattern/action, insertion-ordered so the last matching rule wins.

    MCP tools are gated here too: each one the allow-list omits gets a ``*``-pattern deny, which
    opencode treats as disabling it. This must live in the session ruleset, not the per-message tools
    map — opencode turns that map into permissions and replaces the whole ruleset with it.
    """
    rules: list[dict[str, str]] = []
    for key, value in permission_config(spec, tools, cwd).items():
        if isinstance(value, str):
            rules.append({"permission": key, "pattern": "*", "action": value})
        else:
            for pattern, action in value.items():
                rules.append({"permission": key, "pattern": pattern, "action": action})
    allowed = set(tools)
    for tool_id in mcp_tool_ids():
        if tool_id not in allowed:
            rules.append({"permission": tool_id, "pattern": "*", "action": "deny"})
    return rules


def build_message_body(
    *,
    opencode_model: str,
    system_prompt: str | None,
    user_message: str,
    variant: str | None,
    response_model: type[BaseModel] | None,
) -> dict[str, Any]:
    """Assemble the message body for one prompt.

    No ``tools`` map is sent — opencode would treat it as permissions and replace the session
    ruleset, wiping the confinement. Tool gating lives in that ruleset instead.
    """
    provider = provider_of(opencode_model)
    model_id = opencode_model.split("/", 1)[-1]
    body: dict[str, Any] = {
        "model": {"providerID": provider, "modelID": model_id},
        "parts": [{"type": "text", "text": user_message}],
    }
    if system_prompt:
        body["system"] = system_prompt
    if variant:
        body["variant"] = variant
    if response_model is not None:
        body["format"] = {
            "type": "json_schema",
            "schema": format_schema(response_model, provider),
            "retryCount": _STRUCTURED_RETRY_COUNT,
        }
    return body


# ---------------------------------------------------------------------------
# Wedge watchdog and result mapping
# ---------------------------------------------------------------------------


async def _wait_for_wedge(server: OpencodeServer) -> None:
    """Return once the daemon has failed health probes ``_WEDGE_POLL_MISSES`` times in a row."""
    misses = 0
    while True:
        await asyncio.sleep(_WEDGE_POLL_INTERVAL)
        if await server.is_responsive():
            misses = 0
        else:
            misses += 1
            if misses >= _WEDGE_POLL_MISSES:
                return


def _cut_short_result(
    watcher: SessionWatcher,
    duration_ms: int,
    session_id: str,
    *,
    end: SessionEnd,
    returncode: int | None,
    error: str | None = None,
) -> DriverResult:
    """A session that ended before its reply, reported with the spend and steps it did make."""
    return DriverResult(
        parsed=ParsedRun(
            error=error,
            cost_usd=watcher.cost if watcher.saw_steps else None,
            num_turns=watcher.turns,
        ),
        returncode=returncode,
        duration_ms=duration_ms,
        end=end,
        subagent_cost_usd=watcher.subagent_cost,
        session_id=session_id,
        wrap_up_reason=watcher.wrap_up_reason,
    )


async def _session_total_cost(
    client: httpx.AsyncClient, base_url: str, session_id: str
) -> float | None:
    """The session's aggregate cost, or None if it can't be read. Best-effort — a failure here must
    never turn a completed run into an error."""
    with contextlib.suppress(httpx.HTTPError, RuntimeError, ValueError):
        resp = await client.get(f"{base_url}/session/{session_id}")
        resp.raise_for_status()
        data = resp.json()
        cost = data.get("cost") if isinstance(data, dict) else None
        if isinstance(cost, (int, float)):
            return float(cost)
    return None


def _to_driver_result(
    final_message: dict[str, Any],
    watcher: SessionWatcher,
    duration_ms: int,
    session_id: str,
    fallback_cost: float | None = None,
) -> DriverResult:
    """Map the prompt response and the watched stream onto a DriverResult.

    The watcher supplies cost and turns whenever it saw any steps. If it saw none, the
    session's own aggregate is used when available, since the final message reports only its last
    step's cost.
    """
    info = final_message.get("info") or {}
    parts = final_message.get("parts") or []
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    # An intentional abort (budget or turn cap) leaves opencode's truncated response marked with an
    # error. That is expected, not a failure, so drop it and let the cap flags classify the result.
    aborted = watcher.budget_exceeded or watcher.turns_exceeded
    error = None if aborted else info.get("error")
    cost: float | None
    if watcher.saw_steps:
        cost, turns = watcher.cost, watcher.turns
        session_total = False
    else:
        session_total = fallback_cost is not None
        message_cost = info.get("cost")
        cost = fallback_cost
        if cost is None and isinstance(message_cost, (int, float)):
            cost = float(message_cost)
        turns = sum(1 for p in parts if p.get("type") == "step-finish")
    finish = info.get("finish")
    parsed = ParsedRun(
        text=text,
        structured=info.get("structured"),
        cost_usd=cost,
        num_turns=turns,
        error=json.dumps(error)[:500] if error else None,
        finish=finish if isinstance(finish, str) else None,
        cost_is_session_total=session_total,
        provider_failed=isinstance(error, dict) and error.get("name") in _PROVIDER_ERROR_NAMES,
    )
    return DriverResult(
        parsed=parsed,
        returncode=0,
        duration_ms=duration_ms,
        end=_session_end(watcher),
        subagent_cost_usd=watcher.subagent_cost,
        session_id=session_id,
        wrap_up_reason=watcher.wrap_up_reason,
    )


def _session_end(watcher: SessionWatcher) -> SessionEnd:
    """Which cap cut a replying session short — budget first, since it is the one that spends."""
    if watcher.budget_exceeded:
        return SessionEnd.OVER_BUDGET
    if watcher.turns_exceeded:
        return SessionEnd.TURNS_EXHAUSTED
    return SessionEnd.REPLIED


def _elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)
