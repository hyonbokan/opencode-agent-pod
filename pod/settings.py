"""Env-driven pod settings. Nothing here is hardcoded — ports, the token, and caps all come from
the environment so a deployment configures the pod without code changes."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PodSettings:
    """One pod's runtime configuration, loaded from the environment at startup."""

    # Shared bearer token every caller must present. None = unset, and the pod then refuses every
    # request (fail closed) rather than serving unauthenticated.
    bearer_token: str | None
    # Hard ceiling clamped onto every run's budget, even one that asks for more — anyone who can
    # call can spend, so the pod bounds it. None = no ceiling.
    max_budget_usd: float | None
    # Budget applied when a request names none. None = unbounded unless the ceiling bounds it.
    default_max_budget_usd: float | None
    # Wall-clock cap and turn cap applied to every run.
    session_timeout: float
    max_turns: int
    # Seconds between SSE keep-alive comments while a run is in flight, so the connection and any
    # intermediary don't time out during a minutes-long autonomous run.
    keepalive_seconds: float
    host: str
    port: int


def _float_or_none(name: str) -> float | None:
    raw = os.getenv(name)
    return float(raw) if raw else None


def load_settings() -> PodSettings:
    """Build settings from ``AGENT_POD_*`` environment variables."""
    return PodSettings(
        bearer_token=os.getenv("AGENT_POD_TOKEN") or None,
        max_budget_usd=_float_or_none("AGENT_POD_MAX_BUDGET_USD"),
        default_max_budget_usd=_float_or_none("AGENT_POD_DEFAULT_BUDGET_USD"),
        session_timeout=float(os.getenv("AGENT_POD_SESSION_TIMEOUT", "900")),
        max_turns=int(os.getenv("AGENT_POD_MAX_TURNS", "30")),
        keepalive_seconds=float(os.getenv("AGENT_POD_KEEPALIVE_SECONDS", "15")),
        host=os.getenv("AGENT_POD_HOST", "127.0.0.1"),
        port=int(os.getenv("AGENT_POD_PORT", "8080")),
    )
