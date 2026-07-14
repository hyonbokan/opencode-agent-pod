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
    # Run the key-injecting proxy so provider keys never enter the daemon or the shell it spawns.
    # On by default — it is what makes a shell-enabled run safe. Disable only for local debugging.
    key_proxy_enabled: bool
    # Interface the proxy binds. 127.0.0.1 keeps it unreachable off-host regardless of the pod's bind.
    key_proxy_host: str
    # Ceiling on a workspace pulled over the network — bounds both the download and, for an archive,
    # its expanded size, so a runaway or decompression-bomb source cannot fill the disk.
    workspace_max_bytes: int
    # Wall-clock cap on fetching a remote workspace before staging is abandoned.
    workspace_fetch_timeout: float
    # Hosts a remote (https) workspace source may point at. Empty = any https host (link-local /
    # metadata addresses are always refused); set it to pin fetches to known storage hosts.
    workspace_host_allowlist: tuple[str, ...]
    # Verify TLS when pulling a remote workspace. On by default. Turn off only to accept a
    # self-signed certificate from trusted local storage (e.g. a dev MinIO on the same host).
    workspace_tls_verify: bool


def _float_or_none(name: str) -> float | None:
    raw = os.getenv(name)
    return float(raw) if raw else None


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def _csv(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


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
        key_proxy_enabled=_bool("AGENT_POD_KEY_PROXY", True),
        key_proxy_host=os.getenv("AGENT_POD_KEY_PROXY_HOST", "127.0.0.1"),
        workspace_max_bytes=int(os.getenv("AGENT_POD_WORKSPACE_MAX_BYTES", str(2 * 1024**3))),
        workspace_fetch_timeout=float(os.getenv("AGENT_POD_WORKSPACE_FETCH_TIMEOUT", "60")),
        workspace_host_allowlist=_csv("AGENT_POD_WORKSPACE_HOST_ALLOWLIST"),
        workspace_tls_verify=_bool("AGENT_POD_WORKSPACE_TLS_VERIFY", True),
    )
