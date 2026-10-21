"""What one ``opencode serve`` session reports back: how it ended and what it produced."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SessionEnd(StrEnum):
    """How one session stopped — the single fact a result carries about its ending."""

    REPLIED = "replied"  # the daemon returned a final message, however the model itself finished
    TIMED_OUT = "timed_out"
    # The daemon stopped answering health probes mid-run, distinct from a plain wall-clock timeout
    # where it is still responsive. Callers can recycle the daemon and retry on this signal alone.
    WEDGED = "wedged"
    OVER_BUDGET = "over_budget"
    TURNS_EXHAUSTED = "turns_exhausted"
    TRANSPORT_LOST = "transport_lost"  # the daemon went away under the session


class WrapUpReason(StrEnum):
    """Which limit made a run ask its sessions to wrap up."""

    BUDGET = "budget"
    TURNS = "turns"
    TIME = "time"


@dataclass
class DriverResult:
    parsed: ParsedRun
    returncode: int | None
    duration_ms: int
    end: SessionEnd = SessionEnd.REPLIED
    subagent_cost_usd: float = 0.0  # the part of the run's spend made by subagent steps
    # The opencode session this run used. A wedged run's session outlives the dead daemon (opencode's
    # store is daemon-independent), so the runner can continue it on a fresh daemon.
    session_id: str | None = None
    wrap_up_reason: WrapUpReason | None = None


@dataclass
class ParsedRun:
    """The information the runner needs, reconstructed from the event stream."""

    text: str = ""
    structured: Any | None = None  # structured output from the run (validated by the caller)
    cost_usd: float | None = None  # None when no spend was observed, distinct from a $0 run
    num_turns: int = 0
    error: str | None = None
    finish: str | None = None  # the final message's finish reason; "stop" means the model concluded
    # Whether cost_usd is the session's whole spend rather than this run's own. Set only when the
    # live stream gave nothing and the session's aggregate was read instead — on a resumed session
    # that aggregate already covers the earlier attempts, so the caller must not add them again.
    cost_is_session_total: bool = False
    # The provider failed the call after opencode's own retries, so another model could still answer
    provider_failed: bool = False
