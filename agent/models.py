from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from agent.opencode.driver import WrapUpReason

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOOLS: list[str] = [
    "Read",
    "Glob",
    "Grep",
    "Bash",
]


class ReasoningEffort(StrEnum):
    """Reasoning effort a run asks of the model; MAX is the strongest level the model accepts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


# ---------------------------------------------------------------------------
# Public result model
# ---------------------------------------------------------------------------


class RunSubtype(StrEnum):
    """How one agent run turned out, for callers that diagnose without re-reading the text."""

    SUCCESS = "success"
    MAX_TURNS = "max_turns"  # a designed ceiling; the output up to it stands
    ERROR = "error"
    ERROR_TIMEOUT = "error_timeout"
    ERROR_MAX_BUDGET_USD = "error_max_budget_usd"
    ERROR_TRANSPORT = "error_transport"  # the daemon went away, not the agent
    ERROR_INCOMPLETE = "error_incomplete"  # the session ended without the model concluding
    ERROR_SESSION_START = "error_session_start"
    ERROR_STRUCTURED_OUTPUT = "error_structured_output"
    ERROR_PROVIDER = "error_provider"  # the provider failed the call after opencode's own retries


class OpencodeResult(BaseModel):
    """Result from an agent execution."""

    text: str
    is_error: bool = False
    total_cost_usd: float | None = None
    # The part of total_cost_usd spent by subagents the agent started
    subagent_cost_usd: float | None = None
    duration_ms: int | None = None
    num_turns: int | None = None
    structured_output: Any | None = None
    subtype: RunSubtype | None = None
    # The final message's finish reason as reported by the engine
    finish: str | None = None
    # Set when the run was asked to wrap up before a limit
    wrap_up_reason: WrapUpReason | None = None

    def add_attempt(self, attempt: OpencodeResult, *, cost_already_counted: bool = False) -> None:
        """Fold another attempt of the same run into this one: its spend and time, not its outcome.

        Pass ``cost_already_counted`` when this result's own cost is a session aggregate covering
        ``attempt`` as well, so only its wall-clock is still missing.
        """
        if not cost_already_counted:
            self.total_cost_usd = _summed(self.total_cost_usd, attempt.total_cost_usd)
            self.subagent_cost_usd = _summed(self.subagent_cost_usd, attempt.subagent_cost_usd)
        self.duration_ms = _summed(self.duration_ms, attempt.duration_ms)


def _summed[N: (int, float)](running: N | None, attempt: N | None) -> N | None:
    """The running total plus one attempt's value; None only where neither reported a number."""
    if running is None:
        return attempt
    if attempt is None:
        return running
    return running + attempt
