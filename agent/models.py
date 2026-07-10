from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOOLS: list[str] = [
    "Read",
    "Glob",
    "Grep",
    "Bash",
]

# ---------------------------------------------------------------------------
# Public result model
# ---------------------------------------------------------------------------


class OpencodeResult(BaseModel):
    """Result from an agent execution."""

    text: str
    is_error: bool = False
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    num_turns: int | None = None
    structured_output: Any | None = None
    # How the run ended, for callers that diagnose without re-reading the text: "success",
    # "max_turns", or one of "error_timeout" / "error_max_budget_usd" / "error".
    subtype: str | None = None
