"""Trace-ingestion hook for an agent run's timeline.

A no-op placeholder: observability is added by implementing this function; the
runner already calls it off its critical path, so leaving it inert is safe.
"""

from __future__ import annotations

from typing import Any


def record_opencode_trace(timeline: Any, *, model: str, input_message: str) -> None:
    """Record a run's event timeline to the trace backend. Currently does nothing."""
    return None
