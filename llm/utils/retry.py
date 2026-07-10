"""Backoff configuration for retrying a failed operation."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class RetryConfig:
    """Exponential-backoff parameters for a retry loop."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    def get_delay(self, attempt: int, error: Exception | None = None) -> float:
        """Seconds to wait before the next attempt.

        A provider-supplied retry-after hint on the error wins when present;
        otherwise exponential backoff with optional ±25% jitter, capped.
        """
        retry_after = getattr(error, "retry_after", None)
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            delay = min(float(retry_after), self.max_delay)
        else:
            delay = min(self.base_delay * (self.exponential_base**attempt), self.max_delay)
        if self.jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        return max(0.0, delay)
