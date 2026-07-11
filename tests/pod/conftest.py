from __future__ import annotations

import pytest

from pod.settings import PodSettings


@pytest.fixture
def settings() -> PodSettings:
    """Pod settings for tests: a known token, no budget bounds, a fast keep-alive tick."""
    return PodSettings(
        bearer_token="secret",
        max_budget_usd=None,
        default_max_budget_usd=None,
        session_timeout=900.0,
        max_turns=30,
        keepalive_seconds=0.02,
        host="127.0.0.1",
        port=8080,
    )
