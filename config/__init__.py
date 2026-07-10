"""Runtime configuration for the pod, read from the environment."""

import os


class _RunnerConfig:
    GLOBAL_CONCURRENCY = int(os.getenv("AGENT_GLOBAL_CONCURRENCY", "7"))
    SERVE_STARTUP_TIMEOUT = float(os.getenv("AGENT_SERVE_STARTUP_TIMEOUT", "60"))


class _Config:
    runner = _RunnerConfig()


config = _Config()
