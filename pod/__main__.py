"""Serve the pod: ``python -m pod`` (host/port from ``AGENT_POD_HOST``/``AGENT_POD_PORT``)."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from pod.app import create_app
from pod.settings import load_settings


def _load_dotenv() -> None:
    """Populate the environment from a local ``.env`` next to the project root.

    Only fills variables that are not already set, so real environment values (a
    container's ``-e``/``--env-file``, a secret manager) always win and a missing
    ``.env`` is a silent no-op. This is what lets ``python -m pod`` pick up the
    bearer token and provider keys locally without exporting them by hand.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    _load_dotenv()
    settings = load_settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
