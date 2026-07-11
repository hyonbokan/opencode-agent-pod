"""Serve the pod: ``python -m pod`` (host/port from ``AGENT_POD_HOST``/``AGENT_POD_PORT``)."""

from __future__ import annotations

import uvicorn

from pod.app import create_app
from pod.settings import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
