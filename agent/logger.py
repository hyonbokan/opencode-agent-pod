"""Process-wide logger for the pod."""

import logging
import os

_level = (
    logging.INFO
    if os.getenv("ENVIRONMENT", "development").lower() == "production"
    else logging.DEBUG
)

logger = logging.getLogger("opencode-agent-pod")
logger.setLevel(_level)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False
