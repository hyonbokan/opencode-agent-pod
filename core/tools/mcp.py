"""Tool/MCP identifiers consumed by the daemon and the session client.

Placeholder values pending the pluggable tool/MCP registry — the pod registers
no MCP tools yet, so the allowed-tool set is empty.
"""

from __future__ import annotations

MCP_SERVER_NAME = "opencode-agent"
SERVER_SCRIPT = ""


def get_allowed_tool_names() -> list[str]:
    """Names of MCP tools to expose to an agent. Empty until the registry is added."""
    return []
