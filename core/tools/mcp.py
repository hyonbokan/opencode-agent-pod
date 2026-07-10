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


def is_mcp_configured() -> bool:
    """Whether a real MCP server is wired up to register on the daemon.

    False while the registry is a placeholder (no server script), so the daemon skips registration
    rather than failing to connect a server that does not exist yet. Flips to True — and gates on the
    real registry — when the pluggable tool/MCP registry lands.
    """
    return bool(SERVER_SCRIPT)
