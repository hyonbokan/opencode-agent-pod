"""Deployment-configured MCP servers exposed to agents.

The pod bakes in no MCP tools. A deployment declares its MCP servers in the env var
``AGENT_MCP_SERVERS`` — a JSON array of ``{name, command, tools}`` objects, where ``command`` is
the argv that launches a local server process and ``tools`` names the tools it exposes. The daemon
registers each server on startup (see the daemon pool) and the per-session ruleset gates their
tools by the run's allow-list (see the client). With none declared, an agent has only opencode's
built-in filesystem/exec tools.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

_ENV_VAR = "AGENT_MCP_SERVERS"


@dataclass(frozen=True)
class McpServer:
    """One local MCP server the deployment exposes: how to launch it and what tools it carries."""

    name: str
    command: tuple[str, ...]
    tools: tuple[str, ...] = ()

    def tool_ids(self) -> list[str]:
        """This server's tools in opencode's ``<server>_<tool>`` id form."""
        return [f"{self.name}_{tool}" for tool in self.tools]


def registered_servers() -> list[McpServer]:
    """MCP servers declared in ``AGENT_MCP_SERVERS``; empty when unset (the pod bakes in none)."""
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        return []
    return [
        McpServer(
            name=entry["name"],
            command=tuple(entry["command"]),
            tools=tuple(entry.get("tools", ())),
        )
        for entry in json.loads(raw)
    ]


def mcp_tool_ids() -> list[str]:
    """Every registered server's tool ids (``<server>_<tool>``), for allow-list gating."""
    return [tool_id for server in registered_servers() for tool_id in server.tool_ids()]
