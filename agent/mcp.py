"""Deployment-configured MCP servers exposed to agents.

The pod bakes in no MCP tools. A deployment declares its MCP servers in the env var
``AGENT_MCP_SERVERS`` — a JSON array of ``{name, command, tools}`` objects, where ``command`` is
the argv that launches a local server process and ``tools`` names the tools it exposes. Every
declared server is registered on each daemon at startup; which tools a run may call is decided by
its own allow-list. With none declared, an agent has only opencode's built-in filesystem/exec tools.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

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


def mcp_local_config(server: McpServer) -> dict[str, Any]:
    """Build the ``POST /mcp`` config that registers one local MCP server on the daemon.

    The command comes straight from deployment config; the pod injects nothing, so a server that
    needs the workspace directory must carry it in its own declared argv.
    """
    return {"type": "local", "command": list(server.command), "enabled": True}


async def register_mcp(
    base_url: str,
    name: str,
    config: dict[str, Any],
    *,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    """Register an MCP server on the daemon, failing unless it reports connected.

    Registration is daemon-wide, so it happens once after the daemon is healthy, not per request.
    A server process that never spawns, or spawns and dies, still gets a 200 response — only the
    carried status says so. Left unchecked, every agent on the daemon would silently run without
    the server's tools, so a non-connected status raises here instead.
    """
    resp = await client.post(f"{base_url}/mcp", json={"name": name, "config": config})
    resp.raise_for_status()
    payload = resp.json()
    entry = payload.get(name) if isinstance(payload, dict) else None
    entry = entry if isinstance(entry, dict) else {}
    if entry.get("status") != "connected":
        raise RuntimeError(
            f"MCP server {name!r} failed to connect: {entry.get('error') or entry or payload}"
        )
    return payload
