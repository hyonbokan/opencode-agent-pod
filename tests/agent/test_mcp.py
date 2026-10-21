"""Tests for registering a deployment's MCP servers on a daemon. The daemon is faked with an httpx
MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from agent.mcp import McpServer, mcp_local_config, register_mcp

_DEMO_SERVER = McpServer(name="demo", command=("python", "srv.py"), tools=("lookup", "fetch"))


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_mcp_local_config_builds_local_config_from_server_command():
    config = mcp_local_config(_DEMO_SERVER)
    assert config["type"] == "local"
    assert config["enabled"] is True
    # The command is the server's declared argv, verbatim — the pod injects nothing.
    assert config["command"] == ["python", "srv.py"]


@pytest.mark.asyncio
async def test_register_mcp_posts_config_and_returns_status():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={_DEMO_SERVER.name: {"status": "connected"}})

    client = _mock_client(handler)
    try:
        status = await register_mcp(
            "http://d", _DEMO_SERVER.name, mcp_local_config(_DEMO_SERVER), client=client
        )
    finally:
        await client.aclose()

    assert seen["path"] == "/mcp"
    assert seen["body"]["name"] == _DEMO_SERVER.name
    assert seen["body"]["config"]["type"] == "local"
    assert status[_DEMO_SERVER.name]["status"] == "connected"


@pytest.mark.asyncio
async def test_register_mcp_raises_when_server_fails_to_connect():
    # The daemon answers 200 even when the server process never comes up; only the carried status
    # says so. Registration must fail loudly, or every agent would silently run without the tools.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={_DEMO_SERVER.name: {"status": "failed", "error": "MCP error -32000"}},
        )

    client = _mock_client(handler)
    try:
        with pytest.raises(RuntimeError, match="failed to connect.*-32000"):
            await register_mcp(
                "http://d", _DEMO_SERVER.name, mcp_local_config(_DEMO_SERVER), client=client
            )
    finally:
        await client.aclose()
