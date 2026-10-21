"""Health-probe conformance: the predicate every recycle decision is made on."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from agent.opencode.server import OpencodeServer


async def _serve(handler) -> tuple[asyncio.Server, int]:
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def _attach(port: int) -> OpencodeServer:
    """An OpencodeServer wired to an already-listening port, without launching opencode."""
    srv = OpencodeServer(cwd=".")
    srv._port = port
    srv._client = httpx.AsyncClient()
    return srv


@pytest.mark.asyncio
async def test_wedged_daemon_that_accepts_but_never_answers_is_not_responsive():
    # The real wedge: the process is up and the socket still accepts, so a connect check and a
    # process check both pass. Only an answered request separates it from a healthy daemon.
    release = asyncio.Event()

    async def never_answer(_reader, writer):
        await release.wait()
        writer.close()

    server, port = await _serve(never_answer)
    srv = _attach(port)
    try:
        start = time.monotonic()
        assert await srv.is_responsive(timeout=0.5) is False
        # Bounded by the probe timeout: an unbounded probe would hang the caller it protects.
        assert time.monotonic() - start < 5.0
    finally:
        release.set()
        await srv.client.aclose()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_daemon_answering_health_is_responsive():
    async def answer_ok(reader, writer):
        await reader.read(1024)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        writer.close()

    server, port = await _serve(answer_ok)
    srv = _attach(port)
    try:
        assert await srv.is_responsive(timeout=5.0) is True
    finally:
        await srv.client.aclose()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_non_200_health_is_not_responsive():
    async def answer_503(reader, writer):
        await reader.read(1024)
        writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()

    server, port = await _serve(answer_503)
    srv = _attach(port)
    try:
        assert await srv.is_responsive(timeout=5.0) is False
    finally:
        await srv.client.aclose()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_refused_connection_is_not_responsive():
    async def unreachable(_reader, _writer):
        raise AssertionError("nothing should connect to a closed listener")

    server, port = await _serve(unreachable)
    server.close()
    await server.wait_closed()
    srv = _attach(port)
    try:
        assert await srv.is_responsive(timeout=5.0) is False
    finally:
        await srv.client.aclose()


@pytest.mark.asyncio
async def test_stopped_daemon_is_not_responsive():
    # After stop() clears the shared client there is nothing to probe, and the caller must not crash.
    srv = OpencodeServer(cwd=".")
    assert await srv.is_responsive() is False
