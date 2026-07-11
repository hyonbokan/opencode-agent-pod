"""A local reverse proxy that injects real provider API keys on egress.

The opencode daemon and the shell commands it spawns are pointed at ``http://127.0.0.1:<port>/<provider>``
with a worthless dummy key, so no real key is ever in their environment. This proxy is the only
holder of the real keys: it runs in the pod process, matches each request by its leading
``/<provider>`` path segment, drops whatever auth the daemon sent, injects the real key in that
provider's header format, and streams the request on to the real upstream. A run can therefore use a
model but can never read a key from its own environment.

It binds 127.0.0.1 only, so it is never reachable off the host regardless of how the pod itself
binds. Restricting where the shell can reach on the network is a separate, deploy-time concern.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from agent.opencode.providers import CATALOG_KEY_VARS, load_custom_providers
from core.utils.logger import logger

# Catalog provider segment -> its real upstream and how its key is carried on the wire. The upstream
# includes the API version segment, because the SDK treats it as part of the base URL and appends
# only the endpoint (e.g. "messages"). Custom providers forward to their declared base_url, which
# already carries any version segment, with bearer auth.
_CATALOG_UPSTREAMS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "xai": "https://api.x.ai/v1",
}
_CATALOG_AUTH: dict[str, str] = {
    "anthropic": "anthropic",  # x-api-key header
    "openai": "bearer",  # Authorization: Bearer
    "google": "google",  # x-goog-api-key header
    "xai": "bearer",
}

# Hop-by-hop headers not forwarded, plus the auth headers we replace rather than pass through.
_HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
    }
)
_INCOMING_AUTH = frozenset({"authorization", "x-api-key", "x-goog-api-key", "api-key"})

# The HTTP methods a provider API can use; Starlette needs them named, not a wildcard.
_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]


@dataclass(frozen=True)
class ProviderRoute:
    """Where one provider's traffic goes and the real key to inject on the way out."""

    upstream: str
    api_key: str
    auth: str  # "anthropic" | "google" | "bearer"


def build_routes(env: Mapping[str, str]) -> dict[str, ProviderRoute]:
    """Build the routing table: one entry per provider that has a real key, read from the environment.

    A catalog provider is included only when a real key for it is present; a custom provider is routed
    to the base URL it declared. The real key stays here and is never handed to the daemon.
    """
    routes: dict[str, ProviderRoute] = {}
    for seg, key_vars in CATALOG_KEY_VARS.items():
        key = next((env[v] for v in key_vars if env.get(v)), None)
        if key:
            routes[seg] = ProviderRoute(_CATALOG_UPSTREAMS[seg], key, _CATALOG_AUTH[seg])
    for p in load_custom_providers(env):
        key = (env.get(p.api_key_env) if p.api_key_env else None) or p.api_key or ""
        routes[p.name] = ProviderRoute(p.base_url, key, "bearer")
    return routes


def _inject_auth(headers: dict[str, str], route: ProviderRoute) -> None:
    """Set the real key on the outbound request in the provider's expected header format."""
    if route.auth == "anthropic":
        headers["x-api-key"] = route.api_key
        headers.setdefault("anthropic-version", "2023-06-01")
    elif route.auth == "google":
        headers["x-goog-api-key"] = route.api_key
    else:
        headers["authorization"] = f"Bearer {route.api_key}"


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


class KeyProxy:
    """A running localhost reverse proxy that injects real provider keys on egress.

    Start it once per process; every daemon points at ``base_url``. It holds the real keys for the
    process lifetime and returns them to no one — callers get model responses, never a key.
    """

    def __init__(
        self,
        routes: dict[str, ProviderRoute],
        *,
        host: str = "127.0.0.1",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._routes = routes
        self._host = host
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
        self._owns_client = client is None
        self._app = Starlette(
            routes=[Route("/{provider}/{path:path}", self._handle, methods=_METHODS)]
        )
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._port: int | None = None

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise RuntimeError("KeyProxy is not started")
        return f"http://{self._host}:{self._port}"

    async def _handle(self, request: Request) -> Response:
        """Proxy one request to its provider's upstream, replacing the daemon's dummy key."""
        provider = request.path_params["provider"]
        route = self._routes.get(provider)
        if route is None:
            return JSONResponse({"error": f"unknown provider {provider!r}"}, status_code=404)
        url = f"{route.upstream.rstrip('/')}/{request.path_params['path']}"
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() not in _INCOMING_AUTH
        }
        _inject_auth(headers, route)
        # A dummy key is sometimes carried as a ?key= query param (Google); drop it so ours wins.
        params = httpx.QueryParams(
            [(k, v) for k, v in request.query_params.multi_items() if k != "key"]
        )
        upstream_req = self._client.build_request(
            request.method, url, headers=headers, params=params, content=await request.body()
        )
        try:
            upstream = await self._client.send(upstream_req, stream=True)
        except httpx.HTTPError as e:
            logger.warning("key proxy upstream error for %s: %s", provider, e)
            return JSONResponse({"error": f"upstream request failed: {e}"}, status_code=502)
        resp_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in {"content-length", "transfer-encoding", "connection"}
        }
        return StreamingResponse(
            _relay(upstream),
            status_code=upstream.status_code,
            headers=resp_headers,
            background=BackgroundTask(upstream.aclose),
        )

    async def start(self, timeout: float = 10.0) -> KeyProxy:
        """Bind a free localhost port and serve until stopped."""
        self._port = _free_port(self._host)
        config = uvicorn.Config(
            self._app, host=self._host, port=self._port, log_level="error", lifespan="off"
        )
        self._server = uvicorn.Server(config)
        # Don't let an embedded server touch process signal handlers (fails off the main thread and
        # would fight the pod's own).
        self._server.install_signal_handlers = lambda: None  # type: ignore[attr-defined,method-assign]
        self._task = asyncio.create_task(self._server.serve())
        deadline = timeout / 0.02
        waited = 0
        while not self._server.started:
            if self._task.done():
                self._task.result()  # surface the startup exception
                raise RuntimeError("key proxy exited during startup")
            if waited > deadline:
                raise TimeoutError(f"key proxy not ready within {timeout:.0f}s")
            waited += 1
            await asyncio.sleep(0.02)
        logger.info(
            "key proxy listening at %s for %d provider(s)", self.base_url, len(self._routes)
        )
        return self

    async def stop(self) -> None:
        """Stop serving and close the owned HTTP client. Safe to call twice."""
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception):
                await self._task
        if self._owns_client:
            with contextlib.suppress(Exception):
                await self._client.aclose()
        self._server = None
        self._task = None
        self._port = None


async def _relay(upstream: httpx.Response) -> AsyncIterator[bytes]:
    """Stream the upstream body through unchanged, so SSE token chunks arrive live."""
    async for chunk in upstream.aiter_raw():
        yield chunk
