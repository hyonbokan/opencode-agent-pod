"""The FastAPI app: a bearer-guarded ``POST /agent/run`` that streams one autonomous run as SSE.

Internal, trusted-network service (DESIGN §11): a shared bearer token gates every run, each run is
budget-capped and workspace-isolated, and daemons are drained at shutdown. Not a public endpoint.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from agent.opencode.daemon_pool import shutdown_opencode_daemons
from agent.opencode.providers import KEY_PROXY_ENV
from agent.runner import flush_pending_traces
from core.utils.logger import logger
from pod.key_proxy import KeyProxy, build_routes
from pod.schema import RunRequest
from pod.service import drain_cleanups, run_events
from pod.settings import PodSettings, load_settings
from pod.workspace import supported_scheme


def create_app(settings: PodSettings | None = None) -> FastAPI:
    """Build the app. Settings are injected for tests; production loads them from the environment."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Start the key-injecting proxy first and publish its URL through AGENT_KEY_PROXY_URL, so
        # every daemon launched after this routes providers through it with a dummy key and no real
        # key ever reaches a shell command the daemon spawns.
        proxy: KeyProxy | None = None
        if settings.key_proxy_enabled:
            proxy = KeyProxy(build_routes(dict(os.environ)), host=settings.key_proxy_host)
            await proxy.start()
            os.environ[KEY_PROXY_ENV] = proxy.base_url
        else:
            logger.warning("key proxy disabled: provider keys are visible to the shell tool")
        try:
            yield
        finally:
            # Let in-flight run cleanups finish, drain traces, then reap any daemon still pooled,
            # and only then stop the proxy the daemons were talking to.
            await drain_cleanups()
            await flush_pending_traces()
            await shutdown_opencode_daemons()
            if proxy is not None:
                os.environ.pop(KEY_PROXY_ENV, None)
                await proxy.stop()

    app = FastAPI(title="opencode-agent-pod", lifespan=lifespan)

    async def require_bearer(authorization: str | None = Header(default=None)) -> None:
        """Gate a run on the shared bearer token, failing closed if the pod has none configured."""
        if settings.bearer_token is None:
            raise HTTPException(status_code=503, detail="pod authentication is not configured")
        expected = f"Bearer {settings.bearer_token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/agent/run", dependencies=[Depends(require_bearer)])
    async def agent_run(request: RunRequest) -> StreamingResponse:
        # Reject an unstageable workspace up front so it is a clean 4xx, not a mid-stream error.
        if request.workspace is not None and not supported_scheme(request.workspace.source):
            raise HTTPException(
                status_code=400, detail=f"unsupported workspace source: {request.workspace.source}"
            )
        return StreamingResponse(
            run_events(request, settings),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return app
