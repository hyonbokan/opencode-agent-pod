"""Manage the lifecycle of one ``opencode serve`` daemon.

A daemon is pinned to the directory it launches in, so one is started per scan against the cloned
project and multiplexes the scan's agents as HTTP sessions. This module owns only the process: launch
it on a free port, wait until it answers a health check, and reap the whole tree on shutdown. Nothing
is enforced here — permissions, tools, and structured output are set per session and per request.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from collections import deque
from uuid import uuid4

import httpx

from sdk_agent.opencode.driver import _drain_output, terminate_tree
from sdk_agent.opencode.events import EventConsumer, Watcher
from sdk_agent.opencode.providers import build_daemon_env

from config import config
from core.utils.logger import logger

_HEALTH_PATH = "/api/health"
# ~128 KiB of the daemon's own output kept for crash diagnostics.
_OUTPUT_TAIL_CHUNKS = 16
# Timeout for the shared client's quick control calls (session create/abort/delete).
_CLIENT_TIMEOUT = 30.0
# Env var tagged onto the daemon (and inherited by every tool process it spawns) so a crashed
# daemon's detached, reparented children can still be found and reaped on teardown.
_REAP_ENV_MARKER = "AUDITAGENT_OPENCODE_REAP_TAG"


def _free_port(host: str) -> int:
    """Return a free TCP port on the host.

    Between releasing it here and opencode binding it, the port could be taken; that surfaces as an
    early exit, which the start retry recovers from.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


class OpencodeServer:
    """A running ``opencode serve`` daemon for one scan's working directory.

    Use it as an async context manager, or call start/stop directly. Every provider's key is placed in
    the environment at launch so any model can be addressed, and any opencode config in the scanned
    repo is ignored so tool and permission grants come only from the per-session ruleset.
    """

    def __init__(
        self,
        cwd: str,
        *,
        env: dict[str, str] | None = None,
        host: str = "127.0.0.1",
        startup_timeout: float | None = None,
    ) -> None:
        self._cwd = cwd
        self._host = host
        self._env = build_daemon_env(env)
        self._env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
        self._reap_tag = uuid4().hex
        self._env[_REAP_ENV_MARKER] = self._reap_tag
        self._startup_timeout = (
            startup_timeout
            if startup_timeout is not None
            else config.scan.SDK_SERVE_STARTUP_TIMEOUT
        )
        self._proc: asyncio.subprocess.Process | None = None
        self._port: int | None = None
        self._output: deque[str] = deque(maxlen=_OUTPUT_TAIL_CHUNKS)
        self._drain_task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        self._events: EventConsumer | None = None

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise RuntimeError("OpencodeServer is not started")
        return f"http://{self._host}:{self._port}"

    @property
    def client(self) -> httpx.AsyncClient:
        """One HTTP client shared across the scan's sessions, so calls reuse keep-alive connections."""
        if self._client is None:
            raise RuntimeError("OpencodeServer is not started")
        return self._client

    @property
    def is_alive(self) -> bool:
        """Whether the daemon process is running (started and not yet exited)."""
        return self._proc is not None and self._proc.returncode is None

    @property
    def events_ready(self) -> bool:
        """Whether the shared event stream is connected, so budget/turn caps can be enforced."""
        return self._events is not None and self._events.connected

    def add_watcher(self, session_id: str, watcher: Watcher) -> None:
        """Register a run's watcher to receive its session's events from the shared consumer."""
        if self._events is None:
            raise RuntimeError("OpencodeServer is not started")
        self._events.add_watcher(session_id, watcher)

    def remove_watcher(self, session_id: str) -> None:
        """Drop a run's watcher once its session is finished."""
        if self._events is not None:
            self._events.remove_watcher(session_id)

    async def start(self, attempts: int = 3) -> OpencodeServer:
        """Launch the daemon and wait until it is healthy, retrying on a lost-port race.

        Each attempt picks a fresh port; a daemon that exits early or never answers the health check
        is torn down and retried.
        """
        last_error: Exception | None = None
        for attempt in range(attempts):
            port = _free_port(self._host)
            proc = await asyncio.create_subprocess_exec(
                "opencode",
                "serve",
                "--port",
                str(port),
                "--hostname",
                self._host,
                "--log-level",
                "ERROR",
                cwd=self._cwd,
                env=self._env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self._proc = proc
            self._port = port
            self._drain_task = asyncio.create_task(_drain_output(proc.stdout, self._output))
            try:
                await self._await_healthy()
                self._client = httpx.AsyncClient(timeout=_CLIENT_TIMEOUT)
                # Open the shared event stream before declaring the daemon ready: a run's caps are
                # enforced off it, so a daemon that can't stream events must fail startup (and be
                # retried) rather than serve runs whose budget/turn caps silently can't fire.
                self._events = EventConsumer(self.base_url, self._client)
                await self._events.start()
                logger.info("opencode serve healthy at %s (pid=%d)", self.base_url, proc.pid)
                return self
            except Exception as e:
                last_error = e
                logger.warning(
                    "opencode serve attempt %d/%d on port %d failed: %s",
                    attempt + 1,
                    attempts,
                    port,
                    e,
                )
                await self.stop()

        raise RuntimeError(
            f"opencode serve failed to start after {attempts} attempts: {last_error}"
        )

    async def _await_healthy(self) -> None:
        """Poll the health endpoint until it returns 200 or the startup budget runs out, failing
        early if the daemon process has already exited."""
        assert self._proc is not None
        deadline = time.monotonic() + self._startup_timeout
        url = self.base_url + _HEALTH_PATH
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self._proc.returncode is not None:
                    raise RuntimeError(
                        f"daemon exited during startup (rc={self._proc.returncode}): "
                        f"{self._output_tail()}"
                    )
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass  # not accepting connections yet
                await asyncio.sleep(0.2)
        raise TimeoutError(
            f"daemon not healthy within {self._startup_timeout:.0f}s: {self._output_tail()}"
        )

    def _output_tail(self) -> str:
        return "".join(self._output).strip()[-2000:]

    async def stop(self) -> None:
        """Reap the daemon and its children, close the shared client, and settle the output drain.
        Safe to call twice."""
        if self._events is not None:
            with contextlib.suppress(Exception):
                await self._events.stop()  # stop the stream reader before its client is closed
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
        if self._proc is not None:
            await terminate_tree(self._proc, reap_marker=(_REAP_ENV_MARKER, self._reap_tag))
        if self._drain_task is not None:
            with contextlib.suppress(Exception):
                await self._drain_task
        self._proc = None
        self._port = None
        self._drain_task = None
        self._client = None
        self._events = None

    async def __aenter__(self) -> OpencodeServer:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()
