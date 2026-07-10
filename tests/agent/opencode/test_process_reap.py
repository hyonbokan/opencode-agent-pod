"""Process-reaping tests for terminate_tree — the daemon-crash orphaned-child leak (finding #7).

opencode spawns each bash-tool command detached (its own session/group) with the daemon's environment
inherited. When the daemon crashes, those children reparent to init but keep their own group, so
neither the daemon's process group nor a parent-pid walk can find them. The reap therefore leans on an
inherited environment marker, which these tests exercise against real processes (no opencode needed —
the leak is a process-topology property, reproduced here with plain sleepers)."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import time
import uuid

import psutil
import pytest

from agent.opencode.driver import _reap_by_env_marker, terminate_tree

_MARKER = "OPENCODE_POD_REAP_TAG"
# A sleeper long enough to outlive the test but self-terminating if cleanup ever leaks it.
_SLEEPER = [sys.executable, "-c", "import time; time.sleep(120)"]


def _spawn_detached(env_extra: dict[str, str]) -> subprocess.Popen:
    """A detached (own session/group) sleeper carrying env_extra — mirrors opencode's tool spawn."""
    return subprocess.Popen(
        _SLEEPER,
        env={**os.environ, **env_extra},
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _kill(*procs: subprocess.Popen) -> None:
    for p in procs:
        with contextlib.suppress(Exception):
            p.kill()
            p.wait(timeout=5)


def _await_environ_readable(pid: int, timeout: float = 5.0) -> None:
    """Spin until the process's environ can be read (it has finished exec)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(psutil.Error):
            psutil.Process(pid).environ()
            return
        time.sleep(0.05)
    raise AssertionError(f"environ for pid {pid} never became readable")


def test_reap_by_env_marker_kills_only_the_tagged_process():
    tag = uuid.uuid4().hex
    marked = _spawn_detached({_MARKER: tag})
    # A control process with a *different* tag must be left untouched — the marker is per-daemon.
    other = _spawn_detached({_MARKER: uuid.uuid4().hex})
    try:
        _await_environ_readable(marked.pid)
        _await_environ_readable(other.pid)

        killed = _reap_by_env_marker(_MARKER, tag, exclude=os.getpid())

        assert marked.pid in killed
        assert other.pid not in killed
        marked.wait(timeout=5)
        assert marked.poll() is not None  # the tagged one died
        assert other.poll() is None  # the other-tag one survived
    finally:
        _kill(marked, other)


def test_reap_by_env_marker_excludes_given_pid():
    # The daemon manager passes its own pid as exclude so the sweep can never target the caller, even
    # if the caller happened to carry the marker.
    tag = uuid.uuid4().hex
    marked = _spawn_detached({_MARKER: tag})
    try:
        _await_environ_readable(marked.pid)
        killed = _reap_by_env_marker(_MARKER, tag, exclude=marked.pid)
        assert killed == []  # the only match was excluded
        assert marked.poll() is None  # so it survives
    finally:
        _kill(marked)


@pytest.mark.asyncio
async def test_terminate_tree_reaps_detached_child_after_daemon_crash():
    """The headline case: daemon crashes while a detached tool child runs; teardown must reap it."""
    tag = uuid.uuid4().hex
    # A "daemon" that spawns one detached grandchild (env inherited, so it carries the tag too) and
    # prints its pid. Mirrors opencode launching a detached bash-tool command.
    daemon_src = (
        "import subprocess, sys, time\n"
        "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "                      start_new_session=True,\n"
        "                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "sys.stdout.write(str(gc.pid) + '\\n'); sys.stdout.flush()\n"
        "time.sleep(120)\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        daemon_src,
        env={**os.environ, _MARKER: tag},
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    gc_pid: int | None = None
    try:
        assert proc.stdout is not None
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
        gc_pid = int(line.strip())

        # The grandchild is detached: its own group leader, in a *different* group from the daemon, so
        # a pgid-based reap of the daemon's group can never reach it.
        daemon_pgid = os.getpgid(proc.pid)
        assert os.getpgid(gc_pid) == gc_pid
        assert os.getpgid(gc_pid) != daemon_pgid

        # Crash the daemon and reap it, so the grandchild reparents to init (ppid == 1) and leaks.
        proc.kill()
        await proc.wait()
        assert proc.returncode is not None
        assert psutil.pid_exists(gc_pid)  # still alive: nothing has reaped it yet

        # Old behavior (no marker): group kill + parent-pid walk both miss the orphan — it survives.
        await terminate_tree(proc)
        assert psutil.pid_exists(gc_pid)

        # The fix: the marker sweep finds it by inherited env and reaps it.
        await terminate_tree(proc, reap_marker=(_MARKER, tag))
        for _ in range(50):
            if not psutil.pid_exists(gc_pid):
                break
            await asyncio.sleep(0.1)
        assert not psutil.pid_exists(gc_pid)
    finally:
        if gc_pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(gc_pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()


@pytest.mark.asyncio
async def test_terminate_tree_reaps_detached_child_while_daemon_alive():
    """Normal shutdown: a detached child of a *live* daemon is reaped even without the marker, via the
    parent-pid descendant walk — the marker sweep is the crash backstop, not the only path."""
    tag = uuid.uuid4().hex
    daemon_src = (
        "import subprocess, sys, time\n"
        "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "                      start_new_session=True,\n"
        "                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "sys.stdout.write(str(gc.pid) + '\\n'); sys.stdout.flush()\n"
        "time.sleep(120)\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        daemon_src,
        env={**os.environ, _MARKER: tag},
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    gc_pid: int | None = None
    try:
        assert proc.stdout is not None
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
        gc_pid = int(line.strip())
        assert psutil.pid_exists(gc_pid)

        # Daemon still alive here; no marker passed. The descendant walk (child still parented to the
        # live daemon) plus the group kills should still take the whole tree down.
        await terminate_tree(proc)
        assert proc.returncode is not None
        for _ in range(50):
            if not psutil.pid_exists(gc_pid):
                break
            await asyncio.sleep(0.1)
        assert not psutil.pid_exists(gc_pid)
    finally:
        if gc_pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(gc_pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
