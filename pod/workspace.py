"""Stage a caller's workspace-by-reference into an ephemeral working dir, then reap it.

The pod caches nothing: each run reads the source into a throwaway directory, runs there, and the
directory is torn down on completion or on crash. Only local ``file://`` sources (and bare paths)
are supported today; object-store schemes (``s3://`` …) are the extension point a deployment adds.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlparse

from pod.schema import Workspace

_LOCAL_SCHEMES = ("", "file")


class WorkspaceError(Exception):
    """A workspace could not be staged (unsupported scheme or unreadable source)."""


def supported_scheme(source: str) -> bool:
    """Whether the pod can stage this source. Checked up front so a bad pointer is a 4xx, not a
    mid-stream failure."""
    return urlparse(source).scheme in _LOCAL_SCHEMES


def _local_path(source: str) -> Path:
    parsed = urlparse(source)
    return Path(parsed.path if parsed.scheme == "file" else source)


def _stage_local(source: str, dest: Path) -> None:
    """Copy a local source tree into the ephemeral dir. The copy is what the run mutates; the source
    is never written back to."""
    src = _local_path(source)
    if not src.exists():
        raise WorkspaceError(f"workspace source does not exist: {src}")
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest / src.name)


async def stage_workspace(workspace: Workspace | None) -> str:
    """Create an ephemeral working directory, staging the workspace source into it if one is given.

    A tool-less run passes no workspace and still gets an empty dir: opencode's daemon needs a cwd.
    On any staging failure the just-created dir is reaped before the error propagates, so a failed
    stage never leaks a temp dir.
    """
    tmp = Path(tempfile.mkdtemp(prefix="agent-pod-"))
    try:
        if workspace is not None:
            if not supported_scheme(workspace.source):
                raise WorkspaceError(
                    f"unsupported workspace scheme: {urlparse(workspace.source).scheme!r}"
                )
            await asyncio.to_thread(_stage_local, workspace.source, tmp)
    except BaseException:
        await reap_workspace(str(tmp))
        raise
    return str(tmp)


async def reap_workspace(cwd: str) -> None:
    """Tear down an ephemeral working directory. Best-effort and never raises."""
    with contextlib.suppress(Exception):
        await asyncio.to_thread(shutil.rmtree, cwd, ignore_errors=True)


@contextlib.asynccontextmanager
async def staged_workspace(workspace: Workspace | None) -> AsyncIterator[str]:
    """Stage a workspace for the duration of a block, reaping it on exit. Built on the stage/reap
    primitives; the pod's run path uses those directly so it can reap on a cancellation-proof task."""
    cwd = await stage_workspace(workspace)
    try:
        yield cwd
    finally:
        await reap_workspace(cwd)
