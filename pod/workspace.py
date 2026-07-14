"""Stage a caller's workspace-by-reference into an ephemeral working dir, then reap it.

The pod caches nothing: each run reads the source into a throwaway directory, runs there, and the
directory is torn down on completion or on crash. Two source kinds are staged:

- Local ``file://`` / bare paths — copied from a path the pod can already see. This assumes the
  caller and pod share a filesystem (one host), so it is a single-host convenience only.
- Remote ``https://`` (and ``http://`` for loopback) — pulled over the network into the ephemeral
  dir, so the pod shares no filesystem with the caller. A pre-signed object-store URL is just such
  a GET. The payload is extracted if it is a tar/zip archive, otherwise laid down as a single file.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import shutil
import socket
import tarfile
import tempfile
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from core.utils.logger import logger
from pod.schema import Workspace
from pod.settings import PodSettings, load_settings

_LOCAL_SCHEMES = ("", "file")
_REMOTE_SCHEMES = ("https", "http")


class WorkspaceError(Exception):
    """A workspace could not be staged (unsupported scheme or unreadable/unsafe source)."""


def supported_scheme(source: str) -> bool:
    """Whether the pod can stage this source. Checked up front so a bad pointer is a 4xx, not a
    mid-stream failure."""
    return urlparse(source).scheme in _LOCAL_SCHEMES + _REMOTE_SCHEMES


async def stage_workspace(workspace: Workspace | None, settings: PodSettings | None = None) -> str:
    """Create an ephemeral working directory, staging the workspace source into it if one is given.

    A tool-less run passes no workspace and still gets an empty dir: opencode's daemon needs a cwd.
    A local source is copied; a remote source is pulled over the network. On any staging failure the
    just-created dir is reaped before the error propagates, so a failed stage never leaks a temp dir.
    """
    settings = settings or load_settings()
    tmp = Path(tempfile.mkdtemp(prefix="agent-pod-"))
    try:
        if workspace is not None:
            scheme = urlparse(workspace.source).scheme
            if scheme in _LOCAL_SCHEMES:
                await asyncio.to_thread(_stage_local, workspace.source, tmp)
            elif scheme in _REMOTE_SCHEMES:
                await _stage_remote(workspace.source, tmp, settings)
            else:
                raise WorkspaceError(f"unsupported workspace scheme: {scheme!r}")
    except BaseException:
        await reap_workspace(str(tmp))
        raise
    return str(tmp)


async def reap_workspace(cwd: str) -> None:
    """Tear down an ephemeral working directory. Best-effort and never raises."""
    with contextlib.suppress(Exception):
        await asyncio.to_thread(shutil.rmtree, cwd, ignore_errors=True)


@contextlib.asynccontextmanager
async def staged_workspace(
    workspace: Workspace | None, settings: PodSettings | None = None
) -> AsyncIterator[str]:
    """Stage a workspace for the duration of a block, reaping it on exit. Built on the stage/reap
    primitives; the pod's run path uses those directly so it can reap on a cancellation-proof task."""
    cwd = await stage_workspace(workspace, settings)
    try:
        yield cwd
    finally:
        await reap_workspace(cwd)


# --- local sources ---------------------------------------------------------------------------


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


# --- remote sources --------------------------------------------------------------------------


async def _stage_remote(source: str, dest: Path, settings: PodSettings) -> None:
    """Pull a workspace over the network into the ephemeral dir, so the pod shares no filesystem
    with the caller. Validates the source, streams it under a size cap, then unpacks it in place."""
    await asyncio.to_thread(_validate_remote_source, source, settings)
    downloaded = await _download_remote(source, settings)
    try:
        await asyncio.to_thread(_unpack, downloaded, dest, source, settings.workspace_max_bytes)
    finally:
        with contextlib.suppress(OSError):
            downloaded.unlink()


def _validate_remote_source(source: str, settings: PodSettings) -> None:
    """Refuse a remote source that is unsafe to fetch: a plaintext non-loopback host, a host outside
    the configured allowlist, or a host that resolves to a link-local/metadata address (the cloud
    metadata SSRF path). Host restriction beyond this is the deploy-time egress allowlist's job."""
    parsed = urlparse(source)
    host = parsed.hostname
    if not host:
        raise WorkspaceError(f"workspace source has no host: {source!r}")
    if parsed.scheme == "http" and not _is_loopback_host(host):
        raise WorkspaceError("http workspace sources are allowed only for loopback; use https")
    allowlist = settings.workspace_host_allowlist
    if allowlist and host not in allowlist:
        raise WorkspaceError(
            f"workspace host {host!r} is not in AGENT_POD_WORKSPACE_HOST_ALLOWLIST"
        )
    _reject_link_local(host)


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _reject_link_local(host: str) -> None:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as e:
        raise WorkspaceError(f"cannot resolve workspace host {host!r}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_link_local:
            raise WorkspaceError(
                f"workspace host {host!r} resolves to a blocked link-local address ({ip})"
            )


async def _download_remote(source: str, settings: PodSettings) -> Path:
    """Stream the source to a temp file, aborting if it exceeds the size cap. Redirects are refused:
    a pre-signed object-store GET resolves directly, and following one could dodge the SSRF check."""
    fd, name = tempfile.mkstemp(prefix="agent-pod-dl-")
    os.close(fd)
    downloaded = Path(name)
    max_bytes = settings.workspace_max_bytes
    timeout = httpx.Timeout(settings.workspace_fetch_timeout)
    if not settings.workspace_tls_verify:
        logger.warning(
            "workspace TLS verification is OFF — accepting any certificate for %s", source
        )
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, verify=settings.workspace_tls_verify
        ) as client:
            async with client.stream("GET", source) as resp:
                if resp.is_redirect:
                    raise WorkspaceError(
                        f"workspace source redirected ({resp.status_code}); "
                        "a pre-signed URL must resolve directly"
                    )
                resp.raise_for_status()
                total = 0
                with downloaded.open("wb") as out:
                    async for chunk in resp.aiter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise WorkspaceError(f"workspace exceeds the {max_bytes}-byte size cap")
                        out.write(chunk)
    except httpx.HTTPError as e:
        downloaded.unlink(missing_ok=True)
        raise WorkspaceError(f"failed to fetch workspace: {e}") from e
    except BaseException:
        downloaded.unlink(missing_ok=True)
        raise
    return downloaded


def _unpack(downloaded: Path, dest: Path, source: str, max_bytes: int) -> None:
    """Lay the fetched payload down in the ephemeral dir: extract a tar/zip archive (with member
    paths and expanded size checked), otherwise place it as a single file."""
    if tarfile.is_tarfile(downloaded):
        _extract_tar(downloaded, dest, max_bytes)
    elif zipfile.is_zipfile(downloaded):
        _extract_zip(downloaded, dest, max_bytes)
    else:
        shutil.move(str(downloaded), str(dest / _filename_from_url(source)))


def _extract_tar(archive: Path, dest: Path, max_bytes: int) -> None:
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        expanded = sum(m.size for m in members if m.isreg())
        if expanded > max_bytes:
            raise WorkspaceError(f"workspace archive expands beyond the {max_bytes}-byte cap")
        try:
            # filter="data" refuses absolute paths, ".." traversal, and links outside the tree.
            tar.extractall(dest, filter="data")
        except tarfile.FilterError as e:
            raise WorkspaceError(f"unsafe tar member: {e}") from e


def _extract_zip(archive: Path, dest: Path, max_bytes: int) -> None:
    base = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        expanded = 0
        for info in zf.infolist():
            expanded += info.file_size
            if expanded > max_bytes:
                raise WorkspaceError(f"workspace archive expands beyond the {max_bytes}-byte cap")
            target = (dest / info.filename).resolve()
            if target != base and base not in target.parents:
                raise WorkspaceError(f"zip member escapes the workspace: {info.filename!r}")
        zf.extractall(dest)


def _filename_from_url(source: str) -> str:
    name = Path(unquote(urlparse(source).path)).name
    return name or "workspace.bin"
