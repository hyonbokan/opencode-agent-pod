"""Staging a workspace-by-reference into an ephemeral dir, and reaping it."""

from __future__ import annotations

import dataclasses
import functools
import http.server
import io
import tarfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from pod import workspace as workspace_mod
from pod.schema import Workspace
from pod.settings import PodSettings, load_settings
from pod.workspace import WorkspaceError, staged_workspace, supported_scheme


def test_supported_scheme_accepts_local_and_remote():
    assert supported_scheme("file:///data/x")
    assert supported_scheme("/data/x")  # bare path
    assert supported_scheme("https://host/x")
    assert supported_scheme("http://host/x")
    assert not supported_scheme("s3://bucket/key")
    assert not supported_scheme("ftp://host/x")


@pytest.mark.asyncio
async def test_toolless_run_gets_an_empty_ephemeral_dir_that_is_reaped():
    async with staged_workspace(None) as cwd:
        path = Path(cwd)
        assert path.is_dir()
        assert not any(path.iterdir())  # empty, since no source
    assert not path.exists()  # reaped on exit


@pytest.mark.asyncio
async def test_local_dir_source_is_staged_then_reaped(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "data.txt").write_text("payload")
    (source / "sub").mkdir()
    (source / "sub" / "more.txt").write_text("nested")

    async with staged_workspace(Workspace(source=source.as_uri())) as cwd:
        staged = Path(cwd)
        assert (staged / "data.txt").read_text() == "payload"
        assert (staged / "sub" / "more.txt").read_text() == "nested"
        # The staged copy is independent of the source (writes here never touch the caller's storage).
        assert staged != source
    assert not staged.exists()


@pytest.mark.asyncio
async def test_missing_local_source_raises_and_still_reaps(tmp_path: Path, monkeypatch):
    # Staging fails in __aenter__ (before the dir is yielded), so capture the temp dir at creation to
    # prove the finally still reaped it.
    created: list[str] = []
    real_mkdtemp = workspace_mod.tempfile.mkdtemp

    def _spy(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(workspace_mod.tempfile, "mkdtemp", _spy)

    missing = tmp_path / "nope"
    with pytest.raises(WorkspaceError, match="does not exist"):
        async with staged_workspace(Workspace(source=missing.as_uri())):
            pass
    assert created and not Path(created[0]).exists()  # the temp dir was reaped despite the failure


@pytest.mark.asyncio
async def test_unsupported_scheme_raises():
    with pytest.raises(WorkspaceError, match="unsupported workspace scheme"):
        async with staged_workspace(Workspace(source="s3://bucket/key")):
            pass


# --- remote (https/http) staging -------------------------------------------------------------


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep the test output clean
        pass


@pytest.fixture
def file_server(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """Serve a tmp directory over loopback HTTP; yields (serve_dir, base_url)."""
    serve_dir = tmp_path / "served"
    serve_dir.mkdir()
    handler = functools.partial(_QuietHandler, directory=str(serve_dir))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield serve_dir, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        thread.join()


def _settings(**overrides) -> PodSettings:
    return dataclasses.replace(load_settings(), **overrides)


def _make_tar(path: Path, files: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.mark.asyncio
async def test_remote_tar_archive_is_pulled_extracted_and_reaped(file_server):
    serve_dir, base = file_server
    _make_tar(serve_dir / "ws.tar.gz", {"data.txt": "payload", "sub/more.txt": "nested"})

    async with staged_workspace(Workspace(source=f"{base}/ws.tar.gz")) as cwd:
        staged = Path(cwd)
        assert (staged / "data.txt").read_text() == "payload"
        assert (staged / "sub" / "more.txt").read_text() == "nested"
    assert not staged.exists()


@pytest.mark.asyncio
async def test_remote_single_file_is_placed_under_its_url_name(file_server):
    serve_dir, base = file_server
    (serve_dir / "notes.txt").write_text("just one file")

    async with staged_workspace(Workspace(source=f"{base}/notes.txt")) as cwd:
        assert (Path(cwd) / "notes.txt").read_text() == "just one file"


@pytest.mark.asyncio
async def test_remote_tar_with_path_traversal_member_is_rejected(file_server):
    serve_dir, base = file_server
    _make_tar(serve_dir / "evil.tar.gz", {"../escape.txt": "pwned"})

    with pytest.raises(WorkspaceError, match="unsafe tar member"):
        async with staged_workspace(Workspace(source=f"{base}/evil.tar.gz")):
            pass


@pytest.mark.asyncio
async def test_remote_download_over_size_cap_is_rejected(file_server):
    serve_dir, base = file_server
    (serve_dir / "big.bin").write_bytes(b"x" * 4096)

    with pytest.raises(WorkspaceError, match="size cap"):
        async with staged_workspace(
            Workspace(source=f"{base}/big.bin"), _settings(workspace_max_bytes=1024)
        ):
            pass


@pytest.mark.asyncio
async def test_remote_archive_expanding_over_cap_is_rejected(file_server):
    serve_dir, base = file_server
    _make_tar(serve_dir / "bomb.tar.gz", {"big.txt": "y" * 8192})  # compresses small, expands large

    with pytest.raises(WorkspaceError, match="expands beyond"):
        async with staged_workspace(
            Workspace(source=f"{base}/bomb.tar.gz"), _settings(workspace_max_bytes=1024)
        ):
            pass


@pytest.mark.asyncio
async def test_plaintext_http_to_non_loopback_host_is_rejected():
    # Fails validation before any network call (the scheme/host check runs first).
    with pytest.raises(WorkspaceError, match="loopback"):
        async with staged_workspace(Workspace(source="http://storage.example.com/ws.tar.gz")):
            pass


@pytest.mark.asyncio
async def test_link_local_metadata_host_is_rejected():
    # 169.254.169.254 is the cloud metadata endpoint; must never be fetched.
    with pytest.raises(WorkspaceError, match="link-local"):
        async with staged_workspace(Workspace(source="https://169.254.169.254/latest/meta-data")):
            pass


@pytest.mark.asyncio
async def test_host_outside_allowlist_is_rejected():
    settings = _settings(workspace_host_allowlist=("storage.internal",))
    with pytest.raises(WorkspaceError, match="not in AGENT_POD_WORKSPACE_HOST_ALLOWLIST"):
        async with staged_workspace(
            Workspace(source="https://other.example.com/ws.tar.gz"), settings
        ):
            pass


@pytest.mark.asyncio
async def test_tls_verify_setting_reaches_the_http_client(file_server, monkeypatch):
    # Capture the verify kwarg the client is built with (http loopback ignores it, so the fetch
    # still succeeds — this asserts the flag is plumbed through, not TLS behavior itself).
    serve_dir, base = file_server
    (serve_dir / "notes.txt").write_text("x")
    seen: dict[str, object] = {}
    real_client = workspace_mod.httpx.AsyncClient

    def spy(*args, **kwargs):
        seen["verify"] = kwargs.get("verify")
        return real_client(*args, **kwargs)

    monkeypatch.setattr(workspace_mod.httpx, "AsyncClient", spy)

    async with staged_workspace(
        Workspace(source=f"{base}/notes.txt"), _settings(workspace_tls_verify=False)
    ) as cwd:
        assert (Path(cwd) / "notes.txt").read_text() == "x"
    assert seen["verify"] is False
