"""Staging a workspace-by-reference into an ephemeral dir, and reaping it."""

from __future__ import annotations

from pathlib import Path

import pytest

from pod import workspace as workspace_mod
from pod.schema import Workspace
from pod.workspace import WorkspaceError, staged_workspace, supported_scheme


def test_supported_scheme_accepts_local_only():
    assert supported_scheme("file:///data/x")
    assert supported_scheme("/data/x")  # bare path
    assert not supported_scheme("s3://bucket/key")
    assert not supported_scheme("https://host/x")


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
