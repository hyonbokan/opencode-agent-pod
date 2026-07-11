"""Unit tests for the opencode native permission config. Behavioral parity (deny/allow holding
under --dangerously-skip-permissions) was validated in the migration spike; here we guard that a
PermissionSpec + tool allow-list map to the right allow/deny rules and that they are ordered for
opencode's last-match-wins evaluator — the broad rule first, the specific override last."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.opencode.permission_config import BLOCKED_BASH_GLOBS, permission_config
from agent.permissions import PermissionSpec

# A permissive allow-list so the spec-driven bash/edit rules are exercised; tool gating is asserted
# separately below.
_ALL = ["Read", "Glob", "Grep", "Bash", "Edit", "Write"]


@pytest.fixture
def git_root(tmp_path: Path) -> str:
    """A working directory that is itself a git root, so write-allow globs carry no prefix."""
    (tmp_path / ".git").mkdir()
    return str(tmp_path)


def _wildcard(value: str, pattern: str) -> bool:
    """opencode's matcher: escape regex specials, ``*`` -> ``.*``, ``?`` -> ``.``, anchored."""
    escaped = re.sub(r"[.+^${}()|[\]\\]", lambda m: "\\" + m.group(0), pattern)
    return re.fullmatch(escaped.replace("*", ".*").replace("?", "."), value, re.S) is not None


def _edit_action(edit_map: dict[str, str], path: str) -> str:
    """The action opencode resolves for a path under an edit glob map — last matching rule wins, the
    engine's allow floor if none match."""
    action = "allow"
    for pattern, act in edit_map.items():
        if _wildcard(path, pattern):
            action = act
    return action


def test_default_confines_writes_and_denies_destructive_bash(git_root):
    cfg = permission_config(PermissionSpec(), _ALL, git_root)
    assert cfg["external_directory"] == "deny"
    assert cfg["bash"]["*"] == "allow"
    for glob in BLOCKED_BASH_GLOBS:
        assert cfg["bash"][glob] == "deny"
    assert (
        "edit" not in cfg
    )  # edit allowed, but no path restrictions beyond project-dir confinement


def test_bash_catch_all_allow_precedes_denies(git_root):
    """Last-match-wins: the "*" allow must come before every destructive deny, or the catch-all
    would override them all and nothing would be blocked."""
    bash = permission_config(PermissionSpec(), _ALL, git_root)["bash"]
    keys = list(bash)
    for glob in BLOCKED_BASH_GLOBS:
        assert keys.index("*") < keys.index(glob)


def test_write_deny_blocks_globs_over_the_allow_floor(git_root):
    # With no write_allow the edit floor is allow; write_deny fences specific globs off it. A single
    # "*" traverses "/" in opencode's matcher, so a bare glob fences a name at any depth.
    edit = permission_config(PermissionSpec(write_deny=("*.env", "secrets/**")), _ALL, git_root)[
        "edit"
    ]
    assert _edit_action(edit, "config.env") == "deny"
    assert _edit_action(edit, "deep/config.env") == "deny"
    assert _edit_action(edit, "secrets/key.pem") == "deny"
    assert _edit_action(edit, "src/main.py") == "allow"  # the allow floor holds elsewhere


def test_write_deny_overrides_write_allow(git_root):
    # write_deny is emitted last, so it carves holes out of an exclusive write_allow set.
    edit = permission_config(
        PermissionSpec(write_allow=("src/**",), write_deny=("src/secret.txt",)), _ALL, git_root
    )["edit"]
    assert _edit_action(edit, "src/main.py") == "allow"
    assert _edit_action(edit, "src/secret.txt") == "deny"
    assert _edit_action(edit, "README.md") == "deny"  # outside the write_allow set
    # last-match-wins ordering: blanket deny, then the allow, then the specific deny that overrides it
    keys = list(edit)
    assert keys.index("**") < keys.index("src/**") < keys.index("src/secret.txt")


def test_fork_bomb_is_denied(git_root):
    bash = permission_config(PermissionSpec(), _ALL, git_root)["bash"]
    assert bash["*:(){*"] == "deny"
    assert bash["*:() {*"] == "deny"


def test_write_allow_confines_writes_to_globs(git_root):
    edit = permission_config(
        PermissionSpec(write_allow=(".memory/custom_context/**", "overview.md")), _ALL, git_root
    )["edit"]
    assert edit["**"] == "deny"
    assert _edit_action(edit, ".memory/custom_context/notes/a.md") == "allow"
    assert _edit_action(edit, "overview.md") == "allow"
    assert _edit_action(edit, "src/Token.sol") == "deny"  # the blanket deny holds elsewhere
    # last-match-wins: the blanket deny must precede every allow that overrides it
    keys = list(edit)
    assert keys.index("**") < keys.index(".memory/custom_context/**")
    assert keys.index("**") < keys.index("overview.md")


def test_write_allow_anchored_when_cwd_is_below_the_git_root(git_root):
    # opencode matches edit patterns relative to the git root, so a run whose cwd is a
    # subdirectory (the content-merge agent runs in .memory) must carry that prefix on its globs
    # or the allow never matches and the blanket deny blocks every write.
    memory_dir = Path(git_root) / ".memory"
    memory_dir.mkdir()
    edit = permission_config(PermissionSpec(write_allow=("overview.md",)), _ALL, str(memory_dir))[
        "edit"
    ]
    assert edit[".memory/overview.md"] == "allow"
    assert "overview.md" not in edit
    assert _edit_action(edit, ".memory/overview.md") == "allow"
    assert _edit_action(edit, ".memory/brain_index.md") == "deny"
    assert _edit_action(edit, "src/overview.md") == "deny"


def test_write_allow_anchored_to_filesystem_root_without_git(tmp_path):
    # With no enclosing repository opencode's worktree is the filesystem root (zip-sourced scans),
    # so patterns are near-absolute; the glob must carry the full cwd prefix to ever match.
    project = (tmp_path / "repo").resolve()
    project.mkdir()
    edit = permission_config(
        PermissionSpec(write_allow=(".memory/custom_context/**",)), _ALL, str(project)
    )["edit"]
    anchored = f"{project.as_posix().lstrip('/')}/.memory/custom_context/**"
    assert edit[anchored] == "allow"
    assert ".memory/custom_context/**" not in edit
    assert (
        _edit_action(edit, f"{project.as_posix().lstrip('/')}/.memory/custom_context/a.md")
        == "allow"
    )
    assert _edit_action(edit, f"{project.as_posix().lstrip('/')}/src/Token.sol") == "deny"


def test_empty_allow_list_denies_every_gated_tool(git_root):
    cfg = permission_config(PermissionSpec(), [], git_root)
    for key in ("read", "edit", "glob", "grep", "bash", "task", "webfetch", "websearch"):
        assert cfg[key] == "deny"


def test_tool_gating_denies_only_unlisted_tools(git_root):
    # Read/Edit/Glob allowed; bash and the rest denied.
    cfg = permission_config(PermissionSpec(), ["Read", "Edit", "Glob"], git_root)
    assert cfg["bash"] == "deny"
    assert cfg["grep"] == "deny"
    assert cfg["webfetch"] == "deny"
    assert "read" not in cfg  # an allowed tool carries no deny key
    assert "glob" not in cfg


def test_edit_denied_outright_when_not_allowed_even_with_write_spec(git_root):
    # A write_allow spec must not resurrect edit when the tool itself is off the allow-list.
    cfg = permission_config(PermissionSpec(write_allow=("overview.md",)), ["Read"], git_root)
    assert cfg["edit"] == "deny"


def test_lowercase_and_mcp_tool_names_are_handled(git_root):
    # opencode-lowercase names gate the same as PascalCase; write maps to the edit key; unknown MCP
    # tool names contribute no permission key and so are neither allowed-mapped nor denied.
    cfg = permission_config(PermissionSpec(), ["read", "write", "bash", "demo_lookup"], git_root)
    assert cfg["bash"]["*"] == "allow"
    assert "edit" not in cfg  # write mapped to edit → allowed, no path rules
    assert "read" not in cfg
    assert cfg["grep"] == "deny"  # unlisted built-in still denied
