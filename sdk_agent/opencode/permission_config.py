"""Build opencode's native permission config from a PermissionSpec.

opencode enforces tool permissions declaratively: the runner merges a ``permission`` block into the
per-run config and opencode matches each call against it, so there is no per-call callback to run.

The engine is allow-by-default — its agent loop relies on that floor for internal permissions it
raises during normal operation — so this config denies rather than grants: the tool allow-list turns
into denies for the built-ins it omits, on top of the always-on external-directory guard. Blanket
denying everything unlisted was tried and breaks the agent (it removes engine-internal affordances
like the repeated-call intervention), so the deny stays scoped to the tools we actually model.

Within a tool's glob map the *last* matching rule wins, with no special standing for ``deny``. So
the broad fallback is emitted first and the specific override last: a catch-all ``allow`` ahead of
the destructive-command denies for bash, and a blanket ``deny`` ahead of the writable-path allows
for edit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sdk_agent.permissions import PermissionSpec

_ALLOW = "allow"
_DENY = "deny"

# Destructive command fragments, matched anywhere (opencode matches the whole command, hence the
# wrapping "*"). The fork-bomb globs catch the common ``:(){ ... };:`` forms but not every rewrite,
# and globs can't cap a command's timeout — both lean on the ephemeral, session-timeout-bounded pod.
BLOCKED_BASH_GLOBS: tuple[str, ...] = (
    "*rm -rf /*",
    "*mkfs*",
    "*shutdown*",
    "*reboot*",
    "*dd if=*",
    "*kill -9 -1*",
    "*:(){*",
    "*:() {*",
)

# Map a caller's tool name to opencode's permission key, lower-casing so both PascalCase ("Read")
# and opencode's own lower-case ("read") resolve. edit, write, and patch share the "edit" key.
_TOOL_TO_PERMISSION: dict[str, str] = {
    "read": "read",
    "edit": "edit",
    "write": "edit",
    "patch": "edit",
    "glob": "glob",
    "grep": "grep",
    "bash": "bash",
    "task": "task",
    "webfetch": "webfetch",
    "websearch": "websearch",
    "list": "list",
    "todowrite": "todowrite",
}

# Built-in tools denied when the allow-list omits them, derived from the map above so a new row can't
# leave a gap. MCP tools aren't here — they have no permission key and are gated by server
# registration plus the audit-tool denies added to the session ruleset.
_GATED_PERMISSIONS: tuple[str, ...] = tuple(dict.fromkeys(_TOOL_TO_PERMISSION.values()))


def _allowed_permission_keys(tools: list[str]) -> set[str]:
    """Reduce a run's tool allow-list to the set of opencode permission keys it permits."""
    return {key for t in tools if (key := _TOOL_TO_PERMISSION.get(t.strip().lower()))}


def _worktree_prefix(cwd: str) -> str:
    """The prefix opencode puts before an edit pattern for a file in the run's working directory.

    opencode matches edit permissions against paths made relative to its *worktree* — the enclosing
    git root, or the filesystem root when no repository encloses the directory — not against the
    working directory the daemon was launched in. A run whose working directory sits below the git
    root (or in an unversioned tree) therefore sees every path carrying this prefix, and any glob
    meant to name files in the working directory must carry it too.
    """
    resolved = Path(cwd).resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            rel = resolved.relative_to(candidate)
            return "" if rel == Path(".") else rel.as_posix()
    return resolved.as_posix().lstrip("/")


def permission_config(spec: PermissionSpec, tools: list[str], cwd: str) -> dict[str, Any]:
    """Build the opencode ``permission`` object that enforces one run's rules.

    The tool allow-list is the master gate: every built-in opencode tool not named in it is denied
    outright, and the spec's path and command rules refine only the tools that remain allowed. The
    working directory anchors the write-allow globs to the worktree opencode will match against.
    """
    allowed = _allowed_permission_keys(tools)
    config: dict[str, Any] = {"external_directory": _DENY}

    if "bash" in allowed:
        bash = {"*": _ALLOW}
        bash.update(dict.fromkeys(BLOCKED_BASH_GLOBS, _DENY))
        config["bash"] = bash
    else:
        config["bash"] = _DENY

    if "edit" in allowed:
        # opencode matches paths textually without resolving symlinks, so an in-worktree symlink
        # pointing outside can still be written; these globs confine ordinary paths, and symlink
        # containment relies on pod isolation.
        edit: dict[str, str] = {}
        if spec.write_allow:
            edit["**"] = _DENY
            prefix = _worktree_prefix(cwd)
            for pattern in spec.write_allow:
                pattern = pattern.lstrip("/")
                edit[f"{prefix}/{pattern}" if prefix else pattern] = _ALLOW
        if spec.block_sol_sources:
            # A single "*" traverses "/" in opencode's matcher, so the bare globs cover .sol files
            # at any depth. Denies precede the test/script allows that carve them back out.
            edit["*.sol"] = _DENY
            edit["*.t.sol"] = _ALLOW
            edit["*.s.sol"] = _ALLOW
        if edit:
            config["edit"] = edit
    else:
        config["edit"] = _DENY

    for key in _GATED_PERMISSIONS:
        if key not in ("bash", "edit") and key not in allowed:
            config[key] = _DENY
    return config
