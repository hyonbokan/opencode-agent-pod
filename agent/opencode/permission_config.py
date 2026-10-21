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

import subprocess
from pathlib import Path
from typing import Any

from agent.permissions import PermissionSpec

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

# Commands that print the daemon env, which the agent's tools inherit. The daemon env carries no
# secrets (provider keys go through the key proxy, and service settings are stripped), so this is a
# backstop for one added later. Stops a naive prompt injection, not a determined one: any
# interpreter one-liner gets past a textual deny-list.
ENV_DUMP_BASH_GLOBS: tuple[str, ...] = (
    "env",
    "env *",
    "*; env",
    "*&& env",
    "*| env",
    "set",
    "set |*",
    "*printenv*",
    "*export -p*",
    "*declare -p*",
    "*declare -x*",
    "*/proc/*/environ*",
    "*os.environ*",
    "*process.env*",
    "*_API_KEY*",
    "*_SECRET*",
)

# opencode's built-in tools by tool id, each with the permission key that gates it; edit, write, and
# apply_patch share the "edit" key. Recheck this when bumping opencode: a built-in missing here is
# neither gated by the allow-list nor switched off by a wrap-up.
BUILTIN_TOOL_PERMISSIONS: dict[str, str] = {
    "read": "read",
    "edit": "edit",
    "write": "edit",
    "apply_patch": "edit",
    "glob": "glob",
    "grep": "grep",
    "bash": "bash",
    "task": "task",
    "webfetch": "webfetch",
    "websearch": "websearch",
    "list": "list",
    "todowrite": "todowrite",
    "skill": "skill",
    "question": "question",
}

# Built-ins no allow-list can enable, as neither is safe in a headless run: skill auto-loads external
# instructions the agent spends its turns on, and question blocks the run waiting for an answer no
# one gives until the session wall.
_NEVER_ALLOWED = frozenset({"skill", "question"})

# A caller's tool name to its permission key, matched lower-cased so both PascalCase ("Read") and
# opencode's own ids ("read") resolve; "patch" is kept as a caller-facing alias of apply_patch.
_TOOL_TO_PERMISSION: dict[str, str] = {
    **{tool: key for tool, key in BUILTIN_TOOL_PERMISSIONS.items() if key not in _NEVER_ALLOWED},
    "patch": "edit",
}

# Permission keys denied when the allow-list omits them. MCP tools aren't here — they have no
# permission key and are gated by server registration plus the MCP-tool denies added to the session
# ruleset.
_GATED_PERMISSIONS: tuple[str, ...] = tuple(dict.fromkeys(BUILTIN_TOOL_PERMISSIONS.values()))


def _allowed_permission_keys(tools: list[str]) -> set[str]:
    """Reduce a run's tool allow-list to the set of opencode permission keys it permits."""
    return {key for t in tools if (key := _TOOL_TO_PERMISSION.get(t.strip().lower()))}


def _git_worktree(path: Path) -> Path | None:
    """The git worktree root enclosing ``path``, or ``None`` when ``path`` is not inside a *valid*
    git repository — mirroring opencode's own ``git rev-parse`` discovery.

    A directory whose ``.git`` is present but not a usable repository (e.g. an archive of a clone
    missing ``refs/``) makes that call fail, and opencode then treats the project as non-git with
    worktree ``/``. Probing the same way keeps both sides in agreement; a bare ``.git``-existence
    check would anchor globs where opencode does not, so every write-allow glob would miss.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    top = result.stdout.strip()
    return Path(top) if top else None


def _anchor(prefix: str, pattern: str) -> str:
    """Anchor a caller's project-relative glob to opencode's worktree prefix."""
    pattern = pattern.lstrip("/")
    return f"{prefix}/{pattern}" if prefix else pattern


def _worktree_prefix(cwd: str) -> str:
    """The prefix opencode puts before an edit pattern for a file in the run's working directory.

    opencode matches edit permissions against paths made relative to its *worktree* — the enclosing
    git root, or the filesystem root when no valid repository encloses the directory — not against
    the working directory the daemon was launched in. A run whose working directory sits below the
    git root (or in an unversioned or broken tree) therefore sees every path carrying this prefix, and
    any glob meant to name files in the working directory must carry it too.
    """
    resolved = Path(cwd).resolve()
    worktree = _git_worktree(resolved)
    if worktree is not None:
        try:
            rel = resolved.relative_to(worktree.resolve())
        except ValueError:
            rel = None
        if rel is not None:
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
        bash.update(dict.fromkeys(BLOCKED_BASH_GLOBS + ENV_DUMP_BASH_GLOBS, _DENY))
        config["bash"] = bash
    else:
        config["bash"] = _DENY

    if "edit" in allowed:
        # opencode matches paths textually without resolving symlinks, so an in-worktree symlink
        # pointing outside can still be written; these globs confine ordinary paths, and symlink
        # containment relies on pod isolation.
        edit: dict[str, str] = {}
        if spec.write_allow or spec.write_deny:
            prefix = _worktree_prefix(cwd)
            if spec.write_allow:
                # Exclusive allow-list: deny everything, then re-allow the named globs. The blanket
                # deny is emitted first so the specific allows win under last-match-wins.
                edit["**"] = _DENY
                for pattern in spec.write_allow:
                    edit[_anchor(prefix, pattern)] = _ALLOW
            # Denies are emitted last so they override the allow floor and any write_allow re-allow.
            for pattern in spec.write_deny:
                edit[_anchor(prefix, pattern)] = _DENY
        if edit:
            config["edit"] = edit
    else:
        config["edit"] = _DENY

    # Allowed tools get an explicit allow so none falls back to an opencode default of "ask" (e.g.
    # on reading a dotenv file), which would hang a run that has no one to answer.
    for key in _GATED_PERMISSIONS:
        if key not in ("bash", "edit"):
            config[key] = _ALLOW if key in allowed else _DENY
    return config
