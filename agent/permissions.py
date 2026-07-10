from dataclasses import dataclass


@dataclass(frozen=True)
class PermissionSpec:
    """Per-run tool-permission rules, independent of the engine that enforces them.

    Bash safety and project-directory write confinement always apply. ``block_sol_sources`` adds
    the compilation agent's ban on editing Solidity sources while keeping test and script files
    writable. ``write_allow`` is a tuple of project-relative globs that are the *only* writable
    paths (everything else is denied) — e.g. ``(".memory/custom_context/**",)`` for a sandbox
    subtree or ``("overview.md",)`` to confine writes to a single file.
    """

    block_sol_sources: bool = False
    write_allow: tuple[str, ...] = ()
