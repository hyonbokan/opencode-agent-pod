from dataclasses import dataclass


@dataclass(frozen=True)
class PermissionSpec:
    """Per-run edit-path rules, independent of the engine that enforces them.

    Bash safety and the external-directory write guard always apply. ``write_allow`` and
    ``write_deny`` are project-relative globs a caller declares directly — no domain defaults.
    ``write_allow``, when set, is *exclusive*: only its globs are writable and everything else is
    denied — e.g. ``(".scratch/**",)`` for a sandbox subtree or ``("overview.md",)`` for one file.
    ``write_deny`` names globs to block; it is applied last, so it overrides the allow floor (or
    carves holes out of ``write_allow``) — e.g. ``("*.env", "secrets/**")``.
    """

    write_allow: tuple[str, ...] = ()
    write_deny: tuple[str, ...] = ()
