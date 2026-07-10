"""Error classes for the agent runner."""

from __future__ import annotations


class SessionStartError(Exception):
    """A pre-prompt failure to create the session or open its event stream.

    Raised rather than returned because no prompt has been sent yet, so the run is safe to retry: the
    runner's retry loop re-acquires the daemon (evicting and relaunching a dead one) instead of
    burning the attempt on a terminal result. Once the prompt is in flight a failure is reported as a
    DriverResult instead, so an already-charged run is never silently re-run.
    """
