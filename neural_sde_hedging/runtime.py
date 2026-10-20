"""Explicit opt-in for the paused neural-SDE research infrastructure."""

import os


def require_enabled():
    """Reject new neural-SDE computations unless explicitly enabled; return nothing."""
    if os.environ.get("NEURAL_SDE_ENABLED") != "1":
        raise RuntimeError(
            "Neural-SDE runs are disabled for now. Source is retained. "
            "Set NEURAL_SDE_ENABLED=1 only when intentionally re-enabling this work."
        )
