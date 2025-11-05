#!/usr/bin/env python3
"""Compatibility wrapper for scripts/strategies/gradient/live/emergency_stop.py."""

import warnings

from scripts.strategies.gradient.live.emergency_stop import main


def _run() -> None:
    warnings.warn(
        "Use scripts/strategies/gradient/live/emergency_stop.py",
        DeprecationWarning,
        stacklevel=2,
    )
    main()


if __name__ == "__main__":
    _run()
