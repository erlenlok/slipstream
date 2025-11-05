#!/usr/bin/env python3
"""Compatibility wrapper for scripts/strategies/gradient/live/daily_summary.py."""

import warnings

from scripts.strategies.gradient.live.daily_summary import main


def _run() -> None:
    warnings.warn(
        "Use scripts/strategies/gradient/live/daily_summary.py",
        DeprecationWarning,
        stacklevel=2,
    )
    main()


if __name__ == "__main__":
    _run()
