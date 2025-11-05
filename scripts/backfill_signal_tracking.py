#!/usr/bin/env python3
"""Compatibility wrapper for scripts/strategies/gradient/backfill_signal_tracking.py."""

import warnings

from scripts.strategies.gradient.backfill_signal_tracking import main


def _run() -> None:
    warnings.warn(
        "Use scripts/strategies/gradient/backfill_signal_tracking.py",
        DeprecationWarning,
        stacklevel=2,
    )
    main()


if __name__ == "__main__":
    _run()
