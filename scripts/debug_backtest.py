#!/usr/bin/env python3
"""Compatibility wrapper for scripts/strategies/gradient/debug_backtest.py."""

import warnings

from scripts.strategies.gradient.debug_backtest import main


def _run() -> None:
    warnings.warn(
        "Use scripts/strategies/gradient/debug_backtest.py",
        DeprecationWarning,
        stacklevel=2,
    )
    main()


if __name__ == "__main__":
    _run()
