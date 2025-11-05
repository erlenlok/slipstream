#!/usr/bin/env python
"""Compatibility wrapper for scripts/strategies/gradient/run_backtest.py."""

import warnings

from scripts.strategies.gradient.run_backtest import main


def _run() -> None:
    warnings.warn(
        "Use scripts/strategies/gradient/run_backtest.py (or `uv run gradient-backtest`).",
        DeprecationWarning,
        stacklevel=2,
    )
    main()


if __name__ == "__main__":
    _run()
