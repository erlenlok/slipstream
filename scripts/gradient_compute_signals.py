#!/usr/bin/env python
"""Compatibility wrapper for scripts/strategies/gradient/compute_signals.py."""

import warnings

from scripts.strategies.gradient.compute_signals import main


def _run() -> None:
    warnings.warn(
        "Use scripts/strategies/gradient/compute_signals.py (or `uv run gradient-signals`).",
        DeprecationWarning,
        stacklevel=2,
    )
    main()


if __name__ == "__main__":
    _run()
