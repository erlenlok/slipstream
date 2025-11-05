#!/usr/bin/env python3
"""Compatibility wrapper for scripts/strategies/gradient/concentration_study.py."""

import warnings

from scripts.strategies.gradient.concentration_study import main


def _run() -> None:
    warnings.warn(
        "Use scripts/strategies/gradient/concentration_study.py",
        DeprecationWarning,
        stacklevel=2,
    )
    main()


if __name__ == "__main__":
    _run()
