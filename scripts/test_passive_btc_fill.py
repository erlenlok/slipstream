#!/usr/bin/env python3
"""Compatibility wrapper for scripts/strategies/gradient/test_passive_btc_fill.py."""

import importlib
import warnings


def _run() -> None:
    warnings.warn(
        "Use scripts/strategies/gradient/test_passive_btc_fill.py",
        DeprecationWarning,
        stacklevel=2,
    )
    module = importlib.import_module("scripts.strategies.gradient.test_passive_btc_fill")
    module.main()


if __name__ == "__main__":
    _run()
