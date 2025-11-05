#!/usr/bin/env python3
"""Compatibility wrapper for scripts/strategies/gradient/test_telegram.py."""

import importlib
import warnings


def main() -> None:
    warnings.warn(
        "Use scripts/strategies/gradient/test_telegram.py",
        DeprecationWarning,
        stacklevel=2,
    )
    module = importlib.import_module("scripts.strategies.gradient.test_telegram")
    module.main()


if __name__ == "__main__":
    main()
