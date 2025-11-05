#!/usr/bin/env python
"""
Run a lightweight Gradient backtest using pre-computed returns (and optionally signals).
"""

from slipstream.strategies.gradient.cli import run_backtest_cli


def main() -> None:
    run_backtest_cli()


if __name__ == "__main__":
    main()
