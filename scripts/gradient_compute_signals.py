#!/usr/bin/env python
"""
Compute Gradient trend strength signals from a returns panel.
"""

from slipstream.gradient.cli import compute_signals_cli


def main() -> None:
    compute_signals_cli()


if __name__ == "__main__":
    main()
