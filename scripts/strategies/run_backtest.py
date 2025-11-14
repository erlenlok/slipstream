#!/usr/bin/env python
"""
Dispatch to a strategy-specific backtest CLI.

Example:
    uv run python scripts/strategies/run_backtest.py \
        --strategy gradient \
        -- --returns-csv data/returns.csv --top-n 5
"""

from __future__ import annotations

import argparse
from typing import Iterable, Sequence

from slipstream.strategies import get_strategy_info, list_strategies


def _format_strategy_listing() -> str:
    lines = ["Registered strategies:"]
    for info in list_strategies():
        lines.append(f"  - {info.key}: {info.title} — {info.description}")
    lines.append("Pass `-- --help` to forward a help flag to the chosen strategy CLI.")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=True,
        description="Route common arguments to a specific strategy backtest CLI.",
        epilog="All arguments after `--` are passed directly to the strategy handler.",
    )
    parser.add_argument(
        "--strategy",
        default="gradient",
        help="Strategy key to run (use --list-strategies to view options).",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List registered strategies and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args, remainder = parser.parse_known_args(argv)

    if args.list_strategies:
        print(_format_strategy_listing())
        return 0

    try:
        info = get_strategy_info(args.strategy)
    except KeyError as exc:
        parser.error(str(exc))

    handler = info.load_cli_handler("run_backtest")
    handler(remainder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
