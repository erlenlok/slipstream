#!/usr/bin/env python3
"""
Compare Hyperliquid vs. Binance 24h volumes to spot listings where HL activity lags.

These candidates tend to have wider spreads / less local competition, making them
good fits for the passive Brawler MM. Ratios are normalized against BTC + ETH (by default)
so we evaluate relative gaps rather than absolute volumes.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Optional, Sequence

from slipstream.strategies.brawler.config import load_brawler_config
from slipstream.strategies.brawler.tools.volume_screener import (
    compute_rows,
    fetch_binance_volumes,
    fetch_hl_volumes,
    render_table,
    save_csv,
    save_json,
)


def _load_symbols(config, symbols):
    if symbols:
        missing = [sym for sym in symbols if sym not in config.assets]
        if missing:
            raise ValueError(f"Symbols not found in config assets: {missing}")
        return [config.assets[sym].symbol for sym in symbols]
    return list(config.assets.keys())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flag assets whose Hyperliquid 24h volume badly trails Binance (potential MM edge)."
    )
    parser.add_argument("--config", help="Brawler config to read asset mappings from.")
    parser.add_argument("--symbols", nargs="+", help="Optional subset of Hyperliquid symbols.")
    parser.add_argument(
        "--hl-endpoint",
        default="https://api.hyperliquid.xyz/info",
        help="Hyperliquid /info endpoint (default: https://api.hyperliquid.xyz/info).",
    )
    parser.add_argument(
        "--binance-endpoint",
        default="https://fapi.binance.com/fapi/v1/ticker/24hr",
        help="Binance 24h ticker endpoint (default: futures).",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["BTC", "ETH"],
        help="Reference symbols that define the 'healthy' volume ratio baseline.",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.3,
        help="Flag listings whose HL/Binance ratio is below threshold * baseline ratio (default: 0.3).",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to print (default: 20).")
    parser.add_argument("--csv", type=str, help="Optional path to dump CSV results.")
    parser.add_argument("--json", type=str, help="Optional path to dump JSON results.")
    return parser


async def _run(args: argparse.Namespace) -> None:
    # Need to defer Path import to here or top level? Top level is cleaner but I removed it.
    # Actually argparse 'type=Path' was used in original, let's restore pathlib but only for typing if needed
    # In original script 'save_csv' expected Path.
    from pathlib import Path

    config = load_brawler_config(args.config)
    symbols = _load_symbols(config, args.symbols)
    hl_volumes, binance_volumes = await asyncio.gather(
        fetch_hl_volumes(args.hl_endpoint),
        fetch_binance_volumes(args.binance_endpoint),
    )
    rows = compute_rows(
        config,
        symbols,
        hl_volumes,
        binance_volumes,
        baseline_symbols=args.benchmarks,
        ratio_threshold=args.ratio_threshold,
    )
    print(render_table(rows, limit=args.limit))
    if args.csv:
        save_csv(rows, Path(args.csv))
        print(f"\nCSV saved to {args.csv}")
    if args.json:
        save_json(rows, Path(args.json))
        print(f"JSON saved to {args.json}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    main()
