"""CLI entry point for the Volume Generator Bot."""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Optional, Iterable

from .volume_bot import VolumeGeneratorBot
from .config import load_volume_generator_config, VolumeBotConfig


def run_volume_generator_cli(argv: Optional[Iterable[str]] = None) -> None:
    """Run volume generator bot from CLI."""
    parser = argparse.ArgumentParser(
        description="Volume Generator: Place in-and-out trades to generate volume.",
    )
    parser.add_argument(
        "--config",
        required=False,
        help="Path to a YAML/JSON config file defining volume generation parameters.",
    )
    parser.add_argument(
        "--trade-count",
        type=int,
        help="Number of in-and-out trades to perform [default: 42 from config]",
    )
    parser.add_argument(
        "--trade-size-usd",
        type=float,
        help="USD amount for each trade [default: 20.0 from config]",
    )
    parser.add_argument(
        "--delay-between-trades",
        type=float,
        help="Delay between trades in seconds [default: 1.5 from config]",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Symbol to trade [default: BTC from config]",
    )
    parser.add_argument(
        "--slippage-tolerance-bps",
        type=int,
        help="Slippage tolerance in basis points [default: 100 from config]",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode without placing real orders [default: False from config]",
    )

    args = parser.parse_args(argv)

    # Load config from file if provided, otherwise use defaults
    config = load_volume_generator_config(args.config)

    # Override config values with CLI args if provided
    if args.trade_count is not None:
        config.trade_count = args.trade_count
    if args.trade_size_usd is not None:
        config.trade_size_usd = args.trade_size_usd
    if args.delay_between_trades is not None:
        config.delay_between_trades = args.delay_between_trades
    if args.symbol is not None:
        config.symbol = args.symbol
    if args.slippage_tolerance_bps is not None:
        config.slippage_tolerance_bps = args.slippage_tolerance_bps
    if args.dry_run:
        config.dry_run = args.dry_run

    async def run():
        bot = VolumeGeneratorBot(config)
        try:
            await bot.run_volume_generation()
        except KeyboardInterrupt:
            print("\nVolume generation interrupted by user")
        except Exception as e:
            print(f"Error running volume generator: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(run())