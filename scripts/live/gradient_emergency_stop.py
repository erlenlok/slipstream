#!/usr/bin/env python3
"""
Emergency stop script for Gradient strategy.

Immediately flattens all positions.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from slipstream.gradient.live.config import load_config
from slipstream.gradient.live.execution import (
    _prepare_hyperliquid_context,
    get_current_positions,
    place_market_orders,
)


def flatten_all_positions(config):
    """
    Close all open positions immediately.

    Implementation:
    1. Get current positions
    2. For each position, place opposite market order to close
    3. Report any execution errors
    4. Verify all positions are closed
    """
    print("EMERGENCY STOP: Flattening all positions...")

    try:
        # Get current positions
        positions = get_current_positions(config)

        if len(positions) == 0:
            print("No open positions to close.")
            return

        print(f"Found {len(positions)} open positions:")
        for asset, size in positions.items():
            print(f"  {asset}: ${size:,.2f}")

        # Flatten requires placing opposite market orders for every position.
        deltas = {
            asset: -size
            for asset, size in positions.items()
            if abs(size) > 1e-6
        }

        if not deltas:
            print("All positions below threshold, nothing to close.")
            return

        context = _prepare_hyperliquid_context(config)
        filled, errors = place_market_orders(
            deltas,
            config,
            context.info,
            context.asset_meta,
            context.exchange,
        )

        if filled:
            print("Submitted market orders to flatten positions:")
            for order in filled:
                side = order.get("side")
                asset = order.get("asset")
                usd = order.get("fill_usd", 0.0)
                print(f"  {asset}: {side} ${abs(usd):,.2f}")

        if errors:
            print("Encountered errors while flattening:")
            for err in errors:
                asset = err.get("asset", "UNKNOWN")
                msg = err.get("error", "Unknown error")
                stage = err.get("stage", "n/a")
                print(f"  {asset} [{stage}]: {msg}")
            if not config.dry_run:
                print("WARNING: Some positions may remain open. Verify manually.")

        # Re-fetch positions to confirm status.
        remaining = get_current_positions(config)
        if remaining:
            print("Positions still open after emergency stop:")
            for asset, size in remaining.items():
                print(f"  {asset}: ${size:,.2f}")
        else:
            print("All positions closed.")

    except Exception as e:
        print(f"FATAL ERROR during emergency stop: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Emergency stop for Gradient strategy")
    parser.add_argument("--flatten-all", action="store_true", help="Flatten all positions")
    parser.add_argument("--config", default="config/gradient_live.json", help="Config file")

    args = parser.parse_args()

    if not args.flatten_all:
        print("Usage: gradient_emergency_stop.py --flatten-all")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Execute emergency stop
    flatten_all_positions(config)


if __name__ == "__main__":
    main()
