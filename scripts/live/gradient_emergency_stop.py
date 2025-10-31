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
from slipstream.gradient.live.execution import get_current_positions


def flatten_all_positions(config):
    """
    Close all open positions immediately.

    TODO: Implement
    1. Get current positions
    2. For each position, place opposite market order to close
    3. Wait for fills
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

        # TODO: Implement position closing
        # For each position:
        #   - Place market order in opposite direction
        #   - Size = current position size
        #   - Wait for fill

        if config.dry_run:
            print("DRY-RUN: Would close all positions")
        else:
            print("ERROR: Emergency stop not yet implemented!")
            print("TODO: Implement position closing in this script")
            print("For now, manually close positions via Hyperliquid UI")

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
