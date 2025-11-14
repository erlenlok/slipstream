#!/usr/bin/env python3
"""
Passive flatten script for Gradient strategy.

Closes all positions using limit orders (passive execution) to minimize costs.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from slipstream.strategies.gradient.live.config import load_config
from slipstream.strategies.gradient.live.execution import (
    _prepare_hyperliquid_context,
    get_current_positions,
    place_limit_orders,
)


def flatten_all_positions_passive(config):
    """
    Close all open positions using passive limit orders.

    Implementation:
    1. Get current positions
    2. For each position, place opposite limit order at join_best price
    3. Monitor fills and report unfilled orders
    4. User can re-run or escalate to market orders if needed
    """
    print("PASSIVE FLATTEN: Placing limit orders to close all positions...")

    try:
        # Get current positions
        positions = get_current_positions(config)

        if len(positions) == 0:
            print("No open positions to close.")
            return

        print(f"Found {len(positions)} open positions:")
        for asset, size in positions.items():
            print(f"  {asset}: ${size:,.2f}")

        # Create deltas to flatten (opposite of current positions)
        deltas = {
            asset: -size
            for asset, size in positions.items()
            if abs(size) > 1e-6
        }

        if not deltas:
            print("All positions below threshold, nothing to close.")
            return

        print("\nPlacing passive limit orders (join_best)...")

        context = _prepare_hyperliquid_context(config)
        filled, unfilled, errors = place_limit_orders(
            deltas,
            config,
            context.info,
            context.asset_meta,
            context.exchange,
        )

        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)

        if filled:
            total_filled_usd = sum(abs(o.get("fill_usd", 0)) for o in filled)
            print(f"\n✓ Filled orders ({len(filled)}):")
            for order in filled:
                side = order.get("side")
                asset = order.get("asset")
                usd = order.get("fill_usd", 0.0)
                px = order.get("fill_px", 0.0)
                print(f"  {asset}: {side} ${abs(usd):,.2f} @ ${px:.4f}")
            print(f"  Total: ${total_filled_usd:,.2f}")

        if unfilled:
            total_unfilled_usd = sum(abs(o.get("size_usd", 0)) for o in unfilled)
            print(f"\n⏳ Unfilled orders ({len(unfilled)}):")
            for order in unfilled:
                side = order.get("side")
                asset = order.get("asset")
                usd = order.get("size_usd", 0.0)
                limit_px = order.get("limit_px", 0.0)
                oid = order.get("order_id", "N/A")
                print(f"  {asset}: {side} ${abs(usd):,.2f} @ ${limit_px:.4f} (ID: {oid})")
            print(f"  Total: ${total_unfilled_usd:,.2f}")
            print("\n  Orders are live on the exchange. They will fill when price reaches limit.")
            print("  To cancel unfilled orders, use: gradient-cancel-all-orders")

        if errors:
            print(f"\n✗ Errors ({len(errors)}):")
            for err in errors:
                asset = err.get("asset", "UNKNOWN")
                msg = err.get("error", "Unknown error")
                stage = err.get("stage", "n/a")
                print(f"  {asset} [{stage}]: {msg}")

        # Check remaining positions
        print("\n" + "="*80)
        remaining = get_current_positions(config)
        if remaining:
            total_remaining = sum(abs(v) for v in remaining.values())
            print(f"Positions still open: ${total_remaining:,.2f}")
            for asset, size in remaining.items():
                print(f"  {asset}: ${size:,.2f}")

            if unfilled:
                print("\nUnfilled orders are working. Wait for fills or run with --force-market")
        else:
            print("✓ All positions closed!")

        print("="*80)

    except Exception as e:
        print(f"FATAL ERROR during passive flatten: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Passively flatten all Gradient positions using limit orders"
    )
    parser.add_argument(
        "--flatten-all",
        action="store_true",
        help="Flatten all positions passively (limit orders)"
    )
    parser.add_argument(
        "--config",
        default="config/gradient_live.json",
        help="Config file path"
    )
    parser.add_argument(
        "--force-market",
        action="store_true",
        help="If set, use market orders instead (aggressive)"
    )

    args = parser.parse_args()

    if not args.flatten_all:
        print("Usage: python flatten_passive.py --flatten-all")
        print("       python flatten_passive.py --flatten-all --force-market  (for aggressive)")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    if args.force_market:
        print("--force-market flag detected. Use emergency_stop.py instead.")
        print("Run: python scripts/strategies/gradient/live/emergency_stop.py --flatten-all")
        sys.exit(1)

    # Execute passive flatten
    flatten_all_positions_passive(config)


if __name__ == "__main__":
    main()
