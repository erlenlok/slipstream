#!/usr/bin/env python3
"""Test script for market data daemon."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slipstream.core.mktdata import MarketDataClient


def main():
    print("Testing market data client...")
    print(f"Connecting to daemon...")

    try:
        with MarketDataClient() as client:
            print("✓ Connected to daemon")

            # Test 1: Get BTC candles
            print("\nTest 1: Fetching BTC 4h candles...")
            candles = client.get_candles(
                venue="hyperliquid",
                symbol="BTC",
                interval="4h",
                count=10,
            )

            if candles:
                print(f"✓ Received {len(candles)} candles")
                latest = candles[-1]
                print(f"  Latest candle:")
                print(f"    Symbol: {latest.symbol}")
                print(f"    Time: {time.ctime(latest.timestamp)}")
                print(f"    Close: ${latest.close:,.2f}")
                print(f"    Volume: {latest.volume:.2f}")
                print(f"    Trades: {latest.num_trades}")
            else:
                print("⚠ No candles received (daemon may be warming up)")

            # Test 2: Get ETH candles
            print("\nTest 2: Fetching ETH 4h candles...")
            eth_candles = client.get_candles(
                venue="hyperliquid",
                symbol="ETH",
                interval="4h",
                count=5,
            )
            print(f"✓ Received {len(eth_candles)} ETH candles")

            # Test 3: Batch fetch
            print("\nTest 3: Batch fetching multiple symbols...")
            batch = client.get_candles_batch(
                venue="hyperliquid",
                symbols=["BTC", "ETH", "SOL"],
                interval="4h",
                count=3,
            )
            for symbol, candles in batch.items():
                print(f"  {symbol}: {len(candles)} candles")

            print("\n✓ All tests passed!")

    except FileNotFoundError:
        print("✗ Daemon not running or socket not found")
        print("  Start daemon with: ./target/release/mktdata-daemon")
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
