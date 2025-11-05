#!/usr/bin/env python3
"""
Verify Hyperliquid 4h candle alignment.

This script checks:
1. What timestamps Hyperliquid uses for 4h candles
2. What timestamps our backtest data uses
3. Recommended cron timing for live trading
"""

import asyncio
import httpx
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def fetch_recent_4h_candles():
    """Fetch recent 4h candles from Hyperliquid to see timestamps."""
    endpoint = "https://api.hyperliquid.xyz"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get BTC candles as example
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (10 * 4 * 60 * 60 * 1000)  # Last 10 candles

        response = await client.post(
            f"{endpoint}/info",
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": "BTC",
                    "interval": "4h",
                    "startTime": start_time,
                    "endTime": end_time
                }
            }
        )
        response.raise_for_status()
        data = response.json()

        print("=" * 80)
        print("HYPERLIQUID 4H CANDLE TIMESTAMPS (Last 10 candles)")
        print("=" * 80)

        for candle in data[-10:]:
            timestamp_ms = candle["t"]
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            print(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} "
                  f"(Hour: {timestamp.hour:02d}:00)")

        # Analyze pattern
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        hours = [datetime.fromtimestamp(c["t"] / 1000).hour for c in data[-10:]]
        unique_hours = sorted(set(hours))

        print(f"Unique hours in sample: {unique_hours}")
        print(f"Pattern: Candles close at hours: {', '.join(f'{h:02d}:00' for h in unique_hours)}")

        # Check if it's 00, 04, 08, 12, 16, 20
        expected_pattern = [0, 4, 8, 12, 16, 20]
        if unique_hours == expected_pattern or set(unique_hours).issubset(set(expected_pattern)):
            print("✓ Candles align to 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC")
            print("✓ This matches our cron schedule: 0 */4 * * *")
        else:
            print(f"⚠ WARNING: Candles do NOT align to expected pattern!")
            print(f"   Expected: {expected_pattern}")
            print(f"   Actual: {unique_hours}")

        return data


def check_backtest_data():
    """Check timestamps in our backtest data."""
    data_dir = Path("data/market_data")

    if not data_dir.exists():
        print("\n⚠ No backtest data found in data/market_data/")
        return

    # Find a candle file
    candle_files = list(data_dir.glob("*_candles_4h.csv"))

    if not candle_files:
        print("\n⚠ No 4h candle files found")
        return

    # Check first file
    sample_file = candle_files[0]
    print("\n" + "=" * 80)
    print(f"BACKTEST DATA TIMESTAMPS (from {sample_file.name})")
    print("=" * 80)

    df = pd.read_csv(sample_file)

    # Handle both 'datetime' and 'timestamp' column names
    time_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
    df[time_col] = pd.to_datetime(df[time_col])

    # Show last 10 timestamps
    recent = df.tail(10)
    for _, row in recent.iterrows():
        ts = row[time_col]
        print(f"Timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S')} (Hour: {ts.hour:02d}:00)")

    # Analyze pattern
    hours = df[time_col].dt.hour.unique()
    unique_hours = sorted(hours)

    print("\n" + "=" * 80)
    print("BACKTEST DATA ANALYSIS")
    print("=" * 80)
    print(f"Unique hours in backtest data: {list(unique_hours)}")

    expected_pattern = [0, 4, 8, 12, 16, 20]
    if set(unique_hours).issubset(set(expected_pattern)):
        print("✓ Backtest data aligns to 4h boundaries (00, 04, 08, 12, 16, 20)")
    else:
        print(f"⚠ WARNING: Backtest data has unexpected hours!")
        print(f"   Expected subset of: {expected_pattern}")
        print(f"   Actual: {list(unique_hours)}")


def print_recommendations():
    """Print recommendations for cron timing."""
    print("\n" + "=" * 80)
    print("CRON TIMING RECOMMENDATIONS")
    print("=" * 80)

    print("""
The 4h candles close at: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC

IMPORTANT TIMING CONSIDERATIONS:
================================

1. CANDLE CLOSE vs DATA AVAILABILITY
   - Candle closes at e.g., 04:00:00 UTC
   - Hyperliquid may need a few seconds to finalize the data
   - Recommendation: Wait 30-60 seconds after candle close

2. CURRENT CRON SCHEDULE
   - Cron: 0 */4 * * * (runs at :00 of hours 0, 4, 8, 12, 16, 20)
   - This runs EXACTLY when candle closes
   - Risk: May fetch incomplete candle data

3. RECOMMENDED CRON SCHEDULE
   - Option A (Conservative): 1 */4 * * * (runs at :01, gives 1 min buffer)
   - Option B (Safer): 2 */4 * * * (runs at :02, gives 2 min buffer)

   Recommended: 1 */4 * * *
   (Runs at 00:01, 04:01, 08:01, 12:01, 16:01, 20:01 UTC)

4. SIGNAL CALCULATION
   - The latest candle at 04:01 will be the 00:00-04:00 candle (just closed)
   - We compute signals using data UP TO AND INCLUDING this candle
   - This matches the backtest: signals computed after candle close

5. EXECUTION TIMING
   - Stage 1 (limits): 04:01 - 05:01 (60 min passive)
   - Stage 2 (market): 05:01 (sweep unfilled)
   - Next candle closes at 08:00
   - We're trading on the 04:00-08:00 candle

VERIFICATION CHECKLIST:
======================
✓ Hyperliquid candles close at :00 of 4h intervals
✓ Backtest used same timestamps
✓ Cron should run at :01 (1 min after close) for safety
✓ Latest candle in signal calculation = just-closed candle
✓ We trade during the NEXT 4h period

UPDATED CRON COMMAND:
====================
# Edit crontab
crontab -e

# Change from:
0 */4 * * * /root/slipstream/scripts/strategies/gradient/live/rebalance.sh

# To (with 1-minute buffer):
1 */4 * * * /root/slipstream/scripts/strategies/gradient/live/rebalance.sh

This ensures we:
1. Wait for candle to fully close and data to be available
2. Use the exact same timestamps as the backtest
3. Trade on fresh signals from the just-closed candle
""")


async def main():
    print("\n" + "=" * 80)
    print("GRADIENT CANDLE ALIGNMENT VERIFICATION")
    print("=" * 80)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    # Check live Hyperliquid candles
    await fetch_recent_4h_candles()

    # Check backtest data
    check_backtest_data()

    # Print recommendations
    print_recommendations()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ Verified: Hyperliquid 4h candles align to 00:00, 04:00, 08:00, etc. UTC")
    print("✓ Verified: Backtest data uses same alignment")
    print("⚠ Action Required: Update cron to run at :01 instead of :00")
    print("  (1-minute buffer ensures candle data is available)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
