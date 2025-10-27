#!/usr/bin/env python3
"""
Fetch Hyperliquid S3 historical L2 book data and convert to OHLCV candles.

This script downloads L2 orderbook snapshots from Hyperliquid's S3 archive,
processes them into 4-hour OHLCV candles (via hourly intermediates), and immediately discards raw data
to minimize disk usage. It's resumable - tracks progress and skips already
processed data.

Architecture:
    1. Download single L2 snapshot file (per hour per asset)
    2. Parse to extract mid-prices
    3. Aggregate to 4h OHLCV candle
    4. Save candle to data/s3_historical/
    5. Delete raw L2 file immediately
    6. Update progress tracker

Directory structure:
    data/s3_historical/
        candles/
            BTC_candles_4h.csv
            ETH_candles_4h.csv
            ...
            hourly/
                BTC_candles_1h.csv  # intermediate artifacts (for resampling)
        progress/
            download_state.json  # Tracks what's been processed

Usage:
    # Setup AWS credentials first:
    aws configure  # Enter your AWS access key/secret

    # Download and process historical data:
    python scripts/fetch_s3_historical.py --start 2023-10-01 --end 2024-03-01

    # Resume after interruption (skips already processed):
    python scripts/fetch_s3_historical.py --start 2023-10-01 --end 2024-03-01

    # Validate against existing API data:
    python scripts/fetch_s3_historical.py --validate
"""

import argparse
import gzip
import json
import lzma
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# S3 bucket configuration
S3_BUCKET = "s3://hyperliquid-archive"
S3_MARKET_DATA_PREFIX = "market_data"

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
S3_DATA_DIR = PROJECT_ROOT / "data" / "s3_historical"
S3_CANDLES_DIR = S3_DATA_DIR / "candles"
S3_HOURLY_DIR = S3_CANDLES_DIR / "hourly"
S3_PROGRESS_DIR = S3_DATA_DIR / "progress"
API_DATA_DIR = PROJECT_ROOT / "data" / "market_data"

# Progress tracking
PROGRESS_FILE = S3_PROGRESS_DIR / "download_state.json"


class ProgressTracker:
    """Track which (date, hour, coin) tuples have been successfully processed."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.completed: set[tuple[str, int, str]] = set()
        self.load()

    def load(self):
        """Load progress from disk."""
        if self.filepath.exists():
            with open(self.filepath) as f:
                data = json.load(f)
                self.completed = {tuple(x) for x in data.get("completed", [])}
            print(f"Loaded progress: {len(self.completed)} items already processed")

    def save(self):
        """Save progress to disk."""
        with open(self.filepath, "w") as f:
            json.dump({
                "completed": [list(x) for x in self.completed],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }, f, indent=2)

    def is_completed(self, date_str: str, hour: int, coin: str) -> bool:
        """Check if this item has been processed."""
        return (date_str, hour, coin) in self.completed

    def mark_completed(self, date_str: str, hour: int, coin: str):
        """Mark item as completed and save."""
        self.completed.add((date_str, hour, coin))
        # Save every 10 items to avoid too frequent writes
        if len(self.completed) % 10 == 0:
            self.save()


def download_s3_file(s3_path: str, local_path: Path) -> bool:
    """Download single file from S3 using AWS CLI.

    Returns:
        True if successful, False otherwise
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aws", "s3", "cp",
        s3_path,
        str(local_path),
        "--request-payer", "requester",
        "--quiet"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  Failed to download {s3_path}: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  Timeout downloading {s3_path}")
        return False
    except Exception as e:
        print(f"  Error downloading {s3_path}: {e}")
        return False


def decompress_lz4(lz4_path: Path) -> bytes:
    """Decompress LZ4 file using lz4 command line tool.

    Note: Requires lz4 to be installed (apt install lz4)
    """
    try:
        result = subprocess.run(
            ["lz4", "-d", "-c", str(lz4_path)],
            capture_output=True,
            timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"lz4 decompression failed: {result.stderr.decode()}")
        return result.stdout
    except FileNotFoundError:
        raise RuntimeError(
            "lz4 tool not found. Install with: sudo apt install lz4"
        )


def parse_l2_snapshot_to_candle(lz4_path: Path, coin: str, date_hour: datetime) -> dict[str, Any] | None:
    """Parse L2 snapshot file and extract OHLCV candle data.

    L2 snapshots contain orderbook snapshots throughout the hour.
    We extract mid-prices and aggregate to a single 1h candle.

    Args:
        lz4_path: Path to .lz4 compressed L2 snapshot file
        coin: Asset symbol (e.g., 'BTC')
        date_hour: Datetime representing the start of this hour

    Returns:
        Dict with OHLCV data or None if parsing failed
    """
    try:
        # Decompress
        raw_data = decompress_lz4(lz4_path)

        # Parse newline-delimited JSON
        mid_prices = []
        for line in raw_data.decode('utf-8').strip().split('\n'):
            if not line:
                continue

            try:
                record = json.loads(line)

                # Navigate nested structure: raw.data.levels
                # Format: {"raw": {"data": {"levels": [[bids_array], [asks_array]]}}}
                if "raw" not in record or "data" not in record["raw"]:
                    continue

                data = record["raw"]["data"]
                if "levels" not in data or len(data["levels"]) != 2:
                    continue

                bids_array = data["levels"][0]  # Array of bid orders
                asks_array = data["levels"][1]  # Array of ask orders

                if not bids_array or not asks_array:
                    continue

                # Extract best bid and ask (first element of each array)
                best_bid = float(bids_array[0]["px"])
                best_ask = float(asks_array[0]["px"])
                mid = (best_bid + best_ask) / 2
                mid_prices.append(mid)

            except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
                # Skip malformed records
                continue

        if not mid_prices:
            return None

        # Aggregate to OHLCV
        mid_prices = np.array(mid_prices)
        return {
            "datetime": date_hour,
            "open": mid_prices[0],
            "high": mid_prices.max(),
            "low": mid_prices.min(),
            "close": mid_prices[-1],
            "volume": np.nan,  # L2 snapshots don't have volume, will be NaN
        }

    except Exception as e:
        print(f"  Error parsing {lz4_path}: {e}")
        return None


def append_candle_to_csv(coin: str, candle: dict[str, Any]):
    """Append a single candle to the coin's CSV file."""
    csv_path = S3_HOURLY_DIR / f"{coin}_candles_1h.csv"

    # Create DataFrame for this candle
    df = pd.DataFrame([candle])

    # Append or create file
    if csv_path.exists():
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)


def process_single_snapshot(
    date_str: str,
    hour: int,
    coin: str,
    temp_dir: Path,
    progress: ProgressTracker
) -> bool:
    """Download, process, and delete a single L2 snapshot.

    Returns:
        True if successful, False otherwise
    """
    # Check if already completed
    if progress.is_completed(date_str, hour, coin):
        return True

    # Construct S3 path
    # Format: s3://hyperliquid-archive/market_data/YYYYMMDD/H/l2Book/COIN.lz4
    s3_path = f"{S3_BUCKET}/{S3_MARKET_DATA_PREFIX}/{date_str}/{hour}/l2Book/{coin}.lz4"
    local_path = temp_dir / f"{date_str}_{hour}_{coin}.lz4"

    try:
        # Step 1: Download
        if not download_s3_file(s3_path, local_path):
            return False

        # Step 2: Parse to candle
        date_hour = datetime.strptime(f"{date_str} {hour:02d}", "%Y%m%d %H").replace(tzinfo=timezone.utc)
        candle = parse_l2_snapshot_to_candle(local_path, coin, date_hour)

        if candle is None:
            print(f"  No data extracted from {s3_path}")
            return False

        # Step 3: Append to CSV
        append_candle_to_csv(coin, candle)

        # Step 4: Mark as completed
        progress.mark_completed(date_str, hour, coin)

        return True

    finally:
        # Step 5: Always delete temp file to save disk
        if local_path.exists():
            local_path.unlink()


def get_coin_list() -> list[str]:
    """Get list of coins from existing API data directory."""
    if not API_DATA_DIR.exists():
        return []

    coins = set()
    for f in API_DATA_DIR.glob("*_candles_4h.csv"):
        coin = f.stem.replace("_candles_4h", "")
        coins.add(coin)

    return sorted(coins)


def generate_date_hour_range(start: datetime, end: datetime) -> list[tuple[str, int]]:
    """Generate list of (date_str, hour) tuples covering the range."""
    result = []
    current = start.replace(minute=0, second=0, microsecond=0)

    while current < end:
        date_str = current.strftime("%Y%m%d")
        hour = current.hour
        result.append((date_str, hour))
        current += timedelta(hours=1)

    return result


def validate_against_api_data(coin: str):
    """Validate S3 candles against existing API candles for overlap period.

    Compares close prices in the overlapping time range to ensure data quality.
    """
    api_file = API_DATA_DIR / f"{coin}_candles_4h.csv"
    s3_file = S3_CANDLES_DIR / f"{coin}_candles_4h.csv"

    if not api_file.exists() or not s3_file.exists():
        print(f"  Skipping {coin}: missing files")
        return

    # Load both datasets
    api_df = pd.read_csv(api_file, parse_dates=['datetime'])
    s3_df = pd.read_csv(s3_file, parse_dates=['datetime'])

    # Find overlap
    api_df = api_df.set_index('datetime')
    s3_df = s3_df.set_index('datetime')

    overlap_idx = api_df.index.intersection(s3_df.index)

    if len(overlap_idx) == 0:
        print(f"  {coin}: No overlapping timestamps")
        return

    # Compare close prices
    api_close = api_df.loc[overlap_idx, 'close']
    s3_close = s3_df.loc[overlap_idx, 'close']

    # Calculate differences
    diff = (api_close - s3_close).abs()
    pct_diff = (diff / api_close * 100)

    max_diff = pct_diff.max()
    mean_diff = pct_diff.mean()

    print(f"  {coin}: {len(overlap_idx)} overlapping candles")
    print(f"    Max price diff: {max_diff:.4f}%")
    print(f"    Mean price diff: {mean_diff:.4f}%")

    if max_diff > 1.0:  # More than 1% difference
        print(f"    ⚠️  WARNING: Large price discrepancy detected!")
    else:
        print(f"    ✓ Validation passed")


def resample_hourly_to_4h(coins: list[str]):
    """Convert intermediate 1h candles into 4h aggregates."""
    if not S3_HOURLY_DIR.exists():
        return

    S3_CANDLES_DIR.mkdir(parents=True, exist_ok=True)

    for coin in coins:
        hourly_path = S3_HOURLY_DIR / f"{coin}_candles_1h.csv"
        if not hourly_path.exists():
            continue

        df = pd.read_csv(hourly_path, parse_dates=["datetime"])
        if df.empty:
            continue

        df = df.set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        resampler = df.resample("4h", origin="epoch", offset="0h")
        agg = pd.DataFrame({
            "open": resampler["open"].first(),
            "high": resampler["high"].max(),
            "low": resampler["low"].min(),
            "close": resampler["close"].last(),
            "volume": resampler["volume"].sum(min_count=1),
        })

        agg = agg.dropna(how="all")
        if agg.empty:
            continue

        out_path = S3_CANDLES_DIR / f"{coin}_candles_4h.csv"
        agg.to_csv(out_path, index=True)
        print(f"Resampled hourly candles to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Hyperliquid S3 historical data")
    parser.add_argument(
        "--start",
        type=str,
        required=False,
        help="Start date (ISO format: 2023-10-01)"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=False,
        help="End date (ISO format: 2024-03-01)"
    )
    parser.add_argument(
        "--coins",
        type=str,
        nargs="+",
        help="Specific coins to fetch (default: all from API data dir)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate S3 candles against API candles"
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="/tmp/hyperliquid_s3",
        help="Temporary directory for downloads (default: /tmp/hyperliquid_s3)"
    )

    args = parser.parse_args()

    # Validation mode
    if args.validate:
        print("=== Validating S3 candles against API data ===\n")
        coins = args.coins or get_coin_list()
        for coin in coins:
            validate_against_api_data(coin)
        return

    # Download mode
    if not args.start or not args.end:
        parser.error("--start and --end are required for download mode")

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    coins = args.coins or get_coin_list()
    if not coins:
        print("Error: No coins found. Either specify --coins or ensure API data exists.")
        sys.exit(1)

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Hyperliquid S3 Historical Data Fetcher ===")
    print(f"Date range: {start_dt.date()} to {end_dt.date()}")
    print(f"Coins: {', '.join(coins)}")
    print(f"Temp dir: {temp_dir}")
    print(f"Output: {S3_CANDLES_DIR}\n")

    # Initialize progress tracker
    progress = ProgressTracker(PROGRESS_FILE)

    # Generate all (date, hour) combinations
    date_hours = generate_date_hour_range(start_dt, end_dt)
    total_tasks = len(date_hours) * len(coins)

    print(f"Total tasks: {total_tasks}")
    print(f"Already completed: {len(progress.completed)}")
    print(f"Remaining: {total_tasks - len(progress.completed)}\n")

    # Process each coin/date/hour combination
    completed_count = 0
    failed_count = 0

    try:
        for date_str, hour in date_hours:
            for coin in coins:
                if progress.is_completed(date_str, hour, coin):
                    completed_count += 1
                    continue

                print(f"[{completed_count + failed_count + 1}/{total_tasks}] Processing {coin} {date_str} {hour:02d}:00")

                success = process_single_snapshot(date_str, hour, coin, temp_dir, progress)

                if success:
                    completed_count += 1
                else:
                    failed_count += 1

        # Final save
        progress.save()
        resample_hourly_to_4h(coins)

        print(f"\n=== Summary ===")
        print(f"Completed: {completed_count}")
        print(f"Failed: {failed_count}")
        print(f"Total: {completed_count + failed_count}")
        print(f"\nCandles saved to: {S3_CANDLES_DIR}")
        print(f"\nRun with --validate to check against API data")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress has been saved.")
        print(f"Resume by running the same command again.")
        progress.save()
        sys.exit(0)


if __name__ == "__main__":
    main()
