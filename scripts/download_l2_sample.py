#!/usr/bin/env python3
"""
Download one day of L2 orderbook snapshots for liquidity analysis.

Downloads hourly L2 snapshots for all markets for a single day to analyze
orderbook depth, spreads, and build a slippage cost model.
"""
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import argparse


def decompress_lz4(lz4_path: Path) -> bytes:
    """Decompress an lz4 file and return raw bytes."""
    result = subprocess.run(
        ["lz4", "-d", "-c", str(lz4_path)],
        capture_output=True,
        check=True
    )
    return result.stdout


def download_l2_snapshot(coin: str, date: str, hour: int, output_dir: Path) -> bool:
    """
    Download a single L2 snapshot for a given coin, date, and hour.

    Returns True if successful, False if file doesn't exist.
    """
    # Convert date from YYYY-MM-DD to YYYYMMDD for S3 path
    s3_date = date.replace("-", "")

    # S3 path: s3://hyperliquid-archive/market_data/YYYYMMDD/HH/l2Book/COIN.lz4
    s3_path = f"s3://hyperliquid-archive/market_data/{s3_date}/{hour:02d}/l2Book/{coin}.lz4"

    # Local path
    local_file = output_dir / date / f"{hour:02d}" / f"{coin}.lz4"
    local_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip if already downloaded
    if local_file.exists():
        return True
    
    # Download
    try:
        subprocess.run(
            ["aws", "s3", "cp", s3_path, str(local_file), "--request-payer", "requester"],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError:
        # File doesn't exist in S3 (common for newer coins)
        return False


def get_market_list() -> list[str]:
    """Get list of markets from our downloaded 4h candles."""
    data_dir = Path("data/market_data")
    candles_files = sorted(data_dir.glob("*_candles_4h.csv"))
    
    markets = []
    for f in candles_files:
        parts = f.stem.split("_")
        if len(parts) >= 3 and parts[-2] == "candles":
            coin = "_".join(parts[:-2])
            markets.append(coin)
    
    return markets


def main():
    parser = argparse.ArgumentParser(description="Download L2 snapshots for one day")
    parser.add_argument("--date", type=str, default="2025-10-20", help="Date to download (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, default=Path("data/l2_snapshots"), help="Output directory")
    parser.add_argument("--sample-hours", type=int, default=None, help="Sample every N hours (default: download all 24 hours)")
    args = parser.parse_args()
    
    markets = get_market_list()
    print(f"Found {len(markets)} markets")
    print(f"Downloading L2 snapshots for {args.date}")
    print()
    
    # Determine which hours to download
    if args.sample_hours:
        hours = list(range(0, 24, args.sample_hours))
        print(f"Sampling every {args.sample_hours} hours: {hours}")
    else:
        hours = list(range(24))
        print(f"Downloading all 24 hours")
    print()
    
    success_count = 0
    fail_count = 0
    
    for i, coin in enumerate(markets, 1):
        print(f"[{i}/{len(markets)}] {coin}... ", end="", flush=True)
        
        coin_success = 0
        for hour in hours:
            if download_l2_snapshot(coin, args.date, hour, args.output):
                coin_success += 1
        
        if coin_success > 0:
            print(f"✓ {coin_success}/{len(hours)} snapshots")
            success_count += 1
        else:
            print(f"✗ no data")
            fail_count += 1
    
    print()
    print(f"Summary:")
    print(f"  {success_count} markets with data")
    print(f"  {fail_count} markets without data")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
