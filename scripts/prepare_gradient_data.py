#!/usr/bin/env python3
"""
Prepare wide log returns CSV for Gradient strategy backtest.
Reads individual 1d candle files and pivots them into a wide format.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_all_returns(data_dir: Path, pattern: str = "*_candles_1d.csv") -> pd.DataFrame:
    """
    Load all candle files and compute returns on the fly.
    
    Returns:
        DataFrame with datetime index and one column per asset (coin name from filename).
        NaN for missing/invalid returns.
    """
    all_returns = {}
    
    files = list(data_dir.glob(pattern))
    print(f"Found {len(files)} candle files in {data_dir} matching {pattern}")
    
    for filepath in files:
        # Extract coin name from filename (e.g., "BTC_candles_1d.csv" -> "BTC")
        # Assuming format {COIN}_candles_{INTERVAL}.csv
        coin = filepath.name.replace(pattern.replace("*", ""), "")
        
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            if "close" not in df.columns:
                print(f"Skipping {coin}: 'close' column missing")
                continue
                
            # Compute log returns: ln(p_t / p_{t-1})
            # Handle duplicates just in case
            df = df[~df.index.duplicated(keep='last')].sort_index()
            
            # Simple log return
            rets = np.log(df["close"] / df["close"].shift(1))
            
            # Clean up infinite or insane returns (e.g. from bad data)
            rets = rets.replace([np.inf, -np.inf], np.nan)
            
            all_returns[coin] = rets
            
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
            continue

    if not all_returns:
        raise ValueError("No valid returns data found.")

    # Combine into wide DataFrame
    wide_returns = pd.concat(all_returns, axis=1)
    wide_returns.sort_index(inplace=True)
    
    return wide_returns


def main():
    parser = argparse.ArgumentParser(description="Prepare specific return data for Gradient.")
    parser.add_argument(
        "--data-dir", 
        default="data/market_data/1d",
        help="Directory containing 1d candle CSVs"
    )
    parser.add_argument(
        "--output",
        default="data/gradient/wide_returns_1d.csv",
        help="Output path for wide returns CSV"
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_path = Path(args.output)
    
    if not data_path.exists():
        print(f"Error: Data directory {data_path} does not exist.")
        return

    print(f"Loading returns from {data_path}...")
    wide_returns = load_all_returns(data_path)
    
    print(f"Loaded returns for {len(wide_returns.columns)} assets.")
    print(f"Date range: {wide_returns.index.min()} to {wide_returns.index.max()}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide_returns.to_csv(output_path)
    print(f"Saved wide returns to {output_path}")


if __name__ == "__main__":
    main()
