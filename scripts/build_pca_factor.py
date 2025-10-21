#!/usr/bin/env python3
"""
Build rolling PCA market factor from Hyperliquid returns data.

Computes PC1 loadings on a rolling window to create a market-neutral factor.
Handles variable asset counts over time by only including assets with valid data
in each window.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_all_returns(data_dir: Path, pattern: str = "*_merged_1h.csv") -> pd.DataFrame:
    """
    Load all merged return files and construct a wide returns matrix.

    Returns:
        DataFrame with datetime index and one column per asset (coin name from filename).
        NaN for missing/invalid returns.
    """
    # Look in market_data subdirectory
    market_data_dir = data_dir / "market_data"
    files = sorted(market_data_dir.glob(pattern))
    if not files:
        raise ValueError(f"No files matching {pattern} found in {market_data_dir}")

    print(f"Loading {len(files)} assets from {market_data_dir}")

    returns_dict = {}
    for fpath in files:
        # Extract coin name from filename pattern: COIN_merged_1h.csv
        parts = fpath.stem.split("_")
        if len(parts) >= 3 and parts[-2] == "merged":
            coin = "_".join(parts[:-2])  # e.g., BTC from BTC_merged_1h.csv
        else:
            coin = fpath.stem

        try:
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
            if "ret" in df.columns:
                returns_dict[coin] = df["ret"]
        except Exception as exc:
            print(f"Warning: failed to load {fpath.name}: {exc}")

    if not returns_dict:
        raise ValueError("No valid return data loaded")

    # Combine into wide DataFrame
    returns_wide = pd.DataFrame(returns_dict)
    returns_wide.index = pd.to_datetime(returns_wide.index, utc=True)
    returns_wide = returns_wide.sort_index()

    print(f"Loaded returns matrix: {returns_wide.shape[0]} hours × {returns_wide.shape[1]} assets")
    print(f"Date range: {returns_wide.index[0]} to {returns_wide.index[-1]}")

    return returns_wide


def resample_to_daily(returns_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly returns to daily log returns.

    For each asset, sum hourly log returns within each day.
    Only compute daily return if at least one hourly return exists.
    """
    # Use UTC day boundaries
    daily = returns_1h.groupby(pd.Grouper(freq="D")).sum()

    # Count valid (non-NaN) hourly observations per day
    counts = returns_1h.groupby(pd.Grouper(freq="D")).count()

    # Set daily return to NaN if no valid hourly returns
    daily = daily.where(counts > 0, np.nan)

    print(f"Resampled to daily: {daily.shape[0]} days × {daily.shape[1]} assets")

    return daily


def compute_rolling_pca_loadings(
    returns_daily: pd.DataFrame,
    window_days: int = 60,
    min_assets: int = 10,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling PCA and extract PC1 loadings (market factor weights).

    For each day, compute PCA on the trailing window_days of returns.
    Only include assets with sufficient valid data in the window.

    Args:
        returns_daily: Daily returns matrix (days × assets)
        window_days: Rolling window size in days
        min_assets: Minimum number of assets required to compute PCA
        min_periods: Minimum number of valid observations required per asset

    Returns:
        DataFrame with datetime index and one column per asset containing PC1 loadings.
        NaN for assets excluded from PCA on that day.
    """
    dates = returns_daily.index
    assets = returns_daily.columns

    # Pre-allocate output matrix
    loadings_matrix = np.full((len(dates), len(assets)), np.nan)
    variance_explained = np.full(len(dates), np.nan)
    n_assets_used = np.full(len(dates), 0, dtype=int)

    print(f"Computing rolling PCA with {window_days}-day window...")

    for i, date in enumerate(dates):
        if i < window_days - 1:
            continue  # Not enough history yet

        # Extract window
        window_returns = returns_daily.iloc[max(0, i - window_days + 1) : i + 1]

        # Filter to assets with sufficient valid data in window
        valid_counts = window_returns.count()
        valid_assets = valid_counts[valid_counts >= min_periods].index

        if len(valid_assets) < min_assets:
            continue  # Not enough assets with valid data

        # Extract valid subset and drop any remaining NaNs
        window_clean = window_returns[valid_assets].dropna(axis=0, how="any")

        if len(window_clean) < min_periods or window_clean.shape[1] < min_assets:
            continue  # Insufficient data after cleaning

        # Fit PCA
        try:
            pca = PCA(n_components=1)
            pca.fit(window_clean)

            # Extract PC1 loadings
            pc1_loadings = pca.components_[0]  # Shape: (n_valid_assets,)

            # Map back to full asset list
            asset_indices = [assets.get_loc(asset) for asset in valid_assets]
            loadings_matrix[i, asset_indices] = pc1_loadings

            variance_explained[i] = pca.explained_variance_ratio_[0]
            n_assets_used[i] = len(valid_assets)

        except Exception as exc:
            print(f"Warning: PCA failed on {date}: {exc}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dates)} days")

    # Convert to DataFrame
    loadings_df = pd.DataFrame(
        loadings_matrix,
        index=dates,
        columns=assets,
    )

    # Add summary statistics
    loadings_df["_variance_explained"] = variance_explained
    loadings_df["_n_assets"] = n_assets_used

    valid_days = (n_assets_used > 0).sum()
    print(f"Completed PCA for {valid_days}/{len(dates)} days")
    print(f"Mean variance explained by PC1: {variance_explained[~np.isnan(variance_explained)].mean():.3f}")
    print(f"Mean assets per day: {n_assets_used[n_assets_used > 0].mean():.1f}")

    return loadings_df


def save_factor_weights(loadings_df: pd.DataFrame, output_path: Path) -> None:
    """Save factor weights to CSV."""
    loadings_df.to_csv(output_path, index=True)
    print(f"Saved factor weights to {output_path}")
    print(f"  Shape: {loadings_df.shape[0]} days × {loadings_df.shape[1]} columns")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute rolling PCA market factor from Hyperliquid returns data."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing *_merged_1h.csv files (default: data)",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*_merged_1h.csv",
        help="File pattern to match (default: *_merged_1h.csv)",
    )
    p.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling window size in days (default: 60)",
    )
    p.add_argument(
        "--min-assets",
        type=int,
        default=10,
        help="Minimum number of assets required for PCA (default: 10)",
    )
    p.add_argument(
        "--min-periods",
        type=int,
        default=30,
        help="Minimum valid observations per asset in window (default: 30)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/features/pca_factor_loadings.csv"),
        help="Output path for factor weights (default: data/features/pca_factor_loadings.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load all returns
    returns_1h = load_all_returns(args.data_dir, pattern=args.pattern)

    # Resample to daily
    returns_daily = resample_to_daily(returns_1h)

    # Compute rolling PCA loadings
    loadings = compute_rolling_pca_loadings(
        returns_daily,
        window_days=args.window,
        min_assets=args.min_assets,
        min_periods=args.min_periods,
    )

    # Save results
    save_factor_weights(loadings, args.output)


if __name__ == "__main__":
    main()
