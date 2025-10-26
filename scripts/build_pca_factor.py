#!/usr/bin/env python3
"""
Build rolling PCA market factor from Hyperliquid returns data.

Computes PC1 loadings on a rolling window to create a market-neutral factor.
Handles variable asset counts over time by only including assets with valid data
in each window.

Supports volume-weighted PCA with multiple weighting schemes.
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
    Load all candle files and compute returns on the fly.

    Returns:
        DataFrame with datetime index and one column per asset (coin name from filename).
        NaN for missing/invalid returns.
    """
    # Convert merged pattern to candles pattern
    candles_pattern = pattern.replace("_merged_", "_candles_")

    # Look in market_data subdirectory
    market_data_dir = data_dir / "market_data"
    files = sorted(market_data_dir.glob(candles_pattern))
    if not files:
        raise ValueError(f"No files matching {candles_pattern} found in {market_data_dir}")

    print(f"Loading {len(files)} assets from {market_data_dir}")

    returns_dict = {}
    for fpath in files:
        # Extract coin name from filename pattern: COIN_candles_1h.csv
        parts = fpath.stem.split("_")
        if len(parts) >= 3 and parts[-2] == "candles":
            coin = "_".join(parts[:-2])  # e.g., BTC from BTC_candles_1h.csv
        else:
            coin = fpath.stem

        try:
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
            if "close" in df.columns:
                # Compute log returns from close prices
                log_returns = np.log(df["close"] / df["close"].shift(1))
                returns_dict[coin] = log_returns
        except Exception as exc:
            print(f"Warning: failed to load {fpath.name}: {exc}")

    if not returns_dict:
        raise ValueError("No valid return data loaded")

    # Combine into wide DataFrame
    returns_wide = pd.DataFrame(returns_dict)
    returns_wide.index = pd.to_datetime(returns_wide.index, utc=True)
    returns_wide = returns_wide.sort_index()

    print(f"Loaded returns matrix: {returns_wide.shape[0]} periods × {returns_wide.shape[1]} assets")
    print(f"Date range: {returns_wide.index[0]} to {returns_wide.index[-1]}")

    return returns_wide


def load_volume_data(data_dir: Path, pattern: str = "*_candles_1h.csv") -> pd.DataFrame:
    """
    Load volume data from candle files and construct wide volume matrix.

    Returns:
        DataFrame with datetime index and one column per asset (coin name from filename).
        NaN for missing/invalid volume.
    """
    # Look in market_data subdirectory
    market_data_dir = data_dir / "market_data"
    files = sorted(market_data_dir.glob(pattern))
    if not files:
        raise ValueError(f"No files matching {pattern} found in {market_data_dir}")

    print(f"Loading volume data from {len(files)} assets from {market_data_dir}")

    volume_dict = {}
    for fpath in files:
        # Extract coin name from filename pattern: COIN_candles_1h.csv
        parts = fpath.stem.split("_")
        if len(parts) >= 3 and parts[-2] == "candles":
            coin = "_".join(parts[:-2])  # e.g., BTC from BTC_candles_1h.csv
        else:
            coin = fpath.stem

        try:
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
            if "volume" in df.columns:
                volume_dict[coin] = df["volume"]
            else:
                print(f"  Warning: no volume column in {fpath.name}")
        except Exception as exc:
            print(f"  Warning: failed to load {fpath.name}: {exc}")

    if not volume_dict:
        raise ValueError("No valid volume data loaded")

    # Combine into wide DataFrame
    volumes_wide = pd.DataFrame(volume_dict)
    volumes_wide.index = pd.to_datetime(volumes_wide.index, utc=True)
    volumes_wide = volumes_wide.sort_index()

    print(f"Loaded volume matrix: {volumes_wide.shape[0]} hours × {volumes_wide.shape[1]} assets")
    print(f"Date range: {volumes_wide.index[0]} to {volumes_wide.index[-1]}")

    return volumes_wide


def compute_volume_weights(
    volumes_daily: pd.DataFrame,
    prices_daily: pd.DataFrame | None = None,
    method: str = "sqrt",
    window_periods: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling volume weights for PCA.

    Args:
        volumes_daily: Volume matrix (periods × assets)
        prices_daily: Price matrix (periods × assets), required for dollar volume methods
        method: Weighting scheme - "none", "sqrt", "log", "dollar", "sqrt_dollar"
        window_periods: Rolling window for computing average volume (in number of periods)

    Returns:
        Weight matrix (periods × assets), normalized to mean=1 within each window
    """
    if method == "none":
        # Equal weights
        return pd.DataFrame(1.0, index=volumes_daily.index, columns=volumes_daily.columns)

    # Compute trailing average volume for each asset
    avg_volume = volumes_daily.rolling(window=window_periods, min_periods=window_periods // 2).mean()

    # Replace zero/missing volumes with column median to avoid div by zero
    for col in avg_volume.columns:
        col_median = avg_volume[col].median()
        if pd.isna(col_median) or col_median == 0:
            col_median = 1.0  # Fallback for completely missing data
        avg_volume[col] = avg_volume[col].fillna(col_median).replace(0, col_median)

    # Apply weighting function
    if method == "sqrt":
        weights = np.sqrt(avg_volume)
    elif method == "log":
        weights = np.log1p(avg_volume)  # log(1 + volume) to handle zeros
    elif method == "dollar":
        if prices_daily is None:
            raise ValueError("Dollar volume methods require prices_daily argument")
        # Align prices with volumes
        prices_aligned = prices_daily.reindex_like(avg_volume)
        dollar_volume = prices_aligned * avg_volume
        weights = dollar_volume
    elif method == "sqrt_dollar":
        if prices_daily is None:
            raise ValueError("Dollar volume methods require prices_daily argument")
        prices_aligned = prices_daily.reindex_like(avg_volume)
        dollar_volume = prices_aligned * avg_volume
        weights = np.sqrt(dollar_volume)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Normalize weights to mean=1 within each row (period)
    # This ensures PCA scaling isn't affected by absolute weight magnitudes
    row_means = weights.mean(axis=1, skipna=True)
    weights = weights.div(row_means, axis=0)

    # Handle any remaining NaNs
    weights = weights.fillna(1.0)

    print(f"Computed {method} volume weights (window={window_periods} periods)")
    print(f"  Weight range: [{weights.min().min():.3f}, {weights.max().max():.3f}]")

    return weights


def resample_returns(returns_1h: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Aggregate hourly returns to specified frequency.

    For each asset, sum hourly log returns within each period.
    Only compute period return if at least one hourly return exists.

    Args:
        returns_1h: Hourly returns DataFrame
        freq: Pandas frequency string - "H" (hourly), "4H", "6H", "12H", "D" (daily), "W" (weekly)
    """
    # Use UTC period boundaries
    resampled = returns_1h.groupby(pd.Grouper(freq=freq)).sum()

    # Count valid (non-NaN) hourly observations per period
    counts = returns_1h.groupby(pd.Grouper(freq=freq)).count()

    # Set period return to NaN if no valid hourly returns
    resampled = resampled.where(counts > 0, np.nan)

    print(f"Resampled to {freq}: {resampled.shape[0]} periods × {resampled.shape[1]} assets")

    return resampled


def resample_to_daily(returns_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly returns to daily log returns.

    DEPRECATED: Use resample_returns(returns_1h, freq="D") instead.
    Kept for backward compatibility.
    """
    return resample_returns(returns_1h, freq="D")


def compute_rolling_pca_loadings(
    returns_daily: pd.DataFrame,
    window_days: int = 60,
    min_assets: int = 10,
    min_periods: int = 30,
    n_components: int = 1,
) -> pd.DataFrame:
    """
    Compute rolling PCA and extract PC loadings (market factor weights).

    For each day, compute PCA on the trailing window_days of returns.
    Only include assets with sufficient valid data in the window.

    Args:
        returns_daily: Daily returns matrix (days × assets)
        window_days: Rolling window size in days
        min_assets: Minimum number of assets required to compute PCA
        min_periods: Minimum number of valid observations required per asset
        n_components: Number of principal components to compute (1, 2, or 3)

    Returns:
        DataFrame with datetime index and columns for each PC loading per asset.
        - For n_components=1: One column per asset (PC1 loadings)
        - For n_components=3: Three columns per asset (asset_pc1, asset_pc2, asset_pc3)
        NaN for assets excluded from PCA on that day.
    """
    dates = returns_daily.index
    assets = returns_daily.columns

    # Pre-allocate output matrices - one per component
    loadings_matrices = [np.full((len(dates), len(assets)), np.nan) for _ in range(n_components)]
    variance_explained = np.full((len(dates), n_components), np.nan)
    n_assets_used = np.full(len(dates), 0, dtype=int)

    print(f"Computing rolling PCA with {window_days}-day window...")

    pca_computed = 0  # Track actual PCA computations

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
            pca = PCA(n_components=n_components)
            pca.fit(window_clean)

            # Extract loadings for each component
            for pc_idx in range(n_components):
                pc_loadings = pca.components_[pc_idx]  # Shape: (n_valid_assets,)

                # Map back to full asset list
                asset_indices = [assets.get_loc(asset) for asset in valid_assets]
                loadings_matrices[pc_idx][i, asset_indices] = pc_loadings

            variance_explained[i, :] = pca.explained_variance_ratio_[:n_components]
            n_assets_used[i] = len(valid_assets)

            pca_computed += 1
            if pca_computed % 100 == 0:
                print(f"  Computed {pca_computed} PCAs (at date index {i + 1}/{len(dates)})")

        except Exception as exc:
            print(f"Warning: PCA failed on {date}: {exc}")
            continue

    # Convert to DataFrame
    if n_components == 1:
        # Single component: original format (asset columns)
        loadings_df = pd.DataFrame(
            loadings_matrices[0],
            index=dates,
            columns=assets,
        )
        loadings_df["_variance_explained"] = variance_explained[:, 0]
    else:
        # Multiple components: suffixed columns (asset_pc1, asset_pc2, asset_pc3)
        column_data = {}
        for pc_idx in range(n_components):
            for asset in assets:
                col_name = f"{asset}_pc{pc_idx + 1}"
                column_data[col_name] = loadings_matrices[pc_idx][:, assets.get_loc(asset)]

        loadings_df = pd.DataFrame(column_data, index=dates)

        # Add variance explained for each component
        for pc_idx in range(n_components):
            loadings_df[f"_variance_explained_pc{pc_idx + 1}"] = variance_explained[:, pc_idx]

    # Add asset count metadata
    loadings_df["_n_assets"] = n_assets_used

    valid_days = (n_assets_used > 0).sum()
    print(f"Completed PCA for {valid_days}/{len(dates)} days")

    # Print variance explained statistics
    for pc_idx in range(n_components):
        var_exp = variance_explained[:, pc_idx]
        valid_var = var_exp[~np.isnan(var_exp)]
        if len(valid_var) > 0:
            print(f"Mean variance explained by PC{pc_idx + 1}: {valid_var.mean():.3f}")

    print(f"Mean assets per day: {n_assets_used[n_assets_used > 0].mean():.1f}")

    return loadings_df


def compute_rolling_pca_loadings_weighted(
    returns_daily: pd.DataFrame,
    weights_daily: pd.DataFrame,
    window_days: int = 60,
    min_assets: int = 10,
    min_periods: int = 30,
    n_components: int = 1,
) -> pd.DataFrame:
    """
    Compute rolling PCA with volume weighting and extract PC loadings.

    Volume weighting is applied by scaling returns before PCA fitting.

    Args:
        returns_daily: Daily returns matrix (days × assets)
        weights_daily: Volume weight matrix (days × assets), aligned with returns
        window_days: Rolling window size in days
        min_assets: Minimum number of assets required to compute PCA
        min_periods: Minimum number of valid observations required per asset
        n_components: Number of principal components to compute (1, 2, or 3)

    Returns:
        DataFrame with datetime index and columns for each PC loading per asset.
        - For n_components=1: One column per asset (PC1 loadings)
        - For n_components=3: Three columns per asset (asset_pc1, asset_pc2, asset_pc3)
        NaN for assets excluded from PCA on that day.
    """
    dates = returns_daily.index
    assets = returns_daily.columns

    # Pre-allocate output matrices - one per component
    loadings_matrices = [np.full((len(dates), len(assets)), np.nan) for _ in range(n_components)]
    variance_explained = np.full((len(dates), n_components), np.nan)
    n_assets_used = np.full(len(dates), 0, dtype=int)

    print(f"Computing volume-weighted rolling PCA with {window_days}-day window...")

    pca_computed = 0  # Track actual PCA computations

    for i, date in enumerate(dates):
        if i < window_days - 1:
            continue  # Not enough history yet

        # Extract windows for returns and weights
        window_returns = returns_daily.iloc[max(0, i - window_days + 1) : i + 1]
        window_weights = weights_daily.iloc[max(0, i - window_days + 1) : i + 1]

        # Filter to assets with sufficient valid data in window
        valid_counts = window_returns.count()
        valid_assets = valid_counts[valid_counts >= min_periods].index

        if len(valid_assets) < min_assets:
            continue  # Not enough assets with valid data

        # Extract valid subset and drop any remaining NaNs
        window_returns_valid = window_returns[valid_assets]
        window_weights_valid = window_weights[valid_assets]

        # Align: drop rows where either returns or weights are NaN
        combined = pd.concat([window_returns_valid, window_weights_valid], axis=1, keys=["ret", "weight"])
        combined_clean = combined.dropna(axis=0, how="any")

        if len(combined_clean) < min_periods:
            continue  # Insufficient data after cleaning

        # Split back into returns and weights
        returns_clean = combined_clean["ret"]
        weights_clean = combined_clean["weight"]

        if returns_clean.shape[1] < min_assets:
            continue

        # Apply volume weighting: scale returns by weights
        weighted_returns = returns_clean * weights_clean

        # Fit PCA on weighted returns
        try:
            pca = PCA(n_components=n_components)
            pca.fit(weighted_returns)

            # Extract loadings for each component
            valid_asset_list = returns_clean.columns
            for pc_idx in range(n_components):
                pc_loadings = pca.components_[pc_idx]  # Shape: (n_valid_assets,)

                # Map back to full asset list
                asset_indices = [assets.get_loc(asset) for asset in valid_asset_list]
                loadings_matrices[pc_idx][i, asset_indices] = pc_loadings

            variance_explained[i, :] = pca.explained_variance_ratio_[:n_components]
            n_assets_used[i] = len(valid_asset_list)

            pca_computed += 1
            if pca_computed % 100 == 0:
                print(f"  Computed {pca_computed} PCAs (at date index {i + 1}/{len(dates)})")

        except Exception as exc:
            print(f"Warning: PCA failed on {date}: {exc}")
            continue

    # Convert to DataFrame
    if n_components == 1:
        # Single component: original format (asset columns)
        loadings_df = pd.DataFrame(
            loadings_matrices[0],
            index=dates,
            columns=assets,
        )
        loadings_df["_variance_explained"] = variance_explained[:, 0]
    else:
        # Multiple components: suffixed columns (asset_pc1, asset_pc2, asset_pc3)
        column_data = {}
        for pc_idx in range(n_components):
            for asset in assets:
                col_name = f"{asset}_pc{pc_idx + 1}"
                column_data[col_name] = loadings_matrices[pc_idx][:, assets.get_loc(asset)]

        loadings_df = pd.DataFrame(column_data, index=dates)

        # Add variance explained for each component
        for pc_idx in range(n_components):
            loadings_df[f"_variance_explained_pc{pc_idx + 1}"] = variance_explained[:, pc_idx]

    # Add asset count metadata
    loadings_df["_n_assets"] = n_assets_used

    valid_days = (n_assets_used > 0).sum()
    print(f"Completed volume-weighted PCA for {valid_days}/{len(dates)} days")

    # Print variance explained statistics
    for pc_idx in range(n_components):
        var_exp = variance_explained[:, pc_idx]
        valid_var = var_exp[~np.isnan(var_exp)]
        if len(valid_var) > 0:
            print(f"Mean variance explained by PC{pc_idx + 1}: {valid_var.mean():.3f}")

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
        help="Directory containing market data files (default: data)",
    )
    p.add_argument(
        "--interval",
        type=str,
        default="1h",
        choices=["1h", "4h"],
        help="Data interval: 1h or 4h (default: 1h)",
    )
    p.add_argument(
        "--freq",
        type=str,
        default="D",
        help="Resampling frequency: H (hourly), 4H, 6H, 12H, D (daily), W (weekly) (default: D)",
    )
    p.add_argument(
        "--window",
        type=int,
        default=1440,
        help="Rolling window size in hours (default: 1440 = 60 days)",
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
    p.add_argument(
        "--weight-method",
        type=str,
        default="none",
        choices=["none", "sqrt", "log", "dollar", "sqrt_dollar"],
        help="Volume weighting method: none (equal-weight), sqrt, log, dollar, sqrt_dollar (default: none)",
    )
    p.add_argument(
        "--compare-all",
        action="store_true",
        help="Compute PCA for all weighting methods and save separately with suffixes",
    )
    p.add_argument(
        "--n-components",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Number of principal components to compute (default: 1)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Construct patterns based on interval
    merged_pattern = f"*_merged_{args.interval}.csv"
    candles_pattern = f"*_candles_{args.interval}.csv"

    # Load all returns
    returns_1h = load_all_returns(args.data_dir, pattern=merged_pattern)

    # Resample to specified frequency
    returns_resampled = resample_returns(returns_1h, freq=args.freq)

    # Convert window from hours to number of periods based on frequency
    # Parse frequency string to get period length in hours
    freq_str = args.freq.upper()
    if freq_str == "H":
        period_hours = 1
    elif freq_str == "D":
        period_hours = 24
    elif freq_str == "W":
        period_hours = 168
    elif freq_str.endswith("H"):
        # Handle patterns like "4H", "6H", "12H"
        period_hours = int(freq_str[:-1])
    else:
        raise ValueError(f"Cannot parse frequency: {args.freq}")

    window_periods = args.window // period_hours
    print(f"Window: {args.window} hours = {window_periods} periods at {args.freq} frequency")

    # Determine which methods to compute
    if args.compare_all:
        methods = ["none", "sqrt", "log", "dollar", "sqrt_dollar"]
        print(f"\n{'='*60}")
        print("COMPARE-ALL MODE: Computing PCA for all weighting methods")
        print(f"{'='*60}\n")
    else:
        methods = [args.weight_method]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Processing method: {method.upper()}")
        print(f"{'='*60}\n")

        if method == "none":
            # Equal-weighted PCA (original)
            loadings = compute_rolling_pca_loadings(
                returns_resampled,
                window_days=window_periods,
                min_assets=args.min_assets,
                min_periods=args.min_periods,
                n_components=args.n_components,
            )
        else:
            # Volume-weighted PCA
            # Load volume data
            volumes_1h = load_volume_data(args.data_dir, pattern=candles_pattern)

            # Load prices if needed for dollar volume methods
            prices_resampled = None
            if method in ("dollar", "sqrt_dollar"):
                print("Loading price data for dollar volume calculation...")
                # Load prices from candles
                market_data_dir = args.data_dir / "market_data"
                files = sorted(market_data_dir.glob(candles_pattern))
                prices_dict = {}
                for fpath in files:
                    parts = fpath.stem.split("_")
                    if len(parts) >= 3 and parts[-2] == "candles":
                        coin = "_".join(parts[:-2])
                    else:
                        coin = fpath.stem
                    try:
                        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                        if "close" in df.columns:
                            prices_dict[coin] = df["close"]
                    except Exception:
                        pass
                prices_1h = pd.DataFrame(prices_dict)
                prices_1h.index = pd.to_datetime(prices_1h.index, utc=True)
                prices_1h = prices_1h.sort_index()
                # Resample prices to match frequency (take last price in period)
                prices_resampled = prices_1h.groupby(pd.Grouper(freq=args.freq)).last()
                print(f"Loaded prices: {prices_resampled.shape[0]} periods × {prices_resampled.shape[1]} assets")

            # Resample volume to match frequency (sum over period)
            volumes_resampled = volumes_1h.groupby(pd.Grouper(freq=args.freq)).sum()
            print(f"Resampled volume to {args.freq}: {volumes_resampled.shape[0]} periods × {volumes_resampled.shape[1]} assets")

            # Compute volume weights
            weights_resampled = compute_volume_weights(
                volumes_resampled,
                prices_daily=prices_resampled,
                method=method,
                window_periods=window_periods,
            )

            # Align weights with returns
            weights_aligned = weights_resampled.reindex_like(returns_resampled)

            # Compute volume-weighted PCA
            loadings = compute_rolling_pca_loadings_weighted(
                returns_resampled,
                weights_aligned,
                window_days=window_periods,
                min_assets=args.min_assets,
                min_periods=args.min_periods,
                n_components=args.n_components,
            )

        # Save results
        if args.compare_all:
            # Add method suffix to output path
            suffix = f"_{method}"
            if args.n_components > 1:
                suffix += f"_{args.n_components}pc"
            output_path = args.output.parent / f"{args.output.stem}{suffix}{args.output.suffix}"
        else:
            # Add n_components suffix if > 1
            if args.n_components > 1:
                output_path = args.output.parent / f"{args.output.stem}_{args.n_components}pc{args.output.suffix}"
            else:
                output_path = args.output

        save_factor_weights(loadings, output_path)
        print(f"\n{method.upper()} method complete.\n")


if __name__ == "__main__":
    main()
