"""
Data preparation for alpha model training.

This module handles:
1. Loading idiosyncratic returns and funding rates
2. Computing vol-normalized funding features at multiple spans
3. Computing H-period forward returns (target variable)
4. Creating aligned feature matrix X and target vector y
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

BASE_INTERVAL_HOURS = 4  # All merged market data is sampled every 4 hours

def load_all_returns(data_dir: str = "data/market_data") -> pd.DataFrame:
    """
    Load 4-hourly returns for all assets from merged files.

    Args:
        data_dir: Directory containing *_merged_4h.csv files

    Returns:
        Wide DataFrame with datetime index and asset columns containing log returns
    """
    data_path = Path(data_dir)
    merged_files = list(data_path.glob("*_merged_4h.csv"))

    if not merged_files:
        raise FileNotFoundError(f"No *_merged_4h.csv files found in {data_dir}")

    returns_dict = {}

    for file in merged_files:
        # Extract asset name from filename (e.g., "BTC_merged_4h.csv" -> "BTC")
        asset = file.stem.replace("_merged_4h", "")

        # Load file (first column is datetime index, no name)
        df = pd.read_csv(file, index_col=0, parse_dates=True)

        # Get log returns column (named 'ret' in merged files)
        if 'ret' in df.columns:
            returns_dict[asset] = df['ret']

    # Combine into wide DataFrame
    returns = pd.DataFrame(returns_dict)
    returns = returns.sort_index()

    print(f"Loaded returns for {len(returns.columns)} assets")
    print(f"Date range: {returns.index.min()} to {returns.index.max()}")
    print(f"Shape: {returns.shape}")

    return returns


def load_all_funding(data_dir: str = "data/market_data") -> pd.DataFrame:
    """
    Load 4-hourly funding rates for all assets from merged files.

    Args:
        data_dir: Directory containing *_merged_4h.csv files

    Returns:
        Wide DataFrame with datetime index and asset columns containing funding rates
    """
    data_path = Path(data_dir)
    merged_files = list(data_path.glob("*_merged_4h.csv"))

    if not merged_files:
        raise FileNotFoundError(f"No *_merged_4h.csv files found in {data_dir}")

    funding_dict = {}

    for file in merged_files:
        # Extract asset name
        asset = file.stem.replace("_merged_4h", "")

        # Load file (first column is datetime index, no name)
        df = pd.read_csv(file, index_col=0, parse_dates=True)

        # Get funding rate column (named 'funding' in merged files)
        if 'funding' in df.columns:
            funding_dict[asset] = df['funding']

    # Combine into wide DataFrame
    funding = pd.DataFrame(funding_dict)
    funding = funding.sort_index()

    print(f"Loaded funding rates for {len(funding.columns)} assets")
    print(f"Date range: {funding.index.min()} to {funding.index.max()}")
    print(f"Shape: {funding.shape}")

    return funding


def _hours_to_bars(hours: int, base_interval_hours: int = BASE_INTERVAL_HOURS) -> int:
    """Convert an hours-based span to a number of bars on the base grid."""
    if hours <= 0:
        raise ValueError("Span hours must be positive")
    bars = max(1, int(round(hours / base_interval_hours)))
    return bars


def compute_funding_features(
    funding_rates: pd.DataFrame,
    spans: List[int] = [2, 4, 8, 16, 32, 64],
    std_lookback: int = 90 * 24,  # 90 days in hours
    vol_span: int = 128,  # For normalizing (hours)
    base_interval_hours: int = BASE_INTERVAL_HOURS,
) -> pd.DataFrame:
    """
    Compute EWMA funding features, normalized by historical volatility.

    This creates vol-normalized funding rate signals analogous to momentum signals:
    1. Compute funding volatility for each asset (rolling std)
    2. Vol-normalize funding rates
    3. Compute EWMA at multiple spans

    Args:
        funding_rates: Wide DataFrame with datetime index and asset columns
        spans: List of EWMA spans to compute
        std_lookback: Window for computing funding volatility (hours)
        vol_span: EWMA span for volatility estimation

    Returns:
        Long DataFrame with MultiIndex (timestamp, asset) and columns=funding_2, funding_4, ...
    """
    print(f"Computing funding features for {len(funding_rates.columns)} assets...")

    vol_span_bars = _hours_to_bars(vol_span, base_interval_hours)

    # 1. Compute funding volatility for each asset using EWMA
    funding_vol = funding_rates.ewm(
        span=vol_span_bars,
        min_periods=max(1, vol_span_bars // 2)
    ).std()

    # Avoid division by zero
    funding_vol = funding_vol.replace(0, np.nan)

    # 2. Vol-normalize funding rates
    funding_norm = funding_rates / funding_vol

    # 3. Compute EWMA at multiple spans
    funding_features = {}
    for span in spans:
        print(f"  Computing funding EWMA with span={span}...")
        span_bars = _hours_to_bars(span, base_interval_hours)
        ewma = funding_norm.ewm(
            span=span_bars,
            min_periods=max(1, span_bars // 2)
        ).mean()

        # Convert to long format
        ewma_long = ewma.stack()
        ewma_long.index.names = ['timestamp', 'asset']
        funding_features[f'funding_{span}'] = ewma_long

    # 4. Combine into DataFrame
    result = pd.DataFrame(funding_features)

    # Drop NaN values
    result = result.dropna()

    print(f"Created funding features: {list(result.columns)}")
    print(f"Shape: {result.shape}")

    return result


def compute_forward_returns(
    returns: pd.DataFrame,
    H: int = 24,
    vol_span: int = 128,
    base_interval_hours: int = BASE_INTERVAL_HOURS,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute H-period forward returns, vol-normalized.

    Args:
        returns: Wide DataFrame with datetime index and asset columns (idiosyncratic returns)
        H: Holding period in hours
        vol_span: Span for volatility EWMA

    Returns:
        Tuple of (forward_returns_normalized, volatility):
        - forward_returns_normalized: Series with MultiIndex (timestamp, asset)
        - volatility: Series with MultiIndex (timestamp, asset) for scaling predictions
    """
    if H % base_interval_hours != 0:
        raise ValueError(
            f"H={H}h is not a multiple of the base interval "
            f"({base_interval_hours}h)."
        )

    steps = H // base_interval_hours
    print(f"Computing {H}-hour forward returns over {steps} base steps...")

    # 1. Compute volatility for each asset
    vol_span_bars = _hours_to_bars(vol_span, base_interval_hours)
    volatility = returns.ewm(
        span=vol_span_bars,
        min_periods=max(1, vol_span_bars // 2)
    ).std()
    volatility = volatility.replace(0, np.nan)

    # 2. Compute H-period forward returns (overlapping)
    # For each timestamp t, forward return = sum of returns from t+1 to t+H
    forward_returns = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    if steps <= 0:
        raise ValueError("Number of forward steps must be positive")

    for i in range(len(returns) - steps):
        # Sum returns from i+1 to i+H (next H periods)
        forward_returns.iloc[i] = returns.iloc[i+1:i+steps+1].sum().values

    # 3. Vol-normalize forward returns
    forward_returns_norm = forward_returns / volatility

    # 4. Convert to long format
    forward_long = forward_returns_norm.stack()
    forward_long.index.names = ['timestamp', 'asset']

    vol_long = volatility.stack()
    vol_long.index.names = ['timestamp', 'asset']

    # Drop NaN
    forward_long = forward_long.dropna()
    vol_long = vol_long.dropna()

    print(f"Forward returns shape: {forward_long.shape}")
    print(f"Non-null values: {forward_long.notna().sum()}")

    return forward_long, vol_long


def prepare_alpha_training_data(
    idio_returns: pd.DataFrame,
    funding_rates: pd.DataFrame,
    momentum_panel: pd.DataFrame,  # From idiosyncratic_momentum()
    H: int = 24,
    spans: List[int] = [2, 4, 8, 16, 32, 64],
    vol_span: int = 128,
    base_interval_hours: int = BASE_INTERVAL_HOURS,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare complete feature matrix X and target vector y for alpha training.

    This is the main entry point that combines:
    - Momentum features (from pre-computed signals)
    - Funding features (computed here)
    - Forward returns target (computed here)

    Args:
        idio_returns: Idiosyncratic returns (wide format)
        funding_rates: Funding rates (wide format)
        momentum_panel: Pre-computed momentum signals from idiosyncratic_momentum()
                       MultiIndex (timestamp, asset, span) with 'momentum' column
        H: Holding period for forward returns
        spans: Spans for funding features (should match momentum spans)
        vol_span: Volatility EWMA span

    Returns:
        Tuple of (X, y, vol):
        - X: Feature matrix with columns [mom_2, mom_4, ..., funding_2, funding_4, ...]
        - y: Target vector (vol-normalized H-period forward returns)
        - vol: Volatility estimates for scaling predictions
    """
    print(f"\n{'='*70}")
    print(f"PREPARING ALPHA TRAINING DATA (H={H})")
    print(f"{'='*70}\n")

    # 1. Compute funding features
    funding_features = compute_funding_features(
        funding_rates,
        spans,
        vol_span=vol_span,
        base_interval_hours=base_interval_hours,
    )

    # 2. Compute forward returns target
    forward_returns, volatility = compute_forward_returns(
        idio_returns,
        H,
        vol_span,
        base_interval_hours=base_interval_hours,
    )

    # 3. Reshape momentum panel to match funding features format
    #    momentum_panel has MultiIndex (timestamp, asset, span)
    #    We need MultiIndex (timestamp, asset) with columns [mom_2, mom_4, ...]
    print("Reshaping momentum panel...")
    momentum_features = momentum_panel.unstack(level='span')['momentum']
    momentum_features.columns = [f'mom_{int(s)}' for s in momentum_features.columns]
    momentum_features = momentum_features.sort_index()

    print(f"Momentum features shape: {momentum_features.shape}")
    print(f"Momentum features: {list(momentum_features.columns)}")

    # 4. Align all data on common index
    print("\nAligning features and target...")

    # Get common timestamps and assets
    common_index = (
        momentum_features.index
        .intersection(funding_features.index)
        .intersection(forward_returns.index)
    )

    print(f"Common index size: {len(common_index)}")
    print(f"Date range: {common_index.get_level_values('timestamp').min()} to {common_index.get_level_values('timestamp').max()}")

    # Align
    momentum_aligned = momentum_features.loc[common_index]
    funding_aligned = funding_features.loc[common_index]
    y_aligned = forward_returns.loc[common_index]
    vol_aligned = volatility.loc[common_index]

    # 5. Concatenate momentum and funding features
    X = pd.concat([momentum_aligned, funding_aligned], axis=1)

    # Drop any remaining NaN rows
    valid_mask = X.notna().all(axis=1) & y_aligned.notna()
    X = X[valid_mask]
    y = y_aligned[valid_mask]
    vol = vol_aligned[valid_mask]

    print(f"\n{'='*70}")
    print(f"FINAL DATA SUMMARY")
    print(f"{'='*70}")
    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    print(f"Date range: {X.index.get_level_values('timestamp').min()} to {X.index.get_level_values('timestamp').max()}")
    print(f"Unique assets: {X.index.get_level_values('asset').nunique()}")
    print(f"Assets: {sorted(X.index.get_level_values('asset').unique())}")
    print(f"{'='*70}\n")

    return X, y, vol
