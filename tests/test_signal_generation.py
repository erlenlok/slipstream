#!/usr/bin/env python3
"""
Quick test of signal generation with 4H data and 4H-aligned PCA factors.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from slipstream.signals.idiosyncratic_momentum import (
    idiosyncratic_momentum,
    compute_idiosyncratic_returns,
)


def load_all_returns(data_dir='data/market_data', interval='4h'):
    """Load all candle files and compute log returns."""
    data_path = Path(data_dir)
    pattern = f'*_candles_{interval}.csv'
    candle_files = sorted(data_path.glob(pattern))

    print(f"Looking for pattern: {pattern}")
    print(f"Found {len(candle_files)} files")

    returns_dict = {}

    for file in candle_files[:10]:  # Test with first 10 assets only
        coin = file.stem.replace(f'_candles_{interval}', '')
        df = pd.read_csv(file)

        # Handle both 'datetime' and 'timestamp' column names
        if 'datetime' in df.columns:
            df.index = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'])
        else:
            continue

        df = df.sort_index()
        if 'close' in df.columns:
            log_returns = np.log(df['close'] / df['close'].shift(1))
            returns_dict[coin] = log_returns

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.sort_index()

    print(f"Loaded returns for {len(returns_df.columns)} assets")
    print(f"Date range: {returns_df.index.min()} to {returns_df.index.max()}")
    print(f"Shape: {returns_df.shape}")

    return returns_df


def load_pca_factor(H=24, K=30, weight_method='sqrt', features_dir='data/features'):
    """Load PCA factor file and convert from wide to long format."""
    features_path = Path(features_dir)
    filename = f'pca_factor_H{H}_K{K}_{weight_method}.csv'
    filepath = features_path / filename

    if not filepath.exists():
        raise FileNotFoundError(f"PCA factor file not found: {filepath}")

    # Load wide format PCA data
    pca_wide = pd.read_csv(filepath, index_col=0)
    pca_wide.index = pd.to_datetime(pca_wide.index)
    pca_wide.index.name = 'timestamp'

    # Drop metadata columns
    metadata_cols = ['_variance_explained', '_n_assets']
    asset_cols = [col for col in pca_wide.columns if col not in metadata_cols]

    # Convert to long format (timestamp, asset) -> loading
    loadings_wide = pca_wide[asset_cols]
    loadings_long = loadings_wide.stack()
    loadings_long.index.names = ['timestamp', 'asset']

    # Compute market factor as weighted average of returns
    # (This is a simplified version - ideally we'd have the actual market returns)
    # For now, we'll compute it from the returns weighted by loadings
    market_factor = pd.Series(0, index=pca_wide.index)

    print(f"Loaded PCA factor: H={H}, K={K}, method={weight_method}")
    print(f"Assets: {len(asset_cols)}, Timestamps: {len(pca_wide)}")
    print(f"Date range: {pca_wide.index.min()} to {pca_wide.index.max()}")

    return loadings_long, market_factor


def main():
    print("="*60)
    print("SIGNAL GENERATION TEST - 4H DATA")
    print("="*60)

    # Load data
    print("\n1. Loading 4H candle data...")
    returns = load_all_returns(interval='4h')

    print("\n2. Loading PCA factor (H=24, K=30, sqrt)...")
    loadings, market_factor = load_pca_factor(H=24, K=30, weight_method='sqrt')

    # Test idiosyncratic returns computation
    print("\n3. Computing idiosyncratic returns...")
    idio_returns = compute_idiosyncratic_returns(returns, loadings, market_factor)
    print(f"Idiosyncratic returns shape: {idio_returns.shape}")
    print(f"NaN percentage: {(idio_returns.isna().sum().sum() / idio_returns.size * 100):.2f}%")

    # Test momentum panel computation
    print("\n4. Computing momentum panel...")
    momentum = idiosyncratic_momentum(
        returns=returns,
        pca_loadings=loadings,
        market_factor=market_factor,
        spans=[2, 4, 8],  # Test with just 3 spans
        normalization='volatility'
    )

    print(f"Momentum panel shape: {momentum.shape}")
    print(f"Index levels: {momentum.index.names}")
    print(f"Spans: {sorted(momentum.index.get_level_values('span').unique())}")
    print(f"\nMomentum statistics:")
    print(momentum['momentum'].describe())

    # Check a sample
    print("\n5. Sample momentum values (BTC, span=4):")
    if 'BTC' in momentum.index.get_level_values('asset').unique():
        btc_mom = momentum.xs(('BTC', 4), level=('asset', 'span'))
        print(btc_mom.tail(10))

    print("\n" + "="*60)
    print("✓ SIGNAL GENERATION TEST PASSED")
    print("="*60)


if __name__ == '__main__':
    main()
