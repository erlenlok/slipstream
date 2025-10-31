"""Data fetching and signal generation for live trading."""

import pandas as pd
import numpy as np
from typing import Dict, Any

# TODO: Import or implement these functions from sensitivity.py
# from slipstream.gradient.sensitivity import (
#     compute_vol_normalized_returns,
#     compute_multispan_momentum,
#     compute_adv_usd,
#     filter_universe_by_liquidity,
# )


def fetch_live_data(config) -> Dict[str, Any]:
    """
    Fetch latest 4h candles for all perpetual markets from Hyperliquid API.

    Args:
        config: GradientConfig instance

    Returns:
        Dictionary containing:
            - assets: List of asset symbols
            - candles: DataFrame with OHLCV data (timestamp, asset, open, high, low, close, volume)
            - timestamps: Latest timestamps per asset

    TODO: Implement using Hyperliquid API
    1. Fetch list of all perpetual markets (GET /info with type=meta)
    2. For each market, fetch recent 4h candles (need ~1024 periods for longest lookback)
    3. Combine into panel DataFrame
    4. Handle API rate limits and retries
    """
    raise NotImplementedError(
        "fetch_live_data() not yet implemented. "
        "See scripts/data_load.py for reference implementation."
    )

    # Example structure:
    # return {
    #     "assets": ["BTC", "ETH", ...],
    #     "candles": pd.DataFrame({
    #         "timestamp": [...],
    #         "asset": [...],
    #         "open": [...],
    #         "high": [...],
    #         "low": [...],
    #         "close": [...],
    #         "volume": [...]
    #     })
    # }


def compute_live_signals(market_data: Dict[str, Any], config) -> pd.DataFrame:
    """
    Compute momentum signals from market data.

    Args:
        market_data: Output from fetch_live_data()
        config: GradientConfig instance

    Returns:
        DataFrame with columns: asset, momentum_score, vol_24h, adv_usd, include_in_universe
        Sorted by momentum_score descending

    TODO: Implement signal generation pipeline
    1. Compute log returns from close prices
    2. Calculate ADV in USD (average daily volume)
    3. Filter universe by liquidity (10k < 2.5% of ADV)
    4. Compute EWMA volatility (24h span)
    5. Calculate vol-normalized returns
    6. Compute multi-span EWMA momentum (sum across all lookback periods)
    7. Return signals for latest timestamp only

    Reuse logic from src/slipstream/gradient/sensitivity.py:
    - compute_log_returns()
    - compute_adv_usd()
    - filter_universe_by_liquidity()
    - compute_vol_normalized_returns()
    - compute_multispan_momentum()
    """
    raise NotImplementedError(
        "compute_live_signals() not yet implemented. "
        "Reuse functions from slipstream.gradient.sensitivity module."
    )

    # Example structure:
    # signals = pd.DataFrame({
    #     "asset": ["BTC", "ETH", ...],
    #     "momentum_score": [2.5, -1.3, ...],
    #     "vol_24h": [0.02, 0.03, ...],
    #     "adv_usd": [1e9, 5e8, ...],
    #     "include_in_universe": [True, True, ...],
    # })
    # return signals.sort_values("momentum_score", ascending=False)


def validate_market_data(market_data: Dict[str, Any]) -> None:
    """
    Validate fetched market data.

    Args:
        market_data: Market data dictionary

    Raises:
        ValueError: If data is invalid or insufficient
    """
    if "candles" not in market_data:
        raise ValueError("Market data missing 'candles' key")

    candles = market_data["candles"]

    if len(candles) == 0:
        raise ValueError("No candle data fetched")

    required_cols = ["timestamp", "asset", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in candles.columns]
    if missing_cols:
        raise ValueError(f"Market data missing columns: {missing_cols}")

    # Check for sufficient history (need 1024 periods for longest lookback)
    min_periods = 1024
    periods_per_asset = candles.groupby("asset").size()
    insufficient = periods_per_asset[periods_per_asset < min_periods]

    if len(insufficient) > 0:
        print(f"Warning: {len(insufficient)} assets have < {min_periods} periods of history")


def validate_signals(signals: pd.DataFrame, config) -> None:
    """
    Validate computed signals.

    Args:
        signals: Signal DataFrame
        config: GradientConfig instance

    Raises:
        ValueError: If signals are invalid
    """
    required_cols = ["asset", "momentum_score", "vol_24h", "adv_usd", "include_in_universe"]
    missing_cols = [col for col in required_cols if col not in signals.columns]
    if missing_cols:
        raise ValueError(f"Signals missing columns: {missing_cols}")

    if len(signals) == 0:
        raise ValueError("No signals generated")

    # Check for NaN values in critical columns
    if signals["momentum_score"].isna().any():
        raise ValueError("NaN values in momentum_score")

    if signals["vol_24h"].isna().any():
        raise ValueError("NaN values in vol_24h")

    # Check liquidity filter was applied
    liquid_assets = signals["include_in_universe"].sum()
    print(f"Universe: {liquid_assets}/{len(signals)} assets pass liquidity filter")

    if liquid_assets == 0:
        raise ValueError("No assets pass liquidity filter")
