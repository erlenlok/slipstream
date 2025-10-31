"""Data fetching and signal computation for live Gradient trading."""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import httpx

# Import signal computation functions from sensitivity module
from slipstream.gradient.sensitivity import (
    compute_log_returns,
    compute_ewma_vol,
    compute_adv_usd,
    filter_universe_by_liquidity,
    compute_vol_normalized_returns,
    compute_multispan_momentum,
)


async def fetch_all_perp_markets(endpoint: str) -> List[str]:
    """
    Fetch all perpetual market symbols from Hyperliquid.

    Args:
        endpoint: API endpoint URL

    Returns:
        List of asset symbols (e.g., ["BTC", "ETH", "SOL"])
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{endpoint}/info",
            json={"type": "meta"}
        )
        response.raise_for_status()
        meta = response.json()

        # Extract perpetual markets
        universe = meta.get("universe", [])
        symbols = [asset["name"] for asset in universe if asset.get("szDecimals") is not None]

        return symbols


async def fetch_candles_for_asset(
    asset: str,
    endpoint: str,
    n_candles: int = 1100,
    interval: str = "4h"
) -> pd.DataFrame:
    """
    Fetch historical 4h candles for a single asset.

    Args:
        asset: Asset symbol
        endpoint: API endpoint URL
        n_candles: Number of candles to fetch
        interval: Candle interval (default 4h)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Calculate start time for n_candles
        end_time = int(datetime.now().timestamp() * 1000)
        interval_ms = 4 * 60 * 60 * 1000  # 4 hours in ms
        start_time = end_time - (n_candles * interval_ms)

        response = await client.post(
            f"{endpoint}/info",
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": asset,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time
                }
            }
        )
        response.raise_for_status()
        data = response.json()

        # Parse candles
        if not data:
            return pd.DataFrame()

        candles = []
        for candle in data:
            candles.append({
                "timestamp": pd.to_datetime(candle["t"], unit="ms"),
                "open": float(candle["o"]),
                "high": float(candle["h"]),
                "low": float(candle["l"]),
                "close": float(candle["c"]),
                "volume": float(candle["v"]),
            })

        df = pd.DataFrame(candles)
        df["asset"] = asset

        return df


def fetch_live_data(config) -> Dict[str, Any]:
    """
    Fetch latest 4h candle data for all perpetual markets.

    Args:
        config: GradientConfig instance

    Returns:
        Dictionary with:
            - panel: DataFrame with columns [timestamp, asset, open, high, low, close, volume]
            - assets: List of asset symbols
    """
    endpoint = config.api_endpoint

    # Get all perp markets
    print(f"Fetching perpetual markets from {endpoint}...")
    assets = asyncio.run(fetch_all_perp_markets(endpoint))
    print(f"Found {len(assets)} perpetual markets")

    # Fetch candles for all assets
    print(f"Fetching 1100 4h candles for {len(assets)} assets...")

    async def fetch_all_candles():
        tasks = [
            fetch_candles_for_asset(asset, endpoint, n_candles=1100, interval="4h")
            for asset in assets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    candle_dfs = asyncio.run(fetch_all_candles())

    # Filter out errors and empty DataFrames
    valid_dfs = []
    failed_assets = []
    for asset, df in zip(assets, candle_dfs):
        if isinstance(df, Exception):
            failed_assets.append(asset)
            print(f"Warning: Failed to fetch {asset}: {df}")
        elif not df.empty:
            valid_dfs.append(df)
        else:
            failed_assets.append(asset)

    if failed_assets:
        print(f"Warning: Failed to fetch data for {len(failed_assets)} assets")

    # Combine into single panel
    panel = pd.concat(valid_dfs, ignore_index=True)
    panel = panel.sort_values(["timestamp", "asset"]).reset_index(drop=True)

    successful_assets = [a for a in assets if a not in failed_assets]

    print(f"Successfully fetched {len(panel)} candles for {len(successful_assets)} assets")

    return {
        "panel": panel,
        "assets": successful_assets
    }


def compute_live_signals(market_data: Dict[str, Any], config) -> pd.DataFrame:
    """
    Compute momentum signals from market data.

    Args:
        market_data: Output from fetch_live_data()
        config: GradientConfig instance

    Returns:
        DataFrame with columns: [asset, momentum_score, vol_24h, adv_usd, include_in_universe]
        Sorted by momentum_score descending (for latest timestamp only)
    """
    panel = market_data["panel"].copy()

    print("Computing log returns...")
    panel = compute_log_returns(panel)

    print("Computing ADV and filtering universe...")
    panel = compute_adv_usd(panel, window=6)  # 6 * 4h = 24h
    panel = filter_universe_by_liquidity(
        panel,
        trade_size_usd=config.liquidity_threshold,
        max_impact_pct=config.liquidity_impact_pct
    )

    print("Computing vol-normalized returns...")
    panel = compute_vol_normalized_returns(panel, vol_span=config.vol_span)

    print(f"Computing multi-span momentum (spans: {config.lookback_spans})...")
    panel = compute_multispan_momentum(panel, lookback_spans=config.lookback_spans)

    # Get latest timestamp only
    latest_time = panel["timestamp"].max()
    signals = panel[panel["timestamp"] == latest_time].copy()

    # Select relevant columns
    signals = signals[[
        "asset",
        "momentum_score",
        "vol_24h",
        "adv_usd",
        "include_in_universe"
    ]]

    # Sort by momentum score descending
    signals = signals.sort_values("momentum_score", ascending=False).reset_index(drop=True)

    # Drop any rows with missing values
    signals = signals.dropna()

    print(f"Computed signals for {len(signals)} assets (latest timestamp: {latest_time})")

    return signals


def validate_market_data(market_data: Dict[str, Any]) -> None:
    """
    Validate fetched market data.

    Args:
        market_data: Market data dictionary

    Raises:
        ValueError: If data is invalid or insufficient
    """
    if "panel" not in market_data:
        raise ValueError("Market data missing 'panel' key")

    panel = market_data["panel"]

    if len(panel) == 0:
        raise ValueError("No candle data fetched")

    required_cols = ["timestamp", "asset", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in panel.columns]
    if missing_cols:
        raise ValueError(f"Market data missing columns: {missing_cols}")

    # Check for sufficient history (need 1024 periods for longest lookback)
    min_periods = 1024
    periods_per_asset = panel.groupby("asset").size()
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
