"""Data fetching and signal computation for live Gradient trading."""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from slipstream.gradient.signals import compute_trend_strength
from slipstream.gradient.sensitivity import (
    compute_adv_usd,
    compute_log_returns,
    compute_vol_normalized_returns,
    filter_universe_by_liquidity,
)
from slipstream.gradient.live import cache as cache_module

# Tunable request controls to play nicely with Hyperliquid's rate limits.
# Conservative settings to avoid bans - reduced from 10 to 5 concurrent requests
MAX_CONCURRENT_REQUESTS = 5
MAX_REQUEST_ATTEMPTS = 6
INITIAL_BACKOFF_SECONDS = 2.0  # Increased from 1.0 to be more conservative
BACKOFF_FACTOR = 2.0
BACKOFF_JITTER = 0.3  # Add ±30% jitter to prevent thundering herd


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


async def _post_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    *,
    max_attempts: int = MAX_REQUEST_ATTEMPTS,
    initial_backoff: float = INITIAL_BACKOFF_SECONDS,
    backoff_factor: float = BACKOFF_FACTOR,
    jitter: float = BACKOFF_JITTER,
) -> httpx.Response:
    """POST with exponential backoff + jitter on transport, 429, and 5xx responses."""
    delay = initial_backoff

    def add_jitter(base_delay: float) -> float:
        """Add random jitter to delay to prevent thundering herd."""
        jitter_range = base_delay * jitter
        return base_delay + random.uniform(-jitter_range, jitter_range)

    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(url, json=payload)
        except httpx.HTTPError:
            if attempt == max_attempts:
                raise
            await asyncio.sleep(add_jitter(delay))
            delay *= backoff_factor
            continue

        if response.status_code == 429:
            # Respect server-provided retry hints when possible.
            wait_time = delay
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    wait_time = max(wait_time, float(retry_after))
                except ValueError:
                    pass

            if attempt == max_attempts:
                response.raise_for_status()

            await asyncio.sleep(add_jitter(wait_time))
            delay *= backoff_factor
            continue

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if attempt == max_attempts or not (500 <= status < 600):
                raise
            await asyncio.sleep(add_jitter(delay))
            delay *= backoff_factor
            continue

        return response

    # This should be unreachable because we either returned or raised.
    raise RuntimeError("Failed to obtain successful response after retries")


async def fetch_candles_for_asset(
    asset: str,
    endpoint: str,
    n_candles: int = 1100,
    interval: str = "4h",
    *,
    client: Optional[httpx.AsyncClient] = None,
    max_attempts: int = MAX_REQUEST_ATTEMPTS,
    initial_backoff: float = INITIAL_BACKOFF_SECONDS,
    backoff_factor: float = BACKOFF_FACTOR,
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
    async def _run(active_client: httpx.AsyncClient) -> pd.DataFrame:
        # Calculate start time for n_candles
        end_time = int(datetime.now().timestamp() * 1000)
        interval_ms = 4 * 60 * 60 * 1000  # 4 hours in ms
        start_time = end_time - (n_candles * interval_ms)

        response = await _post_with_backoff(
            active_client,
            f"{endpoint}/info",
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": asset,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time
                }
            },
            max_attempts=max_attempts,
            initial_backoff=initial_backoff,
            backoff_factor=backoff_factor,
        )
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

    if client is None:
        async with httpx.AsyncClient(timeout=30.0) as owned_client:
            return await _run(owned_client)

    return await _run(client)


async def fetch_candles_with_cache(
    asset: str,
    endpoint: str,
    redis_client: Optional[Any],
    http_client: httpx.AsyncClient,
) -> pd.DataFrame:
    """
    Fetch candles for an asset, using cache when available.

    Strategy:
    1. If cache disabled or no cache: fetch all 1100 candles
    2. If cached: only fetch new candles since last cached timestamp
    3. Merge cached + new data
    4. Update cache

    Args:
        asset: Asset symbol
        endpoint: API endpoint
        redis_client: Redis client (or None if caching disabled)
        http_client: HTTP client for API requests

    Returns:
        DataFrame with candles
    """
    # If caching disabled, fetch all candles
    if redis_client is None:
        return await fetch_candles_for_asset(
            asset, endpoint, n_candles=1100, client=http_client
        )

    # Check cache
    cached_df = cache_module.get_cached_candles(asset, redis_client)
    last_time = cache_module.get_last_candle_time(asset, redis_client)

    if last_time is None:
        # No cache: fetch all 1100 candles
        new_df = await fetch_candles_for_asset(
            asset, endpoint, n_candles=1100, client=http_client
        )
    else:
        # Have cache: only fetch new candles since last_time
        # Fetch last 50 candles to ensure we catch any new ones
        new_df = await fetch_candles_for_asset(
            asset, endpoint, n_candles=50, client=http_client
        )

    # Merge cached and new
    combined_df = cache_module.merge_cached_and_new(cached_df, new_df)

    # Update cache
    if not combined_df.empty:
        cache_module.set_cached_candles(asset, combined_df, redis_client)

    return combined_df


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

    # Setup Redis client if caching enabled
    redis_client = None
    if cache_module.cache_enabled():
        try:
            redis_client = cache_module.get_redis_client()
            redis_client.ping()  # Test connection
            print("✓ Redis cache enabled")
        except Exception as e:
            print(f"Warning: Redis connection failed, caching disabled: {e}")
            redis_client = None
    else:
        print("Cache disabled (set REDIS_ENABLED=true to enable)")

    # Fetch candles for all assets (with caching if enabled)
    cache_status = "with cache" if redis_client else "without cache"
    print(f"Fetching candles for {len(assets)} assets ({cache_status})...")

    async def fetch_all_candles():
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        async with httpx.AsyncClient(timeout=30.0) as client:
            async def fetch_with_limit(asset: str):
                async with semaphore:
                    return await fetch_candles_with_cache(
                        asset,
                        endpoint,
                        redis_client,
                        client,
                    )

            tasks = [asyncio.create_task(fetch_with_limit(asset)) for asset in assets]
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
    if not valid_dfs:
        raise RuntimeError("Failed to fetch candle data for all assets")

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
        trade_size_usd=config.liquidity_threshold_usd,
        max_impact_pct=config.liquidity_impact_pct,
    )

    print("Computing vol-normalized returns...")
    panel = compute_vol_normalized_returns(panel, vol_span=config.vol_span)

    print(f"Computing multi-span momentum (spans: {config.lookback_spans})...")
    log_returns_wide = (
        panel.pivot_table(
            index="timestamp",
            columns="asset",
            values="log_return",
            aggfunc="first",
        )
        .sort_index()
    )

    trend_strength = compute_trend_strength(
        log_returns_wide,
        lookbacks=config.lookback_spans,
    )

    # Get latest timestamp only
    latest_time = trend_strength.index.max()
    latest_scores = trend_strength.loc[latest_time]

    signals = panel[panel["timestamp"] == latest_time].copy()

    # Select relevant columns
    signals = signals[[
        "asset",
        "vol_24h",
        "adv_usd",
        "include_in_universe"
    ]]

    signals = signals.drop_duplicates(subset="asset").set_index("asset")
    signals["momentum_score"] = latest_scores
    signals = (
        signals
        .dropna(subset=["momentum_score", "vol_24h"])
        .reset_index()[[
            "asset",
            "momentum_score",
            "vol_24h",
            "adv_usd",
            "include_in_universe",
        ]]
    )

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
