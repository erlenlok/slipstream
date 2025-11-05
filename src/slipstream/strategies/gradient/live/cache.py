"""Redis caching for market data to minimize API calls."""

import json
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
import redis


def get_redis_client() -> redis.Redis:
    """Get Redis client connection."""
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))

    return redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=True,
    )


def get_cached_candles(asset: str, client: redis.Redis) -> Optional[pd.DataFrame]:
    """
    Retrieve cached candles for an asset from Redis.

    Args:
        asset: Asset symbol
        client: Redis client

    Returns:
        DataFrame with cached candles, or None if not cached
    """
    key = f"gradient:candles:{asset}"
    data = client.get(key)

    if not data:
        return None

    try:
        records = json.loads(data)
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Warning: Failed to load cache for {asset}: {e}")
        return None


def set_cached_candles(asset: str, df: pd.DataFrame, client: redis.Redis, ttl_hours: int = 168):
    """
    Store candles for an asset in Redis.

    Args:
        asset: Asset symbol
        df: DataFrame with candles
        client: Redis client
        ttl_hours: Time to live in hours (default 7 days)
    """
    key = f"gradient:candles:{asset}"

    try:
        # Convert to JSON-serializable format
        df_copy = df.copy()
        df_copy['timestamp'] = df_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        records = df_copy.to_dict('records')
        data = json.dumps(records)

        # Set with TTL
        client.setex(key, ttl_hours * 3600, data)
    except Exception as e:
        print(f"Warning: Failed to cache {asset}: {e}")


def get_last_candle_time(asset: str, client: redis.Redis) -> Optional[datetime]:
    """
    Get the timestamp of the last cached candle for an asset.

    Args:
        asset: Asset symbol
        client: Redis client

    Returns:
        Timestamp of last candle, or None if not cached
    """
    df = get_cached_candles(asset, client)
    if df is None or df.empty:
        return None

    return df['timestamp'].max()


def merge_cached_and_new(cached_df: Optional[pd.DataFrame], new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cached candles with newly fetched candles, removing duplicates.

    Args:
        cached_df: Cached candles (may be None)
        new_df: Newly fetched candles

    Returns:
        Combined DataFrame with duplicates removed
    """
    if cached_df is None or cached_df.empty:
        return new_df

    if new_df.empty:
        return cached_df

    # Combine and remove duplicates based on timestamp
    combined = pd.concat([cached_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    # Keep only most recent 1200 candles to prevent unbounded growth
    if len(combined) > 1200:
        combined = combined.tail(1200).reset_index(drop=True)

    return combined


def cache_enabled() -> bool:
    """Check if Redis caching is enabled via environment variable."""
    return os.getenv("REDIS_ENABLED", "true").lower() in ("true", "1", "yes")
