#!/usr/bin/env python3
# slipstream data loading utilities
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import numpy as np
import pandas as pd


API_URL = "https://api.hyperliquid.xyz/info"


# ---------- time helpers ----------
def ms(dt: datetime) -> int:
    """Convert a datetime to UTC milliseconds since epoch."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def to_hour(dt_ms: int) -> datetime:
    """Floor a ms timestamp to the start of its hour (UTC)."""
    dt = datetime.fromtimestamp(dt_ms / 1000, tz=timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


# ---------- http ----------
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
REQUEST_PAUSE_SECONDS = 1.0  # Doubled from 0.5 to 1.0 to be twice as conservative


async def _post_json(client: httpx.AsyncClient, payload: dict[str, Any], *, max_attempts: int = 8) -> Any:
    """POST helper with gentle retry & throttling for rate limits."""
    delay = 2.0  # Start with 2 seconds delay for retries (doubled from 1.0)
    for attempt in range(max_attempts):
        try:
            response = await client.post(API_URL, json=payload)
            response.raise_for_status()
            await asyncio.sleep(REQUEST_PAUSE_SECONDS)
            return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in RETRYABLE_STATUS and attempt + 1 < max_attempts:
                print(f"  Retrying after {status} error (attempt {attempt + 1}/{max_attempts}), waiting {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay *= 2.5  # Increased from 2 to 2.5 for more conservative backoff
                continue
            raise
        except httpx.RequestError:
            if attempt + 1 < max_attempts:
                print(f"  Retrying after request error (attempt {attempt + 1}/{max_attempts}), waiting {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay *= 2.5  # Increased from 2 to 2.5 for more conservative backoff
                continue
            raise

    raise RuntimeError("Exhausted retries calling Hyperliquid API")


# ---------- universe ----------
async def fetch_perp_universe(
    client: httpx.AsyncClient,
    dex: str = "",
) -> pd.DataFrame:
    """Return the live perp universe for a given dex namespace ('' for default)."""
    data = await _post_json(client, {"type": "meta", "dex": dex})
    if not isinstance(data, dict) or "universe" not in data:
        raise TypeError(f"Unexpected meta response: {type(data)} {data!r}")

    uni = pd.DataFrame(data["universe"])
    if uni.empty:
        return uni

    if "isDelisted" in uni:
        is_delisted = (
            uni["isDelisted"]
            .infer_objects(copy=False)
            .fillna(False)
            .astype(bool)
        )
    else:
        is_delisted = pd.Series(False, index=uni.index)
        uni["isDelisted"] = is_delisted

    if "onlyIsolated" in uni:
        uni["onlyIsolated"] = (
            uni["onlyIsolated"]
            .infer_objects(copy=False)
            .fillna(False)
            .astype(bool)
        )
    else:
        uni["onlyIsolated"] = False

    uni = uni.loc[~is_delisted].rename(columns={"name": "coin"})
    uni = uni.reset_index(drop=True)
    return uni  # columns: coin, szDecimals, maxLeverage, onlyIsolated, isDelisted


async def fetch_all_perp_markets(client: httpx.AsyncClient) -> pd.DataFrame:
    """Enumerate all perp dex namespaces and union their live universes."""
    dexs_raw = await _post_json(client, {"type": "perpDexs"})
    if not isinstance(dexs_raw, list):
        raise TypeError(f"Unexpected perpDexs response: {type(dexs_raw)} {dexs_raw!r}")

    dex_names: list[str] = []
    for entry in dexs_raw:
        if isinstance(entry, dict) and entry.get("name"):
            dex_names.append(str(entry["name"]))
    # Always include default dex via empty string
    if "" not in dex_names:
        dex_names = [""] + dex_names

    frames: list[pd.DataFrame] = []
    for dex in dex_names:
        try:
            uni = await fetch_perp_universe(client, dex=dex)
            if not uni.empty:
                uni = uni.copy()
                uni["dex"] = dex or "(default)"
                frames.append(uni)
        except Exception as exc:  # noqa: BLE001 - keep going if one dex fails
            print(f"Warning: failed meta for dex={dex!r}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["coin", "dex"])

    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["dex", "coin"])
        .sort_values(["dex", "coin"])
        .reset_index(drop=True)
    )
    return out


# ---------- market data ----------
async def fetch_candles(
    client: httpx.AsyncClient,
    coin: str,
    start: datetime,
    end: datetime,
    interval: str = "4h",
) -> pd.DataFrame:
    """Paginate candles in chunks to stay under ~5k-candle caps.

    Args:
        interval: Supported intervals: "1m", "5m", "15m", "1h", "4h", "1d"
    """
    # Calculate maximum span to stay under 5000 candles per request
    if interval == "1m":
        chunk_span = timedelta(minutes=5000)  # 5000 candles * 1 minute = 5000 minutes
    elif interval == "5m":
        chunk_span = timedelta(minutes=5000 * 5)  # 5000 candles * 5 minutes = 25000 minutes
    elif interval == "15m":
        chunk_span = timedelta(minutes=5000 * 15)  # 5000 candles * 15 minutes = 75000 minutes
    elif interval == "1h":
        chunk_span = timedelta(hours=5000)  # 5000 candles * 1 hour = 5000 hours
    elif interval == "4h":
        chunk_span = timedelta(hours=5000 * 4)  # 5000 candles * 4 hours = 20000 hours
    elif interval == "1d":
        chunk_span = timedelta(days=5000)  # 5000 candles * 1 day = 5000 days
    else:
        # Conservative default for any unexpected interval
        chunk_span = timedelta(hours=120)  # 5 days worth of 1h candles as fallback

    rows: list[dict[str, Any]] = []
    cur = start

    while cur < end:
        chunk_end = min(cur + chunk_span, end)

        # Ensure we have at least a small range to avoid errors
        if chunk_end <= cur:
            break

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": ms(cur),
                "endTime": ms(chunk_end),
            },
        }
        data = await _post_json(client, payload)
        if not isinstance(data, list):
            raise TypeError(f"Unexpected candle response: {type(data)} {data!r}")
        rows.extend(data)
        cur = chunk_end

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    out = (
        df.loc[:, ["t", "o", "h", "l", "c", "v"]]
        .rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        .assign(
            open=lambda x: x["open"].astype(float),
            high=lambda x: x["high"].astype(float),
            low=lambda x: x["low"].astype(float),
            close=lambda x: x["close"].astype(float),
            volume=lambda x: x["volume"].astype(float),
            ts=lambda x: x["ts"].astype("int64"),
        )
    )

    # Convert timestamp to appropriate datetime based on the interval
    # For all intervals, we use the exact timestamp from the API
    out["datetime"] = pd.to_datetime(out["ts"], unit='ms', utc=True)

    out = out.drop_duplicates("datetime").set_index("datetime").sort_index()
    return out[["open", "high", "low", "close", "volume"]]


async def fetch_funding_hourly(
    client: httpx.AsyncClient,
    coin: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch hourly funding, paginating with advancing startTime."""
    rows: list[dict[str, Any]] = []
    cur = start
    max_loops = 1_000

    for _ in range(max_loops):
        payload = {"type": "fundingHistory", "coin": coin, "startTime": ms(cur), "endTime": ms(end)}
        data = await _post_json(client, payload)
        if not isinstance(data, list) or not data:
            break

        rows.extend(data)

        last_ts = max(int(x["time"]) for x in data if "time" in x)
        prev_cur = cur
        cur = datetime.fromtimestamp((last_ts + 1) / 1000, tz=timezone.utc)
        if cur >= end or cur == prev_cur:
            break
    else:
        raise RuntimeError(f"Funding fetch exceeded max loops of {max_loops}")

    if not rows:
        return pd.DataFrame(columns=["funding"])

    df = pd.DataFrame(rows)
    out = (
        df.loc[:, ["time", "fundingRate"]]
        .rename(columns={"time": "ts"})
        .assign(
            funding=lambda x: x["fundingRate"].astype(float),
            ts=lambda x: x["ts"].astype("int64"),
        )
    )
    out["datetime"] = out["ts"].map(to_hour)
    out = out.groupby("datetime", as_index=True)["funding"].mean().to_frame().sort_index()
    return out


def compute_log_returns(candles: pd.DataFrame) -> pd.DataFrame:
    """Vectorized log returns from 4-hour close prices."""
    log_returns = np.log1p(candles["close"].pct_change())
    return log_returns.to_frame(name="ret")


def _merge_with_existing(new_data: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """
    Safely merge new data with existing CSV file.

    CRITICAL: This function prevents data loss by:
    1. Reading existing CSV if it exists
    2. Concatenating with new data
    3. Removing duplicates (keeping newer data)
    4. Sorting by datetime index

    This ensures we NEVER truncate historical data when updating.

    Args:
        new_data: New DataFrame to merge
        filepath: Path to existing CSV file

    Returns:
        Merged DataFrame with all historical + new data
    """
    import os

    if not os.path.exists(filepath):
        # No existing file, just return new data
        return new_data

    try:
        # Read existing data
        existing_data = pd.read_csv(filepath, index_col=0, parse_dates=True)

        if existing_data.empty:
            return new_data

        # Concatenate existing + new
        combined = pd.concat([existing_data, new_data])

        # Remove duplicates, keeping the last occurrence (newer data)
        # This handles the case where we're updating recent data
        combined = combined[~combined.index.duplicated(keep='last')]

        # Sort by datetime index
        combined = combined.sort_index()

        # Log the merge
        old_range = f"{existing_data.index.min()} to {existing_data.index.max()}"
        new_range = f"{new_data.index.min()} to {new_data.index.max()}"
        final_range = f"{combined.index.min()} to {combined.index.max()}"
        print(f"    Merged data: {len(existing_data)} existing + {len(new_data)} new = {len(combined)} total rows")
        print(f"      Old range: {old_range}")
        print(f"      New range: {new_range}")
        print(f"      Final range: {final_range}")

        return combined

    except Exception as exc:
        print(f"    WARNING: Failed to read existing file {filepath}: {exc}")
        print(f"    Falling back to new data only (this may cause data loss!)")
        return new_data


async def build_datasets(
    client: httpx.AsyncClient,
    coin: str,
    start: datetime,
    end: datetime,
    out_prefix: str,
    interval: str = "4h",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch candles+funding, compute returns, align, and write CSVs to market_data/."""
    # Run requests sequentially rather than in parallel to be more API-friendly
    # This is twice as conservative as running them concurrently
    candles = await fetch_candles(client, coin, start, end, interval)
    await asyncio.sleep(REQUEST_PAUSE_SECONDS)  # Additional pause between requests
    funding = await fetch_funding_hourly(client, coin, start, end)
    rets = compute_log_returns(candles)

    # For intervals other than 1h and 4h, we need to decide how to align funding
    # Since funding rates are typically hourly, we'll resample to match the candle interval
    if interval in ["1m", "5m", "15m"]:
        # For sub-hourly intervals, we'll keep the funding data as hourly
        # because funding rates don't change within the hour
        candles_resampled = candles  # Don't resample candles for sub-hourly
        rets_resampled = compute_log_returns(candles_resampled)
        # Resample funding to hourly since that's how rates are provided
        funding_resampled = funding.resample('1h', origin='epoch', offset='0h').mean().dropna(how='all')
    else:
        # For hourly and longer intervals, align both to the candle interval
        candles_resampled = candles.resample(interval, origin='epoch', offset='0h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(how='all')

        # Recompute returns from resampled candles
        rets_resampled = compute_log_returns(candles_resampled)

        # Resample funding (average) to match the candle interval
        funding_resampled = funding.resample(interval, origin='epoch', offset='0h').mean().dropna(how='all')

    # Use the resampled data's index for alignment (already grid-aligned)
    # This ensures merged data matches candles/funding timestamps
    common_index = rets_resampled.index.union(funding_resampled.index).sort_values()
    aligned = pd.DataFrame(index=common_index).join(rets_resampled, how="left").join(funding_resampled, how="left")

    # CRITICAL: Merge with existing data to prevent truncation
    # This ensures we NEVER lose historical data when updating
    import os
    os.makedirs(f"{out_prefix}/market_data/{interval}", exist_ok=True)

    candles_path = f"{out_prefix}/market_data/{interval}/{coin}_candles_{interval}.csv"
    funding_path = f"{out_prefix}/market_data/{interval}/{coin}_funding_{interval}.csv"
    merged_path = f"{out_prefix}/market_data/{interval}/{coin}_merged_{interval}.csv"

    print(f"  Saving {coin} (interval: {interval}):")
    candles_final = _merge_with_existing(candles_resampled, candles_path)
    funding_final = _merge_with_existing(funding_resampled, funding_path)
    aligned_final = _merge_with_existing(aligned, merged_path)

    # Write merged data
    candles_final.to_csv(candles_path, index=True)
    funding_final.to_csv(funding_path, index=True)
    aligned_final.to_csv(merged_path, index=True)

    return candles_final, funding_final, aligned_final


async def build_for_universe(
    start: datetime,
    end: datetime,
    out_prefix: str,
    interval: str = "4h",
    dex_filter: str | None = None,
) -> None:
    """Fetch & write datasets for all live markets (optionally filter by dex name)."""
    # CRITICAL PRE-FLIGHT CHECK: Warn if fetch window might not cover existing data
    import os
    sample_files = [
        f"{out_prefix}/market_data/{interval}/BTC_candles_{interval}.csv",
        f"{out_prefix}/market_data/{interval}/ETH_candles_{interval}.csv",
    ]

    for sample_file in sample_files:
        if os.path.exists(sample_file):
            try:
                sample_df = pd.read_csv(sample_file, index_col=0, parse_dates=True)
                if not sample_df.empty:
                    existing_start = sample_df.index.min()
                    existing_end = sample_df.index.max()
                    fetch_days = (end - start).days
                    existing_days = (existing_end - existing_start).days

                    print(f"\n{'='*80}")
                    print(f"PRE-FLIGHT CHECK: {os.path.basename(sample_file)}")
                    print(f"  Existing data: {existing_start.date()} to {existing_end.date()} ({existing_days} days)")
                    print(f"  Fetch request:  {start.date()} to {end.date()} ({fetch_days} days)")

                    if start > existing_start:
                        days_at_risk = (start - existing_start).days
                        print(f"  ⚠️  WARNING: Fetch starts {days_at_risk} days AFTER existing data!")
                        print(f"  ✓  SAFE: Merge function will preserve historical data")
                    else:
                        print(f"  ✓  SAFE: Fetch window covers all existing data")
                    print(f"{'='*80}\n")
                    break
            except Exception:
                pass

    async with httpx.AsyncClient(timeout=30.0) as client:
        markets = await fetch_all_perp_markets(client)
        if dex_filter:
            markets = markets.loc[markets["dex"] == dex_filter]

        if markets.empty:
            print("No markets found.")
            return

        print(f"Found {len(markets)} markets.")
        for _, row in markets.iterrows():
            coin = str(row["coin"])
            dex = str(row["dex"])
            # Remove dex prefix from filename since we're organizing by directory
            print(f"- {dex}: {coin}")
            try:
                await build_datasets(client, coin=coin, start=start, end=end, out_prefix=out_prefix, interval=interval)
                await asyncio.sleep(REQUEST_PAUSE_SECONDS * 2)  # Double the pause between assets for more conservatism
            except Exception as exc:  # noqa: BLE001
                print(f"  Warning: failed for {coin} ({dex}): {exc}")


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Hyperliquid perp returns & funding.")
    p.add_argument("--coin", type=str, default="", help="Single market symbol (e.g., BTC). If empty and --all not set, prints markets.")
    p.add_argument("--all", action="store_true", help="Fetch for all live perp markets.")
    p.add_argument("--dex", type=str, default=None, help="Filter to a specific dex namespace (use with --all).")
    p.add_argument("--interval", type=str, default="4h", choices=["1m", "5m", "15m", "1h", "4h", "1d"], help="Candle interval: 1m, 5m, 15m, 1h, 4h, 1d (default: 4h).")
    p.add_argument("--days", type=int, default=180, help="Lookback window in days (default: 180).")
    p.add_argument("--start", type=str, default=None, help="ISO start (UTC). Overrides --days if set.")
    p.add_argument("--end", type=str, default=None, help="ISO end (UTC). Defaults to now.")
    p.add_argument("--out-prefix", type=str, default="data", help="Output directory (default: data).")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    end_dt = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    )
    if args.start:
        start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    else:
        start_dt = end_dt - timedelta(days=int(args.days))

    async with httpx.AsyncClient(timeout=30.0) as client:
        # No args: print markets
        if not args.coin and not args.all:
            markets = await fetch_all_perp_markets(client)
            if markets.empty:
                print("No markets found.")
                return
            print(f"Total live perp markets: {len(markets)}")
            for dex, grp in markets.groupby("dex", dropna=False):
                print(f"\nDEX: {dex or '(default)'}  ({len(grp)} markets)")
                print(", ".join(grp["coin"].tolist()))
            return

        # All markets
        if args.all:
            await build_for_universe(start=start_dt, end=end_dt, out_prefix=args.out_prefix, interval=args.interval, dex_filter=args.dex)
            return

        # Single market
        coin = args.coin.upper()
        print(f"Fetching {coin} ({args.interval}) from {start_dt.isoformat()} to {end_dt.isoformat()} …")
        await build_datasets(client, coin=coin, start=start_dt, end=end_dt, out_prefix=args.out_prefix, interval=args.interval)
        print("Done.")


def app() -> None:
    """Command line entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    app()
