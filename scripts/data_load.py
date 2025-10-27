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
REQUEST_PAUSE_SECONDS = 0.5  # Doubled from 0.25 to be more API-friendly


async def _post_json(client: httpx.AsyncClient, payload: dict[str, Any], *, max_attempts: int = 5) -> Any:
    """POST helper with gentle retry & throttling for rate limits."""
    delay = 1.0  # Start with 1 second delay for retries (doubled from 0.5)
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
                delay *= 2
                continue
            raise
        except httpx.RequestError:
            if attempt + 1 < max_attempts:
                print(f"  Retrying after request error (attempt {attempt + 1}/{max_attempts}), waiting {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay *= 2
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
        interval: "1h" or "4h" (API supports 1m, 5m, 15m, 1h, 4h, 1d)
    """
    # Adjust chunk size based on interval to stay under 5000 candle limit
    if interval == "4h":
        chunk_days = 500  # 500*24/4=3000 < 5000
    elif interval == "1h":
        chunk_days = 120  # 120*24=2880 < 5000
    else:
        chunk_days = 120  # Conservative default

    rows: list[dict[str, Any]] = []
    cur = start

    while cur < end:
        chunk_end = min(cur + timedelta(days=chunk_days), end)
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
    out["datetime"] = out["ts"].map(to_hour)
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


async def build_datasets(
    client: httpx.AsyncClient,
    coin: str,
    start: datetime,
    end: datetime,
    out_prefix: str,
    interval: str = "4h",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch candles+funding, compute returns, align, and write CSVs to market_data/."""
    candles, funding = await asyncio.gather(
        fetch_candles(client, coin, start, end, interval),
        fetch_funding_hourly(client, coin, start, end),
    )
    rets = compute_log_returns(candles)

    # Resample both to common grid aligned to epoch (UTC midnight)
    # This ensures candles and funding are on the same timestamps
    freq = interval if interval in ["1h", "4h"] else "4h"

    # Resample candles (use last for OHLCV)
    candles_resampled = candles.resample(freq, origin='epoch', offset='0h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(how='all')

    # Recompute returns from resampled candles
    rets_resampled = compute_log_returns(candles_resampled)

    # Resample funding (average)
    funding_resampled = funding.resample(freq, origin='epoch', offset='0h').mean().dropna(how='all')

    # Use the resampled data's index for alignment (already grid-aligned)
    # This ensures merged data matches candles/funding timestamps
    common_index = rets_resampled.index.union(funding_resampled.index).sort_values()
    aligned = pd.DataFrame(index=common_index).join(rets_resampled, how="left").join(funding_resampled, how="left")

    # Save resampled data
    candles_resampled.to_csv(f"{out_prefix}/market_data/{coin}_candles_{freq}.csv", index=True)
    funding_resampled.to_csv(f"{out_prefix}/market_data/{coin}_funding_{freq}.csv", index=True)
    aligned.to_csv(f"{out_prefix}/market_data/{coin}_merged_{freq}.csv", index=True)

    return candles_resampled, funding_resampled, aligned


async def build_for_universe(
    start: datetime,
    end: datetime,
    out_prefix: str,
    interval: str = "4h",
    dex_filter: str | None = None,
) -> None:
    """Fetch & write datasets for all live markets (optionally filter by dex name)."""
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
                await asyncio.sleep(REQUEST_PAUSE_SECONDS)
            except Exception as exc:  # noqa: BLE001
                print(f"  Warning: failed for {coin} ({dex}): {exc}")


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Hyperliquid perp returns & funding.")
    p.add_argument("--coin", type=str, default="", help="Single market symbol (e.g., BTC). If empty and --all not set, prints markets.")
    p.add_argument("--all", action="store_true", help="Fetch for all live perp markets.")
    p.add_argument("--dex", type=str, default=None, help="Filter to a specific dex namespace (use with --all).")
    p.add_argument("--interval", type=str, default="4h", choices=["1h", "4h"], help="Candle interval: 4h (default) or 1h.")
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
