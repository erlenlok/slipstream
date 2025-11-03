#!/usr/bin/env python3
"""
Backfill signal history and tracking error metrics from historical market data.

This script reconstructs the forecast signals that would have been produced at
each rebalance timestamp and compares them with realized 4h returns to populate
the new signal_history.jsonl and signal_performance.jsonl logs.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Dict, List, Tuple

import pandas as pd

from slipstream.gradient.live.config import load_config
from slipstream.gradient.live.data import compute_live_signals
from slipstream.gradient.live.performance import (
    PerformanceTracker,
    compute_signal_tracking_metrics,
)
from slipstream.gradient.sensitivity import compute_log_returns


DATA_DIR = Path("data/market_data")
REBALANCE_LOG = Path("/var/log/gradient/rebalance_history.jsonl")


def load_panel() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for csv_path in sorted(DATA_DIR.glob("*_candles_4h.csv")):
        asset = csv_path.name.replace("_candles_4h.csv", "")
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        if df.empty:
            continue
        df = df.rename(
            columns={
                "datetime": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        df["asset"] = asset
        frames.append(df[["timestamp", "asset", "open", "high", "low", "close", "volume"]])

    if not frames:
        raise RuntimeError(f"No candle files found under {DATA_DIR}")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["asset", "timestamp"])
    return panel


def load_rebalance_times() -> List[pd.Timestamp]:
    if not REBALANCE_LOG.exists():
        raise RuntimeError(f"Rebalance log not found at {REBALANCE_LOG}")

    times: List[pd.Timestamp] = []
    with open(REBALANCE_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ts = pd.Timestamp(record["timestamp"]).tz_localize(None)
            times.append(ts)

    times.sort()
    return times


def main() -> None:
    os.environ.setdefault("HYPERLIQUID_API_KEY", "0x0000000000000000000000000000000000000000")
    os.environ.setdefault("HYPERLIQUID_API_SECRET", "0x0000000000000000000000000000000000000000000000000000000000000000")
    config = load_config()
    tracker = PerformanceTracker(log_dir=config.log_dir)

    # Remove existing backfill artifacts so we start fresh.
    tracker.signals_log.unlink(missing_ok=True)
    tracker.signal_performance_log.unlink(missing_ok=True)

    panel = load_panel()
    panel_returns = compute_log_returns(panel.copy())

    rebalance_times = load_rebalance_times()
    if not rebalance_times:
        raise RuntimeError("No rebalance timestamps available for backfill")

    assets_all = sorted(panel["asset"].unique())

    previous_signals: List[Dict[str, float]] | None = None
    previous_signal_ts: pd.Timestamp | None = None

    processed_signal_times: set[pd.Timestamp] = set()

    for rebalance_time in rebalance_times:
        signal_time = rebalance_time.floor("4h")

        # Skip duplicate signal timestamps (multiple rebalances inside the same 4h window)
        if signal_time in processed_signal_times:
            continue

        subset = panel[panel["timestamp"] <= signal_time].copy()
        if subset.empty:
            continue

        market_data = {"panel": subset, "assets": assets_all}
        signals = compute_live_signals(market_data, config)

        if signals.empty:
            continue

        signal_timestamp_series = signals["signal_timestamp"]
        if signal_timestamp_series.empty:
            continue
        latest_signal_time = pd.to_datetime(signal_timestamp_series.iloc[0]).tz_localize(None)
        if latest_signal_time != signal_time:
            # Insufficient data to produce a fresh forecast for this window.
            continue

        tracker.log_signals(signal_time.to_pydatetime(), signals)

        if previous_signals is not None and previous_signal_ts is not None:
            realized_slice = panel_returns[panel_returns["timestamp"] == signal_time]
            realized_map = (
                realized_slice.set_index("asset")["log_return"].to_dict()
                if not realized_slice.empty
                else {}
            )

            asset_records: List[Dict[str, float]] = []
            for entry in previous_signals:
                asset = entry.get("asset")
                if asset is None:
                    continue
                forecast = entry.get("momentum_score")
                realized = realized_map.get(asset)
                if forecast is None or realized is None:
                    continue

                try:
                    forecast_val = float(forecast)
                    realized_val = float(realized)
                except (TypeError, ValueError):
                    continue

                if pd.isna(forecast_val) or pd.isna(realized_val):
                    continue

                asset_records.append(
                    {
                        "asset": asset,
                        "forecast": forecast_val,
                        "realized": realized_val,
                        "abs_error": abs(realized_val - forecast_val),
                        "squared_error": (realized_val - forecast_val) ** 2,
                    }
                )

            if asset_records:
                metrics = compute_signal_tracking_metrics(asset_records)
                tracker.log_signal_performance(
                    evaluation_timestamp=signal_time.to_pydatetime(),
                    forecast_timestamp=previous_signal_ts.to_pydatetime(),
                    metrics=metrics,
                    asset_records=asset_records,
                )

        previous_signals = signals[
            ["asset", "momentum_score"]
        ].to_dict(orient="records")
        previous_signal_ts = signal_time
        processed_signal_times.add(signal_time)

    print("Backfill complete.")


if __name__ == "__main__":
    main()
