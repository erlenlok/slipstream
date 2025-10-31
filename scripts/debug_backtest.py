#!/usr/bin/env python3
"""Debug a single backtest run."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from slipstream.gradient.sensitivity import run_concentration_backtest

# Load panel
panel = pd.read_csv("data/gradient/sensitivity/panel_data.csv")
panel["timestamp"] = pd.to_datetime(panel["timestamp"])

# Load sample periods
samples = pd.read_csv("data/gradient/sensitivity/sample_periods.csv")
samples["start_time"] = pd.to_datetime(samples["start_time"])
samples["end_time"] = pd.to_datetime(samples["end_time"])

# Run one backtest
sample = samples.iloc[0]
print(f"Testing period: {sample['start_time']} to {sample['end_time']}")

# Check data availability
period_data = panel[
    (panel["timestamp"] >= sample["start_time"]) &
    (panel["timestamp"] <= sample["end_time"])
]

print(f"Period data: {len(period_data)} rows")
print(f"Assets in period: {period_data['asset'].nunique()}")
print(f"Timestamps: {period_data['timestamp'].nunique()}")

# Check momentum distribution
print("\nMomentum score distribution:")
print(period_data.groupby("timestamp")["momentum_score"].describe().head())

# Check some example rankings
first_time = period_data["timestamp"].min()
first_data = period_data[period_data["timestamp"] == first_time].sort_values("momentum_score", ascending=False)

print(f"\nTop 5 assets at {first_time}:")
print(first_data.head()[["asset", "momentum_score", "vol_24h"]])

print(f"\nBottom 5 assets at {first_time}:")
print(first_data.tail()[["asset", "momentum_score", "vol_24h"]])

# Run backtest
print("\n" + "="*80)
print("Running backtest...")
result = run_concentration_backtest(
    panel=panel,
    start_time=sample["start_time"],
    end_time=sample["end_time"],
    n_pct=5.0,
    rebalance_freq_hours=24,
    weight_scheme="equal"
)

print(f"Annualized return: {result:.2f}%")
