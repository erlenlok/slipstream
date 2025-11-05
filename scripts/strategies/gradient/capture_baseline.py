#!/usr/bin/env python3
"""Compute baseline metrics for the Gradient strategy backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from slipstream.strategies.gradient.backtest import run_gradient_backtest
from slipstream.strategies.gradient.signals import DEFAULT_LOOKBACKS


def _parse_lookbacks(raw: Optional[Iterable[str]]) -> list[int]:
    if not raw:
        return list(DEFAULT_LOOKBACKS)
    values = [int(v) for v in raw]
    if any(v <= 0 for v in values):
        raise ValueError("Lookbacks must be positive integers")
    return values


def compute_metrics(result) -> dict:
    equity_curve = (1 + result.portfolio_returns).cumprod()
    cumulative_return = result.portfolio_returns.sum()
    annualized = result.annualized_sharpe(periods_per_year=24 * 365)
    drawdown = float("nan")
    if not equity_curve.empty:
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1).min()

    weights = result.weights.fillna(0.0)
    abs_weights = weights.abs()
    avg_gross_exposure = abs_weights.sum(axis=1).mean()
    active_assets = (abs_weights > 0).sum(axis=1).mean()
    turnover = weights.diff().abs().sum(axis=1).mean()

    metrics = {
        "periods": int(len(result.portfolio_returns)),
        "cumulative_return": float(cumulative_return),
        "annualized_sharpe": float(annualized),
        "max_drawdown": float(drawdown) if drawdown == drawdown else None,
        "avg_gross_exposure": float(avg_gross_exposure),
        "avg_active_assets": float(active_assets),
        "avg_turnover": float(turnover),
    }
    return metrics


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returns-csv", required=True, help="Path to wide log returns CSV")
    parser.add_argument("--signals-csv", help="Optional precomputed trend strength CSV")
    parser.add_argument("--output-json", help="Where to store resulting metrics (defaults to stdout)")
    parser.add_argument("--lookbacks", nargs="*", help="Optional list of lookback windows for signal generation")
    parser.add_argument("--top-n", type=int, default=44, help="Number of strongest assets to hold long")
    parser.add_argument("--bottom-n", type=int, help="Number of weakest assets to hold short (defaults to top-n)")
    parser.add_argument("--vol-span", type=int, default=64, help="EWMA span for volatility normalisation")
    parser.add_argument("--target-side-dollar-vol", type=float, default=1.0,
                        help="Target dollar vol allocated to each side of the book")

    args = parser.parse_args(argv)

    lookbacks = _parse_lookbacks(args.lookbacks)
    returns = pd.read_csv(args.returns_csv, index_col=0, parse_dates=True)

    signals = None
    if args.signals_csv:
        signals = pd.read_csv(args.signals_csv, index_col=0, parse_dates=True)
        signals = signals.reindex(index=returns.index, columns=returns.columns)

    result = run_gradient_backtest(
        returns,
        trend_strength=signals,
        lookbacks=lookbacks,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        vol_span=args.vol_span,
        target_side_dollar_vol=args.target_side_dollar_vol,
    )

    metrics = compute_metrics(result)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))
        print(f"Baseline metrics written to {output_path}")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
