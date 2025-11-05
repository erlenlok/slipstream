"""
Command-line entry points for the Gradient strategy utilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from . import (
    DEFAULT_LOOKBACKS,
    compute_trend_strength,
    run_gradient_backtest,
)


def _parse_lookbacks(raw: Optional[Iterable[str]]) -> List[int]:
    if not raw:
        return list(DEFAULT_LOOKBACKS)
    lookbacks = [int(value) for value in raw]
    if any(lb <= 0 for lb in lookbacks):
        raise ValueError("Lookbacks must be positive integers.")
    return lookbacks


def compute_signals_cli(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute Gradient trend strength signals from wide log returns.",
    )
    parser.add_argument(
        "--returns-csv",
        required=True,
        help="Path to a CSV file containing wide log returns (index=datetime).",
    )
    parser.add_argument(
        "--output",
        default="data/gradient/signals/trend_strength.csv",
        help="Output CSV file for aggregated trend strength.",
    )
    parser.add_argument(
        "--lookbacks",
        nargs="*",
        default=None,
        help="Optional list of lookback windows. Defaults to 2,4,...,1024.",
    )
    parser.add_argument(
        "--normalization",
        choices=["ewm", "rolling"],
        default="ewm",
        help="Volatility normalization method to use.",
    )

    args = parser.parse_args(argv)

    lookbacks = _parse_lookbacks(args.lookbacks)
    returns = pd.read_csv(args.returns_csv, index_col=0, parse_dates=True)

    trend_strength = compute_trend_strength(
        returns,
        lookbacks=lookbacks,
        normalization_method=args.normalization,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trend_strength.to_csv(output_path)

    print(f"Gradient trend strength saved to {output_path}")


def run_backtest_cli(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a lightweight Gradient backtest using wide log returns.",
    )
    parser.add_argument(
        "--returns-csv",
        required=True,
        help="Path to a CSV file containing wide log returns (index=datetime).",
    )
    parser.add_argument(
        "--signals-csv",
        default=None,
        help="Optional precomputed trend strength CSV.",
    )
    parser.add_argument(
        "--lookbacks",
        nargs="*",
        default=None,
        help="Optional list of lookback windows when computing signals.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of strongest assets to hold long.",
    )
    parser.add_argument(
        "--bottom-n",
        type=int,
        default=None,
        help="Number of weakest assets to hold short (defaults to top-n).",
    )
    parser.add_argument(
        "--vol-span",
        type=int,
        default=64,
        help="EWMA span used for volatility estimation.",
    )
    parser.add_argument(
        "--target-side-dollar-vol",
        type=float,
        default=1.0,
        help="Target dollar volatility allocated to each side of the book.",
    )
    parser.add_argument(
        "--returns-output",
        default="data/gradient/backtests/portfolio_returns.csv",
        help="Where to store the per-period portfolio returns.",
    )

    args = parser.parse_args(argv)

    lookbacks = _parse_lookbacks(args.lookbacks)
    returns = pd.read_csv(args.returns_csv, index_col=0, parse_dates=True)

    signals: Optional[pd.DataFrame]
    if args.signals_csv:
        signals = pd.read_csv(args.signals_csv, index_col=0, parse_dates=True)
        signals = signals.reindex(index=returns.index, columns=returns.columns)
    else:
        signals = compute_trend_strength(
            returns,
            lookbacks=lookbacks,
        )

    result = run_gradient_backtest(
        returns,
        trend_strength=signals,
        lookbacks=lookbacks,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        vol_span=args.vol_span,
        target_side_dollar_vol=args.target_side_dollar_vol,
    )

    output_path = Path(args.returns_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.portfolio_returns.to_csv(output_path, header=["portfolio_return"])

    sharpe = result.annualized_sharpe(periods_per_year=24 * 365)
    if np.isnan(sharpe):
        sharpe_msg = "Sharpe unavailable (insufficient data)."
    else:
        sharpe_msg = f"Annualized Sharpe (8760 periods/year assumption): {sharpe:.2f}"

    print(f"Saved portfolio returns to {output_path}")
    print(sharpe_msg)
