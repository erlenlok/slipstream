"""CLI helpers for the template strategy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .backtest import run_template_backtest
from .config import load_template_config
from .signals import compute_template_signal

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a YAML/JSON file overriding template defaults.",
    )
    parser.add_argument(
        "--returns-csv",
        required=True,
        help="Path to a CSV file containing wide log returns (index=datetime).",
    )
    return parser


def _load_config_mapping(path: Optional[str]):
    if path is None:
        return None
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{path}' does not exist.")
    text = config_path.read_text()
    suffix = config_path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is not installed. Install it or use JSON for template configs."
            )
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _override_config_from_args(config, args) -> None:
    if args.lookback is not None:
        config.lookback = args.lookback
    if args.volatility_span is not None:
        config.volatility_span = args.volatility_span
    if args.top_n is not None:
        config.top_n = args.top_n
    if args.bottom_n is not None:
        config.bottom_n = args.bottom_n


def compute_signals_cli(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser("Compute template momentum signals.")
    parser.add_argument(
        "--signals-output",
        default="data/template/signals/momentum.csv",
        help="Where to store computed signals.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Override the rolling lookback for momentum.",
    )
    parser.add_argument(
        "--volatility-span",
        type=int,
        default=None,
        help="Override the EWMA span used for volatility normalization.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Override number of strongest assets on the long side (documentation aid).",
    )
    parser.add_argument(
        "--bottom-n",
        type=int,
        default=None,
        help="Override number of weakest assets on the short side (documentation aid).",
    )
    args = parser.parse_args(argv)

    config_mapping = _load_config_mapping(args.config)
    config = load_template_config(config_mapping)
    _override_config_from_args(config, args)

    returns = pd.read_csv(args.returns_csv, index_col=0, parse_dates=True)
    signals = compute_template_signal(
        returns,
        lookback=config.lookback,
        volatility_span=config.volatility_span,
    )

    output_path = Path(args.signals_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output_path)
    print(f"Template signals saved to {output_path}")


def run_backtest_cli(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser("Run the template strategy backtest.")
    parser.add_argument(
        "--signals-csv",
        default=None,
        help="Optional CSV containing pre-computed template signals.",
    )
    parser.add_argument(
        "--output",
        default="data/template/backtests/portfolio_returns.csv",
        help="Where to store the template backtest returns.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Override the rolling lookback for momentum.",
    )
    parser.add_argument(
        "--volatility-span",
        type=int,
        default=None,
        help="Override the EWMA span used for volatility normalization.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Override number of strongest assets on the long side.",
    )
    parser.add_argument(
        "--bottom-n",
        type=int,
        default=None,
        help="Override number of weakest assets on the short side.",
    )
    args = parser.parse_args(argv)

    config_mapping = _load_config_mapping(args.config)
    config = load_template_config(config_mapping)
    _override_config_from_args(config, args)

    returns = pd.read_csv(args.returns_csv, index_col=0, parse_dates=True)

    signals = None
    if args.signals_csv:
        signals = pd.read_csv(args.signals_csv, index_col=0, parse_dates=True)
        signals = signals.reindex(index=returns.index, columns=returns.columns)

    result = run_template_backtest(
        returns,
        signals=signals,
        lookback=config.lookback,
        volatility_span=config.volatility_span,
        top_n=config.top_n,
        bottom_n=config.bottom_n,
        target_side_risk=config.target_side_risk,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.portfolio_returns.to_csv(output_path, header=["portfolio_return"])

    sharpe = result.annualized_sharpe()
    sharpe_msg = (
        f"Annualized Sharpe (8760 periods/year assumption): {sharpe:.2f}"
        if np.isfinite(sharpe)
        else "Sharpe unavailable (insufficient data)."
    )

    print(f"Template backtest returns saved to {output_path}")
    print(sharpe_msg)


__all__ = ["compute_signals_cli", "run_backtest_cli"]
