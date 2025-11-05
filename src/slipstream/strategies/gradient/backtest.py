"""
Lightweight backtest wrapper for the Gradient strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .portfolio import construct_gradient_portfolio
from .signals import DEFAULT_LOOKBACKS, compute_trend_strength


@dataclass
class GradientBacktestResult:
    """Container for Gradient backtest artefacts."""

    weights: pd.DataFrame
    trend_strength: pd.DataFrame
    portfolio_returns: pd.Series

    def cumulative_returns(self) -> pd.Series:
        """Cumulative sum of per-period portfolio returns."""
        return self.portfolio_returns.cumsum()

    def annualized_sharpe(
        self,
        periods_per_year: int,
    ) -> float:
        """Compute a simple annualized Sharpe ratio."""
        mean = self.portfolio_returns.mean()
        vol = self.portfolio_returns.std(ddof=0)

        if vol == 0 or np.isnan(vol):
            return float("nan")

        return (mean * periods_per_year) / (vol * np.sqrt(periods_per_year))


def run_gradient_backtest(
    log_returns: pd.DataFrame,
    *,
    trend_strength: Optional[pd.DataFrame] = None,
    lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
    top_n: int = 5,
    bottom_n: Optional[int] = None,
    min_abs_strength: float = 0.0,
    vol_span: int = 64,
    target_side_dollar_vol: float = 1.0,
) -> GradientBacktestResult:
    """Run a basic gradient backtest using the provided returns."""
    if trend_strength is None:
        trend_strength = compute_trend_strength(
            log_returns,
            lookbacks=lookbacks,
        )

    weights = construct_gradient_portfolio(
        trend_strength,
        log_returns,
        top_n=top_n,
        bottom_n=bottom_n,
        min_abs_strength=min_abs_strength,
        vol_span=vol_span,
        target_side_dollar_vol=target_side_dollar_vol,
    )

    shifted_weights = weights.shift(1).fillna(0.0)
    portfolio_returns = (shifted_weights * log_returns).sum(axis=1)

    return GradientBacktestResult(
        weights=weights,
        trend_strength=trend_strength,
        portfolio_returns=portfolio_returns,
    )

