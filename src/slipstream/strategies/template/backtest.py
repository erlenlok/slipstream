"""Minimal backtest loop used by the template strategy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import compute_template_signal


@dataclass
class TemplateBacktestResult:
    """Container for template backtest artefacts."""

    positions: pd.DataFrame
    portfolio_returns: pd.Series

    def annualized_sharpe(self, periods_per_year: int = 365 * 24) -> float:
        """Compute a simple Sharpe ratio assuming `periods_per_year` rebalancing steps."""
        mean = self.portfolio_returns.mean()
        std = self.portfolio_returns.std(ddof=0)
        if std == 0 or np.isnan(std):
            return float("nan")
        return (mean / std) * np.sqrt(periods_per_year)


def _select_positions(
    signals: pd.DataFrame,
    top_n: int,
    bottom_n: int,
    target_side_risk: float,
) -> pd.DataFrame:
    """Convert cross-sectional signals into equal-weight long/short allocations."""
    long_slice = top_n if top_n > 0 else 0
    short_slice = bottom_n if bottom_n > 0 else 0
    total_columns = signals.shape[1]

    ranks = signals.rank(axis=1, method="first", ascending=False)
    long_mask = ranks <= long_slice
    short_mask = ranks >= (total_columns - short_slice + 1) if short_slice else pd.DataFrame(
        False, index=signals.index, columns=signals.columns
    )

    positions = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    if long_slice:
        positions = positions.where(~long_mask, target_side_risk / long_slice)
    if short_slice:
        positions = positions.where(~short_mask, -target_side_risk / short_slice)
    return positions


def run_template_backtest(
    returns: pd.DataFrame,
    *,
    signals: pd.DataFrame | None = None,
    lookback: int = 24,
    volatility_span: int = 64,
    top_n: int = 3,
    bottom_n: int | None = None,
    target_side_risk: float = 0.5,
) -> TemplateBacktestResult:
    """
    Backtest that:
    1. Builds normalized momentum signals (if not provided)
    2. Ranks signals cross-sectionally
    3. Allocates equal risk to the strongest/weakest assets with one-bar lag
    """

    aligned_returns = returns.sort_index()
    inferred_bottom_n = bottom_n if bottom_n is not None else top_n

    if signals is None:
        signals = compute_template_signal(
            aligned_returns,
            lookback=lookback,
            volatility_span=volatility_span,
        )
    aligned_signals = signals.reindex_like(aligned_returns).ffill()

    positions = _select_positions(
        aligned_signals,
        top_n=max(top_n, 0),
        bottom_n=max(inferred_bottom_n or 0, 0),
        target_side_risk=target_side_risk,
    )

    # Apply one-period lag to avoid lookahead bias.
    lagged_positions = positions.shift().fillna(0.0)
    portfolio_returns = (lagged_positions * aligned_returns).sum(axis=1)

    return TemplateBacktestResult(
        positions=positions,
        portfolio_returns=portfolio_returns,
    )


__all__ = ["TemplateBacktestResult", "run_template_backtest"]
