"""Sample signal helpers for template strategies."""

from __future__ import annotations

import pandas as pd


def compute_template_signal(
    returns: pd.DataFrame,
    lookback: int = 24,
    volatility_span: int = 64,
) -> pd.DataFrame:
    """
    Compute a simple normalized momentum signal for demonstration purposes.

    - Momentum: rolling mean of log returns over `lookback`.
    - Volatility: exponential moving standard deviation over `volatility_span`.
    """

    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    if volatility_span <= 1:
        raise ValueError("volatility_span must be greater than 1.")

    momentum = returns.rolling(window=lookback, min_periods=lookback).mean()
    volatility = returns.ewm(span=volatility_span, adjust=False).std(bias=False)
    normalized = momentum / volatility.replace(0.0, pd.NA)
    return normalized.dropna(how="all")


__all__ = ["compute_template_signal"]
