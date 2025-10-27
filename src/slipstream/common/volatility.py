"""
Volatility estimation utilities shared between strategies.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def ewm_volatility(
    returns: pd.DataFrame,
    span: int,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Compute exponentially weighted volatility."""
    if span <= 0:
        raise ValueError("span must be positive")

    if min_periods is None:
        min_periods = span

    return returns.ewm(span=span, min_periods=min_periods).std()


def annualize_volatility(
    volatility: pd.DataFrame,
    periods_per_year: int,
) -> pd.DataFrame:
    """Annualize per-period volatility estimates."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    scale = np.sqrt(periods_per_year)
    return volatility * scale

