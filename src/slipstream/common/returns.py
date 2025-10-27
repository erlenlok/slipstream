"""
Return-related helpers shared across strategies.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


def vol_normalize_returns(
    returns: pd.DataFrame,
    window: int,
    method: Literal["ewm", "rolling"] = "ewm",
    min_periods: Optional[int] = None,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Scale returns by an estimate of their own volatility.

    Args:
        returns: Wide DataFrame of log returns (timestamp index, asset columns).
        window: Lookback window (bars) used for the volatility estimate.
        method: Volatility estimator. ``"ewm"`` uses exponential weighting while
            ``"rolling"`` uses a simple moving standard deviation.
        min_periods: Optional minimum periods required for a valid estimate.
            Defaults to ``window`` to avoid noisy early samples.
        epsilon: Small value to avoid division by zero.

    Returns:
        DataFrame with the same shape as ``returns`` containing volatility-normalized
        values. Entries without a valid volatility estimate remain NaN.
    """
    if window <= 0:
        raise ValueError("window must be positive")

    if min_periods is None:
        min_periods = window

    if method == "ewm":
        vol = returns.ewm(span=window, min_periods=min_periods).std()
    elif method == "rolling":
        vol = returns.rolling(window=window, min_periods=min_periods).std()
    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'ewm' or 'rolling'.")

    vol = vol.replace(0, np.nan)

    normalized = returns.div(vol + epsilon)
    normalized = normalized.where(np.isfinite(normalized), np.nan)

    return normalized
