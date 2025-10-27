"""
Data preparation helpers for funding-rate forecasting.

These utilities mirror the alpha pipeline but focus on predicting
forward funding payments instead of price-driven alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

from slipstream.alpha.data_prep import (
    compute_funding_features,
    load_all_funding,
    BASE_INTERVAL_HOURS,
)


def _hours_to_bars(hours: int, base_interval_hours: int = BASE_INTERVAL_HOURS) -> int:
    if hours <= 0:
        raise ValueError("Span hours must be positive")
    return max(1, int(round(hours / base_interval_hours)))


def compute_forward_funding(
    funding_rates: pd.DataFrame,
    H: int = 24,
    vol_span: int = 128,
    base_interval_hours: int = BASE_INTERVAL_HOURS,
    target_clip: float = 10.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute H-hour forward funding payments, normalised by funding volatility.

    Returns a tuple of (forward_funding_norm, funding_volatility) where both
    are indexed by MultiIndex (timestamp, asset).
    """
    if H % base_interval_hours != 0:
        raise ValueError(
            f"H={H}h is not a multiple of the base interval "
            f"({base_interval_hours}h)."
        )

    steps = H // base_interval_hours
    if steps <= 0:
        raise ValueError("Number of forward steps must be positive")

    vol_span_bars = _hours_to_bars(vol_span, base_interval_hours)
    funding_vol = funding_rates.ewm(
        span=vol_span_bars,
        min_periods=vol_span_bars,
    ).std()
    funding_vol = funding_vol.replace(0, np.nan)

    # Sum future funding over the next `steps` intervals
    # Using shift-based accumulation avoids explicit Python loops
    forward = sum(funding_rates.shift(-k) for k in range(1, steps + 1))

    forward_norm = (forward / funding_vol).clip(lower=-target_clip, upper=target_clip)

    forward_long = forward_norm.stack().dropna()
    forward_long.index.names = ['timestamp', 'asset']

    vol_long = funding_vol.stack().reindex(forward_long.index)
    vol_long.index.names = ['timestamp', 'asset']
    vol_long = vol_long.dropna()

    # Align indices after dropna
    common_idx = forward_long.index.intersection(vol_long.index)
    forward_long = forward_long.loc[common_idx]
    vol_long = vol_long.loc[common_idx]

    return forward_long, vol_long


def prepare_funding_training_data(
    funding_rates: pd.DataFrame,
    H: int = 24,
    spans: List[int] = [2, 4, 8, 16, 32, 64],
    vol_span: int = 128,
    base_interval_hours: int = BASE_INTERVAL_HOURS,
    min_history_hours: int | None = None,
    feature_clip: float = 5.0,
    target_clip: float = 10.0,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build the feature matrix and target vector for funding-rate forecasting.

    Args:
        funding_rates: Wide DataFrame of funding rates (base interval columns).
        H: Forward horizon in hours.
        spans: EWMA spans (in hours) for funding features.
        vol_span: EWMA span used for normalisation.
        base_interval_hours: underlying cadence of funding_rates (default 4h).
        min_history_hours: minimum historical coverage required per asset.
        feature_clip: clipping applied to EWMA funding features.
        target_clip: clipping applied to forward funding target (normalised space).

    Returns:
        Tuple (X, y, vol) where:
            X: MultiIndex feature DataFrame.
            y: Series of normalised forward funding sums.
            vol: Series of funding vol estimates for re-scaling predictions.
    """
    features = compute_funding_features(
        funding_rates,
        spans=spans,
        vol_span=vol_span,
        base_interval_hours=base_interval_hours,
        std_lookback=90 * 24,
        clip=feature_clip,
    )

    forward_funding, funding_vol = compute_forward_funding(
        funding_rates,
        H=H,
        vol_span=vol_span,
        base_interval_hours=base_interval_hours,
        target_clip=target_clip,
    )

    # Align indices across features, targets, and vol scaling
    common_index = (
        features.index
        .intersection(forward_funding.index)
        .intersection(funding_vol.index)
    )

    features = features.loc[common_index]
    y = forward_funding.loc[common_index]
    vol = funding_vol.loc[common_index]

    if min_history_hours is None:
        min_history_hours = vol_span
    min_history_bars = _hours_to_bars(min_history_hours, base_interval_hours)

    history_counts = features.groupby(level='asset').cumcount()
    history_mask = history_counts >= min_history_bars

    features = features[history_mask]
    y = y[history_mask]
    vol = vol[history_mask]

    valid_mask = features.notna().all(axis=1) & y.notna() & vol.notna()
    features = features[valid_mask]
    y = y[valid_mask]
    vol = vol[valid_mask]

    if features.empty:
        raise ValueError("No samples remain after alignment, clipping, and warm-up filters.")

    return features, y, vol


__all__ = [
    "prepare_funding_training_data",
    "compute_forward_funding",
    "load_all_funding",
]
