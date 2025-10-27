"""
Portfolio construction for the Gradient strategy.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from slipstream.common import ewm_volatility

from .universe import select_top_bottom_assets

EPSILON = 1e-8


def construct_gradient_portfolio(
    trend_strength: pd.DataFrame,
    log_returns: pd.DataFrame,
    *,
    top_n: int = 5,
    bottom_n: Optional[int] = None,
    min_abs_strength: float = 0.0,
    vol_span: int = 64,
    target_side_dollar_vol: float = 1.0,
) -> pd.DataFrame:
    """Build long/short weights that balance dollar volatility on both sides.

    Args:
        trend_strength: Wide DataFrame of aggregated trend scores.
        log_returns: Matching wide DataFrame of log returns (used for vol estimates).
        top_n: Number of strongest assets to consider for longs.
        bottom_n: Number of weakest assets for shorts (defaults to ``top_n``).
        min_abs_strength: Optional absolute-strength filter.
        vol_span: Span (bars) for EWMA volatility estimation.
        target_side_dollar_vol: Target dollar volatility for each side (long/short).

    Returns:
        DataFrame of portfolio weights aligned with ``trend_strength``.
    """
    if not trend_strength.index.equals(log_returns.index):
        raise ValueError("trend_strength and log_returns must share the same index")
    if not trend_strength.columns.equals(log_returns.columns):
        raise ValueError("trend_strength and log_returns must share the same columns")

    bottom_n = bottom_n or top_n
    if target_side_dollar_vol <= 0:
        raise ValueError("target_side_dollar_vol must be positive")

    vol_estimate = ewm_volatility(
        log_returns,
        span=vol_span,
        min_periods=vol_span,
    )

    long_mask, short_mask = select_top_bottom_assets(
        trend_strength,
        top_n=top_n,
        bottom_n=bottom_n,
        min_abs_strength=min_abs_strength,
    )

    weight_rows = []

    for timestamp in trend_strength.index:
        row_weights: Dict[str, float] = {}
        vol_row = vol_estimate.loc[timestamp]

        long_assets = long_mask.loc[timestamp]
        long_weights = _allocate_side(
            assets=long_assets.index[long_assets].tolist(),
            vol_row=vol_row,
            target_dollar_vol=target_side_dollar_vol,
            sign=1,
        )
        row_weights.update(long_weights)

        short_assets = short_mask.loc[timestamp]
        short_weights = _allocate_side(
            assets=short_assets.index[short_assets].tolist(),
            vol_row=vol_row,
            target_dollar_vol=target_side_dollar_vol,
            sign=-1,
        )
        row_weights.update(short_weights)

        weight_rows.append(pd.Series(row_weights, name=timestamp))

    weights = (
        pd.DataFrame(weight_rows)
        .reindex(index=trend_strength.index, columns=trend_strength.columns)
        .fillna(0.0)
    )

    # Zero out weeks with NaN volatility (no allocation)
    invalid_vol = vol_estimate.isna()
    weights = weights.mask(invalid_vol, 0.0)

    return weights


def _allocate_side(
    assets: Iterable[str],
    vol_row: pd.Series,
    target_dollar_vol: float,
    sign: int,
) -> Dict[str, float]:
    """Allocate weights for a single side (long or short)."""
    valid_assets = []
    for asset in assets:
        vol_value = vol_row.get(asset, np.nan)
        if pd.isna(vol_value) or vol_value <= 0:
            continue
        valid_assets.append((asset, vol_value))

    if not valid_assets:
        return {}

    n_assets = len(valid_assets)
    per_asset_risk = target_dollar_vol / n_assets

    weights: Dict[str, float] = {}
    for asset, vol_value in valid_assets:
        weight = sign * per_asset_risk / max(vol_value, EPSILON)
        weights[asset] = weight

    return weights
