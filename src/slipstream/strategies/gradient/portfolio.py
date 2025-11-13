"""
Portfolio construction for the Gradient strategy.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from slipstream.common import (
    ewm_volatility,
    compute_daily_returns,
    estimate_covariance_rie,
    compute_portfolio_var,
)

from .universe import select_top_bottom_assets

EPSILON = 1e-8


def construct_gradient_portfolio(
    trend_strength: pd.DataFrame,
    log_returns: pd.DataFrame,
    *,
    top_n: int = 5,
    bottom_n: Optional[int] = None,
    min_abs_strength: float = 0.0,
    # VAR parameters
    risk_method: str = "dollar_vol",
    target_side_var: float = 0.02,
    var_lookback_days: int = 60,
    # Legacy dollar-vol parameters
    vol_span: int = 64,
    target_side_dollar_vol: float = 1.0,
) -> pd.DataFrame:
    """Build long/short weights with dollar-vol or VAR-based risk balancing.

    Args:
        trend_strength: Wide DataFrame of aggregated trend scores.
        log_returns: Matching wide DataFrame of 4h log returns.
        top_n: Number of strongest assets to consider for longs.
        bottom_n: Number of weakest assets for shorts (defaults to ``top_n``).
        min_abs_strength: Optional absolute-strength filter.
        risk_method: "dollar_vol" (legacy) or "var" (new VAR-based balancing).
        target_side_var: Target one-day VAR for each side (95% confidence).
        var_lookback_days: Days of history for RIE covariance estimation.
        vol_span: Span (bars) for EWMA volatility estimation (legacy mode).
        target_side_dollar_vol: Target dollar volatility for each side (legacy mode).

    Returns:
        DataFrame of portfolio weights aligned with ``trend_strength``.

    Notes:
        When risk_method="var", weights within each side are proportional to
        |signal_strength| and both sides are scaled to achieve equal VAR using
        RIE-cleaned covariance matrices.
    """
    if not trend_strength.index.equals(log_returns.index):
        raise ValueError("trend_strength and log_returns must share the same index")
    if not trend_strength.columns.equals(log_returns.columns):
        raise ValueError("trend_strength and log_returns must share the same columns")

    bottom_n = bottom_n or top_n

    if risk_method not in ("dollar_vol", "var"):
        raise ValueError(f"risk_method must be 'dollar_vol' or 'var', got {risk_method}")

    if risk_method == "dollar_vol" and target_side_dollar_vol <= 0:
        raise ValueError("target_side_dollar_vol must be positive")
    if risk_method == "var" and target_side_var <= 0:
        raise ValueError("target_side_var must be positive")

    # Compute daily returns if using VAR method
    daily_returns = None
    if risk_method == "var":
        daily_returns = compute_daily_returns(log_returns, window=6)  # 6 * 4h = 24h

    # Legacy: compute volatility estimate for dollar_vol method
    vol_estimate = None
    if risk_method == "dollar_vol":
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

        long_assets = long_mask.loc[timestamp]
        long_asset_list = long_assets.index[long_assets].tolist()

        short_assets = short_mask.loc[timestamp]
        short_asset_list = short_assets.index[short_assets].tolist()

        if risk_method == "dollar_vol":
            # Legacy dollar-vol allocation
            vol_row = vol_estimate.loc[timestamp]

            long_weights = _allocate_side(
                assets=long_asset_list,
                vol_row=vol_row,
                target_dollar_vol=target_side_dollar_vol,
                sign=1,
            )
            row_weights.update(long_weights)

            short_weights = _allocate_side(
                assets=short_asset_list,
                vol_row=vol_row,
                target_dollar_vol=target_side_dollar_vol,
                sign=-1,
            )
            row_weights.update(short_weights)

        else:  # risk_method == "var"
            # VAR-based allocation with signal weighting
            strength_row = trend_strength.loc[timestamp]

            long_weights = _allocate_side_var_targeted(
                assets=long_asset_list,
                signal_strengths={a: strength_row[a] for a in long_asset_list},
                daily_returns=daily_returns.loc[:timestamp],  # Up to current time
                target_var=target_side_var,
                lookback_days=var_lookback_days,
                sign=1,
            )
            row_weights.update(long_weights)

            short_weights = _allocate_side_var_targeted(
                assets=short_asset_list,
                signal_strengths={a: strength_row[a] for a in short_asset_list},
                daily_returns=daily_returns.loc[:timestamp],  # Up to current time
                target_var=target_side_var,
                lookback_days=var_lookback_days,
                sign=-1,
            )
            row_weights.update(short_weights)

        weight_rows.append(pd.Series(row_weights, name=timestamp))

    weights = (
        pd.DataFrame(weight_rows)
        .reindex(index=trend_strength.index, columns=trend_strength.columns)
        .fillna(0.0)
    )

    # Zero out periods with invalid data
    if risk_method == "dollar_vol":
        invalid_vol = vol_estimate.isna()
        weights = weights.mask(invalid_vol, 0.0)
    elif risk_method == "var":
        # Zero out periods where we don't have enough daily return history
        invalid_daily = daily_returns.isna()
        weights = weights.mask(invalid_daily, 0.0)

    return weights


def _allocate_side(
    assets: Iterable[str],
    vol_row: pd.Series,
    target_dollar_vol: float,
    sign: int,
) -> Dict[str, float]:
    """Allocate weights for a single side using dollar-vol method (legacy)."""
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


def _allocate_side_var_targeted(
    assets: list[str],
    signal_strengths: dict[str, float],
    daily_returns: pd.DataFrame,
    target_var: float,
    lookback_days: int,
    sign: int,
) -> Dict[str, float]:
    """
    Allocate weights to achieve target VAR, proportional to signal strength.

    Algorithm:
    1. Compute signal-proportional base weights: w_i = |signal_i| / sum(|signals|)
    2. Estimate RIE-cleaned covariance matrix from daily returns
    3. Compute portfolio VAR using parametric formula
    4. Scale all weights to achieve target_var

    Args:
        assets: List of assets for this side
        signal_strengths: Dict of {asset: signal_strength}
        daily_returns: Wide DataFrame of daily returns (up to current timestamp)
        target_var: Target one-day VAR (95% confidence)
        lookback_days: Days of history for covariance estimation
        sign: +1 for long, -1 for short

    Returns:
        Dict of {asset: weight}
    """
    if not assets:
        return {}

    # Filter for assets with valid signals
    valid_assets = [a for a in assets if a in signal_strengths and not pd.isna(signal_strengths[a])]

    if not valid_assets:
        return {}

    # Step 1: Signal-proportional base weights
    abs_signals = {a: abs(signal_strengths[a]) for a in valid_assets}
    total_signal = sum(abs_signals.values())

    if total_signal <= EPSILON:
        # All signals near zero, use equal weighting
        base_weights_dict = {a: 1.0 / len(valid_assets) for a in valid_assets}
    else:
        base_weights_dict = {a: abs_signals[a] / total_signal for a in valid_assets}

    # Convert to array for linear algebra
    base_weights_array = np.array([base_weights_dict[a] for a in valid_assets])

    # Step 2: Estimate RIE covariance for these assets
    try:
        recent_returns = daily_returns[valid_assets]
        cov_matrix = estimate_covariance_rie(
            recent_returns,
            lookback_days=lookback_days,
            fallback_diagonal=True,
        )
    except Exception:
        # Fallback: if covariance estimation fails, return empty weights
        return {}

    # Step 3: Compute portfolio VAR
    try:
        computed_var = compute_portfolio_var(
            weights=base_weights_array,
            cov_matrix=cov_matrix,
            confidence=0.95,
        )
    except Exception:
        # Fallback: if VAR computation fails, return empty weights
        return {}

    if computed_var <= EPSILON or np.isnan(computed_var):
        # Can't compute meaningful VAR, return empty
        return {}

    # Step 4: Scale to achieve target VAR
    scale = target_var / computed_var

    final_weights = {a: sign * base_weights_dict[a] * scale for a in valid_assets}

    return final_weights
