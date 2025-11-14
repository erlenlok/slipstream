"""Portfolio construction for live Gradient trading."""

import math
from typing import Dict

import numpy as np
import pandas as pd

# Import unified portfolio construction from backtest module
from slipstream.strategies.gradient.portfolio import construct_gradient_portfolio


def construct_target_portfolio(
    signals: pd.DataFrame,
    log_returns: pd.DataFrame,  # NEW: needed for VAR method
    config
) -> Dict[str, float]:
    """
    Construct target portfolio from momentum signals using unified portfolio construction.

    Args:
        signals: DataFrame with columns: asset, momentum_score, vol_24h, adv_usd, include_in_universe
        log_returns: Wide DataFrame of recent 4h log returns (for VAR covariance estimation)
        config: GradientConfig instance

    Returns:
        Dictionary mapping asset -> target position size in USD
        Positive values = long, negative values = short
    """
    # Filter to liquid assets only
    liquid = (
        signals[signals["include_in_universe"] == True]
        .sort_values("momentum_score", ascending=False)
        .reset_index(drop=True)
        .copy()
    )

    n_liquid = len(liquid)
    if n_liquid == 0:
        print("Warning: No liquid assets available")
        return {}

    if n_liquid < 2:
        print("Warning: Not enough liquid assets to build both long and short books")
        return {}

    raw_select = math.ceil(n_liquid * config.concentration_pct / 100.0)
    n_select = max(1, raw_select)
    n_select = min(n_select, n_liquid // 2)

    if n_select == 0:
        print("Warning: Concentration setting results in empty buckets")
        return {}

    print(
        f"Selecting top/bottom {n_select} assets "
        f"({config.concentration_pct}% of {n_liquid} liquid assets)"
    )

    # Convert signals to format expected by construct_gradient_portfolio
    # Need: trend_strength (wide) and log_returns (wide)

    # Create single-row wide DataFrames for current rebalance
    trend_strength_wide = pd.DataFrame(
        [liquid.set_index('asset')['momentum_score'].to_dict()],
        index=[pd.Timestamp.now()]
    )

    # Ensure log_returns has same columns as trend_strength
    # Use only columns that exist in both
    common_assets = list(set(trend_strength_wide.columns) & set(log_returns.columns))
    trend_strength_wide = trend_strength_wide[common_assets]
    log_returns_aligned = log_returns[common_assets]

    # Call unified portfolio construction
    if config.risk_method == "var":
        # Use VAR-based allocation
        print(f"  Using VAR-based risk balancing (target: {config.target_side_var*100:.1f}% per side)")
        weights_df = construct_gradient_portfolio(
            trend_strength=trend_strength_wide,
            log_returns=log_returns_aligned,
            top_n=n_select,
            bottom_n=n_select,
            risk_method="var",
            target_side_var=config.target_side_var,
            var_lookback_days=config.var_lookback_days,
        )
    else:
        # Use legacy dollar-vol allocation
        print(f"  Using legacy dollar-vol balancing (scheme: {config.weight_scheme})")
        # Map weight_scheme to equivalent vol_span usage
        # inverse_vol uses volatility, equal doesn't
        target_dollar_vol = 1.0  # Normalized, will scale later
        weights_df = construct_gradient_portfolio(
            trend_strength=trend_strength_wide,
            log_returns=log_returns_aligned,
            top_n=n_select,
            bottom_n=n_select,
            risk_method="dollar_vol",
            target_side_dollar_vol=target_dollar_vol,
            vol_span=config.vol_span,
        )

    # Extract weights from last row (only row)
    weights_series = weights_df.iloc[-1]
    weights_dict = weights_series[weights_series != 0].to_dict()

    # Scale to dollar amounts
    # Each side gets 100% of capital (gross exposure = 2x)
    capital_per_side = config.capital_usd
    positions = {}

    for asset, weight in weights_dict.items():
        # Weights are already normalized by construct_gradient_portfolio
        # Scale to capital
        positions[asset] = weight * capital_per_side

    # Apply position size limits
    positions = apply_position_limits(positions, config)

    # Validate
    validate_target_portfolio(positions, config)

    return positions


def compute_weights(assets: pd.DataFrame, weight_scheme: str, side: str) -> Dict[str, float]:
    """
    Compute position weights for a bucket of assets.

    Args:
        assets: DataFrame with columns: asset, vol_24h
        weight_scheme: "equal" or "inverse_vol"
        side: "long" or "short"

    Returns:
        Dictionary mapping asset -> weight (normalized to sum to 1.0, signed)
    """
    if assets.empty:
        return {}

    if weight_scheme == "equal":
        weights = pd.Series(1.0 / len(assets), index=assets["asset"])
    elif weight_scheme == "inverse_vol":
        inv_vol = 1.0 / assets["vol_24h"]
        weights = inv_vol / inv_vol.sum()
        weights.index = assets["asset"]
    else:
        raise ValueError(f"Unknown weight scheme: {weight_scheme}")

    # Apply sign
    if side == "short":
        weights = -weights

    return weights.to_dict()


def apply_position_limits(positions: Dict[str, float], config) -> Dict[str, float]:
    """
    Apply position size limits to ensure risk controls.

    Args:
        positions: Target positions in USD
        config: GradientConfig instance

    Returns:
        Limited positions
    """
    max_position_usd = config.capital_usd * (config.max_position_pct / 100.0)

    limited_positions = {}
    for asset, size in positions.items():
        if abs(size) > max_position_usd:
            limited_positions[asset] = np.sign(size) * max_position_usd
        else:
            limited_positions[asset] = size

    # Check total leverage
    total_exposure = sum(abs(p) for p in limited_positions.values())
    current_leverage = total_exposure / config.capital_usd

    if current_leverage > config.max_total_leverage:
        # Scale down proportionally
        scale_factor = config.max_total_leverage / current_leverage
        limited_positions = {
            asset: size * scale_factor for asset, size in limited_positions.items()
        }

    return limited_positions


def validate_target_portfolio(positions: Dict[str, float], config) -> None:
    """
    Validate target portfolio.

    Args:
        positions: Target positions
        config: Configuration

    Raises:
        ValueError: If portfolio is invalid
    """
    if len(positions) == 0:
        raise ValueError("Target portfolio is empty")

    # Check for NaN or inf
    for asset, size in positions.items():
        if not np.isfinite(size):
            raise ValueError(f"Invalid position size for {asset}: {size}")

    # Check leverage (with small tolerance for floating point precision)
    total_exposure = sum(abs(p) for p in positions.values())
    leverage = total_exposure / config.capital_usd
    TOLERANCE = 0.01  # Allow 1% buffer to absorb floating point noise

    if leverage > config.max_total_leverage * (1 + TOLERANCE):
        raise ValueError(
            f"Total leverage {leverage:.2f}x exceeds limit {config.max_total_leverage}x"
        )

    # Check position sizes
    max_position_usd = config.capital_usd * (config.max_position_pct / 100.0)
    oversized = {
        asset: size for asset, size in positions.items() if abs(size) > max_position_usd
    }

    if oversized:
        raise ValueError(f"Positions exceed size limit: {oversized}")

    # Log summary
    n_long = sum(1 for p in positions.values() if p > 0)
    n_short = sum(1 for p in positions.values() if p < 0)
    gross_exposure = total_exposure
    net_exposure = sum(positions.values())

    print(f"Portfolio validation passed:")
    print(f"  Positions: {n_long} long, {n_short} short")
    print(f"  Gross exposure: ${gross_exposure:,.2f} ({leverage:.2f}x)")
    print(f"  Net exposure: ${net_exposure:,.2f}")
