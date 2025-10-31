"""Portfolio construction for live Gradient trading."""

import pandas as pd
import numpy as np
from typing import Dict


def construct_target_portfolio(signals: pd.DataFrame, config) -> Dict[str, float]:
    """
    Construct target portfolio from momentum signals.

    Args:
        signals: DataFrame with columns: asset, momentum_score, vol_24h, adv_usd, include_in_universe
        config: GradientConfig instance

    Returns:
        Dictionary mapping asset -> target position size in USD
        Positive values = long, negative values = short

    TODO: Implement portfolio construction
    1. Filter to liquid assets only (include_in_universe == True)
    2. Rank by momentum_score
    3. Select top N% (long) and bottom N% (short) where N = concentration_pct
    4. Calculate weights based on weight_scheme:
       - "equal": 1/N per asset in each bucket
       - "inverse_vol": w_i ∝ 1/vol_24h, normalized
    5. Scale to dollar amounts based on config.capital_usd
    6. Apply position size limits (max_position_pct)
    7. Verify total leverage within limits
    """
    raise NotImplementedError(
        "construct_target_portfolio() not yet implemented. "
        "Reuse logic from slipstream.gradient.sensitivity.run_concentration_backtest()."
    )

    # Example implementation structure:
    # liquid = signals[signals["include_in_universe"] == True].copy()
    # liquid = liquid.sort_values("momentum_score", ascending=False)
    # n_assets = len(liquid)
    # n_select = max(1, int(np.ceil(n_assets * config.concentration_pct / 100.0)))
    #
    # long_assets = liquid.head(n_select)
    # short_assets = liquid.tail(n_select)
    #
    # weights_long = compute_weights(long_assets, config.weight_scheme, "long")
    # weights_short = compute_weights(short_assets, config.weight_scheme, "short")
    #
    # # Scale to dollar amounts
    # capital_per_side = config.capital_usd  # 100% long, 100% short
    # positions = {}
    # for asset, weight in weights_long.items():
    #     positions[asset] = weight * capital_per_side
    # for asset, weight in weights_short.items():
    #     positions[asset] = weight * capital_per_side  # Already negative
    #
    # # Apply position size limits
    # positions = apply_position_limits(positions, config)
    #
    # return positions


def compute_weights(assets: pd.DataFrame, weight_scheme: str, side: str) -> Dict[str, float]:
    """
    Compute position weights for a bucket of assets.

    Args:
        assets: DataFrame with columns: asset, vol_24h
        weight_scheme: "equal" or "inverse_vol"
        side: "long" or "short"

    Returns:
        Dictionary mapping asset -> weight (normalized to sum to 1.0, signed)

    TODO: Implement weighting schemes
    """
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

    TODO: Implement position limits
    1. Cap each position at max_position_pct of capital
    2. Verify total leverage <= max_total_leverage
    3. Scale down proportionally if limits exceeded
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

    # Check leverage
    total_exposure = sum(abs(p) for p in positions.values())
    leverage = total_exposure / config.capital_usd

    if leverage > config.max_total_leverage:
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
