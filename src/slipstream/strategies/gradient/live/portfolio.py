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
    # Need: trend_strength (wide) and log_returns (wide) with matching indices

    # Get the timestamp from the signals DataFrame to ensure alignment
    if 'signal_timestamp' in signals.columns and not signals.empty:
        signal_timestamp = signals['signal_timestamp'].iloc[0]
    else:
        # Fallback: use the most recent timestamp from log_returns
        signal_timestamp = log_returns.index[-1]

    # For live rebalancing, we need to use historical log_returns data to calculate risk metrics,
    # but only current signals for asset selection. To maximize historical data per asset,
    # we should preserve as much history as each asset has individually.

    # Get historical data up to (and including) the signal timestamp for risk calculations
    historical_log_returns = log_returns[log_returns.index <= signal_timestamp]

    # Get only assets that are in both the liquid universe AND have historical data
    liquid_assets = set(liquid['asset'])
    available_assets = set(historical_log_returns.columns)
    common_assets = list(liquid_assets & available_assets)

    # Focus on liquid assets that passed the universe filter
    liquid_assets_set = set(liquid[liquid['include_in_universe']]['asset'])
    final_assets = list(liquid_assets_set & set(historical_log_returns.columns))

    if not final_assets:
        print("  No assets available for portfolio construction - all liquid assets missing historical data")
        return {}

    # Select only the liquid assets that have historical data
    historical_log_returns = historical_log_returns[final_assets]

    # Create single-row trend strength for signal calculation (current moment only)
    liquid_filtered = liquid[liquid['asset'].isin(final_assets)].copy()
    current_signals_dict = liquid_filtered.set_index('asset')['momentum_score'].to_dict()

    # For risk calculations, we want to preserve maximum history per asset
    # Instead of dropping NaN values (which removes time periods), we should handle them appropriately
    # The log_returns may have NaN for different assets at different times due to different start dates
    # We should forward-fill within each asset column but not across time periods

    # Fill any NaN values that are due to different start dates for different assets
    historical_log_returns = historical_log_returns.ffill().fillna(0.0)  # Use 0.0 for returns where no data exists

    # Ensure we have sufficient data for risk calculations by checking minimum history required
    min_required_periods = config.var_lookback_days * 6  # 6 periods per day for 4h data
    actual_periods = len(historical_log_returns)

    print(f"    - Original log returns shape: {log_returns.shape}")
    print(f"    - Selected liquid assets with data: {len(final_assets)}")
    print(f"    - Historical log returns shape: {historical_log_returns.shape}")
    print(f"    - Available historical periods: {actual_periods}")
    print(f"    - Required lookback for VAR: {config.var_lookback_days} days (~{min_required_periods} periods)")

    if actual_periods < min_required_periods:
        print(f"    ⚠️  Insufficient historical data for VAR calculation ({actual_periods} < {min_required_periods})")
        print(f"    ⚠️  VAR method may not perform optimally with limited history")

    # Create trend strength with same index as historical data, filled with current signal values
    # This aligns the current signals with historical data for risk calculations
    # Create a DataFrame with the same index as historical_log_returns but with current signal values
    trend_strength_for_calc = pd.DataFrame(
        index=historical_log_returns.index,
        columns=historical_log_returns.columns,
        data=0.0  # Initialize with zeros
    )

    # Fill in the current signal values for all timestamps (this is how construct_gradient_portfolio expects it)
    for asset, signal_value in current_signals_dict.items():
        if asset in trend_strength_for_calc.columns:
            trend_strength_for_calc[asset] = signal_value  # Fill entire column with current signal value

    print(f"    - Trend strength for calculation shape: {trend_strength_for_calc.shape}")

    # Use the properly aligned DataFrames for portfolio construction
    trend_strength_wide = trend_strength_for_calc
    log_returns_aligned = historical_log_returns

    # Call unified portfolio construction
    try:
        if config.risk_method == "var":
            # Use VAR-based allocation
            print(f"  Using VAR-based risk balancing (target: {config.target_side_var*100:.1f}% per side)")
            print(f"    - Trend strength shape: {trend_strength_wide.shape}")
            print(f"    - Log returns shape: {log_returns_aligned.shape}")
            print(f"    - Top/Bottom: {n_select} each")
            print(f"    - VAR lookback days: {config.var_lookback_days}")

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
            print(f"    - Trend strength shape: {trend_strength_wide.shape}")
            print(f"    - Log returns shape: {log_returns_aligned.shape}")
            print(f"    - Top/Bottom: {n_select} each")
            print(f"    - Vol span: {config.vol_span}")

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

        # Debug: Check if weights are all zero
        if weights_dict:
            non_zero_count = len(weights_dict)
            total_assets = len(weights_series)
            print(f"    - Generated positions for {non_zero_count}/{total_assets} assets")
            if non_zero_count < total_assets:
                zero_weight_count = total_assets - non_zero_count
                print(f"    - {zero_weight_count} assets had zero weights")

            # Add position size statistics
            position_sizes = [abs(pos) for pos in weights_dict.values()]
            avg_position_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0
            max_position_size = max(position_sizes) if position_sizes else 0
            min_position_size = min(position_sizes) if position_sizes else 0

            print(f"    - Position size stats: avg=${avg_position_size:.2f}, min=${min_position_size:.2f}, max=${max_position_size:.2f}")

            # Sample of positions for debugging
            sample_positions = dict(list(weights_dict.items())[:5])
            print(f"    - Sample target positions: {sample_positions}")
        else:
            print(f"    - All weights were zero")

    except Exception as e:
        print(f"  Portfolio construction failed: {e}")
        import traceback
        print(f"  Full traceback: {traceback.format_exc()}")
        weights_dict = {}

    # If no positions were created (e.g., due to insufficient history or other issues),
    # fall back to simple equal weighting
    if not weights_dict:
        print(f"  No positions created - starting diagnostic analysis...")

        # Diagnostic: Check if we have sufficient data for VAR method
        if config.risk_method == "var":
            print("  VAR method diagnostics:")
            print(f"    - Required lookback days: {config.var_lookback_days}")
            print(f"    - Log returns date range: {log_returns_aligned.index.min()} to {log_returns_aligned.index.max()}")
            date_range = log_returns_aligned.index.max() - log_returns_aligned.index.min()
            print(f"    - Available date range: {date_range.days} days")
            print(f"    - Number of assets in lookback: {log_returns_aligned.shape[1]}")
            print(f"    - Number of time periods in lookback: {log_returns_aligned.shape[0]}")

            if log_returns_aligned.shape[0] < config.var_lookback_days:
                print(f"    - INSUFFICIENT DATA: Only {log_returns_aligned.shape[0]} periods, need {config.var_lookback_days}")

            # Check for NaN values in log returns
            nan_count = log_returns_aligned.isna().sum().sum()
            if nan_count > 0:
                print(f"    - Found {nan_count} NaN values in log returns that might cause VAR calc to fail")

            # Try dollar-vol as fallback
            print("  Attempting dollar-vol fallback method...")
            try:
                weights_df = construct_gradient_portfolio(
                    trend_strength=trend_strength_wide,
                    log_returns=log_returns_aligned,
                    top_n=n_select,
                    bottom_n=n_select,
                    risk_method="dollar_vol",
                    target_side_dollar_vol=1.0,  # Normalized, will scale later
                    vol_span=config.vol_span,
                )
                weights_series = weights_df.iloc[-1]
                weights_dict = weights_series[weights_series != 0].to_dict()

                if weights_dict:
                    print("  ✓ Dollar-vol fallback successful")
                else:
                    print("  ✗ Dollar-vol also resulted in zero positions")
            except Exception as e:
                print(f"  ✗ Dollar-vol fallback failed: {e}")
                import traceback
                print(f"    Full traceback: {traceback.format_exc()}")

        # If both methods failed, use simple equal weighting
        if not weights_dict:
            print("  Using equal-weighting fallback...")
            # Fallback: create simple equal-weighted positions based on top/bottom assets
            weights_dict = {}

            # Get top and bottom assets based on momentum scores
            liquid_sorted = liquid.sort_values("momentum_score", ascending=False)
            top_assets = liquid_sorted.head(n_select)["asset"].tolist()
            bottom_assets = liquid_sorted.tail(n_select)["asset"].tolist()  # Use actual bottom assets based on ranking

            # Equal weight within each side
            if top_assets:
                long_weight = 1.0 / len(top_assets)  # Equal weight for longs
                for asset in top_assets:
                    weights_dict[asset] = long_weight

            if bottom_assets:
                short_weight = -1.0 / len(bottom_assets)  # Equal weight for shorts
                for asset in bottom_assets:
                    weights_dict[asset] = short_weight

            # If we still don't have positions, use the top and bottom assets from the sorted list
            if not weights_dict and len(liquid_sorted) >= 2:
                # At least create positions for the top and bottom assets
                if len(top_assets) == 0 and len(liquid_sorted) > 0:
                    weights_dict[liquid_sorted.iloc[0]["asset"]] = 1.0  # Top asset long
                if len(bottom_assets) == 0 and len(liquid_sorted) > 1:
                    weights_dict[liquid_sorted.iloc[-1]["asset"]] = -1.0  # Bottom asset short

            # Final check
            if not weights_dict:
                print("  ❌ No positions could be created even with equal-weighting fallback!")
                print(f"    - Total liquid assets: {len(liquid_sorted)}")
                print(f"    - Selected n: {n_select}")
                print(f"    - Top assets: {top_assets}")
                print(f"    - Bottom assets: {bottom_assets}")
            else:
                print(f"  ✓ Equal-weighting fallback successful: {len(weights_dict)} positions")

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

    # Check for NaN or inf, and ensure type is numeric
    for asset, size in positions.items():
        # Check if position is a number first
        if not isinstance(size, (int, float)):
            raise ValueError(f"Invalid position size for {asset}: {size} (not a number)")
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
