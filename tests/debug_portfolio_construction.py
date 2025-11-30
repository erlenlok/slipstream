"""
Debug test for the specific portfolio construction issue.

This test recreates the exact conditions that lead to "Target portfolio is empty" error.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from slipstream.strategies.gradient.live.portfolio import construct_target_portfolio
from slipstream.strategies.gradient.live.config import GradientConfig


def create_test_config():
    """Create a test configuration for gradient strategy."""
    config = GradientConfig(
        capital_usd=100000.0,
        concentration_pct=30.0,  # 30% concentration
        rebalance_freq_hours=4,
        weight_scheme="equal",
        lookback_spans=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        vol_span=24,
        max_position_pct=10.0,
        max_total_leverage=2.0,
        emergency_stop_drawdown_pct=0.15,
        liquidity_threshold_usd=10000.0,
        liquidity_impact_pct=2.5,
        api_endpoint="https://api.test.hyperliquid",
        api_key="test_key",
        api_secret="test_secret",
        mainnet=False,
        api={
            "endpoint": "https://api.test.hyperliquid",
            "mainnet": False
        },
        execution={
            "passive_timeout_seconds": 3600,
            "limit_order_aggression": "join_best",
            "cancel_before_market_sweep": True,
            "min_order_size_usd": 10
        },
        dry_run=True,
        log_dir="/tmp",
        log_level="INFO",
        alerts_enabled=False,
        risk_method="var",  # This is now set as a keyword argument with default
        target_side_var=0.02,
        var_lookback_days=60
    )
    return config


def create_debug_data():
    """Create debug data that mimics the live scenario."""
    # Create a mock signal DataFrame similar to what would be produced
    signal_data = {
        'asset': [f'ASSET{i}' for i in range(106)],  # 106 liquid assets
        'momentum_score': np.random.randn(106) * 0.1,  # Random momentum scores
        'vol_24h': np.abs(np.random.randn(106)) * 0.01,  # Positive vol values
        'adv_usd': np.random.uniform(15000, 50000, 106),  # Above liquidity threshold
        'include_in_universe': [True] * 106,  # All included in universe
    }
    
    signals = pd.DataFrame(signal_data)
    
    # Add signal timestamp
    signals['signal_timestamp'] = pd.Timestamp('2025-11-23 12:00:00')
    
    # Create mock log_returns DataFrame with same timestamp as signal
    assets = signals['asset'].tolist()
    timestamps = [pd.Timestamp('2025-11-23 12:00:00')]  # Same as signal
    
    log_returns = pd.DataFrame(
        np.random.randn(1, len(assets)) * 0.001,  # Small returns
        index=timestamps,
        columns=assets
    )
    
    return signals, log_returns


def test_portfolio_construction_debug():
    """Debug test for the portfolio construction issue."""
    print("Creating test data...")
    signals, log_returns = create_debug_data()
    
    print(f"Signals shape: {signals.shape}")
    print(f"Log returns shape: {log_returns.shape}")
    print(f"Signal timestamp: {signals['signal_timestamp'].iloc[0]}")
    print(f"Log returns index: {log_returns.index}")
    
    config = create_test_config()
    print(f"Config concentration: {config.concentration_pct}%")
    print(f"Config risk method: {config.risk_method}")
    print(f"Config target side var: {config.target_side_var}")
    
    # Debug: Check how many assets we're selecting
    liquid = signals[signals["include_in_universe"] == True]
    print(f"Liquid assets: {len(liquid)}")
    
    n_liquid = len(liquid)
    raw_select = int(n_liquid * config.concentration_pct / 100.0)
    n_select = max(1, raw_select)
    n_select = min(n_select, n_liquid // 2)  # Ensure we don't exceed half the assets
    
    print(f"Raw select: {raw_select}")
    print(f"Adjusted n_select: {n_select}")
    print(f"n_liquid // 2: {n_liquid // 2}")
    
    # Test the construction step by step
    trend_strength_wide = pd.DataFrame(
        [liquid.set_index('asset')['momentum_score'].to_dict()],
        index=[signals['signal_timestamp'].iloc[0]]
    )
    
    print(f"Trend strength wide shape: {trend_strength_wide.shape}")
    print(f"Trend strength wide index: {trend_strength_wide.index}")
    print(f"Trend strength wide columns: {list(trend_strength_wide.columns[:5])}...")  # Show first 5
    
    # Check alignment
    common_assets = list(set(trend_strength_wide.columns) & set(log_returns.columns))
    print(f"Common assets count: {len(common_assets)}")
    
    trend_strength_wide = trend_strength_wide[common_assets]
    if signals['signal_timestamp'].iloc[0] in log_returns.index:
        log_returns_filtered = log_returns.loc[[signals['signal_timestamp'].iloc[0]]]
        log_returns_aligned = log_returns_filtered[common_assets]
    else:
        log_returns_aligned = log_returns[common_assets].tail(1)
    
    print(f"Aligned shapes - trend: {trend_strength_wide.shape}, log_returns: {log_returns_aligned.shape}")
    print(f"Indices match: {trend_strength_wide.index.equals(log_returns_aligned.index)}")
    
    try:
        print("Attempting to call construct_gradient_portfolio...")
        from slipstream.strategies.gradient.portfolio import construct_gradient_portfolio
        
        weights_df = construct_gradient_portfolio(
            trend_strength=trend_strength_wide,
            log_returns=log_returns_aligned,
            top_n=n_select,
            bottom_n=n_select,
            risk_method="var",
            target_side_var=config.target_side_var,
            var_lookback_days=config.var_lookback_days,
        )
        
        print(f"Weight matrix shape: {weights_df.shape}")
        print(f"Weight matrix:\n{weights_df}")
        
        # Extract weights from last row (only row)
        weights_series = weights_df.iloc[-1]
        non_zero_weights = weights_series[weights_series != 0]
        print(f"Non-zero weights count: {len(non_zero_weights)}")
        print(f"Non-zero weights: {non_zero_weights.to_dict()}")
        
    except Exception as e:
        print(f"Error in construct_gradient_portfolio: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Now try the full function
    print("\nTrying full construct_target_portfolio...")
    try:
        target_positions = construct_target_portfolio(signals, log_returns, config)
        print(f"Target positions: {target_positions}")
        print(f"Target positions count: {len(target_positions)}")
    except ValueError as e:
        print(f"ValueError in construct_target_portfolio: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Other error in construct_target_portfolio: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_portfolio_construction_debug()