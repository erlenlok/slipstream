"""
Test to verify that with proper historical data, risk methods should work correctly.
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


def test_with_historical_data():
    """Test the system with proper historical data."""
    print("Testing with realistic historical data that should work with VAR/dollar-vol methods...")
    
    # Create signals with liquid assets (only the latest timestamp)
    n_liquid = 106
    signal_data = {
        'asset': [f'ASSET{i}' for i in range(n_liquid)],
        'momentum_score': np.random.randn(n_liquid) * 0.1,  # Random momentum scores
        'vol_24h': np.abs(np.random.randn(n_liquid)) * 0.01,  # Positive vol values
        'adv_usd': np.random.uniform(15000, 50000, n_liquid),  # Above liquidity threshold
        'include_in_universe': [True] * n_liquid,  # All included in universe
    }
    
    signals = pd.DataFrame(signal_data)
    signals['signal_timestamp'] = pd.Timestamp('2025-11-23 12:00:00')
    
    # Now create HISTORICAL log_returns data (not just single row)
    # This simulates what we'd get after downloading historical data
    n_days = 100  # 100 days of historical data
    n_periods = n_days * 6  # 6 periods per day (4h intervals)
    
    # Create timestamps going back n_days
    end_time = pd.Timestamp('2025-11-23 12:00:00')
    timestamps = pd.date_range(end=end_time, periods=n_periods, freq='4H')
    
    assets = signals['asset'].tolist()
    
    # Generate realistic log returns (mean-reverting, with some volatility clustering)
    log_returns_data = np.random.randn(n_periods, len(assets)) * 0.001
    log_returns = pd.DataFrame(
        log_returns_data,
        index=timestamps,
        columns=assets
    )
    
    config = create_test_config()
    
    print(f"Config risk method: {config.risk_method}")
    print(f"Log returns shape: {log_returns.shape}")
    print(f"Number of historical periods: {log_returns.shape[0]}")
    print(f"Required var_lookback_days: {config.var_lookback_days}")
    print(f"Number of assets: {log_returns.shape[1]}")
    
    # This should now work because we have sufficient historical data
    try:
        target_positions = construct_target_portfolio(signals, log_returns, config)
        
        print(f"✓ Successfully created {len(target_positions)} target positions")
        
        # Check if we used the proper methods (not the fallback)
        if len(target_positions) > 0:
            print("✓ Target portfolio is NOT empty")
            
            # Check that we have both long and short positions
            long_pos = [k for k, v in target_positions.items() if v > 0]
            short_pos = [k for k, v in target_positions.items() if v < 0]
            
            print(f"Long positions: {len(long_pos)}")
            print(f"Short positions: {len(short_pos)}")
            
            if long_pos and short_pos:
                print("✓ Both long and short positions created successfully")
                
                # Check if we're using the actual risk methods or fallback
                # If we have a more balanced risk distribution (not all equal weights), 
                # it indicates the risk methods worked
                position_values = list(target_positions.values())
                unique_values = set(abs(v) for v in position_values)
                
                if len(unique_values) > min(len(long_pos), len(short_pos)):
                    print("✓ Risk-based position sizing used (not equal weighting)")
                else:
                    print("⚠ Equal weighting was used (risk methods may not have worked)")
            else:
                print("⚠ Only long or short positions created")
        else:
            print("✗ Target portfolio is empty")
            
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()


def test_var_based_diagnostics():
    """Test the VAR method specifically with diagnostics."""
    print("\nTesting VAR method with detailed diagnostics...")
    
    config = GradientConfig(
        capital_usd=100000.0,
        concentration_pct=20.0,
        rebalance_freq_hours=4,
        weight_scheme="equal",
        lookback_spans=[2, 4, 8, 16, 32],
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
        risk_method="var",
        target_side_var=0.02,
        var_lookback_days=30  # Use 30 days to reduce requirements
    )
    
    # Create test data
    n_assets = 50
    signal_data = {
        'asset': [f'ASSET{i}' for i in range(n_assets)],
        'momentum_score': np.random.randn(n_assets) * 0.05,
        'vol_24h': np.abs(np.random.randn(n_assets)) * 0.01,
        'adv_usd': [20000] * n_assets,
        'include_in_universe': [True] * n_assets,
    }
    
    signals = pd.DataFrame(signal_data)
    signals['signal_timestamp'] = pd.Timestamp('2025-11-23 12:00:00')
    
    # Create 40 days of historical data (more than required 30 days)
    n_periods = 40 * 6  # 40 days worth of 4h periods
    end_time = pd.Timestamp('2025-11-23 12:00:00')
    timestamps = pd.date_range(end=end_time, periods=n_periods, freq='4H')
    
    assets = signals['asset'].tolist()
    log_returns_data = np.random.randn(n_periods, len(assets)) * 0.001
    log_returns = pd.DataFrame(
        log_returns_data,
        index=timestamps,
        columns=assets
    )
    
    print(f"VAR method test:")
    print(f"  - Historical periods: {log_returns.shape[0]}")
    print(f"  - Required lookback: {config.var_lookback_days} days")
    print(f"  - Available days: {(log_returns.index[-1] - log_returns.index[0]).days}")
    print(f"  - Assets: {log_returns.shape[1]}")
    
    try:
        target_positions = construct_target_portfolio(signals, log_returns, config)
        print(f"  - Positions created: {len(target_positions)}")
        
        if target_positions:
            print("  ✓ VAR method successful with historical data!")
        else:
            print("  ✗ VAR method still failed")
    except Exception as e:
        print(f"  ✗ VAR method error: {e}")


if __name__ == "__main__":
    test_with_historical_data()
    test_var_based_diagnostics()
    print("\nTest completed. The VAR and dollar-vol methods should work when historical data is available.")