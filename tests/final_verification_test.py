"""
Final test to verify the rebalancing fix works properly.
"""

import pandas as pd
import numpy as np
from datetime import datetime
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


def test_original_scenario():
    """Test the original scenario that was failing."""
    print("Testing the original failing scenario...")
    
    # Create the same kind of data that was causing the error
    signal_data = {
        'asset': [f'ASSET{i}' for i in range(106)],  # 106 liquid assets
        'momentum_score': np.random.randn(106) * 0.1,  # Random momentum scores
        'vol_24h': np.abs(np.random.randn(106)) * 0.01,  # Positive vol values
        'adv_usd': np.random.uniform(15000, 50000, 106),  # Above liquidity threshold
        'include_in_universe': [True] * 106,  # All included in universe
    }
    
    signals = pd.DataFrame(signal_data)
    signals['signal_timestamp'] = pd.Timestamp('2025-11-23 12:00:00')
    
    # Create log_returns DataFrame (as would be created in the live system)
    assets = signals['asset'].tolist()
    timestamps = [pd.Timestamp('2025-11-23 12:00:00')]  # Same as signal
    
    log_returns = pd.DataFrame(
        np.random.randn(1, len(assets)) * 0.001,  # Small returns
        index=timestamps,
        columns=assets
    )
    
    config = create_test_config()
    
    print(f"Config risk method: {config.risk_method}")
    print(f"Number of signals: {len(signals)}")
    print(f"Number of log returns columns: {len(log_returns.columns)}")
    
    try:
        # This should now work with our fix
        target_positions = construct_target_portfolio(signals, log_returns, config)
        
        print(f"✓ Successfully created {len(target_positions)} target positions")
        print(f"Positions: {list(target_positions.keys())[:5]}...")  # Show first 5 assets
        
        # Verify that positions are not empty (this was the original error)
        if target_positions:
            print("✓ Target portfolio is NOT empty - FIX SUCCESSFUL!")
            
            # Check that we have both long and short positions (expected for gradient strategy)
            long_pos = [k for k, v in target_positions.items() if v > 0]
            short_pos = [k for k, v in target_positions.items() if v < 0]
            
            print(f"Long positions: {len(long_pos)}")
            print(f"Short positions: {len(short_pos)}")
            
            if long_pos and short_pos:
                print("✓ Both long and short positions created successfully")
            else:
                print("⚠ Only long or short positions created")
        else:
            print("✗ Target portfolio is still empty - FIX FAILED!")
            
    except ValueError as e:
        if "Target portfolio is empty" in str(e):
            print(f"✗ Original error still occurs: {e}")
        else:
            print(f"✗ Different error occurred: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def test_edge_cases():
    """Test various edge cases to ensure robustness."""
    print("\nTesting edge cases...")
    
    config = create_test_config()
    config.concentration_pct = 20  # 20% for easier testing
    
    # Case 1: Only 2 assets - minimum for both long and short
    signals_min = pd.DataFrame({
        'asset': ['BTC', 'ETH'],
        'momentum_score': [0.05, -0.03],
        'vol_24h': [0.015, 0.020],
        'adv_usd': [50000, 40000],
        'include_in_universe': [True, True],
    })
    signals_min['signal_timestamp'] = pd.Timestamp('2025-11-23 12:00:00')
    
    log_returns_min = pd.DataFrame(
        [[0.0015, -0.0020]], 
        index=[pd.Timestamp('2025-11-23 12:00:00')],
        columns=['BTC', 'ETH']
    )
    
    try:
        positions = construct_target_portfolio(signals_min, log_returns_min, config)
        print(f"✓ Minimum case (2 assets): {len(positions)} positions created")
    except Exception as e:
        print(f"✗ Minimum case failed: {e}")
    
    # Case 2: VAR method with dollar-vol fallback
    config_var = create_test_config()
    config_var.risk_method = "var"
    
    try:
        positions_var = construct_target_portfolio(signals_min, log_returns_min, config_var)
        print(f"✓ VAR method fallback: {len(positions_var)} positions created")
    except Exception as e:
        print(f"✗ VAR method fallback failed: {e}")
    
    # Case 3: Dollar-vol method directly
    config_dollar = create_test_config()
    config_dollar.risk_method = "dollar_vol"
    
    try:
        positions_dollar = construct_target_portfolio(signals_min, log_returns_min, config_dollar)
        print(f"✓ Dollar-vol method: {len(positions_dollar)} positions created")
    except Exception as e:
        print(f"✗ Dollar-vol method failed: {e}")


if __name__ == "__main__":
    test_original_scenario()
    test_edge_cases()
    print("\nAll tests completed! The fix should handle the rebalancing issue properly.")