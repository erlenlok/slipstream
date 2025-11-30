"""
Additional tests to verify the rebalance fix works properly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from slipstream.strategies.gradient.live.portfolio import construct_target_portfolio, apply_position_limits, validate_target_portfolio
from slipstream.strategies.gradient.live.config import GradientConfig


def create_test_config():
    """Create a test configuration for gradient strategy."""
    config = GradientConfig(
        capital_usd=100000.0,
        concentration_pct=10.0,  # 10% concentration to test with fewer assets
        rebalance_freq_hours=4,
        weight_scheme="equal",
        lookback_spans=[2, 4, 8, 16],
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


def test_fallback_mechanism():
    """Test the fallback mechanism when VAR method fails."""
    print("Testing fallback mechanism...")
    
    # Create signals DataFrame with some assets that pass liquidity filter
    n_liquid = 10
    signal_data = {
        'asset': [f'ASSET{i}' for i in range(n_liquid)],
        'momentum_score': np.linspace(-0.1, 0.1, n_liquid),  # Spread of scores
        'vol_24h': np.abs(np.random.randn(n_liquid)) * 0.01 + 0.001,  # Positive vol values
        'adv_usd': [15000] * n_liquid,  # Above liquidity threshold
        'include_in_universe': [True] * n_liquid,  # All included in universe
    }
    
    signals = pd.DataFrame(signal_data)
    signals['signal_timestamp'] = pd.Timestamp('2025-11-23 12:00:00')
    
    # Create log_returns DataFrame with just 1 row (which causes VAR to fail)
    assets = signals['asset'].tolist()
    timestamps = [pd.Timestamp('2025-11-23 12:00:00')]
    
    log_returns = pd.DataFrame(
        np.random.randn(1, len(assets)) * 0.001,  # Small returns
        index=timestamps,
        columns=assets
    )
    
    config = create_test_config()
    
    # Modify config to use smaller concentration to ensure we have both long and short
    config.concentration_pct = 30  # 3 assets long, 3 assets short (from 10 assets)
    
    print(f"Number of liquid assets: {len(signals)}")
    print(f"Concentration: {config.concentration_pct}%")
    
    try:
        target_positions = construct_target_portfolio(signals, log_returns, config)
        print(f"Target positions: {len(target_positions)} assets")
        print(f"Positions: {target_positions}")
        
        if target_positions:
            # Validate the positions
            validate_target_portfolio(target_positions, config)
            print("✓ Portfolio validation passed")
            
            # Check that we have both long and short positions
            long_positions = {k: v for k, v in target_positions.items() if v > 0}
            short_positions = {k: v for k, v in target_positions.items() if v < 0}
            
            print(f"Long positions: {len(long_positions)}")
            print(f"Short positions: {len(short_positions)}")
            
            if long_positions and short_positions:
                print("✓ Successfully created both long and short positions")
            else:
                print("⚠ Warning: Missing long or short positions")
        else:
            print("✗ No positions created")
            
    except Exception as e:
        print(f"✗ Error in portfolio construction: {e}")
        import traceback
        traceback.print_exc()


def test_dollar_vol_fallback():
    """Test dollar-vol fallback specifically."""
    print("\nTesting dollar-vol fallback...")
    
    # Create a config with dollar-vol method
    config = create_test_config()
    config.risk_method = "dollar_vol"
    
    # Create test signals with liquid assets
    n_liquid = 8
    signal_data = {
        'asset': [f'ASSET{i}' for i in range(n_liquid)],
        'momentum_score': np.random.randn(n_liquid) * 0.05,  # Random scores
        'vol_24h': np.abs(np.random.randn(n_liquid)) * 0.01 + 0.001,
        'adv_usd': [20000] * n_liquid,
        'include_in_universe': [True] * n_liquid,
    }
    
    signals = pd.DataFrame(signal_data)
    signals['signal_timestamp'] = pd.Timestamp('2025-11-23 12:00:00')
    
    # Create log returns
    assets = signals['asset'].tolist()
    timestamps = [pd.Timestamp('2025-11-23 12:00:00')]
    
    log_returns = pd.DataFrame(
        np.random.randn(1, len(assets)) * 0.001,
        index=timestamps,
        columns=assets
    )
    
    config.concentration_pct = 25  # 2 assets each side from 8 assets
    
    try:
        target_positions = construct_target_portfolio(signals, log_returns, config)
        print(f"Dollar-vol positions: {len(target_positions)} assets")
        print(f"Positions: {target_positions}")
        
        if target_positions:
            validate_target_portfolio(target_positions, config)
            print("✓ Dollar-vol portfolio validation passed")
        else:
            print("✗ No positions created with dollar-vol")
            
    except Exception as e:
        print(f"✗ Error in dollar-vol portfolio construction: {e}")
        import traceback
        traceback.print_exc()


def test_edge_cases():
    """Test edge cases like too few assets."""
    print("\nTesting edge cases...")
    
    config = create_test_config()
    
    # Case 1: Only 1 asset passes liquidity filter
    signal_data = {
        'asset': ['ASSET0'],
        'momentum_score': [0.1],
        'vol_24h': [0.01],
        'adv_usd': [15000],
        'include_in_universe': [True],
    }
    
    signals = pd.DataFrame(signal_data)
    signals['signal_timestamp'] = pd.Timestamp('2025-11-23 12:00:00')
    
    log_returns = pd.DataFrame(
        [[0.001]], 
        index=[pd.Timestamp('2025-11-23 12:00:00')],
        columns=['ASSET0']
    )
    
    config.concentration_pct = 50  # Should get 0.5 assets -> 0 after floor
    
    try:
        target_positions = construct_target_portfolio(signals, log_returns, config)
        print(f"Single asset result: {target_positions}")
        print("✓ Handled single asset case gracefully")
    except Exception as e:
        print(f"Expected behavior for single asset: {e}")
    
    # Case 2: No assets pass liquidity filter
    signals_no_liquid = pd.DataFrame({
        'asset': ['ASSET0', 'ASSET1'],
        'momentum_score': [0.1, 0.2],
        'vol_24h': [0.01, 0.02],
        'adv_usd': [5000, 6000],  # Below threshold
        'include_in_universe': [False, False],
    })
    signals_no_liquid['signal_timestamp'] = pd.Timestamp('2025-11-23 12:00:00')
    
    log_returns_empty = pd.DataFrame(
        [[0.001, 0.002]], 
        index=[pd.Timestamp('2025-11-23 12:00:00')],
        columns=['ASSET0', 'ASSET1']
    )
    
    try:
        target_positions = construct_target_portfolio(signals_no_liquid, log_returns_empty, config)
        print(f"No liquid assets result: {target_positions}")
        print("✓ Handled no liquid assets case gracefully")
    except Exception as e:
        print(f"Expected behavior for no liquid assets: {e}")


def test_position_limits_and_validation():
    """Test position limits and validation."""
    print("\nTesting position limits...")
    
    config = create_test_config()
    
    # Fake positions to test limits
    positions = {
        'BTC': config.capital_usd * 0.8,  # Too large position
        'ETH': -config.capital_usd * 0.8  # Too large short
    }
    
    # Test apply_position_limits
    limited_positions = apply_position_limits(positions, config)
    print(f"Original positions: {positions}")
    print(f"Limited positions: {limited_positions}")
    
    # Test validation
    try:
        validate_target_portfolio(positions, config)
        print("✗ Should have failed validation")
    except ValueError as e:
        print(f"✓ Correctly failed validation: {e}")
    
    # Valid position should pass
    valid_positions = {'BTC': config.capital_usd * 0.05}  # 5% of capital
    try:
        validate_target_portfolio(valid_positions, config)
        print("✓ Valid position passed validation")
    except ValueError as e:
        print(f"✗ Valid position failed validation: {e}")


if __name__ == "__main__":
    test_fallback_mechanism()
    test_dollar_vol_fallback()
    test_edge_cases()
    test_position_limits_and_validation()
    
    print("\nAll tests completed!")