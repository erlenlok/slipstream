"""
Test to check actual position sizes with realistic parameters
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from slipstream.strategies.gradient.live.portfolio import construct_target_portfolio
from slipstream.strategies.gradient.live.config import GradientConfig


def create_realistic_test_data():
    """Create test data that simulates real market conditions."""
    # Create 50 assets (representing 25% of a larger universe)
    n_assets = 50
    assets = [f'ASSET{i}' for i in range(n_assets)]
    
    # Create realistic trend strengths (some strong, some weak)
    # Simulate trend scores from -1.0 to +1.0 roughly
    np.random.seed(42)  # For reproducibility
    trend_scores = np.random.uniform(-1.0, 1.0, n_assets)
    
    # Sort to clearly see top/bottom
    sorted_indices = np.argsort(trend_scores)
    
    # Create liquid signals DataFrame (all pass liquidity filter)
    signals_data = {
        'asset': assets,
        'momentum_score': trend_scores,
        'vol_24h': np.abs(np.random.normal(0.02, 0.005, n_assets)),  # Realistic 24h vol
        'adv_usd': np.random.uniform(20000, 100000, n_assets),  # Realistic ADV
        'include_in_universe': [True] * n_assets,  # All liquid
    }
    signals = pd.DataFrame(signals_data)
    signals['signal_timestamp'] = pd.Timestamp('2025-11-23 16:00:00')
    
    # Create realistic historical log returns (60+ days of 4h data)
    n_periods = 90 * 6  # 90 days of 4h periods = 540 periods
    end_time = pd.Timestamp('2025-11-23 16:00:00')
    timestamps = pd.date_range(end=end_time, periods=n_periods, freq='4h')
    
    # Generate realistic log returns (volatility clustering, correlations)
    log_returns_data = np.random.randn(n_periods, n_assets) * 0.0015  # Daily vol ~ 0.23%
    log_returns = pd.DataFrame(
        log_returns_data,
        index=timestamps,
        columns=assets
    )
    
    return signals, log_returns


def test_realistic_position_sizing():
    """Test position sizing with realistic parameters."""
    print("Testing realistic position sizing with your parameters...")
    print("Capital: $450, Leverage: 2x, VAR target: 2% per side ($9)")
    print("Concentration: top/bottom 25% of perps")
    print("Min trade size: $10")
    
    signals, log_returns = create_realistic_test_data()
    
    # Create config matching your requirements
    config = GradientConfig(
        capital_usd=450.0,  # $450 capital
        concentration_pct=25.0,  # 25% concentration
        rebalance_freq_hours=4,
        weight_scheme="equal",  # For comparison
        lookback_spans=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        vol_span=24,
        max_position_pct=100.0,  # Allow larger positions
        max_total_leverage=2.0,  # 2x leverage
        emergency_stop_drawdown_pct=0.15,
        liquidity_threshold_usd=10000.0,
        liquidity_impact_pct=2.5,
        api_endpoint="https://api.hyperliquid.xyz",
        api_key="test_key",
        api_secret="test_secret",
        mainnet=False,
        api={
            "endpoint": "https://api.hyperliquid.xyz",
            "mainnet": False
        },
        execution={
            "passive_timeout_seconds": 3600,
            "limit_order_aggression": "join_best",
            "cancel_before_market_sweep": True,
            "min_order_size_usd": 10.0  # $10 minimum trade size
        },
        dry_run=True,
        log_dir="/tmp",
        log_level="INFO",
        alerts_enabled=False,
        risk_method="var",  # VAR-based risk management
        target_side_var=0.02,  # 2% daily VAR per side
        var_lookback_days=60
    )
    
    print(f"\nSignals shape: {signals.shape}")
    print(f"Log returns shape: {log_returns.shape}")
    print(f"Available assets: {len(signals)}")
    
    # Test VAR-based position construction
    print(f"\nUsing VAR method with 2% VAR target per side")
    print(f"Target exposure per side: ${config.capital_usd * 1.0:.2f}")  # Each side gets $450
    print(f"Target VAR per side: ${config.target_side_var * config.capital_usd:.2f}")  # 2% of $450 = $9
    
    try:
        target_positions = construct_target_portfolio(signals, log_returns, config)
        
        if target_positions:
            print(f"\nGenerated {len(target_positions)} target positions")
            
            # Analyze position sizes
            position_sizes = [abs(pos) for pos in target_positions.values()]
            avg_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0
            min_size = min(position_sizes) if position_sizes else 0
            max_size = max(position_sizes) if position_sizes else 0
            
            print(f"Position size statistics:")
            print(f"  Average: ${avg_size:.3f}")
            print(f"  Minimum: ${min_size:.3f}")
            print(f"  Maximum: ${max_size:.3f}")
            
            # Count positions above minimum threshold
            above_min = sum(1 for pos in position_sizes if abs(pos) >= 10.0)
            below_min = sum(1 for pos in position_sizes if abs(pos) < 10.0)
            
            print(f"  Above $10 minimum: {above_min} positions")
            print(f"  Below $10 minimum: {below_min} positions")
            
            # Show long/short breakdown
            long_positions = {k: v for k, v in target_positions.items() if v > 0}
            short_positions = {k: abs(v) for k, v in target_positions.items() if v < 0}
            
            print(f"\nLong positions: {len(long_positions)}")
            print(f"Short positions: {len(short_positions)}")
            
            if long_positions:
                print(f"  Long avg: ${sum(long_positions.values()) / len(long_positions):.3f}")
            if short_positions:
                print(f"  Short avg: ${sum(short_positions.values()) / len(short_positions):.3f}")
                
            # Sample of positions
            print(f"\nSample positions:")
            for asset, pos in list(target_positions.items())[:10]:
                print(f"  {asset}: ${pos:.3f}")
                
        else:
            print(f"\nNo positions generated!")
            
        # Also test dollar-vol method for comparison
        print(f"\n" + "="*60)
        print("Testing with dollar-vol method for comparison:")
        config.risk_method = "dollar_vol"
        target_positions_dollar_vol = construct_target_portfolio(signals, log_returns, config)
        
        if target_positions_dollar_vol:
            print(f"\nDollar-vol method generated {len(target_positions_dollar_vol)} positions")
            
            pos_sizes = [abs(pos) for pos in target_positions_dollar_vol.values()]
            avg_size = sum(pos_sizes) / len(pos_sizes) if pos_sizes else 0
            min_size = min(pos_sizes) if pos_sizes else 0
            max_size = max(pos_sizes) if pos_sizes else 0
            
            print(f"Dollar-vol position size statistics:")
            print(f"  Average: ${avg_size:.3f}")
            print(f"  Minimum: ${min_size:.3f}")
            print(f"  Maximum: ${max_size:.3f}")
            
            # Count positions above minimum threshold
            above_min = sum(1 for pos in pos_sizes if abs(pos) >= 10.0)
            below_min = sum(1 for pos in pos_sizes if abs(pos) < 10.0)
            
            print(f"  Above $10 minimum: {above_min} positions")
            print(f"  Below $10 minimum: {below_min} positions")
            
        else:
            print(f"\nNo positions with dollar-vol method!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_realistic_position_sizing()