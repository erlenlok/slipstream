"""
Comprehensive test suite for Gradient strategy rebalancing.

This test suite ensures the rebalance process works correctly from data fetching
through portfolio construction to execution.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the modules we need to test
from slipstream.strategies.gradient.live.rebalance import run_rebalance
from slipstream.strategies.gradient.live.data import compute_live_signals, validate_market_data, validate_signals
from slipstream.strategies.gradient.live.portfolio import construct_target_portfolio, validate_target_portfolio
from slipstream.strategies.gradient.portfolio import construct_gradient_portfolio
from slipstream.strategies.gradient.live.config import load_config, GradientConfig


def create_test_market_data(n_assets=5, n_periods=100):
    """
    Create realistic test market data for gradient strategy.
    
    Args:
        n_assets: Number of assets to include
        n_periods: Number of 4h periods of data
    
    Returns:
        Dictionary with market data structure
    """
    # Create asset names
    assets = [f"ASSET{i}" for i in range(n_assets)]
    
    # Create timestamps (4h intervals)
    base_time = datetime.now() - timedelta(hours=n_periods * 4)
    timestamps = [base_time + timedelta(hours=4*i) for i in range(n_periods)]
    
    # Create test data
    data_list = []
    for asset in assets:
        # Generate realistic price data with some trend
        base_price = 100 + np.random.uniform(-20, 20)  # Random base price
        returns = np.random.normal(0.001, 0.02, n_periods)  # Daily returns ~0.1% mean, 2% std
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        volumes = np.random.uniform(1000, 10000, n_periods)  # Random trading volumes
        
        for i, ts in enumerate(timestamps):
            data_list.append({
                'timestamp': ts,
                'asset': asset,
                'open': prices[i],
                'high': prices[i] * (1 + abs(np.random.normal(0, 0.01))),
                'low': prices[i] * (1 - abs(np.random.normal(0, 0.01))),
                'close': prices[i],
                'volume': volumes[i]
            })
    
    panel = pd.DataFrame(data_list)
    
    return {
        'panel': panel,
        'assets': assets
    }


def create_test_config():
    """Create a test configuration for gradient strategy."""
    config = GradientConfig(
        api_endpoint="https://api.test.hyperliquid",
        wallet_address="0x0000000000000000000000000000000000000000",
        private_key="0x0000000000000000000000000000000000000000000000000000000000000000",
        risk_method="var",
        target_side_var=0.02,
        vol_span=24,
        lookback_spans=[2, 4, 8, 16],
        concentration_pct=10.0,
        capital_usd=100000.0,
        max_position_pct=5.0,
        max_total_leverage=2.0,
        liquidity_threshold_usd=10000.0,
        liquidity_impact_pct=2.5,
        var_lookback_days=60,
        min_abs_strength=0.01,
        dry_run=True,
        log_level="INFO",
        log_dir="/tmp",
        alerts_enabled=False
    )
    return config


class TestMarketDataAndSignals:
    """Test market data fetching and signal computation."""
    
    def test_create_valid_market_data(self):
        """Test that test market data has correct structure."""
        market_data = create_test_market_data(n_assets=3, n_periods=20)
        
        assert 'panel' in market_data
        assert 'assets' in market_data
        assert len(market_data['assets']) == 3
        assert len(market_data['panel']) == 60  # 3 assets * 20 periods
        assert list(market_data['panel'].columns) == ['timestamp', 'asset', 'open', 'high', 'low', 'close', 'volume']
        
        # Validate the data
        validate_market_data(market_data)
    
    def test_compute_live_signals(self):
        """Test signal computation from market data."""
        config = create_test_config()
        market_data = create_test_market_data(n_assets=5, n_periods=100)
        
        signals, log_returns = compute_live_signals(market_data, config)
        
        # Check that we get signals
        assert isinstance(signals, pd.DataFrame)
        assert isinstance(log_returns, pd.DataFrame)
        assert 'momentum_score' in signals.columns
        assert 'asset' in signals.columns
        assert 'signal_timestamp' in signals.columns
        
        # Validate the signals
        validate_signals(signals, config)
        
        # Check log returns has proper index structure
        assert len(log_returns.index) > 0
        assert len(log_returns.columns) > 0
    
    def test_signal_validation(self):
        """Test signal validation logic."""
        config = create_test_config()
        market_data = create_test_market_data(n_assets=3, n_periods=50)
        
        signals, log_returns = compute_live_signals(market_data, config)
        
        # This should pass validation
        validate_signals(signals, config)
        
        # Test validation failure cases
        bad_signals = signals.copy()
        bad_signals = bad_signals.drop(columns=['momentum_score'])
        
        with pytest.raises(ValueError):
            validate_signals(bad_signals, config)


class TestPortfolioConstruction:
    """Test portfolio construction logic."""
    
    def test_construct_gradient_portfolio_basic(self):
        """Test basic portfolio construction."""
        # Create test data with known structure
        timestamps = pd.date_range(start='2023-01-01', periods=10, freq='D')
        assets = ['A', 'B', 'C']
        
        # Create trend strength data (signals)
        trend_strength = pd.DataFrame(
            np.random.rand(10, 3), 
            index=timestamps, 
            columns=assets
        )
        
        # Create log returns data
        log_returns = pd.DataFrame(
            np.random.randn(10, 3) * 0.01, 
            index=timestamps, 
            columns=assets
        )
        
        # Test VAR method
        weights_df = construct_gradient_portfolio(
            trend_strength=trend_strength,
            log_returns=log_returns,
            top_n=2,
            bottom_n=2,
            risk_method="var",
            target_side_var=0.02,
            var_lookback_days=5
        )
        
        assert isinstance(weights_df, pd.DataFrame)
        assert weights_df.shape == trend_strength.shape
        assert not weights_df.empty
        
        # Test dollar_vol method
        weights_df2 = construct_gradient_portfolio(
            trend_strength=trend_strength,
            log_returns=log_returns,
            top_n=2,
            bottom_n=2,
            risk_method="dollar_vol",
            target_side_dollar_vol=1.0
        )
        
        assert isinstance(weights_df2, pd.DataFrame)
        assert weights_df2.shape == trend_strength.shape
    
    def test_construct_gradient_portfolio_index_validation(self):
        """Test that index validation works properly."""
        # Create DataFrames with different indices
        trend_strength = pd.DataFrame(
            [[0.1, 0.2, 0.3]], 
            index=[pd.Timestamp('2023-01-01')], 
            columns=['A', 'B', 'C']
        )
        
        log_returns = pd.DataFrame(
            [[0.01, 0.02, 0.03]], 
            index=[pd.Timestamp('2023-01-02')],  # Different timestamp
            columns=['A', 'B', 'C']
        )
        
        with pytest.raises(ValueError, match="trend_strength and log_returns must share the same index"):
            construct_gradient_portfolio(
                trend_strength=trend_strength,
                log_returns=log_returns,
                top_n=1,
                bottom_n=1
            )
    
    def test_construct_gradient_portfolio_column_validation(self):
        """Test that column validation works properly."""
        # Create DataFrames with different columns
        trend_strength = pd.DataFrame(
            [[0.1, 0.2]], 
            index=[pd.Timestamp('2023-01-01')], 
            columns=['A', 'B']
        )
        
        log_returns = pd.DataFrame(
            [[0.01, 0.02, 0.03]], 
            index=[pd.Timestamp('2023-01-01')],  # Same timestamp
            columns=['A', 'B', 'C']  # Different columns
        )
        
        with pytest.raises(ValueError, match="trend_strength and log_returns must share the same columns"):
            construct_gradient_portfolio(
                trend_strength=trend_strength,
                log_returns=log_returns,
                top_n=1,
                bottom_n=1
            )
    
    def test_construct_target_portfolio(self):
        """Test the live portfolio construction function."""
        config = create_test_config()
        market_data = create_test_market_data(n_assets=10, n_periods=50)
        
        signals, log_returns = compute_live_signals(market_data, config)
        
        # Adjust config for test - use smaller concentration
        config.concentration_pct = 20.0
        
        # Call the main function
        target_positions = construct_target_portfolio(signals, log_returns, config)
        
        # Should have positions
        assert isinstance(target_positions, dict)
        assert len(target_positions) > 0
        assert all(isinstance(k, str) for k in target_positions.keys())
        assert all(isinstance(v, (int, float, np.number)) for v in target_positions.values())


class TestRebalanceValidation:
    """Test validation logic."""
    
    def test_validate_target_portfolio(self):
        """Test portfolio validation."""
        config = create_test_config()
        
        # Valid positions
        positions = {'BTC': 1000.0, 'ETH': -500.0}
        validate_target_portfolio(positions, config)
        
        # Empty positions should fail
        with pytest.raises(ValueError, match="Target portfolio is empty"):
            validate_target_portfolio({}, config)
        
        # Invalid positions (NaN) should fail
        with pytest.raises(ValueError, match="Invalid position size"):
            validate_target_portfolio({'BTC': float('nan')}, config)
        
        # Too much leverage should fail
        high_leverage = {f'ASSET{i}': config.capital_usd * 0.6 for i in range(4)}  # 2.4x leverage
        with pytest.raises(ValueError, match="Total leverage"):
            validate_target_portfolio(high_leverage, config)
        
        # Too large individual positions should fail
        large_pos = {'BTC': config.capital_usd * config.max_position_pct / 100.0 * 2}  # 2x max position size
        with pytest.raises(ValueError, match="Positions exceed size limit"):
            validate_target_portfolio(large_pos, config)


class TestIntegration:
    """Integration tests for the rebalance process."""
    
    def test_complete_rebalance_flow(self):
        """Test the complete rebalance flow with mocked data."""
        # This would normally call run_rebalance() but we need to mock external dependencies
        # Since run_rebalance() calls API functions that we don't want to hit in tests
        pass
    
    @patch('slipstream.strategies.gradient.live.data.fetch_live_data')
    @patch('slipstream.strategies.gradient.live.config.load_config')
    def test_rebalance_with_mocked_data(self, mock_load_config, mock_fetch_data):
        """Test rebalance flow with mocked external dependencies."""
        # Create mock config
        config = create_test_config()
        mock_load_config.return_value = config
        
        # Create mock market data
        market_data = create_test_market_data(n_assets=8, n_periods=50)
        mock_fetch_data.return_value = market_data
        
        # Mock other functions to avoid calling external APIs
        with patch('slipstream.strategies.gradient.live.data.validate_market_data'), \
             patch('slipstream.strategies.gradient.live.data.validate_signals'), \
             patch('slipstream.strategies.gradient.live.rebalance.validate_execution_results'), \
             patch('slipstream.strategies.gradient.live.rebalance.get_current_positions') as mock_get_positions, \
             patch('slipstream.strategies.gradient.live.rebalance.execute_rebalance_with_stages') as mock_execute, \
             patch('slipstream.strategies.gradient.live.rebalance.PerformanceTracker'), \
             patch('slipstream.strategies.gradient.live.rebalance.send_telegram_rebalance_alert_sync'):
            
            # Mock current positions
            mock_get_positions.return_value = {'BTC': 1000.0, 'ETH': -500.0}
            
            # Mock execution results to return success
            mock_execute.return_value = {
                'target_order_count': 2,
                'stage1_filled': 2,
                'stage2_filled': 0,
                'total_turnover': 1500.0,
                'stage1_asset_fills': 2,
                'stage2_asset_fills': 0,
                'stage1_fill_notional': 1500.0,
                'stage2_fill_notional': 0.0,
                'total_target_usd': 1500.0,
                'errors': [],
                'passive_fill_rate': 1.0,
                'total_slippage': {'weighted_bps': 2.5, 'total_usd': 1500.0}
            }
            
            # This should run without errors
            # NOTE: We can't actually call run_rebalance here without more mocking
            # as it tries to import and run the full pipeline
            pass


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_no_liquid_assets(self):
        """Test behavior when no assets pass liquidity filter."""
        # Create signals where no assets pass liquidity filter
        signals = pd.DataFrame({
            'asset': ['A', 'B', 'C'],
            'momentum_score': [0.1, 0.2, 0.3],
            'vol_24h': [0.01, 0.02, 0.03],
            'adv_usd': [1000.0, 2000.0, 3000.0],  # Below liquidity threshold
            'include_in_universe': [False, False, False],  # No assets included
        })
        
        # Create corresponding log_returns
        timestamps = [datetime.now()]
        log_returns = pd.DataFrame(
            [[0.01, 0.02, 0.03]], 
            index=timestamps, 
            columns=['A', 'B', 'C']
        )
        
        config = create_test_config()
        
        # This should return empty portfolio
        target_positions = construct_target_portfolio(signals, log_returns, config)
        assert target_positions == {}
    
    def test_insufficient_assets_for_both_sides(self):
        """Test behavior when there aren't enough assets for both long and short."""
        signals = pd.DataFrame({
            'asset': ['A'],
            'momentum_score': [0.1],
            'vol_24h': [0.01],
            'adv_usd': [100000.0],  # Above liquidity threshold
            'include_in_universe': [True],
        })
        
        timestamps = [datetime.now()]
        log_returns = pd.DataFrame(
            [[0.01]], 
            index=timestamps, 
            columns=['A']
        )
        
        config = create_test_config()
        
        # This should return empty portfolio
        target_positions = construct_target_portfolio(signals, log_returns, config)
        assert target_positions == {}
    
    def test_empty_signals(self):
        """Test behavior with empty signals."""
        signals = pd.DataFrame({
            'asset': [],
            'momentum_score': [],
            'vol_24h': [],
            'adv_usd': [],
            'include_in_universe': [],
        })
        
        timestamps = [datetime.now()]
        log_returns = pd.DataFrame(
            [], 
            index=timestamps, 
            columns=[]
        )
        
        config = create_test_config()
        
        # This should return empty portfolio
        target_positions = construct_target_portfolio(signals, log_returns, config)
        assert target_positions == {}


# Run the tests
if __name__ == "__main__":
    test_suite = TestMarketDataAndSignals()
    test_suite.test_create_valid_market_data()
    test_suite.test_compute_live_signals()
    test_suite.test_signal_validation()
    
    test_portfolio = TestPortfolioConstruction()
    test_portfolio.test_construct_gradient_portfolio_basic()
    test_portfolio.test_construct_gradient_portfolio_index_validation()
    test_portfolio.test_construct_gradient_portfolio_column_validation()
    
    test_validation = TestRebalanceValidation()
    test_validation.test_validate_target_portfolio()
    
    test_edge = TestEdgeCases()
    test_edge.test_no_liquid_assets()
    test_edge.test_insufficient_assets_for_both_sides()
    test_edge.test_empty_signals()
    
    print("All tests passed!")