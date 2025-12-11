"""
Unit tests for Spectrum Execution Bridge - Module E

Tests for execution.py following the requirements in spectrum_spec.md
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from src.slipstream.strategies.spectrum.execution import (
    ExecutionTask,
    PortfolioPosition,
    HedgeManager,
    SpectrumExecutionBridge,
    create_execution_schedule
)


def test_execution_task():
    """Test ExecutionTask dataclass."""
    task = ExecutionTask(
        symbol='BTC',
        side='BUY',
        quantity=1.5,
        target_price=45000.0
    )
    
    assert task.symbol == 'BTC'
    assert task.side == 'BUY'
    assert task.quantity == 1.5
    assert task.target_price == 45000.0


def test_portfolio_position():
    """Test PortfolioPosition dataclass."""
    pos = PortfolioPosition(
        symbol='ETH',
        quantity=2.5,
        avg_price=3000.0,
        market_value=7500.0,
        beta_exposure=0.0
    )
    
    assert pos.symbol == 'ETH'
    assert pos.quantity == 2.5
    assert pos.avg_price == 3000.0
    assert pos.market_value == 7500.0


def test_hedge_manager():
    """Test HedgeManager functionality."""
    hedge_manager = HedgeManager(hedge_threshold=0.01)
    
    # Create sample positions
    positions = {
        'SOL': PortfolioPosition('SOL', 100, 100.0, 10000.0, 0.0),
        'ADA': PortfolioPosition('ADA', 1000, 1.0, 1000.0, 0.0)
    }
    
    # Create sample betas
    betas = {
        'SOL': 0.8,  # SOL has 0.8 beta to BTC
        'ADA': 0.5   # ADA has 0.5 beta to BTC
    }
    
    # Calculate beta exposure
    btc_beta, eth_beta = hedge_manager.calculate_beta_exposure(positions, betas)
    
    # SOL: 100 shares * 0.8 beta = 80 beta units
    # ADA: 1000 shares * 0.5 beta = 500 beta units
    # Total: 580 beta units
    expected_btc_beta = 100 * 0.8 + 1000 * 0.5  # 80 + 500 = 580
    assert abs(btc_beta - expected_btc_beta) < 1e-6
    
    # Generate hedge orders
    current_prices = {'BTC': 50000.0}
    hedge_orders = hedge_manager.generate_hedge_orders(positions, betas, current_prices)
    
    # Should generate hedge orders if net beta > threshold
    assert isinstance(hedge_orders, list)


def test_spectrum_execution_bridge_initialization():
    """Test SpectrumExecutionBridge initialization."""
    bridge = SpectrumExecutionBridge(
        account_equity=1000000.0,
        hedge_threshold=0.02,
        twap_duration_minutes=10
    )
    
    assert bridge.account_equity == 1000000.0
    assert bridge.hedge_manager.hedge_threshold == 0.02
    assert bridge.twap_duration_minutes == 10
    assert len(bridge.current_positions) == 0


def test_convert_weights_to_orders():
    """Test conversion of weights to execution orders."""
    bridge = SpectrumExecutionBridge(account_equity=1000000.0)
    
    # Create sample target weights
    target_weights = pd.Series({
        'BTC': 0.3,   # 30% in BTC
        'ETH': -0.2,  # -20% in ETH (short)
        'SOL': 0.1    # 10% in SOL
    })
    
    # Current prices
    current_prices = {
        'BTC': 50000.0,
        'ETH': 3000.0,
        'SOL': 100.0
    }
    
    # Current positions (all zero)
    position_quantities = {
        'BTC': 0.0,
        'ETH': 0.0,
        'SOL': 0.0
    }
    
    # Convert weights to orders
    orders = bridge.convert_weights_to_orders(target_weights, current_prices, position_quantities)
    
    # Verify orders
    assert len(orders) == 3  # One for each asset
    
    # Check BTC order: 30% of $1M = $300K at $50K = 6 shares
    btc_order = next(o for o in orders if o.symbol == 'BTC')
    assert btc_order.side == 'BUY'
    assert abs(btc_order.quantity - 6.0) < 0.01  # 300K / 50K = 6
    
    # Check ETH order: -20% of $1M = -$200K at $3K = -66.67 shares (SELL 66.67)
    eth_order = next(o for o in orders if o.symbol == 'ETH')
    assert eth_order.side == 'SELL'
    assert abs(eth_order.quantity - 66.67) < 0.1  # 200K / 3K = 66.67


def test_handle_asset_universe_changes():
    """Test handling of universe changes (new entrants and dropouts)."""
    bridge = SpectrumExecutionBridge(account_equity=1000000.0)

    # Current active assets
    current_assets = ['BTC', 'ETH', 'SOL']  # XRP is no longer active

    # Target weights that include a dropout (XRP) and missing assets
    target_weights = pd.Series({
        'BTC': 0.3,
        'ETH': -0.2,
        'SOL': 0.1,
        'XRP': 0.05  # This is now a dropout
    })

    # Beta coefficients
    betas = pd.DataFrame({
        'BTC': [0.8] * 10,
        'ETH': [0.6] * 10,
        'SOL': [0.7] * 10,
        'XRP': [0.9] * 10
    })

    # Handle universe changes
    adjusted_weights = bridge.handle_asset_universe_changes(current_assets, target_weights, betas)

    # Dropout (XRP) should be removed from the output entirely
    assert 'XRP' not in adjusted_weights.index

    # Active assets should keep their original weights
    assert adjusted_weights['BTC'] == 0.3
    assert adjusted_weights['ETH'] == -0.2
    assert adjusted_weights['SOL'] == 0.1

    # Only currently active assets should be in the result
    assert set(adjusted_weights.index) == set(current_assets)


def test_execution_bridge_full_cycle():
    """Test full execution cycle with both stages."""
    # Create sample data
    projected_weights = pd.Series({
        'BTC': 0.2,
        'ETH': -0.1,
        'SOL': 0.15
    })
    
    final_weights = pd.Series({
        'BTC': 0.25,  # Slightly different after final calculation
        'ETH': -0.15,
        'SOL': 0.12
    })
    
    betas = pd.DataFrame({
        'BTC': [0.8] * 5,
        'ETH': [0.6] * 5,
        'SOL': [0.7] * 5
    })
    
    projected_prices = {
        'BTC': 50000.0,
        'ETH': 3000.0,
        'SOL': 100.0
    }
    
    final_prices = {
        'BTC': 50100.0,  # Prices changed slightly
        'ETH': 3020.0,
        'SOL': 99.5
    }
    
    initial_positions = {
        'BTC': PortfolioPosition('BTC', 0, 50000.0, 0, 0.0),
        'ETH': PortfolioPosition('ETH', 0, 3000.0, 0, 0.0),
        'SOL': PortfolioPosition('SOL', 0, 100.0, 0, 0.0)
    }
    
    bridge = SpectrumExecutionBridge(account_equity=1000000.0)
    
    # Run the full execution cycle (asynchronous)
    async def run_test():
        results = await bridge.run_full_execution_cycle(
            projected_weights=projected_weights,
            final_weights=final_weights,
            betas=betas,
            projected_prices=projected_prices,
            final_prices=final_prices,
            initial_positions=initial_positions
        )
        return results
    
    # Run the async function
    results = asyncio.run(run_test())
    
    # Verify results structure
    assert 'stage_1' in results
    assert 'stage_2' in results
    assert 'hedge_orders' in results
    assert results['total_execution_cycle'] == 'completed'


def test_create_execution_schedule():
    """Test creation of execution schedule."""
    schedule = create_execution_schedule()
    
    assert 'stage_1_projected' in schedule
    assert 'stage_2_correction' in schedule
    
    # Verify they are datetime objects
    assert isinstance(schedule['stage_1_projected'], datetime)
    assert isinstance(schedule['stage_2_correction'], datetime)
    
    # Stage 2 should be after Stage 1
    assert schedule['stage_2_correction'] > schedule['stage_1_projected']


def test_execution_edge_cases():
    """Test edge cases in execution."""
    bridge = SpectrumExecutionBridge(account_equity=1000000.0)
    
    # Test with empty weights
    empty_weights = pd.Series(dtype=float)
    empty_prices = {}
    orders = bridge.convert_weights_to_orders(empty_weights, empty_prices)
    assert len(orders) == 0
    
    # Test with zero weights (no positions needed)
    zero_weights = pd.Series({'BTC': 0.0, 'ETH': 0.0})
    zero_prices = {'BTC': 50000.0, 'ETH': 3000.0}
    zero_quantities = {'BTC': 0.0, 'ETH': 0.0}
    orders = bridge.convert_weights_to_orders(zero_weights, zero_prices, zero_quantities)
    # Should have no meaningful orders (or very small ones due to rounding)
    assert len([o for o in orders if o.quantity > 1e-10]) == 0
    
    # Test handling universe with no overlap
    current_assets = ['BTC']
    target_weights = pd.Series({'ETH': 0.1})  # ETH not in current universe
    betas = pd.DataFrame({'BTC': [0.8], 'ETH': [0.6]})
    adjusted = bridge.handle_asset_universe_changes(current_assets, target_weights, betas)
    # ETH should be removed or set to 0, and BTC should be added with 0
    assert 'BTC' in adjusted.index
    assert 'ETH' not in adjusted.index or adjusted.get('ETH', 0) == 0.0


if __name__ == "__main__":
    # Run tests
    test_execution_task()
    test_portfolio_position()
    test_hedge_manager()
    test_spectrum_execution_bridge_initialization()
    test_convert_weights_to_orders()
    test_handle_asset_universe_changes()
    test_execution_bridge_full_cycle()
    test_create_execution_schedule()
    test_execution_edge_cases()
    print("All execution bridge tests passed!")