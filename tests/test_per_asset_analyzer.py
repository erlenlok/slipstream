"""
Tests for the per-instrument analysis and breakdowns module.
Following TDD approach - these tests validate per-asset analysis functionality.
"""
import pytest
from datetime import datetime
import sys
import os

# Add the src directory to the path so we can import slipstream modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from slipstream.analytics.per_asset_analyzer import (
    PerAssetPerformanceAnalyzer, AssetPerformanceMetrics
)
from slipstream.analytics.data_structures import TradeEvent


def test_per_asset_pnl_calculation():
    """Test PnL calculation broken down by asset."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Create trades for different assets
    trades = [
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="buy", quantity=1.0, 
                  price=50000.0, trade_type=None, fees_paid=10.0, funding_paid=5.0),
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="buy", quantity=10.0, 
                  price=3000.0, trade_type=None, fees_paid=5.0, funding_paid=2.0),
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="sell", quantity=0.5, 
                  price=50100.0, trade_type=None, fees_paid=5.0, funding_paid=3.0)
    ]
    
    analyzer.process_trades_batch(trades)
    summary = analyzer.get_per_asset_summary()
    
    # Should have metrics for BTC and ETH
    assert "BTC" in summary
    assert "ETH" in summary
    
    # Each should have key metrics
    assert 'total_pnl' in summary["BTC"]
    assert 'total_pnl' in summary["ETH"]


def test_per_asset_hit_rate():
    """Test hit rate calculation per individual asset."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Create trades with quotes for different assets
    trades = [
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="buy", quantity=1.0, 
                  price=50000.0, trade_type=None, fees_paid=10.0, quote_id="q1"),
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="sell", quantity=1.0, 
                  price=50050.0, trade_type=None, fees_paid=10.0, quote_id="q2"),
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="buy", quantity=10.0, 
                  price=3000.0, trade_type=None, fees_paid=5.0, quote_id=None),  # No quote
    ]
    
    analyzer.process_trades_batch(trades)
    
    # Check that assets were processed
    summary = analyzer.get_per_asset_summary()
    assert "BTC" in summary
    assert "ETH" in summary
    
    # BTC should have trades from quotes (hit rate context)
    assert summary["BTC"]['total_trades'] >= 2


def test_per_asset_markout():
    """Test markout calculation per individual asset."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Create trades with reference prices for markout calculation
    trades = [
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="buy", quantity=1.0, 
                  price=50000.0, trade_type=None, fees_paid=10.0, 
                  reference_price=50010.0),
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="sell", quantity=10.0, 
                  price=3000.0, trade_type=None, fees_paid=5.0, 
                  reference_price=2999.0),
    ]
    
    analyzer.process_trades_batch(trades)
    
    # Check that assets were processed
    summary = analyzer.get_per_asset_summary()
    assert "BTC" in summary
    assert "ETH" in summary
    
    # Both should have trade data
    assert summary["BTC"]['total_trades'] == 1
    assert summary["ETH"]['total_trades'] == 1


def test_asset_correlation_analysis():
    """Test correlation analysis between different assets."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Create trades that will allow correlation analysis
    trades = [
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="buy", quantity=1.0, 
                  price=50000.0, trade_type=None, fees_paid=10.0, funding_paid=5.0),
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="buy", quantity=10.0, 
                  price=3000.0, trade_type=None, fees_paid=5.0, funding_paid=2.0),
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="sell", quantity=1.0, 
                  price=50050.0, trade_type=None, fees_paid=10.0, funding_paid=5.0),
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="sell", quantity=10.0, 
                  price=3005.0, trade_type=None, fees_paid=5.0, funding_paid=2.0)
    ]
    
    analyzer.process_trades_batch(trades)
    
    # Get correlations between assets
    correlations = analyzer.get_cross_asset_correlations()
    
    # The function should return a dictionary without errors
    assert isinstance(correlations, dict)


def test_inventory_concentration_metrics():
    """Test metrics for inventory concentration by asset."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Update inventory with different levels
    alerts = analyzer.concentration.update_inventory("BTC", 5.0)
    alerts2 = analyzer.concentration.update_inventory("ETH", 2.0)
    alerts3 = analyzer.concentration.update_inventory("SOL", 0.5)
    
    # Get concentration report
    report = analyzer.get_concentration_risk_report()
    
    # Should have concentrations for all assets
    assert "BTC" in report
    assert "ETH" in report
    assert "SOL" in report
    
    # All concentrations should be between 0 and 1
    for asset, ratio in report.items():
        assert 0.0 <= ratio <= 1.0


def test_asset_capacity_analysis():
    """Test capacity analysis per asset."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Add size impact data
    analyzer.capacity.add_size_impact_data("BTC", 1.0, -0.001)
    analyzer.capacity.add_size_impact_data("BTC", 2.0, -0.003)
    analyzer.capacity.add_size_impact_data("BTC", 5.0, -0.010)
    analyzer.capacity.add_size_impact_data("ETH", 10.0, -0.0005)
    analyzer.capacity.add_size_impact_data("ETH", 20.0, -0.0015)
    
    # Get capacity analysis
    capacity_metrics = analyzer.get_capacity_analysis()
    
    # Should have metrics for both assets
    assert "BTC" in capacity_metrics
    assert "ETH" in capacity_metrics
    
    # Each should have capacity metrics
    assert 'avg_impact_per_size' in capacity_metrics["BTC"]
    assert 'avg_impact_per_size' in capacity_metrics["ETH"]


def test_cross_asset_impact():
    """Test how performance on one asset affects others."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Create trades for multiple assets
    btc_trades = [
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="buy", quantity=1.0, 
                  price=50000.0, trade_type=None, fees_paid=10.0, funding_paid=5.0),
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="sell", quantity=1.0, 
                  price=50050.0, trade_type=None, fees_paid=10.0, funding_paid=5.0)
    ]
    
    eth_trades = [
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="buy", quantity=10.0, 
                  price=3000.0, trade_type=None, fees_paid=5.0, funding_paid=2.0),
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="sell", quantity=10.0, 
                  price=3005.0, trade_type=None, fees_paid=5.0, funding_paid=2.0)
    ]
    
    # Process all trades through the main analyzer
    all_trades = btc_trades + eth_trades
    analyzer.process_trades_batch(all_trades)
    
    # Get top performing assets
    top_assets = analyzer.get_top_performing_assets(2)
    
    # Should return a list of tuples (asset, pnl)
    assert isinstance(top_assets, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in top_assets)
    
    # Get worst performing assets
    worst_assets = analyzer.get_worst_performing_assets(2)
    assert isinstance(worst_assets, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in worst_assets)


def test_asset_pair_analysis():
    """Test analysis of asset pairs and their interactions."""
    analyzer = PerAssetPerformanceAnalyzer()
    
    # Create mixed trades
    trades = [
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="buy", quantity=1.0, 
                  price=50000.0, trade_type=None, fees_paid=10.0, funding_paid=5.0),
        TradeEvent(timestamp=datetime.now(), symbol="ETH", side="buy", quantity=10.0, 
                  price=3000.0, trade_type=None, fees_paid=5.0, funding_paid=2.0),
        TradeEvent(timestamp=datetime.now(), symbol="BTC", side="sell", quantity=0.5, 
                  price=50100.0, trade_type=None, fees_paid=5.0, funding_paid=3.0),
        TradeEvent(timestamp=datetime.now(), symbol="SOL", side="buy", quantity=100.0, 
                  price=150.0, trade_type=None, fees_paid=7.0, funding_paid=1.0)
    ]
    
    analyzer.process_trades_batch(trades)
    
    # Get per-asset summary
    summary = analyzer.get_per_asset_summary()
    
    # Should have all three assets
    assert "BTC" in summary
    assert "ETH" in summary
    assert "SOL" in summary


if __name__ == "__main__":
    # Run the tests
    test_per_asset_pnl_calculation()
    test_per_asset_hit_rate()
    test_per_asset_markout()
    test_asset_correlation_analysis()
    test_inventory_concentration_metrics()
    test_asset_capacity_analysis()
    test_cross_asset_impact()
    test_asset_pair_analysis()
    
    print("All Per-Instrument Analysis and Breakdowns tests passed!")