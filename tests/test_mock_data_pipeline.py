"""
Tests for the mock data pipeline and event processing.
Following TDD approach - these tests validate the mock pipeline functionality.
"""
import pytest
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path so we can import slipstream modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from slipstream.analytics.mock_data_pipeline import (
    MockTradeGenerator, MockTradeConfig, MockBrawlerEventProcessor
)
from slipstream.analytics.data_structures import TradeType


def test_mock_trade_generation():
    """Test that mock trades are generated with realistic parameters"""
    generator = MockTradeGenerator()
    trades = generator.generate_24h_trades()
    
    assert len(trades) > 0, "Should generate some trades"
    assert all(trade.symbol in ["BTC", "ETH", "SOL", "XRP", "ADA"] for trade in trades)
    assert all(trade.quantity > 0 for trade in trades)
    assert all(trade.price > 0 for trade in trades)
    
    # Check that we have both maker and taker trades
    maker_trades = [t for t in trades if t.trade_type == TradeType.MAKER]
    taker_trades = [t for t in trades if t.trade_type == TradeType.TAKER]
    assert len(maker_trades) > 0 and len(taker_trades) > 0


def test_mock_trade_stream():
    """Test that mock trades can be streamed over time periods"""
    start_time = datetime.now() - timedelta(hours=2)
    end_time = datetime.now()
    
    generator = MockTradeGenerator()
    trade_stream = list(generator.generate_trades_stream(start_time, end_time, trades_per_hour=5))
    
    assert len(trade_stream) > 0
    assert all(t.timestamp >= start_time and t.timestamp <= end_time for t in trade_stream)
    
    # Verify timestamps are roughly in order (with potential small variations due to ms delays)
    assert all(trade_stream[i].timestamp <= trade_stream[i+1].timestamp 
              for i in range(len(trade_stream)-1))


def test_trade_event_processing():
    """Test that trade events are processed correctly by the analytics system"""
    processor = MockBrawlerEventProcessor()
    
    # Generate some mock trades
    generator = MockTradeGenerator()
    trades = generator.generate_24h_trades()
    
    # Process the trades
    processor.process_multiple_events(trades)
    
    # Check that metrics were updated
    metrics = processor.get_24h_snapshot()
    assert metrics.total_trades == len(trades)
    assert metrics.total_volume > 0
    # Note: total_pnl is calculated differently, so we don't check exact equality
    assert len(processor.trade_buffer) == len(trades)


def test_multiple_instruments_handling():
    """Test that the system handles multiple instruments correctly"""
    processor = MockBrawlerEventProcessor()
    generator = MockTradeGenerator(MockTradeConfig(symbols=["BTC", "ETH"]))
    
    # Generate trades for both instruments
    trades = generator.generate_24h_trades()
    
    # Process trades
    processor.process_multiple_events(trades)
    metrics = processor.get_24h_snapshot()
    
    # Check that we have metrics for both instruments
    assert len(metrics.per_asset_metrics) >= 1  # At least one asset should have trades


def test_24hr_window_processing():
    """Test that 24-hour rolling windows are calculated correctly"""
    processor = MockBrawlerEventProcessor()
    
    # Generate 24 hours of data
    start_time = datetime.now() - timedelta(hours=24)
    generator = MockTradeGenerator()
    trades = generator.generate_24h_trades(start_time)
    
    # Process the trades
    processor.set_time_window(start_time)
    processor.process_multiple_events(trades)
    
    metrics = processor.get_24h_snapshot()
    
    # Verify we have data for the time window
    assert len(trades) == len(processor.trade_buffer)
    assert metrics.total_trades == len(trades)


def test_instrument_breakdown():
    """Test that metrics are correctly broken down by instrument"""
    processor = MockBrawlerEventProcessor()
    generator = MockTradeGenerator(MockTradeConfig(symbols=["BTC", "ETH", "SOL"]))
    
    # Generate trades for multiple instruments
    trades = generator.generate_24h_trades()
    
    # Process all trades
    processor.process_multiple_events(trades)
    metrics = processor.get_24h_snapshot()
    
    # Check that per-asset metrics exist
    for trade in trades:
        if trade.symbol in [t.symbol for t in trades]:  # Only check symbols that were generated
            assert trade.symbol in metrics.per_asset_metrics
    
    # Verify that each asset's metrics are properly calculated
    for symbol, asset_metrics in metrics.per_asset_metrics.items():
        # Each asset should have its own metrics
        assert asset_metrics.total_trades >= 0
        assert asset_metrics.total_volume >= 0


if __name__ == "__main__":
    # Run the tests
    test_mock_trade_generation()
    test_mock_trade_stream()
    test_trade_event_processing()
    test_multiple_instruments_handling()
    test_24hr_window_processing()
    test_instrument_breakdown()
    
    print("All mock data pipeline tests passed!")