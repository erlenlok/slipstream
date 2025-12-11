"""
Tests for the Brawler performance tracking data structures.
Following TDD approach - tests are written before implementation validation.
"""
import pytest
from datetime import datetime, timedelta
from dataclasses import dataclass
import sys
import os

# Add the src directory to the path so we can import slipstream modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from slipstream.analytics.data_structures import TradeEvent, MarkoutAnalysis, PerformanceMetrics, TradeType


def test_trade_event_creation():
    """Test that TradeEvent can be properly instantiated with all required fields"""
    timestamp = datetime.now()
    trade = TradeEvent(
        timestamp=timestamp,
        symbol="BTC",
        side="buy",
        quantity=1.0,
        price=50000.0,
        trade_type=TradeType.MAKER,
        fees_paid=10.0,
        funding_paid=5.0,
        position_before=0.0,
        position_after=1.0
    )
    
    assert trade.timestamp == timestamp
    assert trade.symbol == "BTC"
    assert trade.side == "buy"
    assert trade.quantity == 1.0
    assert trade.price == 50000.0
    assert trade.trade_type == TradeType.MAKER
    assert trade.fees_paid == 10.0
    assert trade.funding_paid == 5.0
    assert trade.position_before == 0.0
    assert trade.position_after == 1.0


def test_trade_event_serialization():
    """Test that TradeEvent can be serialized/deserialized for storage"""
    timestamp = datetime(2023, 1, 1, 12, 0, 0)
    trade = TradeEvent(
        timestamp=timestamp,
        symbol="ETH",
        side="sell",
        quantity=2.0,
        price=3000.0,
        trade_type=TradeType.TAKER,
        reference_price=3001.0,
        fees_paid=5.0
    )
    
    trade_dict = trade.to_dict()
    
    assert trade_dict['timestamp'] == '2023-01-01T12:00:00'
    assert trade_dict['symbol'] == 'ETH'
    assert trade_dict['side'] == 'sell'
    assert trade_dict['quantity'] == 2.0
    assert trade_dict['price'] == 3000.0
    assert trade_dict['trade_type'] == 'taker'
    assert trade_dict['reference_price'] == 3001.0
    assert trade_dict['fees_paid'] == 5.0


def test_markout_calculation_basic():
    """Test basic markout calculation between trade and reference price"""
    timestamp = datetime.now()
    trade = TradeEvent(
        timestamp=timestamp,
        symbol="BTC",
        side="buy",
        quantity=1.0,
        price=50000.0,
        trade_type=TradeType.MAKER,
        reference_price=50010.0,  # Higher reference price means positive markout for buy
        fees_paid=10.0
    )
    
    markout = trade.calculate_markout()
    expected_markout = (50010.0 - 50000.0) * 1.0 - 10.0  # (ref_price - trade_price) * qty - fees
    assert markout == expected_markout


def test_markout_calculation_with_fees():
    """Test markout calculation that accounts for fees"""
    timestamp = datetime.now()
    trade = TradeEvent(
        timestamp=timestamp,
        symbol="BTC",
        side="sell",
        quantity=1.0,
        price=50000.0,
        trade_type=TradeType.MAKER,
        reference_price=49990.0,  # Lower reference price means positive markout for sell
        fees_paid=15.0
    )
    
    markout = trade.calculate_markout()
    expected_markout = (50000.0 - 49990.0) * 1.0 - 15.0  # (trade_price - ref_price) * qty - fees
    assert markout == expected_markout


def test_performance_metrics_initialization():
    """Test that PerformanceMetrics can be initialized properly"""
    metrics = PerformanceMetrics()
    
    assert metrics.total_pnl == 0.0
    assert metrics.total_quotes == 0
    assert metrics.total_fills == 0
    assert metrics.hit_rate == 0.0
    assert metrics.total_volume == 0.0
    assert metrics.total_trades == 0
    assert isinstance(metrics.markout_analysis, MarkoutAnalysis)


def test_metrics_aggregation_empty():
    """Test metrics aggregation with no data returns appropriate defaults"""
    metrics = PerformanceMetrics()
    
    assert metrics.calculate_hit_rate() == 0.0
    assert metrics.calculate_fill_rate() == 0.0
    assert metrics.calculate_pnl_per_quote() == 0.0


def test_trade_pnl_calculation():
    """Test PnL calculation for trades"""
    timestamp = datetime.now()
    trade = TradeEvent(
        timestamp=timestamp,
        symbol="BTC",
        side="buy",
        quantity=1.0,
        price=50000.0,
        trade_type=TradeType.MAKER,
        fees_paid=10.0,
        funding_paid=5.0
    )
    
    # Test with a hypothetical exit price
    exit_price = 50100.0
    pnl = trade.calculate_pnl(exit_price)
    expected_pnl = (exit_price - trade.price) * trade.quantity - trade.fees_paid - trade.funding_paid
    assert pnl == expected_pnl


def test_markout_calculation_no_reference_price():
    """Test that markout returns None when reference price is not available"""
    timestamp = datetime.now()
    trade = TradeEvent(
        timestamp=timestamp,
        symbol="BTC",
        side="buy",
        quantity=1.0,
        price=50000.0,
        trade_type=TradeType.MAKER,
        reference_price=None,  # No reference price
        fees_paid=10.0
    )
    
    markout = trade.calculate_markout()
    assert markout is None


def test_markout_calculation_sells():
    """Test markout calculation for sell trades"""
    timestamp = datetime.now()
    trade = TradeEvent(
        timestamp=timestamp,
        symbol="BTC",
        side="sell",
        quantity=1.0,
        price=50000.0,
        trade_type=TradeType.MAKER,
        reference_price=49990.0,  # Lower reference price means positive markout for sell
        fees_paid=10.0
    )
    
    markout = trade.calculate_markout()
    expected_markout = (trade.price - trade.reference_price) * trade.quantity - trade.fees_paid
    assert markout == expected_markout


if __name__ == "__main__":
    # Run the tests
    test_trade_event_creation()
    test_trade_event_serialization()
    test_markout_calculation_basic()
    test_markout_calculation_with_fees()
    test_performance_metrics_initialization()
    test_metrics_aggregation_empty()
    test_trade_pnl_calculation()
    test_markout_calculation_no_reference_price()
    test_markout_calculation_sells()
    
    print("All tests passed!")