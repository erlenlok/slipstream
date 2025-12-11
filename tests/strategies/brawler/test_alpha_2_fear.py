
import pytest
import time
from unittest.mock import MagicMock
from slipstream.strategies.brawler.feeds import LocalQuote
from slipstream.strategies.brawler.alpha_engine import ReplenishmentTracker

@pytest.fixture
def tracker():
    return ReplenishmentTracker(
        recovery_threshold=0.8,
        timeout_seconds=2.0
    )

def test_replenishment_success(tracker):
    """Test standard MM refill behavior (No Fear)."""
    t0 = time.time()
    
    # 1. Steady State
    q1 = LocalQuote(symbol="SOL", bid=100.0, ask=101.0, ts=t0, bid_sz=1000.0, ask_sz=1000.0)
    tracker.on_quote(q1)
    assert not tracker.active_fear['bid']
    
    # 2. Consumption (Size drops 1000 -> 200)
    q2 = LocalQuote(symbol="SOL", bid=100.0, ask=101.0, ts=t0 + 0.1, bid_sz=200.0, ask_sz=1000.0)
    tracker.on_quote(q2)
    
    # Should be tracking the bid level 100.0
    assert tracker.active_consumption['bid'] is not None
    assert tracker.active_consumption['bid']['price'] == 100.0
    assert tracker.active_consumption['bid']['pre_size'] == 1000.0
    
    # 3. Replenishment (Size returns to 900 > 800) within 1 sec
    q3 = LocalQuote(symbol="SOL", bid=100.0, ask=101.0, ts=t0 + 0.5, bid_sz=900.0, ask_sz=1000.0)
    tracker.on_quote(q3)
    
    # Should have cleared the tracking
    assert tracker.active_consumption['bid'] is None
    assert not tracker.active_fear['bid']

def test_replenishment_failure_fear_signal(tracker):
    """Test slow refill triggering Fear."""
    t0 = time.time()
    
    # 1. Steady
    q1 = LocalQuote(symbol="SOL", bid=100.0, ask=101.0, ts=t0, bid_sz=1000.0, ask_sz=1000.0)
    tracker.on_quote(q1)
    
    # 2. Consumption
    q2 = LocalQuote(symbol="SOL", bid=100.0, ask=101.0, ts=t0 + 0.1, bid_sz=200.0, ask_sz=1000.0)
    tracker.on_quote(q2)
    
    # 3. Time passes > 2.0s without refill
    q3 = LocalQuote(symbol="SOL", bid=100.0, ask=101.0, ts=t0 + 2.5, bid_sz=200.0, ask_sz=1000.0)
    tracker.on_quote(q3)
    
    # Active consumption still there? Or resolved as FAILURE?
    # Logic: If timeout happens, we flag FEAR.
    assert tracker.active_fear['bid']
    assert tracker.active_consumption['bid'] is None # reset after flagging

def test_full_consumption_level_change(tracker):
    """Test when level is fully eaten (price updates)."""
    t0 = time.time()
    q1 = LocalQuote(symbol="SOL", bid=100.0, ask=101.0, ts=t0, bid_sz=1000.0, ask_sz=1000.0)
    tracker.on_quote(q1)
    
    # Price drops to 99.0 (Full consumption of 100.0)
    # The BID moved from 100.0 to 99.0
    
    q2 = LocalQuote(symbol="SOL", bid=99.0, ask=101.0, ts=t0 + 0.1, bid_sz=500.0, ask_sz=1000.0)
    tracker.on_quote(q2)
    
    # Should NOT be tracking 100.0 anymore
    assert tracker.active_consumption['bid'] is None

def test_symmetric_fear_ask_side(tracker):
    """Test that Ask side logic works independently."""
    t0 = time.time()
    # 1000 on Ask
    q1 = LocalQuote(symbol="SOL", bid=100.0, ask=101.0, ts=t0, bid_sz=1000.0, ask_sz=1000.0)
    tracker.on_quote(q1)
    
    # Ask consumed to 200
    q2 = LocalQuote(symbol="SOL", bid=100.0, ask=101.0, ts=t0 + 0.1, bid_sz=1000.0, ask_sz=200.0)
    tracker.on_quote(q2)
    
    assert tracker.active_consumption['ask'] is not None
    
    # Timeout
    q3 = LocalQuote(symbol="SOL", bid=100.0, ask=101.0, ts=t0 + 2.5, bid_sz=1000.0, ask_sz=200.0)
    tracker.on_quote(q3)
    
    assert tracker.active_fear['ask']
    assert not tracker.active_fear['bid']

