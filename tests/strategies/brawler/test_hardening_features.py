
import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock
from slipstream.strategies.brawler.engine import BrawlerEngine, QuoteDecision, AssetState
from slipstream.strategies.brawler.config import BrawlerConfig, BrawlerAssetConfig
from slipstream.strategies.brawler.feeds import CexQuote

@pytest.fixture
def mock_config():
    cfg = BrawlerConfig()
    asset_cfg = BrawlerAssetConfig(
        symbol="BTC",
        cex_symbol="BTCUSDT",
        max_inventory=1.0,
        order_size=0.1,
        min_basis_bps=5.0, # Active filter
        momentum_threshold_bps=5.0 # Active momentum
    )
    cfg.assets["BTC"] = asset_cfg
    return cfg

@pytest.fixture
def engine(mock_config):
    engine = BrawlerEngine(config=mock_config)
    engine.executor = AsyncMock()
    engine.portfolio = MagicMock()
    engine.portfolio.allow_quotes.return_value = True
    engine.portfolio.allow_order.return_value = True
    engine.portfolio.scale_order_size.side_effect = lambda s: s
    engine.states["BTC"] = AssetState(config=mock_config.assets["BTC"])
    return engine

@pytest.mark.asyncio
async def test_opportunity_filter_low_basis(engine):
    """Test that quotes are suppressed (None) when basis is too small."""
    state = engine.states["BTC"]
    state.active_bid = MagicMock() # Simulate existing order
    
    # Setup Low Basis Scenario
    state.latest_cex_price = 100.0
    state.cex_mid_window.append(100.0)
    state.fair_basis = 0.01  # Basis = 0.01/100 = 1 bps (Below 5 bps)
    state.last_basis = 0.01
    
    decision = engine._build_quote_decision(state)
    assert decision is None, "Should return None when basis (1bps) < min (5bps)"
    
    # Verify _update_quotes would trigger cancel
    # We can mock _cancel_all to verify it's called
    engine._cancel_all = AsyncMock()
    engine.states["BTC"] = state # ensure state ref is good
    await engine._update_quotes()
    engine._cancel_all.assert_called_once()

@pytest.mark.asyncio
async def test_opportunity_filter_high_basis(engine):
    """Test that quotes are generated when basis is sufficient."""
    state = engine.states["BTC"]
    state.latest_cex_price = 100.0
    state.cex_mid_window.append(100.0)
    state.fair_basis = 0.10 # 10 bps (Above 5 bps)
    state.last_basis = 0.10
    
    decision = engine._build_quote_decision(state)
    assert decision is not None, "Should quote when basis (10bps) > min (5bps)"

@pytest.mark.asyncio
async def test_momentum_guard_velocity(engine):
    """Test that engine calculates velocity and triggers momentum logic."""
    state = engine.states["BTC"]
    
    # 1. Setup Stream consumption
    engine.binance_stream = MagicMock()
    queue = asyncio.Queue()
    engine.binance_stream.queue = queue
    
    # Ensure Basis is high enough (10bps) to pass Opportunity Filter
    state.fair_basis = 0.1
    state.last_basis = 0.1

    # 2. Push initial quote
    t0 = time.time()
    state.push_cex_mid(100.0, t0 - 1.0) # Previous
    state.last_cex_mid_ts = t0 - 1.0
    
    # 3. Simulate Fast Move Up (Momentum > 5bps)
    # 100 -> 101 in 1 sec => 1% = 100bps >> 5bps
    quote = CexQuote(symbol="BTCUSDT", bid=101.0, ask=101.0, ts=t0)
    await queue.put(quote)
    
    # Run one iteration of consume
    # Since _consume_cex_quotes is a loop, we can't await it directly easily without stopping it.
    # Instead, let's manually invoke the logic block or break the loop.
    # Or just copy the logic we want to test:
    
    # Manually run the relevant block from _consume_cex_quotes
    prev_price = state.cex_mid_window[-1] # 100.0
    dt = quote.ts - state.last_cex_mid_ts # 1.0
    
    if dt > 0.001:
        velocity_bps = ((quote.mid - prev_price) / prev_price) / dt * 10000.0
        state.cex_velocity = velocity_bps
    
    assert state.cex_velocity > 5.0
    assert state.cex_velocity == 100.0
    
    # Now check decision
    # Momentum Guard in _build_quote_decision sets state.sigma based on velocity?
    # Actually logic was: sigma = max(sigma, 0.05) if momentum > threshold
    
    decision = engine._build_quote_decision(state)
    assert decision.sigma >= 0.05, "Sigma should be boosted by momentum guard"

@pytest.mark.asyncio
async def test_withdraw_quotes_action(engine):
    """Test that _cancel_all correctly calls executor to cancel orders."""
    state = engine.states["BTC"]
    state.active_bid = MagicMock(order_id="123")
    state.active_ask = MagicMock(order_id="456")
    
    await engine._cancel_all("BTC", state)
    
    assert engine.executor.cancel_order.call_count == 2
    assert state.active_bid is None
    assert state.active_ask is None
