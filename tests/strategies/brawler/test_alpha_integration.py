
import pytest
from unittest.mock import MagicMock, AsyncMock
from slipstream.strategies.brawler.engine import BrawlerEngine, AssetState
from slipstream.strategies.brawler.config import BrawlerConfig, BrawlerAssetConfig
from slipstream.strategies.brawler.feeds import LocalQuote
import time

@pytest.fixture
def engine():
    cfg = BrawlerConfig()
    cfg.assets["SOL"] = BrawlerAssetConfig(symbol="SOL", cex_symbol="SOLUSDT")
    
    engine = BrawlerEngine(config=cfg)
    engine.executor = AsyncMock()
    # Mock states
    engine.states["SOL"] = AssetState(config=cfg.assets["SOL"])
    engine.states["SOL"].cex_mid_window.append(100.0)
    engine.states["SOL"].last_cex_mid_ts = time.time()
    
    return engine

@pytest.mark.asyncio
async def test_alpha_fear_suspension(engine):
    """Test that Fear Signal suspends quoting."""
    symbol = "SOL"
    state = engine.states[symbol]
    
    # 1. Steady State
    t0 = time.time()
    q1 = LocalQuote(symbol=symbol, bid=100.0, ask=101.0, ts=t0, bid_sz=1000.0, ask_sz=1000.0)
    engine.alpha_engine.on_local_quote(q1)
    
    # Check decision -> Should be valid (Mock valid conditions)
    # We need to ensure feed_suspension_reason is None
    state.last_local_mid_ts = t0 # Fresh
    
    decision = engine._build_quote_decision(state)
    assert decision is not None, "Should quote in steady state"
    
    # 2. Trigger Fear Logic (Consumption + Timeout)
    # Convert to standard flow:
    # A. Consumption
    q2 = LocalQuote(symbol=symbol, bid=100.0, ask=101.0, ts=t0 + 0.1, bid_sz=200.0, ask_sz=1000.0)
    engine.alpha_engine.on_local_quote(q2)
    
    # B. Timeout (Fear Triggered on BID)
    # Start: bid=100, ask=101.
    # q2 consumption at t=3.0? No, wait.
    # Our test setups in q2 was t0+0.1.
    # We update t to t0+3.0.
    q3 = LocalQuote(symbol=symbol, bid=100.0, ask=101.0, ts=t0 + 3.0, bid_sz=200.0, ask_sz=1000.0)
    engine.alpha_engine.on_local_quote(q3)
    
    alpha_state = engine.alpha_engine.states[symbol]
    assert alpha_state.fear_side == 'bid', "Fear side should be 'bid'"
    
    # 3. Check Decision -> Should be VALID but with BID=0.0
    state.last_local_mid_ts = t0 + 3.0 
    state.last_cex_mid_ts = t0 + 3.0
    
    state.fair_basis = 0.0
    state.latest_cex_price = 100.5
    
    decision = engine._build_quote_decision(state)
    assert decision is not None, "Should NOT suspend fully"
    assert decision.bid_price == 0.0, "Bid should be cancelled (0.0)"
    assert decision.ask_price > 0.0, "Ask should be active"
    
    # Also verify that _ensure_orders would call replace with 0.0 -> Cancel
    # We can't easily mock the internal Ensure call unless we run the loop step,
    # but proving _build_quote_decision returns 0.0 + our code audit of _maybe_replace_order
    # confirms the chain.

