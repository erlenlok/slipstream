
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from slipstream.strategies.brawler.engine import BrawlerEngine, QuoteDecision, AssetState
from slipstream.strategies.brawler.config import BrawlerConfig, BrawlerAssetConfig
from slipstream.strategies.brawler.connectors import HyperliquidOrderSide

@pytest.fixture
def mock_config():
    cfg = BrawlerConfig()
    asset_cfg = BrawlerAssetConfig(
        symbol="BTC",
        cex_symbol="BTCUSDT",
        max_inventory=1.0,
        order_size=0.1,
    )
    cfg.assets["BTC"] = asset_cfg
    return cfg

@pytest.fixture
def engine(mock_config):
    engine = BrawlerEngine(config=mock_config)
    engine.executor = AsyncMock()
    # Mock portfolio to always allow
    engine.portfolio = MagicMock()
    engine.portfolio.allow_quotes.return_value = True
    engine.portfolio.allow_order.return_value = True
    engine.portfolio.scale_order_size.side_effect = lambda s: s
    
    # Setup state
    engine.states["BTC"] = AssetState(config=mock_config.assets["BTC"])
    import time
    now = time.time()
    # Fake some prices
    engine.states["BTC"].cex_mid_window.append(100.0)
    engine.states["BTC"].last_cex_mid_ts = now
    engine.states["BTC"].last_local_mid_ts = now
    return engine


@pytest.mark.asyncio
async def test_inventory_overflow_long_behavior(engine):
    """Test that when inventory > max, we still quote SELL but block BUY."""
    state = engine.states["BTC"]
    state.inventory = 2.0  # Way over max of 1.0
    
    # Determine what decision would be built
    # We need to manually invoke logic or bypass the suspension check if we haven't fixed it yet.
    # To test the FIX, we expect _build_quote_decision to NOT return None.
    
    # But wait, we haven't applied the fix yet, so currently this SHOULD fail (return None)
    # verifying the "bug" exists.
    
    decision = engine._build_quote_decision(state)
    
    # Desired behavior: Should NOT be None. We want to quote to reduce position.
    assert decision is not None, "Engine should NOT suspend quoting on overflow, should allow reduce-only"
    assert decision.order_size > 0


@pytest.mark.asyncio
async def test_inventory_reduce_only_logic(engine):
    """Test the Ensure Orders logic respects the reduce-only flag internally."""
    state = engine.states["BTC"]
    state.inventory = 2.0
    
    # Manually construct a decision as if the engine generated one
    decision = QuoteDecision(
        bid_price=99.0,
        ask_price=101.0,
        half_spread=1.0,
        fair_value=100.0,
        sigma=0.01,
        gamma=0.0,
        order_size=0.1
    )
    
    await engine._ensure_orders("BTC", state, decision)
    
    # Check calls to executor
    # BUY side should be BLOCKED (no place_limit_order)
    # SELL side should be PLACED
    
    calls = engine.executor.place_limit_order.call_args_list
    print(calls)
    
    buy_calls = [c for c in calls if c[0][0].side == "buy"]
    sell_calls = [c for c in calls if c[0][0].side == "sell"]
    
    assert len(buy_calls) == 0, "Buy order should be blocked because inventory > max"
    assert len(sell_calls) == 1, "Sell order should be placed to reduce inventory"

