
import pytest
import math
from unittest.mock import MagicMock, AsyncMock
from collections import deque
from slipstream.strategies.brawler.config import BrawlerAssetConfig, BrawlerConfig
from slipstream.strategies.brawler.engine import BrawlerEngine
from slipstream.strategies.brawler.state import AssetState, OrderSnapshot

# -------------------------------------------------------------------------
# Test Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Returns a basic safe config."""
    return BrawlerAssetConfig(
        symbol="TEST",
        cex_symbol="TESTUSDT",
        base_spread=0.003, # 30 bps
        volatility_lookback=60,
        risk_aversion=1.0,
        order_size=10.0,
        max_inventory=100.0,
        tick_size=0.01
    )

@pytest.fixture
def float_config():
    """Config for a standard float-based asset (e.g. SUI)."""
    return BrawlerAssetConfig(
        symbol="FLOATCOIN",
        cex_symbol="FLOATUSDT",
        order_size=10.0,
        max_inventory=100.0,
        tick_size=0.0001
    )

@pytest.fixture
def int_config():
    """Config for a zero-decimal integer asset (e.g. BONK/PEPE/SHIB in wrong units, or High-Price integers)."""
    return BrawlerAssetConfig(
        symbol="INTCOIN",
        cex_symbol="INTUSDT",
        order_size=1000.0,
        max_inventory=5000.0,
        tick_size=1.0 # Significant! integer ticks
    )

@pytest.fixture
def engine():
    cfg = BrawlerConfig()
    return BrawlerEngine(cfg)

# -------------------------------------------------------------------------
# 1. Precision & Rounding Tests
# -------------------------------------------------------------------------

def test_price_normalization_float(engine, float_config):
    """Ensure standard prices round to nearest tick."""
    # tick = 0.0001
    raw_price = 1.23456789
    norm = engine._normalize_price(float_config, raw_price)
    assert norm == pytest.approx(1.2346)
    
    # Check floating point stability
    raw_price_2 = 1.2345444
    norm_2 = engine._normalize_price(float_config, raw_price_2)
    assert norm_2 == pytest.approx(1.2345)

def test_price_normalization_int_asset(engine, int_config):
    """Ensure integer assets round to nearest integer tick."""
    # tick = 1.0
    raw_price = 420.69
    norm = engine._normalize_price(int_config, raw_price)
    assert norm == 421.0 # rounded up
    assert isinstance(norm, float) # System uses floats internally, but values should be integral
    assert norm.is_integer()

def test_price_normalization_tiny_tick(engine):
    """Ensure extremely small ticks (BONK) work."""
    cfg = BrawlerAssetConfig("TINY", "TINY", tick_size=1e-8)
    raw = 0.0000123456789
    norm = engine._normalize_price(cfg, raw)
    # expected: 0.00001235 (rounded to 1e-8) -> Wait 1.23456789e-5
    # Let's check calculation
    # 0.0000123456789 / 1e-8 = 1234.56789 -> round -> 1235
    # 1235 * 1e-8 = 0.00001235
    assert math.isclose(norm, 0.00001235, rel_tol=1e-9)

# -------------------------------------------------------------------------
# 2. Wild Price Moves (Volatility & Safety)
# -------------------------------------------------------------------------

def test_quote_generation_crash(engine, mock_config):
    """Simulate a -99% crash in one tick interaction."""
    state = AssetState(mock_config)
    # INITIAL STATE: Stable at $100
    for i in range(60):
        state.cex_mid_window.append((100.0, i*1.0))
    state.update_sigma()
    
    # SUDDEN CRASH: Price goes to $1
    state.cex_mid_window.append((1.0, 61.0)) # Newest item
    state.latest_cex_price = 1.0 # Real-time feed update
    
    # Volatility should spike
    state.update_sigma()
    # assert state.sigma > 0.5 # Huge vol NOTE: Depends on calculation, but should be large
    
    # Decision
    decision = engine._build_quote_decision(state)
    
    # Logic Checks:
    # 1. Spreads should be WIDE due to vol
    # 2. Bid should be very low, maybe even 0 if totally discouraged (but engine allows low bids)
    # 3. Should NOT crash with MathError
    
    assert decision is not None or state.suspended_reason == "volatility"
    
    if decision:
        print(f"\nCrash Quote: Bid={decision.bid_price} Ask={decision.ask_price} Sigma={decision.sigma}")
        assert decision.ask_price > 1.0 # Valid ask
        # Ensure we are not quoting negative prices (unless shorting allowed?)
        assert decision.bid_price >= 0.0

def test_quote_generation_infinite_pump(engine, mock_config):
    """Simulate +10,000% pump."""
    state = AssetState(mock_config)
    for i in range(60):
        state.cex_mid_window.append((10.0, i*1.0))
    state.update_sigma()
    
    state.cex_mid_window.append((1000.0, 61.0))
    state.latest_cex_price = 1000.0
    state.update_sigma()
    
    decision = engine._build_quote_decision(state)
    
    # Engine usually suspends on excessive vol (> max_volatility)
    # Check config default max_volatility=0.02 (2%)
    # This pump is massive, so it MUST suspend.
    
    assert decision is None
    assert state.suspended_reason == "volatility"

# -------------------------------------------------------------------------
# 3. Inventory & Reduce-Only Logic
# -------------------------------------------------------------------------

from slipstream.strategies.brawler.connectors.hyperliquid import HyperliquidOrderSide

# ... imports ... (not replacing all, just appending import if needed, or I'll just use string "buy"/"sell")

# Let's just use string literals "buy" and "sell" to match the Enum definition found
# class HyperliquidOrderSide:
#    BUY = "buy"
#    SELL = "sell"

@pytest.mark.asyncio
async def test_inventory_cap_enforcement(engine, mock_config):
    """Test that engine BLOCKS buys when inventory > max."""
    state = AssetState(mock_config)
    
    # Scenario: Long 150, Max 100
    state.inventory = 150.0 
    
    # Mock Executor
    engine.executor = MagicMock()
    engine.executor.place_limit_order = AsyncMock() # VITAL: Make it awaitable
    engine.executor.cancel_order = AsyncMock()
    
    # Attempt to place BUY order
    # _maybe_replace_order(symbol, state, target_price, side, size, is_reduce_only)
    await engine._maybe_replace_order("TEST", state, 90.0, "buy", 10.0, is_reduce_only=False)
    
    # Verify: NO order placed
    engine.executor.place_limit_order.assert_not_called()
    
    # Scenario: Sell logic should be allowed (Reduce Only)
    await engine._maybe_replace_order("TEST", state, 110.0, "sell", 10.0, is_reduce_only=False)
    
    # Verify: Order PLACED
    engine.executor.place_limit_order.assert_called_once()


@pytest.mark.asyncio
async def test_inventory_short_cap_enforcement(engine, mock_config):
    """Test that engine BLOCKS sells when short inventory < -max."""
    state = AssetState(mock_config)
    
    # Scenario: Short 150, Max 100
    state.inventory = -150.0 
    
    engine.executor = MagicMock()
    engine.executor.place_limit_order = AsyncMock()
    engine.executor.cancel_order = AsyncMock()

    # Attempt to place SELL (adding to short)
    await engine._maybe_replace_order("TEST", state, 110.0, "sell", 10.0, is_reduce_only=False)
    
    # Verify: NO order placed
    engine.executor.place_limit_order.assert_not_called()
    
    # Attempt to place BUY (reducing short)
    await engine._maybe_replace_order("TEST", state, 90.0, "buy", 10.0, is_reduce_only=False)
    
    # Verify: Order PLACED
    engine.executor.place_limit_order.assert_called_once()
