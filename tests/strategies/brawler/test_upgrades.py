
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from slipstream.strategies.brawler.engine import BrawlerEngine, QuoteDecision
from slipstream.strategies.brawler.config import BrawlerConfig, BrawlerAssetConfig
from slipstream.strategies.brawler.economics import ToleranceController
from slipstream.strategies.brawler.state import AssetState

def test_tolerance_controller_method():
    """Verify calculate_spread_penalty exists and works."""
    ctrl = ToleranceController(min_tolerance_ticks=1.0)
    # Budget 100 -> 0 penalty
    assert ctrl.calculate_spread_penalty(100.0) == 0.0
    # Budget -200 -> 1bp penalty (0.0001)
    assert ctrl.calculate_spread_penalty(-200.0) == 0.0001
    # Budget -100000 -> Cap at 100bps (0.01)
    assert ctrl.calculate_spread_penalty(-100000.0) == 0.01

@patch("slipstream.strategies.brawler.engine.HyperliquidInfoClient")
def test_engine_initialization(mock_info):
    """Verify Engine initializes without crashing (imports check)."""
    cfg = BrawlerConfig()
    cfg.hyperliquid_main_wallet = "0xWallet"
    engine = BrawlerEngine(cfg)
    assert engine.reconciler is not None
    assert engine.discovery is not None

@pytest.mark.asyncio
async def test_build_quote_decision_schema():
    """Verify _build_quote_decision runs without NameErrors."""
    # Setup Engine with Mocks
    cfg = BrawlerConfig()
    # We need to mock components that _build_quote_decision uses
    engine = BrawlerEngine(cfg)
    engine.portfolio = MagicMock()
    engine.portfolio.allow_quotes.return_value = True
    engine.portfolio.scale_order_size.side_effect = lambda x: x # Pass through
    engine.controller = ToleranceController(1.0)
    engine.purse = MagicMock()
    engine.purse.request_budget = 100.0
    
    # Setup State
    asset_cfg = BrawlerAssetConfig(symbol="TEST", cex_symbol="TESTUSDT", tick_size=0.001)
    state = AssetState(asset_cfg)
    state.fair_basis = 0.1
    state.sigma = 0.01
    state.cex_mid_window = [100.0]
    setattr(state, "latest_cex_price", 100.0)
    
    # Run
    decision = engine._build_quote_decision(state)
    
    # Check
    assert decision is not None
    assert isinstance(decision, QuoteDecision)
    assert decision.bid_price < 100.1  # Less than fair value
    assert decision.ask_price > 100.1  # Greater than fair value
    # Check if half_spread was correctly calculated and passed
    assert decision.half_spread > 0
