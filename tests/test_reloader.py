
import pytest
from unittest.mock import AsyncMock, MagicMock
from slipstream.strategies.brawler.economics import RequestPurse
from slipstream.strategies.brawler.config import BrawlerEconomicsConfig
from slipstream.strategies.brawler.reloader import ReloaderAgent
from slipstream.strategies.brawler.connectors import HyperliquidOrderSide

@pytest.fixture
def mock_executor():
    return AsyncMock()

@pytest.fixture
def mock_stream():
    return MagicMock()

@pytest.fixture
def purse():
    return RequestPurse(cost_per_request=0.00035)

@pytest.fixture
def config():
    return BrawlerEconomicsConfig(
        reload_threshold_budget=100.0,
        reload_target_budget=5000.0,
        reload_symbol="BTC",
        max_spread_bps=5.0
    )

class TestReloaderAgent:
    
    @pytest.mark.asyncio
    async def test_trigger_condition(self, config, purse, mock_executor, mock_stream):
        agent = ReloaderAgent(config, purse, mock_executor, mock_stream)
        
        # Budget is 0 (start)
        assert purse.request_budget == 0.0
        
        # Threshold is 100, so 0 < 100 -> Should trigger
        # Mock price and spread
        mid_price = 50000.0
        spread = 1.0 # very tight
        
        await agent.check_and_reload(mid_price, spread)
        
        assert mock_executor.place_limit_order.call_count == 2 # Buy then Sell
        
        # Verify call args
        calls = mock_executor.place_limit_order.call_args_list
        buy_order = calls[0][0][0]
        sell_order = calls[1][0][0]
        
        assert buy_order.side == HyperliquidOrderSide.BUY
        assert sell_order.side == HyperliquidOrderSide.SELL
        assert buy_order.symbol == "BTC"

    @pytest.mark.asyncio
    async def test_spread_safety(self, config, purse, mock_executor, mock_stream):
        agent = ReloaderAgent(config, purse, mock_executor, mock_stream)
        
        # Budget critical
        assert purse.request_budget == 0.0
        
        # Spread too wide
        mid_price = 50000.0
        # Max bps 5.0 -> 0.05% -> 50000 * 0.0005 = 25.0
        # Let's set spread to 30.0
        spread = 30.0 
        
        await agent.check_and_reload(mid_price, spread)
        
        # Should NOT trigger
        assert mock_executor.place_limit_order.call_count == 0

    @pytest.mark.asyncio
    async def test_sizing_logic(self, config, purse, mock_executor, mock_stream):
        agent = ReloaderAgent(config, purse, mock_executor, mock_stream)
        
        # Needed budget: 5000 - 0 = 5000
        # Target Volume USD: 5000 * 1000 = 5,000,000
        # Leg Size USD: 2,500,000
        # Price: 50,000
        # Size Tokens: 2,500,000 / 50,000 = 50.0
        
        await agent.check_and_reload(50000.0, 5.0)
        
        calls = mock_executor.place_limit_order.call_args_list
        buy_order = calls[0][0][0]
        
        # Target Vol 5M / 2 legs = 2.5M USD. Price 50k. Size = 50.0
        assert abs(buy_order.size - 50.0) < 0.1
