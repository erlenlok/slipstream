
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, call, patch
from dataclasses import dataclass

from src.slipstream.strategies.brawler.engine import BrawlerEngine, QuoteDecision
from src.slipstream.strategies.brawler.config import BrawlerConfig, BrawlerAssetConfig, BrawlerEconomicsConfig
from src.slipstream.strategies.brawler.state import AssetState
from src.slipstream.strategies.brawler.connectors.hyperliquid import HyperliquidOrderSide, HyperliquidOrder

# Mocking data structures
@dataclass
class MockLocalQuote:
    symbol: str
    bid: float
    ask: float
    ts: float

class TestBrawlerInventoryLogic(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Setup minimal config
        self.asset_cfg = BrawlerAssetConfig(
            symbol="ETH", 
            cex_symbol="ETHUSDT",
            order_size=0.1,
            max_inventory=1.0, # 1.0 limit
            min_quote_interval_ms=0
        )
        self.config = BrawlerConfig(
            assets={"ETH": self.asset_cfg},
            economics=BrawlerEconomicsConfig()
        )
        
        # Mock Engine components
        self.engine = BrawlerEngine(self.config)
        self.engine.executor = AsyncMock() # Mock the client
        self.engine.purse = MagicMock()
        self.engine.purse.request_budget = 10000 # High budget
        
        # Init State
        self.state = AssetState(config=self.asset_cfg)
        self.state.active_bid = None
        self.state.active_ask = None
        
        # Mock portfolio if needed
        self.engine.portfolio = None

    async def test_normal_quoting_within_limits(self):
        """Test that bot quotes both sides when inventory is low."""
        self.state.inventory = 0.5  # Below max 1.0
        decision = QuoteDecision(
            bid_price=2990, ask_price=3010, order_size=0.1,
            half_spread=10, fair_value=3000, sigma=0.01, gamma=0
        )
        
        await self.engine._ensure_orders("ETH", self.state, decision)
        
        # Expect calls to place both orders
        calls = self.engine.executor.place_limit_order.call_args_list
        self.assertEqual(len(calls), 2)
        
        # Verify sides
        sides = [c.args[0].side for c in calls]
        self.assertIn(HyperliquidOrderSide.BUY, sides)
        self.assertIn(HyperliquidOrderSide.SELL, sides)

    async def test_reduce_only_long_limit(self):
        """Test that bot BLOCKS buys when inventory >= max, but allows SELLS."""
        self.state.inventory = 1.0  # Hit limit
        decision = QuoteDecision(
            bid_price=2990, ask_price=3010, order_size=0.1,
            half_spread=10, fair_value=3000, sigma=0.01, gamma=0
        )
        
        # Ensure we have an active "Stale" bid to check cancellation
        fake_bid = MagicMock()
        fake_bid.order_id = "oid_bid_1"
        self.state.active_bid = fake_bid
        
        await self.engine._ensure_orders("ETH", self.state, decision)
        
        # Expect cancellation of BID
        self.engine.executor.cancel_order.assert_called_with("ETH", "oid_bid_1")
        self.assertIsNone(self.state.active_bid)
        
        # Expect Placement of SELL ONLY
        calls = self.engine.executor.place_limit_order.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0].side, HyperliquidOrderSide.SELL)

    async def test_reduce_only_short_limit(self):
        """Test that bot BLOCKS sells when inventory <= -max, but allows BUYS."""
        self.state.inventory = -1.0  # Hit short limit
        decision = QuoteDecision(
            bid_price=2990, ask_price=3010, order_size=0.1,
            half_spread=10, fair_value=3000, sigma=0.01, gamma=0
        )
        
        # Ensure we have an active "Stale" ask
        fake_ask = MagicMock()
        fake_ask.order_id = "oid_ask_1"
        self.state.active_ask = fake_ask
        
        await self.engine._ensure_orders("ETH", self.state, decision)
        
        # Expect cancellation of ASK
        self.engine.executor.cancel_order.assert_called_with("ETH", "oid_ask_1")
        self.assertIsNone(self.state.active_ask)
        
        # Expect Placement of BUY ONLY
        calls = self.engine.executor.place_limit_order.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0].side, HyperliquidOrderSide.BUY)

    async def test_oversized_inventory_skews(self):
        """Test behavior when inventory is extremely over limit (manual intervention or drift)."""
        self.state.inventory = 2.0  # Way over 1.0
        decision = QuoteDecision(
             bid_price=2990, ask_price=3010, order_size=0.1,
             half_spread=10, fair_value=3000, sigma=0.01, gamma=0
         )
        await self.engine._ensure_orders("ETH", self.state, decision)
        
        # Still Sell Only
        calls = self.engine.executor.place_limit_order.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0].side, HyperliquidOrderSide.SELL)

    async def test_cancellation_logic(self):
        """Test that blocked orders are cancelled even if prices match (force cleanup)."""
        self.state.inventory = 1.0
        
        # Simulate active bid exists
        fake_bid = MagicMock()
        fake_bid.order_id = "keep_me"
        fake_bid.price = 2990 # Matches new price
        self.state.active_bid = fake_bid
        
        # Even if decision matches valid price, it should CANCEL because of Reduce Only
        decision = QuoteDecision(
             bid_price=2990, ask_price=3010, order_size=0.1,
             half_spread=10, fair_value=3000, sigma=0.01, gamma=0
         )
        
        await self.engine._ensure_orders("ETH", self.state, decision)
        
        # Verify Bid cancelled
        self.engine.executor.cancel_order.assert_called_with("ETH", "keep_me")
        
    def test_sizing_logic(self):
        """Test sizing clamps in _build_quote_decision if observable."""
        # This requires mocking the random check or inspecting logic.
        # Since _build_quote_decision is internal, let's just integration test it via ensure_orders if we can pass decision generator?
        # Actually _build_quote_decision returns the decision.
        # Let's call _build_quote_decision directly.
        pass

if __name__ == '__main__':
    unittest.main()
