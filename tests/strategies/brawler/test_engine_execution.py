
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from slipstream.strategies.brawler.engine import BrawlerEngine
from slipstream.strategies.brawler.config import BrawlerConfig, BrawlerAssetConfig
from slipstream.strategies.brawler.state import AssetState, QuoteDecision, OrderSnapshot
from slipstream.strategies.brawler.connectors import HyperliquidExecutionClient, HyperliquidOrder, HyperliquidOrderSide

class TestBrawlerEngineExecution(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = BrawlerConfig()
        self.engine = BrawlerEngine(self.config)
        self.engine.executor = AsyncMock(spec=HyperliquidExecutionClient)
        self.engine.purse = MagicMock()
        self.engine.purse.request_budget = 2000
        self.engine.controller = MagicMock()
        self.engine.controller.calculate_tolerance.return_value = 10
        self.engine.portfolio = MagicMock()
        self.engine.portfolio.allow_order.return_value = True
        
        # Setup Asset State
        self.asset_config = BrawlerAssetConfig(
            symbol="SOL",
            cex_symbol="SOLUSDT",
            tick_size=0.01,
            max_inventory=10.0
        )
        self.state = AssetState(config=self.asset_config)
        self.state.inventory = 0.0

    async def test_maybe_replace_order_success(self):
        """Test happy path order placement."""
        mock_update = MagicMock()
        mock_update.order_id = "oid-123"
        self.engine.executor.place_limit_order.return_value = mock_update
        
        await self.engine._maybe_replace_order(
            symbol="SOL",
            state=self.state,
            target_price=100.0,
            side=HyperliquidOrderSide.BUY,
            size=1.0,
            is_reduce_only=False
        )
        
        self.engine.executor.place_limit_order.assert_called_once()
        self.assertIsNotNone(self.state.active_bid)
        self.assertEqual(self.state.active_bid.order_id, "oid-123")

    async def test_post_only_rejection_handled(self):
        """Test that Post-Only RuntimeError is caught and suppressed."""
        # Setup: Existing order to cancel
        self.state.active_bid = OrderSnapshot("old-oid", 99.0, 1.0, "buy")
        
        # Mock cancel to succeed
        self.engine._throttled_cancel = AsyncMock()
        
        # Mock placement to fail with Post-Only
        self.engine.executor.place_limit_order.side_effect = RuntimeError("Post only order would have immediately matched")
        
        await self.engine._maybe_replace_order(
            symbol="SOL",
            state=self.state,
            target_price=100.0,
            side=HyperliquidOrderSide.BUY,
            size=1.0,
            is_reduce_only=False
        )
        
        # Verify cancel was awaited
        self.engine._throttled_cancel.assert_called_once()
        # Verify exception suppressed (test should not fail)

    async def test_reduce_only_block_inventory(self):
        """Test logic blocks opening new positions if inventory limit hit."""
        self.state.inventory = 10.0 # Max
        self.state.config.max_inventory = 10.0
        
        # Try to buy more
        await self.engine._maybe_replace_order(
            symbol="SOL",
            state=self.state,
            target_price=100.0,
            side=HyperliquidOrderSide.BUY,
            size=1.0,
            is_reduce_only=False
        )
        
        # Verify NO placement
        self.engine.executor.place_limit_order.assert_not_called()

if __name__ == '__main__':
    unittest.main()
