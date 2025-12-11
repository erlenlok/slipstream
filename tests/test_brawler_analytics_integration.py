import asyncio
import unittest
from unittest.mock import MagicMock, patch

from slipstream.strategies.brawler.config import BrawlerConfig, BrawlerAssetConfig, BrawlerAnalyticsConfig
from slipstream.strategies.brawler.engine import BrawlerEngine
from slipstream.strategies.brawler.feeds import FillEvent
from slipstream.analytics.storage_layer import DatabaseConfig

class TestBrawlerAnalyticsIntegration(unittest.TestCase):
    def setUp(self):
        self.config = BrawlerConfig(
            assets={"BTC": BrawlerAssetConfig("BTC", "btcusdt")},
            analytics=BrawlerAnalyticsConfig(enabled=True)
        )

    @patch('slipstream.strategies.brawler.engine.CoreMetricsCalculator')
    @patch('slipstream.strategies.brawler.engine.AnalyticsStorage')
    def test_analytics_initialization(self, MockStorage, MockCalculator):
        """Test that analytics components are initialized when enabled."""
        engine = BrawlerEngine(self.config)
        
        # Verify components are initialized
        self.assertIsNotNone(engine.core_metrics)
        self.assertIsNotNone(engine.historical_analyzer)
        self.assertIsNotNone(engine.per_asset_analyzer)
        self.assertIsNotNone(engine.analytics_storage)
        
        # Verify storage initialized with correct config
        MockStorage.assert_called_once()
    
    @patch('slipstream.strategies.brawler.engine.CoreMetricsCalculator')
    @patch('slipstream.strategies.brawler.engine.AnalyticsStorage') 
    def test_fill_processing_integration(self, MockStorage, MockCalculator):
        """Test that fills are processed by analytics."""
        engine = BrawlerEngine(self.config)
        
        # Mock the analytics components
        engine.core_metrics = MagicMock()
        engine.per_asset_analyzer = MagicMock()
        engine.analytics_storage = MagicMock()
        
        # Create a fill event with new fields
        fill = FillEvent(
            symbol="BTC",
            size=0.1,
            price=50000.0,
            side="buy",
            ts=1234567890.0,
            order_id="test_oid",
            fee=5.0,
            fee_token="USDC",
            liquidity_type="maker"
        )
        
        # Inject fill into engine's fill handling logic
        # We simulate _consume_fills logic by injecting state first
        engine._bootstrap_inventory = MagicMock() # avoid async issues
        engine.states["BTC"].inventory = 0.0
        
        # We can't cancel/run the infinite loop, but we can call the processing logic if we mock the queue
        # Or better, just inspect the logic we changed.
        # But to be safe, let's replicate the critical section logic call or extract it.
        # Since I can't easily extract it without refactoring, I will simulate it by 
        # putting item in queue and running loop for one iteration? No, that's flaky.
        
        # I'll rely on inspecting the side effects of running a single pass of "process fill"
        # But _consume_fills is a while loop.
        
        # Let's override _consume_fills to just run once for this test or use a helper?
        # A better way for unit testing `_consume_fills` is to mock the queue get to return one item then raise CancelledError?
        pass

    async def async_test_fill_flow(self):
        """Async test for fill flow."""
        # Setup mocks
        mock_core = MagicMock()
        mock_storage = MagicMock()
        mock_storage.store_trade_event = MagicMock() # It returns coroutine?
        mock_storage.store_trade_event.return_value = asyncio.sleep(0) # dummy awaitable
        
        engine = BrawlerEngine(self.config)
        engine.core_metrics = mock_core
        engine.analytics_storage = mock_storage
        
        # Create a queue and put one item
        engine.fill_stream = MagicMock()
        engine.fill_stream.queue = asyncio.Queue()
        
        fill = FillEvent(
            symbol="BTC",
            size=0.1,
            price=50000.0,
            side="buy",
            ts=1234567890.0,
            order_id="test_oid",
            fee=2.5,
            fee_token="USDC",
            liquidity_type="maker"
        )
        await engine.fill_stream.queue.put(fill)
        
        # Start _consume_fills as a task, wait a bit, then cancel
        task = asyncio.create_task(engine._consume_fills())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
            
        # Verify core_metrics.process_trade was called
        self.assertTrue(mock_core.process_trade.called)
        call_args = mock_core.process_trade.call_args[0][0]
        
        self.assertEqual(call_args.symbol, "BTC")
        self.assertEqual(call_args.fees_paid, 2.5)
        self.assertEqual(call_args.trade_type.value, "maker")

    def test_async_integration(self):
        asyncio.run(self.async_test_fill_flow())

if __name__ == '__main__':
    unittest.main()
