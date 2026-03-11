
import unittest
from unittest.mock import MagicMock
from slipstream.strategies.brawler.strategy import StandardBrawlerStrategy, QuoteDecision
from slipstream.strategies.brawler.state import AssetState
from slipstream.strategies.brawler.config import BrawlerAssetConfig, BrawlerEconomicsConfig, BrawlerRiskConfig
from slipstream.strategies.brawler.economics import RequestPurse, ToleranceController
from slipstream.strategies.brawler.alpha_engine import AlphaEngine

class TestStandardBrawlerStrategy(unittest.TestCase):
    def test_calculate_quote_signature(self):
        """Verify calculate_quote accepts all required arguments and returns valid decision."""
        
        # Setup Mocks
        strategy = StandardBrawlerStrategy()
        
        # Mock Configs
        econ_config = BrawlerEconomicsConfig(max_spread_bps=500.0)
        asset_config = BrawlerAssetConfig(
            symbol="SOL", 
            cex_symbol="SOLUSDT",
            order_size=1.0, 
            base_spread=10.0,
            tick_size=0.01
        )
        
        # Mock State
        state = AssetState(config=asset_config)
        state.latest_cex_price = 100.0
        state.active_bid = None
        state.active_ask = None
        state.inventory = 0.0
        state.sigma = 0.0
        
        # Mock Dependencies
        alpha_engine = MagicMock(spec=AlphaEngine)
        alpha_engine.states = {} # Return empty dict for get()
        
        purse = MagicMock(spec=RequestPurse)
        purse.request_budget = 2000
        
        controller = MagicMock(spec=ToleranceController)
        controller.calculate_spread_penalty.return_value = 0.0
        
        # Execute
        decision = strategy.calculate_quote(
            state=state,
            alpha_engine=alpha_engine,
            purse=purse,
            controller=controller,
            economics=econ_config
        )
        
        # Verify
        self.assertIsNotNone(decision, "Strategy returned None unexpectedly")
        self.assertIsInstance(decision, QuoteDecision)
        self.assertGreater(decision.bid_price, 0)
        self.assertGreater(decision.ask_price, decision.bid_price)
        print(f"Decision: Bid={decision.bid_price}, Ask={decision.ask_price}, Size={decision.order_size}")

if __name__ == '__main__':
    unittest.main()
