
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from slipstream.strategies.brawler.engine import BrawlerEngine
from slipstream.strategies.brawler.config import BrawlerConfig, BrawlerAssetConfig, BrawlerEconomicsConfig

@pytest.fixture
def mock_config():
    econ = BrawlerEconomicsConfig(
        cost_per_request_usd=1.0, # High cost to drain fast
        reload_threshold_budget=10.0,
        reload_target_budget=100.0,
        reload_symbol="BTC",
        tolerance_dilation_k=1000.0,
    )
    asset = BrawlerAssetConfig(symbol="BTC", cex_symbol="btcusdt")
    return BrawlerConfig(
        assets={"BTC": asset},
        economics=econ,
        hyperliquid_main_wallet="0xMOCK"
    )

class TestOptimalQuotingCycle:
    
    @pytest.mark.asyncio
    async def test_full_economic_cycle(self, mock_config):
        """
        Verify: Rich -> Poor -> Reload -> Recovery
        """
        # 1. Setup Engine using mocks
        with patch('slipstream.strategies.brawler.engine.HyperliquidInfoClient') as MockInfoClient, \
             patch('slipstream.strategies.brawler.engine.HyperliquidQuoteStream') as MockQuoteStream:
            
            engine = BrawlerEngine(mock_config)
            
            # Mock Executor for Reloader
            engine.executor = AsyncMock()
            engine.reloader.executor = engine.executor # Link mock to reloader
            
            # PHASE 1: RICH
            # Inject high volume credit manually
            # 1 USD Vol = 0.001 credit (heuristic in code)
            # We want budget > 100. Target=100 in config.
            # Let's give it 200 budget. 200 / 0.001 = 200,000 Volume
            engine.purse.add_fill_credit(200_000.0)
            
            # Budget = 200. Cost = 0.
            assert engine.purse.request_budget == 200.0
            
            # Check Tolerance
            # T = max(1.0, 1000 / 200 = 5.0) -> 5.0 ticks.
            # Wait, default min might be 1.0. 
            # Controller internal min is 1.0. Asset config defaults to 1.0.
            # So expected is 5.0.
            tol_rich = engine.controller.calculate_tolerance(200.0)
            assert tol_rich == 5.0
            
            # PHASE 2: DRAIN
            # We need to drop budget below 10.0
            # 200 - 10 = 190 budget units to burn.
            # Cost per request = 1.0
            # Need 191 requests.
            for _ in range(191):
                engine.purse.deduct_request()
                
            current_budget = engine.purse.request_budget
            assert current_budget <= 9.0
            
            # PHASE 3: RELOAD TRIGGER
            # Check reload logic
            # We need to simulate market state for spread check
            # engine.states['BTC'].update_bbo(99.0, 100.0, ts) -> Spread 1.0. Mid 99.5.
            # Spread bps = 1.0 / 99.5 ~ 1%. Max is 5bps. Too wide!
            # Let's set tight spread. 100.00, 100.01. Spread 0.01.
            engine.states["BTC"].update_bbo(100.00, 100.01, 123456789.0)
            
            # Call monitor manually (bypass infinite loop)
            # await engine._monitor_reload_needs() --> Replicating logic:
            state = engine.states["BTC"]
            mid = (state.best_bid + state.best_ask) / 2.0
            spread = abs(state.best_ask - state.best_bid)
            await engine.reloader.check_and_reload(mid, spread)
            
            # Verify Executor was called
            assert engine.executor.place_limit_order.call_count == 2 # Buy + Sell
            
            # PHASE 4: RECOVERY
            # Verify that AFTER the reload cycle (which we mocked), we would have volume.
            # The test here just mocks the updated volume from "exchange" or manual fill credit.
            # In real ReloadAgent, we wait for fills properly or rely on subsequent sync.
            # Here we simulate the effect:
            # Target Budget 100. Shortfall was ~91 (from 9 to 100).
            # Agent should have traded enough volume to get back to 100.
            # Let's manually add that expected volume to verify "Recovery" state helper.
            engine.purse.add_fill_credit(91_000.0) 
            
            assert engine.purse.request_budget >= 100.0
            
            # Check Tolerance reset
            # Budget ~100. T = 1000 / 100 = 10.0. 
            # It's better than Critical/Survival.
            tol_recovered = engine.controller.calculate_tolerance(100.0)
            assert tol_recovered == 10.0
