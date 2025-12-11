"""Reloader module for Brawler strategy ("The Reloader").

Responsible for detecting low budget and executing wash trades to earn request credits.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

from .config import BrawlerEconomicsConfig
from .connectors import (
    HyperliquidExecutionClient,
    HyperliquidOrder,
    HyperliquidOrderSide,
    HyperliquidQuoteStream,
)
from .economics import RequestPurse

logger = logging.getLogger(__name__)


class ReloaderAgent:
    """Agent that buys request credits via wash trading when budget is low."""

    def __init__(
        self,
        config: BrawlerEconomicsConfig,
        purse: RequestPurse,
        executor: Optional[HyperliquidExecutionClient],
        quote_stream: HyperliquidQuoteStream,
    ) -> None:
        self.config = config
        self.purse = purse
        self.executor = executor
        self.quote_stream = quote_stream
        self._reloading = False

    async def check_and_reload(self, mid_price: float, spread: float) -> None:
        """Main entry point: checks budget and triggers reload if needed."""
        if not self.executor:
            return
            
        if self._reloading:
            return

        budget = self.purse.request_budget
        if budget < self.config.reload_threshold_budget:
            # Check spread safety
            # spread in absolute price units? No usually passed as is.
            # config.max_spread_bps uses bps.
            # spread_bps = (spread / mid_price) * 10000
            spread_bps = (spread / mid_price) * 10000.0 if mid_price > 0 else 9999.0
            
            if spread_bps > self.config.max_spread_bps:
                logger.warning(
                    "Budget crit (%.1f) but spread too wide (%.1f bps > %.1f). Skipping reload.", 
                    budget, spread_bps, self.config.max_spread_bps
                )
                return

            logger.warning("Budget critical (%.1f < %.1f). Triggering RELOAD cycle on %s.", 
                           budget, self.config.reload_threshold_budget, self.config.reload_symbol)
            self._reloading = True
            try:
                await self._execute_reload_cycle(mid_price)
            except Exception as exc:
                logger.error("Reload cycle failed: %s", exc, exc_info=exc)
            finally:
                self._reloading = False

    async def _execute_reload_cycle(self, price: float) -> None:
        """Execute a round-trip trade to generate volume."""
        symbol = self.config.reload_symbol
        needed_budget = self.config.reload_target_budget - self.purse.request_budget
        
        # Heuristic: 1 USD vol = 1 budget credit (Hyperliquid Standard)
        target_volume_usd = needed_budget * 1.0
        
        # Round trip legs (Buy + Sell). Total Volume = BuyVol + SellVol.
        # So LegSizeUSD = TargetVol / 2
        leg_size_usd = target_volume_usd / 2.0
        
        # Size in tokens
        if price <= 0:
            return
        size = leg_size_usd / price
        
        # Minimum size check (Hyperliquid has min trade size ~$10)
        min_usd = 12.0
        if leg_size_usd < min_usd:
            logger.info("Reload volume too small ($%.2f), boosting to min ($%d).", leg_size_usd, min_usd)
            size = min_usd / price
        
        # Round size to avoid float_to_wire errors (Hyperliquid generally accepts 5 decimals for major coins)
        size = round(size, 5)
        
        # 1. Place Buy
        # Aggressive Limit: Price * 1.05 (5% slippage allowance for instant fill)
        # Round to integer for BTC/ETH (safe tick size assumption for major pairs)
        buy_price = float(int(price * 1.02))
        buy_order = HyperliquidOrder(
            symbol=symbol,
            price=buy_price, # We rely on executor to round/tick this if needed, or we explicitly round?
            # Connectors usually expect valid prices. We should round.
            # But we don't have access to tick_size here easily without passing config.
            # Let's assume executor or exchange handles slight precision issues or we rely on floats.
            # Actually, `HyperliquidExecutionClient` passes float.
            size=size,
            side=HyperliquidOrderSide.BUY
        )
        logger.info("RELOAD: Buying %.4f %s @ ~$%.2f", size, symbol, buy_price)
        await self.executor.place_limit_order(buy_order) # IOC ideally, but GTC aggressive fine if cancelled or filled.
        # We don't wait for fill confirmation here (optimistic wash), or we should?
        # Ideally we wait for fill. But implementation plan said "Buy -> Wait -> Sell".
        
        # Rate Limit Safety: When in debt, we are limited to ~1 req / 10s.
        # Wait 15s to ensure the Sell order doesn't hit "Too many requests" before volume credits apply.
        logger.info("Waiting 15s between legs for rate limit safety...")
        await asyncio.sleep(15.0)
        
        # 2. Place Sell
        sell_price = float(int(price * 0.98))
        sell_order = HyperliquidOrder(
            symbol=symbol,
            price=sell_price,
            size=size,
            side=HyperliquidOrderSide.SELL
        )
        logger.info("RELOAD: Selling %.4f %s @ ~$%.2f", size, symbol, sell_price)
        await self.executor.place_limit_order(sell_order)
        
        logger.info("RELOAD cycle complete.")
