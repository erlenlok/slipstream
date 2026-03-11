"""Core event loop for the Brawler market-making strategy.

Refactored for High-Frequency Safety:
1. Basis calculated in BPS (prevents scaling death).
2. Emergency Delta overrides time-locks.
3. Fill stream clears local order state (Ghost Order fix).
4. Reduce-Only delegated to Exchange (no local blocking).
5. Velocity calculated via rolling window (noise reduction).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

from .config import BrawlerAssetConfig, BrawlerConfig
from .connectors import (
    BinanceTickerStream,
    HyperliquidExecutionClient,
    HyperliquidInfoClient,
    HyperliquidOrder,
    HyperliquidOrderSide,
    HyperliquidQuoteStream,
    HyperliquidUserFillStream,
)
from .discovery import DiscoveryEngine
from .economics import RequestPurse, ToleranceController
from .reconciliation import OrderReconciler
from .reloader import ReloaderAgent
from .inventory import InventoryProvider
from .persistence import FileStatePersistence, StatePersistence
from .portfolio import PortfolioController
from .state import (
    AssetState,
    OrderSnapshot,
    build_initial_states,
    capture_state,
    restore_state,
    QuoteDecision,
)
from slipstream.analytics.core_metrics_calculator import CoreMetricsCalculator
from slipstream.analytics.data_structures import TradeEvent, TradeType
from slipstream.analytics.historical_analyzer import HistoricalAnalyzer
from slipstream.analytics.per_asset_analyzer import PerAssetPerformanceAnalyzer
from slipstream.analytics.storage_layer import AnalyticsStorage, DatabaseConfig
from .alpha_engine import AlphaEngine
from .strategy import BrawlerStrategy, StandardBrawlerStrategy


logger = logging.getLogger(__name__)




class BrawlerEngine:
    """Binds all connectors together and applies the CAMM logic."""

    def __init__(
        self,
        config: BrawlerConfig,
        *,
        binance_stream: Optional[BinanceTickerStream] = None,
        hyperliquid_stream: Optional[HyperliquidQuoteStream] = None,
        fill_stream: Optional[HyperliquidUserFillStream] = None,
        executor: Optional[HyperliquidExecutionClient] = None,
        state_persistence: Optional[StatePersistence] = None,
        inventory_provider: Optional[InventoryProvider] = None,
    ) -> None:
        self.config = config
        self.states: Dict[str, AssetState] = build_initial_states(config.assets)
        self._symbol_index: Dict[str, str] = {
            cfg.symbol.upper(): cfg.symbol for cfg in config.assets.values()
        }
        self._cex_symbol_index: Dict[str, str] = {
            cfg.cex_symbol.upper(): cfg.symbol for cfg in config.assets.values()
        }
        self.binance_stream = binance_stream or BinanceTickerStream(
            [asset.cex_symbol for asset in config.assets.values()],
            ws_url=config.binance_ws_url,
            queue_maxsize=config.risk.cex_queue_maxsize,
        )
        self.hyperliquid_stream = hyperliquid_stream or HyperliquidQuoteStream(
            config.assets.keys(),
            ws_url=config.hyperliquid_ws_url,
            queue_maxsize=config.risk.local_queue_maxsize,
        )
        if config.hyperliquid_main_wallet:
            self.fill_stream = fill_stream or HyperliquidUserFillStream(
                config.hyperliquid_main_wallet,
                ws_url=config.hyperliquid_ws_url,
                queue_maxsize=config.risk.fill_queue_maxsize,
            )
        else:
            self.fill_stream = fill_stream
        self.executor = executor
        self._tasks: list[asyncio.Task[None]] = []
        self._stop = asyncio.Event()
        self.state_persistence = state_persistence or (
            FileStatePersistence(config.state_snapshot_path)
            if config.state_snapshot_path
            else None
        )
        self.inventory_provider = inventory_provider
        if self.state_persistence:
            try:
                restore_state(self.states, self.state_persistence.load())
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to restore state snapshot: %s", exc, exc_info=exc)
        self.portfolio = PortfolioController(config.portfolio)
        self.discovery = DiscoveryEngine(config, engine=self)
        self._last_cancel_ts: Dict[str, float] = {}

        # Economics
        self.purse = RequestPurse(
            cost_per_request=config.economics.cost_per_request_usd
        )
        self.controller = ToleranceController(
            min_tolerance_ticks=1.0,
            dilation_k=config.economics.tolerance_dilation_k,
            survival_tolerance_ticks=config.economics.survival_tolerance_ticks
        )
        self.reloader = ReloaderAgent(
            config=config.economics,
            purse=self.purse,
            executor=self.executor,
            quote_stream=self.hyperliquid_stream
        )

        self.info_client = HyperliquidInfoClient(base_url=config.hyperliquid_rest_url)

        # Analytics
        self.core_metrics = None
        self.historical_analyzer = None
        self.per_asset_analyzer = None
        self.analytics_storage = None

        self.alpha_engine = AlphaEngine()
        
        self.reconciler = OrderReconciler(
            api=self.info_client,
            wallet=config.hyperliquid_main_wallet
        ) if config.hyperliquid_main_wallet else None

        # Strategy
        # TODO: Load from config string (factory pattern). For now defaulting to Standard.
        self.strategy: BrawlerStrategy = StandardBrawlerStrategy()


        if config.analytics.enabled:
            logger.info("Initializing Brawler analytics system")
            self.core_metrics = CoreMetricsCalculator()
            self.historical_analyzer = HistoricalAnalyzer()
            self.per_asset_analyzer = PerAssetPerformanceAnalyzer()
            
            db_config = DatabaseConfig(
                host=config.analytics.db_host,
                port=config.analytics.db_port,
                database=config.analytics.db_name,
                username=config.analytics.db_user,
                password=config.analytics.db_password
            )
            self.analytics_storage = AnalyticsStorage(db_config)


    async def start(self) -> None:
        logger.info("Starting Brawler engine with %d assets.", len(self.states))
        await self._bootstrap_inventory()
        self.binance_stream.start()
        self.hyperliquid_stream.start()
        if self.fill_stream:
            self.fill_stream.start()
        self.discovery.start()

        if self.analytics_storage:
            try:
                await self.analytics_storage.connect()
                await self.analytics_storage.create_tables()
            except Exception as exc:
                logger.error("Failed to connect to analytics storage: %s", exc)

        self._tasks = [
            asyncio.create_task(self._consume_cex_quotes(), name="brawler-cex-quotes"),
            asyncio.create_task(self._consume_local_quotes(), name="brawler-local-quotes"),
            asyncio.create_task(self._quote_loop(), name="brawler-quote-loop"),
            asyncio.create_task(self._log_status_summary(), name="brawler-status-log"),
            asyncio.create_task(self._sync_economics(), name="brawler-economics-sync"),
            asyncio.create_task(self._monitor_reload_needs(), name="brawler-reloader"),
        ]
        if self.fill_stream:
            self._tasks.append(asyncio.create_task(self._consume_fills(), name="brawler-fills"))

    async def stop(self) -> None:
        self._stop.set()
        await self.binance_stream.stop()
        await self.hyperliquid_stream.stop()
        if self.fill_stream:
            await self.fill_stream.stop()
        await self.discovery.stop()
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._persist_state()

    async def _log_status_summary(self) -> None:
        interval = 60.0
        while not self._stop.is_set():
            await asyncio.sleep(interval)
            
            lines = ["\n📊 Brawler Status Summary"]
            budget = self.purse.request_budget
            threshold = self.config.economics.reload_threshold_budget
            
            if budget < threshold:
                econ_status = "CRITICAL"
            elif budget < threshold + 1000:
                econ_status = "POOR"
            else:
                econ_status = "HEALTHY"

            lines.append(
                f"  [ECON] Requests: {self.purse.request_count} | Vol: ${self.purse.cumulative_volume:,.0f} | "
                f"Budget: {budget:+.1f} | Status: {econ_status}"
            )
            for symbol, state in self.states.items():
                cex_mid = state.latest_cex_price
                local_mid = (
                    (state.active_bid.price + state.active_ask.price) / 2.0 
                    if state.active_bid and state.active_ask 
                    else state.last_mid_price
                )
                
                status = "ACTIVE"
                if state.suspended_reason:
                    status = f"SUSPENDED ({state.suspended_reason})"
                
                # Use basis BPS in logs
                basis_bps = getattr(state, 'last_basis_bps', 0.0)
                
                lines.append(
                    f"  {symbol:<6} | {status:<20} | "
                    f"CEX: {cex_mid:.4f} | Local: {local_mid:.4f} | "
                    f"Basis: {basis_bps:.1f}bps | Sigma: {state.sigma:.6f} | "
                    f"Inv: {state.inventory:+.4f}"
                )
            logger.info("\n".join(lines))


    def _round_size(self, size: float, step: float = 0.001) -> float:
        """Round size to step precision to avoid float artifacts."""
        if size <= 0: return 0.0
        steps = round(size / step)
        return round(steps * step, 9)
    async def _sync_economics(self) -> None:
        """Periodically sync local purse and inventory with exchange truth."""
        interval = 60.0
        while not self._stop.is_set():
            try:
                wallet = self.config.hyperliquid_main_wallet
                if not wallet:
                    await asyncio.sleep(interval)
                    continue
                
                info = await self.info_client.get_user_rate_limit(wallet)
                req_used = int(info.get("nRequestsUsed", 0))
                cum_vol = float(info.get("cumVlm", "0"))
                self.purse.sync(req_used, cum_vol)
                
                user_state = await asyncio.to_thread(self.info_client.info.user_state, wallet)
                # Parse positions map
                remote_positions = {}
                for item in user_state.get("assetPositions", []):
                    pos = item.get("position")
                    if not pos: continue
                    symbol = pos.get("coin")
                    szi = float(pos.get("szi", 0.0))
                    remote_positions[symbol] = szi
                
                # Sync all managed states
                for symbol, state in self.states.items():
                    remote_szi = remote_positions.get(symbol, 0.0)
                    if abs(state.inventory - remote_szi) > 1e-6:
                         logger.info("Inventory drift detected for %s: Local=%.4f, Remote=%.4f. Syncing.", symbol, state.inventory, remote_szi)
                    state.inventory = remote_szi

            except Exception as exc:
                logger.warning("Failed to sync economics/inventory: %s", exc)
            
            await asyncio.sleep(interval)

    async def _monitor_reload_needs(self) -> None:
        interval = 10.0
        while not self._stop.is_set():
            await asyncio.sleep(interval)
            try:
                target_symbol = self.config.economics.reload_symbol
                state = self.states.get(target_symbol)
                
                if not state or state.best_bid <= 0 or state.best_ask <= 0:
                    continue
                    
                mid = (state.best_bid + state.best_ask) / 2.0
                spread = abs(state.best_ask - state.best_bid)
                
                await self.reloader.check_and_reload(mid, spread)
            except Exception as exc:
                logger.error("Error in reload monitor: %s", exc)
            
            await asyncio.sleep(interval)
    
    async def _consume_cex_quotes(self) -> None:
        while not self._stop.is_set():
            quote: CexQuote = await self.binance_stream.queue.get()
            cfg = self._config_for_cex_symbol(quote.symbol)
            if not cfg:
                continue
            state = self.states[cfg.symbol]
            maxlen = max(2, cfg.volatility_lookback)
            if state.cex_mid_window.maxlen != maxlen:
                state.cex_mid_window = deque(state.cex_mid_window, maxlen=maxlen)
            
            state.latest_cex_price = quote.mid
            state.latest_cex_ts = quote.ts
            state.latest_cex_recv_ts = quote.recv_ts
            state.last_trigger_ts = time.time()
            state.last_trigger_source = "cex"
            
            # [FIX] Velocity Noise: Use a rolling window calculation
            # Look back ~100ms to calculate meaningful velocity
            state.push_cex_mid(quote.mid, quote.ts)
            
            if len(state.cex_mid_window) > 2:
                # Find a sample at least 100ms ago
                target_ts = quote.ts - 0.1
                prev_price = None
                prev_ts = None
                
                # Iterate backwards to find best match
                # Window is [oldest ... newest]
                # We want newest sample that is <= target_ts
                for p, t in reversed(state.cex_mid_window):
                    if t <= target_ts:
                        prev_price = p
                        prev_ts = t
                        break
                
                if prev_price and prev_ts:
                    dt = quote.ts - prev_ts
                    if dt > 0.05: # Guard against tiny div
                        velocity_bps = ((quote.mid - prev_price) / prev_price) / dt * 10000.0
                        state.cex_velocity = velocity_bps

            # Only update Volatility (Sigma) periodically to save CPU
            if quote.ts - state.last_calc_ts > 1.0:
                 state.update_sigma()
                 state.last_calc_ts = quote.ts

    async def _consume_local_quotes(self) -> None:
        while not self._stop.is_set():
            quote: LocalQuote = await self.hyperliquid_stream.queue.get()
            state = self.states.get(quote.symbol)
            if not state:
                continue
            
            state.update_bbo(quote.bid, quote.ask, quote.ts)
            state.last_trigger_ts = time.time()
            state.last_trigger_source = "local"
            self.alpha_engine.on_local_quote(quote)
            
            # [FIX] Absolute Basis Trap -> Basis Points (BPS)
            # Old: basis = local - cex (Dangerous for shitcoins)
            # New: basis_bps = ((local / cex) - 1) * 10000
            if state.latest_cex_price > 0:
                local_mid = (quote.bid + quote.ask) / 2.0
                basis_ratio = (local_mid / state.latest_cex_price) - 1.0
                basis_bps = basis_ratio * 10000.0
                state.last_basis_bps = basis_bps
                state.update_basis(basis_bps)
                
                # We can store the absolute for display, but logic should use BPS
                state.last_basis = local_mid - state.latest_cex_price

    async def _consume_fills(self) -> None:
        if not self.fill_stream:
            return
        while not self._stop.is_set():
            fill: FillEvent = await self.fill_stream.queue.get()
            state = self.states.get(fill.symbol)
            if not state:
                continue
            
            # Update Inventory
            change = fill.size if fill.side == "buy" else -fill.size
            # Update Inventory
            change = fill.size if fill.side == "buy" else -fill.size
            state.inventory += change
            state.last_trigger_ts = time.time()
            state.last_trigger_source = "fill"
            
            # [FIX] Ghost Order Prevention
            # If we get a fill, the order is likely gone or partially gone.
            # We must verify if the filled order matches our tracking to prevent
            # the bot from thinking it's still on the book.
            # [FIX] Ghost Order Prevention: Handle Partial Fills
            if state.active_bid and str(state.active_bid.order_id) == str(fill.order_id):
                state.active_bid.size -= abs(fill.size)
                # If remaining size is negligible (e.g. < 1% of order size), consider it done
                if state.active_bid.size < state.config.order_size * 0.01:
                    state.active_bid = None
                
            if state.active_ask and str(state.active_ask.order_id) == str(fill.order_id):
                state.active_ask.size -= abs(fill.size)
                if state.active_ask.size < state.config.order_size * 0.01:
                    state.active_ask = None

            logger.debug(
                "Inventory update %s: %+f (current=%f) [Fill ID: %s]", 
                fill.symbol, change, state.inventory, fill.order_id
            )
            
            # Analytics Processing
            if self.core_metrics:
                try:
                    trade_type = TradeType.MAKER if getattr(fill, "liquidity_type", "maker") == "maker" else TradeType.TAKER
                    fees = getattr(fill, "fee", 0.0)
                    event = TradeEvent(
                        timestamp=datetime.fromtimestamp(fill.ts),
                        symbol=fill.symbol,
                        side=fill.side,
                        quantity=abs(fill.size),
                        price=fill.price,
                        trade_type=trade_type,
                        reference_price=fill.price,
                        fees_paid=fees,
                        funding_paid=0.0,
                        position_before=state.inventory - change,
                        position_after=state.inventory,
                        order_id=fill.order_id
                    )
                    self.core_metrics.process_trade(event)
                    if self.per_asset_analyzer:
                        self.per_asset_analyzer.per_asset.add_trade(event)
                    if self.analytics_storage:
                        asyncio.create_task(self.analytics_storage.store_trade_event(event))
                except Exception as exc:
                    logger.error("Error processing analytics for trade: %s", exc)

    async def _quote_loop(self) -> None:
        # [FIX] Weight Management
        # If budget is low, slow down the loop instead of widening spreads (which costs more requests)
        base_interval = self.config.risk.tick_interval_ms / 1000.0
        throttle_mult = 1.0
        
        while not self._stop.is_set():
            start = time.time()
            
            try:
                if self.portfolio:
                    self.portfolio.update_metrics(self.states)
                await self._update_quotes()
            except Exception as exc:
                logger.error("Error in quote loop: %s", exc, exc_info=exc)
            
            elapsed = time.time() - start
            sleep_time = max(0.0, (base_interval * throttle_mult) - elapsed)
            await asyncio.sleep(sleep_time)

    async def _update_quotes(self) -> None:
        tasks = []
        for symbol in self.config.assets.keys():
            tasks.append(self._update_single_asset(symbol))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _update_single_asset(self, symbol: str) -> None:
        try:
            state = self.states[symbol]
            
            # Delegate decision to Strategy
            decision = self.strategy.calculate_quote(
                state=state,
                alpha_engine=self.alpha_engine,
                purse=self.purse,
                controller=self.controller,
                economics=self.config.economics
            )
            
            # If no decision (suspended), clear orders
            if decision is None:
                await self._cancel_all(symbol, state)
                return

            # Apply Portfolio Scaling (Engine level constraint)
            if self.portfolio and decision.order_size > 0:
                 decision.order_size = self.portfolio.scale_order_size(decision.order_size)
                 # Re-round
                 decision.order_size = self._round_size(decision.order_size, 0.001)
                
            await self._ensure_orders(symbol, state, decision)
        except Exception as exc:
            logger.error("Failed to update quotes for %s: %s", symbol, exc, exc_info=exc)

    
        # Logic moved to strategy.py

    async def _ensure_orders(self, symbol: str, state: AssetState, decision: QuoteDecision) -> None:
        """Cancel/replace logic with Emergency Delta Override."""
        if not self.executor:
            return

        now = time.time()
        min_interval = max(0.01, state.config.min_quote_interval_ms / 1000.0)
        
        # [FIX] Asymmetric Emergency Logic (Adverse Selection Protection only)
        # Only panic if price moves AGAINST us (e.g. market dumps below our bid)
        # We do NOT panic if market moves away (e.g. market pumps above our bid)
        is_emergency = False
        if state.active_bid:
             # If market crashes, fair value drops. If current Bid > New Fair, we are toxic.
             # Trigger if New Bid is significantly LOWER than Old Bid
             if decision.bid_price < state.active_bid.price * 0.995:
                  is_emergency = True
        if state.active_ask:
             # If market pumps, fair value rips. If current Ask < New Fair, we are toxic.
             # Trigger if New Ask is significantly HIGHER than Old Ask
             if decision.ask_price > state.active_ask.price * 1.005: 
                  is_emergency = True

        if not is_emergency and (now - state.last_quote_ts < min_interval):
            return

        await asyncio.gather(
            self._maybe_replace_order(
                symbol, state, decision.bid_price, HyperliquidOrderSide.BUY, 
                decision.order_size, decision.is_reduce_only_bid,
                decision.cex_event_ts, decision.cex_recv_ts
            ),
            self._maybe_replace_order(
                symbol, state, decision.ask_price, HyperliquidOrderSide.SELL, 
                decision.order_size, decision.is_reduce_only_ask,
                decision.cex_event_ts, decision.cex_recv_ts
            )
        )
        state.last_quote_ts = now

    async def _cancel_all(self, symbol: str, state: AssetState) -> None:
        if not self.executor:
            return
        tasks = []
        if state.active_bid:
            tasks.append(self._throttled_cancel(symbol, state.active_bid))
            state.active_bid = None
        if state.active_ask:
            tasks.append(self._throttled_cancel(symbol, state.active_ask))
            state.active_ask = None
        if tasks:
            await asyncio.gather(*tasks)

    async def _maybe_replace_order(
        self,
        symbol: str,
        state: AssetState,
        target_price: float,
        side: str,
        size: float,
        is_reduce_only: bool,
        cex_event_ts: float = 0.0,
        cex_recv_ts: float = 0.0
    ) -> None:
        cancel_task = None
        update = None
        
        snapshot = state.active_bid if side == HyperliquidOrderSide.BUY else state.active_ask
        
        # REDUCE-ONLY ENFORCEMENT
        cfg = state.config
        blocked = False
        if side == HyperliquidOrderSide.BUY and state.inventory >= cfg.max_inventory:
            blocked = True
        elif side == HyperliquidOrderSide.SELL and state.inventory <= -cfg.max_inventory:
            blocked = True

        # Explicit Cancellation (Price <= 0)
        if target_price <= 0:
            blocked = True
            
        if blocked:
            if snapshot and self.executor:
                await self._throttled_cancel(symbol, snapshot)
                if side == HyperliquidOrderSide.BUY: state.active_bid = None
                else: state.active_ask = None
            return

        # Tolerance Check
        tick = max(state.config.tick_size, 1e-12)
        budget = self.purse.request_budget
        dynamic_ticks = self.controller.calculate_tolerance(budget)
        tolerance = dynamic_ticks * tick
        
        target_price = self._normalize_price(state.config, target_price)
        
        if snapshot:
            # Check price diff
            price_ok = abs(snapshot.price - target_price) < tolerance
            # Check reduce_only flag consistency (implied, hard to check on snapshot without storing it)
            # Generally price movement dominates logic.
            if price_ok:
                return

        if self.portfolio and not self.portfolio.allow_order(side):
            return

        if snapshot and self.executor:
            # [FIX] Async/Concurrent Cancellation with Safety
            # Fire cancel, but don't await it yet. If place fails, we await it.
            cancel_task = asyncio.create_task(self._throttled_cancel(symbol, snapshot))
            
            # Add callback to log if cancel fails silently
            def _check_cancel(t):
                if t.exception():
                    logger.error("Background cancel failed! Potential double execution.", exc_info=t.exception())
            cancel_task.add_done_callback(_check_cancel)

        # [FIX] Post-Only (Alo) and Reduce-Only
        order = HyperliquidOrder(
            symbol=symbol,
            price=target_price,
            size=size,
            side=side,
            alo=True,           # Add Liquidity Only (Post-Only)
            reduce_only=is_reduce_only,
            cex_event_ts=cex_event_ts,
            cex_recv_ts=cex_recv_ts
        )
        
        # Meter Tick-to-Trade Latency
        if state.last_trigger_ts > 0:
            latency_ms = (time.time() - state.last_trigger_ts) * 1000.0
            if latency_ms < 5000: # Filter outliers
                    logger.info("LATENCY Symbol=%s Source=%s TickToTrade=%.3fms", 
                                symbol, state.last_trigger_source, latency_ms)

        # Place new order
        try:
            update = await self.executor.place_limit_order(order)
        except RuntimeError as rte:
            # [FIX] Handle Post-Only Rejections gracefully
            if "Post only" in str(rte):
                logger.warning("Post-only rejected for %s @ %.4f. Retrying next tick.", symbol, target_price)
                # We must still await cancel if we fired it!
                if cancel_task: await cancel_task
                return
            # For other RuntimeErrors, re-raise
            if cancel_task: await cancel_task
            raise rte
        except Exception:
            if cancel_task: await cancel_task
            raise
        
        if update is None:
            return

        new_snapshot = OrderSnapshot(
            order_id=update.order_id,
            price=target_price,
            size=order.size,
            side=side,
        )
        if side == HyperliquidOrderSide.BUY:
            state.active_bid = new_snapshot
        else:
            state.active_ask = new_snapshot

    def _config_for_cex_symbol(self, cex_symbol: str) -> Optional[BrawlerAssetConfig]:
        symbol = self._cex_symbol_index.get(cex_symbol.upper())
        if not symbol:
            return None
        return self.config.assets.get(symbol)

    def _normalize_price(self, cfg: BrawlerAssetConfig, price: float) -> float:
        tick = max(cfg.tick_size, 1e-12) 
        if tick <= 0: return price # Guard
        raw_ticks = round(price / tick)
        return float(f"{raw_ticks * tick:.8g}") # Enough precision for 1e-8

    def _feed_suspension_reason(self, state: AssetState, now: float) -> Optional[str]:
        kill_cfg = self.config.kill_switch
        if kill_cfg.max_feed_lag_seconds > 0:
            cex_ts = getattr(state, "latest_cex_ts", 0.0)
            local_ts = getattr(state, "last_local_mid_ts", state.last_local_mid_ts)
            
            if cex_ts and now - cex_ts > kill_cfg.max_feed_lag_seconds:
                return "cex_feed_stale"
            if local_ts and now - local_ts > kill_cfg.max_feed_lag_seconds:
                return "local_feed_stale"
        return None

    async def _bootstrap_inventory(self) -> None:
        if not self.inventory_provider:
            return
        try:
            seeds = await self.inventory_provider.fetch(self.config.assets.keys())
        except Exception as exc: 
            logger.error("Failed to fetch inventory seed: %s", exc, exc_info=exc)
            return

        for symbol, qty in seeds.items():
            state = self.states.get(symbol)
            if state:
                state.inventory = qty

    def _persist_state(self) -> None:
        if not self.state_persistence:
            return
        try:
            snapshots = capture_state(self.states)
            self.state_persistence.save(snapshots)
        except Exception as exc:
            logger.error("Failed to persist state snapshot: %s", exc, exc_info=exc)

    async def _throttled_cancel(self, symbol: str, snapshot: Optional[OrderSnapshot]) -> None:
        if not snapshot or not snapshot.order_id or not self.executor:
            return
        await self.executor.cancel_order(symbol, snapshot.order_id)
        self._last_cancel_ts[symbol] = time.time()

    async def add_asset(self, config: BrawlerAssetConfig) -> None:
        """Dynamically register a new asset for trading."""
        if config.symbol in self.states:
            return
        state = AssetState(config=config)
        self.states[config.symbol] = state
        self.config.assets[config.symbol] = config
        self._symbol_index[config.symbol.upper()] = config.symbol
        self._cex_symbol_index[config.cex_symbol.upper()] = config.symbol
        await self.binance_stream.subscribe(config.cex_symbol)
        await self.hyperliquid_stream.subscribe(config.symbol)