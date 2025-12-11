"""Core event loop for the Brawler market-making strategy."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from .config import BrawlerAssetConfig, BrawlerConfig
from .connectors import (
    BinanceTickerStream,
    HyperliquidExecutionClient,
    HyperliquidInfoClient,
    HyperliquidOrder,
    HyperliquidOrderSide,
    HyperliquidOrderUpdate,
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
)
from slipstream.analytics.core_metrics_calculator import CoreMetricsCalculator
from slipstream.analytics.data_structures import TradeEvent, TradeType
from slipstream.analytics.historical_analyzer import HistoricalAnalyzer
from slipstream.analytics.per_asset_analyzer import PerAssetPerformanceAnalyzer
from slipstream.analytics.storage_layer import AnalyticsStorage, DatabaseConfig
from .alpha_engine import AlphaEngine


logger = logging.getLogger(__name__)


@dataclass
class QuoteDecision:
    bid_price: float
    ask_price: float
    half_spread: float
    fair_value: float
    sigma: float
    gamma: float
    order_size: float


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
            except Exception as exc:  # pragma: no cover - surfacing to logs
                logger.error("Failed to restore state snapshot: %s", exc, exc_info=exc)
        self.portfolio = PortfolioController(config.portfolio)
        self.discovery = DiscoveryEngine(config, engine=self)
        self._last_cancel_ts: Dict[str, float] = {}

        # Economics
        self.purse = RequestPurse(
            cost_per_request=config.economics.cost_per_request_usd
        )
        self.controller = ToleranceController(
            min_tolerance_ticks=1.0,  # We will override this per-asset call actually, but init details here
            dilation_k=config.economics.tolerance_dilation_k,
            survival_tolerance_ticks=config.economics.survival_tolerance_ticks
        )
        self.reloader = ReloaderAgent(
            config=config.economics,
            purse=self.purse,
            executor=self.executor,
            quote_stream=self.hyperliquid_stream
        )

        
        # Info Client for Reconciliation
        self.info_client = HyperliquidInfoClient(base_url=config.hyperliquid_rest_url)

        # Analytics
        self.core_metrics = None
        self.historical_analyzer = None
        self.per_asset_analyzer = None
        self.analytics_storage = None


        # Alphas
        self.alpha_engine = AlphaEngine()
        
        # Reconciliation
        self.reconciler = OrderReconciler(
            api=self.info_client,
            wallet=config.hyperliquid_main_wallet
        ) if config.hyperliquid_main_wallet else None


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
                logger.info("Connected to analytics storage")
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
        """Periodically log a summary of the engine's state."""
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
                f"Budget: {budget:+.1f} | Tol: {self.controller.calculate_tolerance(budget):.0f}t | Status: {econ_status}"
            )
            for symbol, state in self.states.items():
                cex_mid = state.cex_mid_window[-1] if state.cex_mid_window else 0.0
                local_mid = (
                    (state.active_bid.price + state.active_ask.price) / 2.0 
                    if state.active_bid and state.active_ask 
                    else state.last_mid_price
                )
                
                status = "ACTIVE"
                if state.suspended_reason:
                    status = f"SUSPENDED ({state.suspended_reason})"
                
                lines.append(
                    f"  {symbol:<6} | {status:<20} | "
                    f"CEX: {cex_mid:.4f} | Local: {local_mid:.4f} | "
                    f"Basis: {state.last_basis:.4f} | Sigma: {state.sigma:.6f} | "
                    f"N: {len(state.cex_mid_window)} | Inv: {state.inventory:+.4f}"
                )
            logger.info("\n".join(lines))

    async def _sync_economics(self) -> None:
        """Periodically sync local purse and inventory with exchange truth."""
        interval = 60.0
        while not self._stop.is_set():
            try:
                wallet = self.config.hyperliquid_main_wallet
                if not wallet:
                    await asyncio.sleep(interval)
                    continue
                
                # 1. Sync Rate Limits
                info = await self.info_client.get_user_rate_limit(wallet)
                req_used = int(info.get("nRequestsUsed", 0))
                cum_vol = float(info.get("cumVlm", "0"))
                self.purse.sync(req_used, cum_vol)
                
                # 2. Sync Inventory (Failsafe for websocket drops)
                # We use the raw info client wrapper to fetch state
                user_state = await asyncio.to_thread(self.info_client.info.user_state, wallet)
                for item in user_state.get("assetPositions", []):
                    pos = item.get("position")
                    if not pos:
                        continue
                    symbol = pos.get("coin")
                    szi = float(pos.get("szi", 0.0))
                    
                    # Update local state if we are tracking this asset
                    state = self.states.get(symbol)
                    if state:
                        # Log if significant drift found (optional)
                        if abs(state.inventory - szi) > state.config.order_size * 0.1:
                            logger.info("Inventory drift detected for %s: Local=%.4f, Remote=%.4f. Syncing.", symbol, state.inventory, szi)
                        state.inventory = szi

                logger.debug("Synced economics and inventory. Req=%d, Vol=%.2f", req_used, cum_vol)

            except Exception as exc:
                logger.warning("Failed to sync economics/inventory: %s", exc)
            
            await asyncio.sleep(interval)

    async def _monitor_reload_needs(self) -> None:
        """Periodically check if we need to reload request credits."""
        interval = 10.0
        while not self._stop.is_set():
            await asyncio.sleep(interval)
            
            try:
                # Identify target asset state
                target_symbol = self.config.economics.reload_symbol
                # If symbol is not in our main brawler assets, we can't get quote easily.
                # Assuming user configures reload_symbol to be one of the traded assets.
                state = self.states.get(target_symbol)
                
                if not state:
                    # If not tracking locally, we can't easily check spread
                    continue
                    
                # Get Mid/Spread from BBO
                if state.best_bid <= 0 or state.best_ask <= 0:
                    continue
                    
                mid = (state.best_bid + state.best_ask) / 2.0
                spread = abs(state.best_ask - state.best_bid)
                
                await self.reloader.check_and_reload(mid, spread)
                
            except Exception as exc:
                logger.error("Error in reload monitor: %s", exc)
            
            await asyncio.sleep(interval)
    
    async def _consume_cex_quotes(self) -> None:
        from collections import deque

        while not self._stop.is_set():
            quote: CexQuote = await self.binance_stream.queue.get()
            cfg = self._config_for_cex_symbol(quote.symbol)
            if not cfg:
                continue
            state = self.states[cfg.symbol]
            maxlen = max(2, cfg.volatility_lookback)
            if state.cex_mid_window.maxlen != maxlen:
                state.cex_mid_window = deque(state.cex_mid_window, maxlen=maxlen)
            
            # [OPTIMIZATION 1] Real-Time Updates (Removed 1s Throttle)
            # Update the latest price immediately for low-latency quoting.
            state.latest_cex_price = quote.mid
            state.latest_cex_ts = quote.ts
            
            # Momentum Calculation
            if state.cex_mid_window:
                prev_price = state.cex_mid_window[-1]
                dt = quote.ts - state.last_cex_mid_ts
                if dt > 0.001:
                    velocity_bps = ((quote.mid - prev_price) / prev_price) / dt * 10000.0
                    state.cex_velocity = velocity_bps
            
            # Only rate-limit the expensive history push/sigma calc, not the price ref
            if quote.ts - state.last_cex_mid_ts < 1.0:
                 continue

            state.push_cex_mid(quote.mid, quote.ts)
            state.update_sigma()

    async def _consume_local_quotes(self) -> None:
        while not self._stop.is_set():
            quote: LocalQuote = await self.hyperliquid_stream.queue.get()
            cfg = self.config.assets.get(quote.symbol)
            if not cfg:
                continue
            state = self.states[quote.symbol]
            if state.config.symbol != quote.symbol:
                # Mismatch - ignore or log warning
                continue
            
            # Update state with BBO
            state.update_bbo(quote.bid, quote.ask, quote.ts)
            
            # [ALPHA] Update Alpha Engine
            self.alpha_engine.on_local_quote(quote)
            
            # Basis handling
            if state.last_cex_mid_ts > 0:
                cex_mid = state.cex_mid_window[-1]
                # basis = log(local_mid / cex_mid)
                # Use the just-updated mid
                local_mid = (quote.bid + quote.ask) / 2.0
                if cex_mid > 0 and local_mid > 0:
                    basis = local_mid - cex_mid
                    state.last_basis = basis
                    state.update_basis(basis)

    async def _consume_fills(self) -> None:
        if not self.fill_stream:
            return
        while not self._stop.is_set():
            fill: FillEvent = await self.fill_stream.queue.get()
            state = self.states.get(fill.symbol)
            if not state:
                continue
            state.inventory += fill.size if fill.side == "buy" else -fill.size
            logger.debug(
                "Inventory update %s: %+f (current=%f)", fill.symbol, fill.size, state.inventory
            )
            
            # Analytics Processing
            if self.core_metrics:
                try:
                    trade_type = TradeType.MAKER if getattr(fill, "liquidity_type", "maker") == "maker" else TradeType.TAKER
                    # Fees/Funding from fill event (extended event)
                    fees = getattr(fill, "fee", 0.0)
                    
                    event = TradeEvent(
                        timestamp=datetime.fromtimestamp(fill.ts),
                        symbol=fill.symbol,
                        side=fill.side,
                        quantity=abs(fill.size),
                        price=fill.price,
                        trade_type=trade_type,
                        reference_price=fill.price, # Use fill price as ref for now unless we have better
                        fees_paid=fees,
                        funding_paid=0.0, # Not in fill stream usually
                        position_before=state.inventory - fill.size if fill.side == "buy" else state.inventory + fill.size,
                        position_after=state.inventory,
                        order_id=fill.order_id
                    )
                    
                    self.core_metrics.process_trade(event)
                    if self.per_asset_analyzer:
                        self.per_asset_analyzer.per_asset.add_trade(event)
                    
                    if self.analytics_storage:
                        # Fire and forget storage to not block loop
                        asyncio.create_task(self.analytics_storage.store_trade_event(event))
                        
                except Exception as exc:
                    logger.error("Error processing analytics for trade: %s", exc, exc_info=exc)


    async def _quote_loop(self) -> None:
        interval = max(0.1, self.config.risk.tick_interval_ms / 1000.0)
        while not self._stop.is_set():
            start = time.time()
            try:
                if self.portfolio:
                    self.portfolio.update_metrics(self.states)
                await self._update_quotes()
            except Exception as exc:
                logger.error("Error in quote loop: %s", exc, exc_info=exc)
            
            elapsed = time.time() - start
            await asyncio.sleep(max(0.0, interval - elapsed))

    # [OPTIMIZATION 2] Parallelize the Asset Loop
    # Added new helper _update_single_asset to allow asyncio.gather
    async def _update_quotes(self) -> None:
        tasks = []
        for symbol in self.config.assets.keys():
            tasks.append(self._update_single_asset(symbol))
        
        # Run all asset updates in parallel, ignore exceptions so one failure doesn't halt others
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _update_single_asset(self, symbol: str) -> None:
        """Helper to process a single asset safely within the parallel gather."""
        try:
            state = self.states[symbol]
            decision = self._build_quote_decision(state)
            if decision is None:
                await self._cancel_all(symbol, state)
                return
            await self._ensure_orders(symbol, state, decision)
        except Exception as exc:
            logger.error("Failed to update quotes for %s: %s", symbol, exc, exc_info=exc)

    def _build_quote_decision(self, state: AssetState) -> Optional[QuoteDecision]:
        cfg = state.config

        if not state.cex_mid_window:
            return None

        if not state.cex_mid_window:
            return None

        # [HARDENING] Use the REAL-TIME latest price, not the windowed 1Hz price
        mid_cex = getattr(state, "latest_cex_price", state.cex_mid_window[-1])
        pfair = mid_cex + state.fair_basis
        sigma = state.sigma
        now = time.time()

        feed_reason = self._feed_suspension_reason(state, now)
        if feed_reason:
            if state.suspended_reason != feed_reason:
                logger.warning("Suspending %s: %s", state.symbol, feed_reason)
            state.suspended_reason = feed_reason
            return None
            
        # [ALPHA] Check Fear Signal
        alpha_state = self.alpha_engine.states.get(cfg.symbol)
        fear_side = alpha_state.fear_side if alpha_state else None
        
        if fear_side == 'both':
            if state.suspended_reason != "alpha_fear_both":
                logger.warning("Suspending %s: Fear Signal Active (Both Sides)", state.symbol)
            state.suspended_reason = "alpha_fear_both"
            return None
        elif fear_side:
            # Directional Fear -> Don't suspend entirely, just flag active direction
            # We want to CONTINUE quoting the other side.
            if state.suspended_reason:
                 # If previously fully suspended, we might need to clear it to allow partial quoting?
                 logger.info("Resuming %s (partial fear: %s)", state.symbol, fear_side)
                 state.suspended_reason = None
        else:
             # No fear
             pass

        # Clear past suspension if we are here
        if state.suspended_reason:
            logger.info("Resuming %s (was %s)", state.symbol, state.suspended_reason)
            state.suspended_reason = None
        
        if sigma > cfg.max_volatility:
             if state.suspended_reason != "volatility":
                 logger.warning("Suspending %s: Volatility %.4f > %.4f", cfg.symbol, sigma, cfg.max_volatility)
             state.suspended_reason = "volatility"
             return None

        # [REMOVED] Preemptive inventory check. We now allow generation of quotes
        # so that _ensure_orders can apply "Reduce Only" logic.
        # if abs(state.inventory) > cfg.max_inventory:
        #    state.mark_suspended("inventory")
        #    return None

        if cfg.max_basis_deviation > 0 and state.last_basis:
            basis_delta = abs(state.last_basis - state.fair_basis)
            if basis_delta > cfg.max_basis_deviation:
                if state.suspended_reason != "basis":
                    logger.warning(
                        "%s suspended: basis delta=%.6f exceeds %.6f",
                        cfg.symbol,
                        basis_delta,
                        cfg.max_basis_deviation,
                    )
                state.mark_suspended("basis")
                return None

        if state.suspended_reason == "inventory":
            # If we were suspended for inventory, clear it now since we support ReduceOnly
            state.clear_suspension()
        #     ratio = abs(state.inventory) / cfg.max_inventory
        #     if ratio <= cfg.reduce_only_ratio:
        #         state.clear_suspension()
        #     else:
        #         return None
        elif state.suspended_reason and time.time() - state.last_suspend_ts > self.config.risk.resume_backoff_seconds:
            logger.info("Auto-resuming %s from %s", cfg.symbol, state.suspended_reason)
            state.clear_suspension()

        if self.portfolio and not self.portfolio.allow_quotes(state):
            return None

        # [HARDENING] Opportunity Filter: Minimum Basis Check
        if cfg.min_basis_bps > 0:
            basis_bps = 0.0
            if mid_cex > 0:
                basis_bps = abs(state.fair_basis / mid_cex) * 10000.0
            
            if basis_bps < cfg.min_basis_bps:
                if state.suspended_reason != "low_edge":
                    # Only log occasionally to avoid spam
                    pass 
                # We return None, which triggers _cancel_all in _update_quotes
                return None

        # [HARDENING] Momentum Guard
        momentum_skew = 0.0
        if cfg.momentum_threshold_bps > 0:
            velocity = getattr(state, "cex_velocity", 0.0)
            if abs(velocity) > cfg.momentum_threshold_bps:
                # Momentum Guard effectively adds EXTRA spread during chaos.
                sigma = max(sigma, 0.05) # Boost perceived vol
                
                # Optional: log momentum event
                if random.random() < 0.05:
                   logger.info("Momentum detected for %s: %.1f bps/s", cfg.symbol, velocity)

        # -------------------------------------------------------------------------
        # Dynamic Spread Calculation (Algorithmic Improvement)
        # -------------------------------------------------------------------------
        # Pfair * base_spread + Pfair * sigma * k + EconomicsPenalty
        
        # 1. Base + Volatility Component
        # Config default 'vol_spread_multiplier' (k) is 5.0
        # If sigma=0.0005 (low), add 0.0025 (25bps).
        # If sigma=0.01 (high), add 0.05 (500bps).
        dynamic_bps = cfg.base_spread + (sigma * cfg.vol_spread_multiplier)
        
        # 2. Economics Penalty (Low Budget -> Widen Spread)
        budget_penalty = self.controller.calculate_spread_penalty(self.purse.request_budget)
        
        total_spread_bps = dynamic_bps + budget_penalty
        
        # 3. Cap at Max Spread
        max_bps = self.config.economics.max_spread_bps / 10000.0
        if total_spread_bps > max_bps:
            total_spread_bps = max_bps
            
        half_spread_val = (total_spread_bps * pfair) / 2.0
        
        # Inventory Skew (Gamma)
        inv_ratio = 0.0
        if cfg.max_inventory > 0:
            inv_ratio = max(-1.0, min(1.0, state.inventory / cfg.max_inventory))
        gamma = cfg.inventory_aversion * inv_ratio * pfair # Convert to price delta

        bid_price = self._normalize_price(cfg, pfair - half_spread_val - gamma)
        ask_price = self._normalize_price(cfg, pfair + half_spread_val - gamma)
        
        # [ALPHA] Apply Directional Fear Cancellation
        if fear_side == 'bid':
            # Support vanished -> Don't Buy -> Cancel Bid
            bid_price = 0.0
        elif fear_side == 'ask':
            # Resistance vanished -> Don't Sell -> Cancel Ask
            ask_price = 0.0

        # Order Sizing
        if cfg.vol_sizing_risk_dollars > 0:
            effective_sigma = max(sigma, 0.01)
            raw_size = cfg.vol_sizing_risk_dollars / (pfair * 2.0 * effective_sigma)
            order_size = min(raw_size, cfg.max_inventory)

            # Log sizing logic periodically
            if random.random() < 0.05:
                logger.info(
                    "Sizing %s: sigma=%.5f eff_sigma=%.4f risk=$%.2f pfair=%.4f -> raw=%.4f clamped=%.4f",
                    cfg.symbol, sigma, effective_sigma, cfg.vol_sizing_risk_dollars, pfair, raw_size, order_size
                )
            
            val = order_size * pfair
            if val < 10.0:
                order_size = 10.0 / pfair
                
        else:
            order_size = state.config.order_size

        if self.portfolio:
            order_size = self.portfolio.scale_order_size(order_size)
        if order_size <= 0:
            return None

        # Build Decision
        return QuoteDecision(
            bid_price=bid_price,
            ask_price=ask_price,
            half_spread=half_spread_val,
            fair_value=pfair,
            sigma=sigma,
            gamma=gamma,
            order_size=order_size,
        )

    async def add_asset(self, config: BrawlerAssetConfig) -> None:
        """Dynamically register a new asset for trading."""
        if config.symbol in self.states:
            return
            
        logger.info("Adding new asset to Brawler: %s", config.symbol)
        
        # 1. Init State
        state = AssetState(config=config)
        self.states[config.symbol] = state
        
        # 2. Update Indexes
        self.config.assets[config.symbol] = config
        self._symbol_index[config.symbol.upper()] = config.symbol
        self._cex_symbol_index[config.cex_symbol.upper()] = config.symbol
        
        # 3. Subscribe Feeds
        await self.binance_stream.subscribe(config.cex_symbol)
        await self.hyperliquid_stream.subscribe(config.symbol)
        
        # 4. Bootstrap Inventory
        pass

    async def _run_reconciliation(self) -> None:
        """Periodic reconciliation loop."""
        if not self.reconciler:
            return
            
        while True:
            try:
                await asyncio.sleep(self.reconciler.interval)
                await self.reconciler.reconcile(self.states)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Reconciliation error: %s", e, exc_info=True)
                await asyncio.sleep(5.0)

    async def _ensure_orders(self, symbol: str, state: AssetState, decision: QuoteDecision) -> None:
        """Cancel/replace logic; only one resting order per side."""
        if not self.executor:
            return

        now = time.time()
        min_interval = max(0.1, state.config.min_quote_interval_ms / 1000.0)
        if now - state.last_quote_ts < min_interval:
            return

        # [OPTIMIZATION 5] Removed side jitter (hardcoded to 0)
        side_offset = 0.0 
        
        if self.executor:
            # [OPTIMIZATION 3] Parallelize Bid and Ask Placement
            # Fire both requests simultaneously to halve RTT.
            await asyncio.gather(
                self._maybe_replace_order(
                    symbol,
                    state,
                    decision.bid_price,
                    HyperliquidOrderSide.BUY,
                    decision.order_size,
                    side_delay=side_offset,
                ),
                self._maybe_replace_order(
                    symbol,
                    state,
                    decision.ask_price,
                    HyperliquidOrderSide.SELL,
                    decision.order_size,
                    side_delay=side_offset,
                )
            )
        state.last_quote_ts = now

    async def _cancel_all(self, symbol: str, state: AssetState) -> None:
        """Cancel all active orders for the symbol (used when suspended)."""
        if not self.executor:
            return
        
        if not state.active_bid and not state.active_ask:
            return

        # [OPTIMIZATION 3.5] Parallelize Cancels as well
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
        side_delay: float = 0.0,
    ) -> None:
        snapshot = state.active_bid if side == HyperliquidOrderSide.BUY else state.active_ask
        
        # REDUCE-ONLY ENFORCEMENT
        # If inventory exceeds limit, prevent opening new positions on the increasing side.
        # But ensure we CANCEL existing orders on that side.
        cfg = state.config
        blocked = False
        if side == HyperliquidOrderSide.BUY and state.inventory >= cfg.max_inventory:
            blocked = True
        elif side == HyperliquidOrderSide.SELL and state.inventory <= -cfg.max_inventory:
            blocked = True
        
        # Explicit Cancellation Instruction (Price <= 0)
        if target_price <= 0:
            blocked = True
            
        if blocked:
            if snapshot and self.executor:
                await self._throttled_cancel(symbol, snapshot)
                if side == HyperliquidOrderSide.BUY:
                    state.active_bid = None
                else:
                    state.active_ask = None
            return

        tick = max(state.config.tick_size, 1e-12)
        
        # Dynamic Tolerance Calculation
        budget = self.purse.request_budget
        dynamic_ticks = self.controller.calculate_tolerance(budget)
        final_ticks = max(dynamic_ticks, state.config.quote_reprice_tolerance_ticks)
        
        tolerance = final_ticks * tick
        
        target_price = self._normalize_price(state.config, target_price)
        if snapshot and math.isclose(snapshot.price, target_price, rel_tol=1e-5):
            return
        if snapshot and abs(snapshot.price - target_price) < tolerance:
            return

        if self.portfolio and not self.portfolio.allow_order(side):
            return

        order_size = size
        if order_size <= 0:
            return

        if side_delay > 0:
            await asyncio.sleep(side_delay)

        if snapshot and self.executor:
            await self._throttled_cancel(symbol, snapshot)
        order = HyperliquidOrder(
            symbol=symbol,
            price=target_price,
            size=order_size,
            side=side,
        )
        update = await self.executor.place_limit_order(order)
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
        logger.debug(
            "Placed %s order for %s @ %.4f (id=%s)",
            side,
            symbol,
            target_price,
            update.order_id,
        )

    def _config_for_cex_symbol(self, cex_symbol: str) -> Optional[BrawlerAssetConfig]:
        symbol = self._cex_symbol_index.get(cex_symbol.upper())
        if not symbol:
            return None
        return self.config.assets.get(symbol)

    def _normalize_price(self, cfg: BrawlerAssetConfig, price: float) -> float:
        tick = max(cfg.tick_size, 1e-12)
        return round(price / tick) * tick

    def _feed_suspension_reason(self, state: AssetState, now: float) -> Optional[str]:
        kill_cfg = self.config.kill_switch
        last_cex = state.last_cex_mid_ts
        last_local = state.last_local_mid_ts

        max_feed_lag = kill_cfg.max_feed_lag_seconds
        if max_feed_lag > 0:
            if last_cex and now - last_cex > max_feed_lag:
                return "cex_feed_stale"
            if last_local and now - last_local > max_feed_lag:
                return "local_feed_stale"

        max_disconnect = kill_cfg.max_disconnection_seconds
        if max_disconnect > 0:
            if last_cex and now - last_cex > max_disconnect:
                return "cex_disconnected"
            if last_local and now - last_local > max_disconnect:
                return "local_disconnected"

        return None

    async def _bootstrap_inventory(self) -> None:
        if not self.inventory_provider:
            return
        try:
            seeds = await self.inventory_provider.fetch(self.config.assets.keys())
        except Exception as exc:  # pragma: no cover - depends on provider
            logger.error("Failed to fetch inventory seed: %s", exc, exc_info=exc)
            return

        for symbol, qty in seeds.items():
            state = self.states.get(symbol)
            if not state:
                continue
            state.inventory = qty
            logger.info("Seeded %s inventory to %.6f contracts", symbol, qty)

    def _persist_state(self) -> None:
        if not self.state_persistence:
            return
        try:
            snapshots = capture_state(self.states)
            self.state_persistence.save(snapshots)
        except Exception as exc:  # pragma: no cover - I/O errors surfaced
            logger.error("Failed to persist state snapshot: %s", exc, exc_info=exc)

    async def _throttled_cancel(self, symbol: str, snapshot: Optional[OrderSnapshot]) -> None:
        if not snapshot or not snapshot.order_id or not self.executor:
            return
        
        # [OPTIMIZATION 4] Removed Sleep on Cancel
        # Never sleep during a cancellation/retreat.
        # If rate limited, let the API error handler deal with it.
        # min_interval = max(0.0, self.config.risk.min_cancel_interval_ms / 1000.0)
        # last = self._last_cancel_ts.get(symbol, 0.0)
        # now = time.time()
        # delay = (last + min_interval) - now
        # if delay > 0:
        #    await asyncio.sleep(delay)
        
        await self.executor.cancel_order(symbol, snapshot.order_id)
        self._last_cancel_ts[symbol] = time.time()

    def _next_side_delay(self, symbol: str) -> float:
        # [OPTIMIZATION 5] Remove Jitter
        return 0.0
