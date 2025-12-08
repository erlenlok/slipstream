"""Core event loop for the Brawler market-making strategy."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional

from .config import BrawlerAssetConfig, BrawlerConfig
from .connectors import (
    BinanceTickerStream,
    HyperliquidExecutionClient,
    HyperliquidOrder,
    HyperliquidOrderSide,
    HyperliquidOrderUpdate,
    HyperliquidQuoteStream,
    HyperliquidUserFillStream,
)
from .discovery import DiscoveryEngine
from .feeds import CexQuote, FillEvent, LocalQuote
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
        self.discovery = DiscoveryEngine(config)
        self._last_cancel_ts: Dict[str, float] = {}

    async def start(self) -> None:
        logger.info("Starting Brawler engine with %d assets.", len(self.states))
        await self._bootstrap_inventory()
        self.binance_stream.start()
        self.hyperliquid_stream.start()
        if self.fill_stream:
            self.fill_stream.start()
        self.discovery.start()

        self._tasks = [
            asyncio.create_task(self._consume_cex_quotes(), name="brawler-cex-quotes"),
            asyncio.create_task(self._consume_local_quotes(), name="brawler-local-quotes"),
            asyncio.create_task(self._quote_loop(), name="brawler-quote-loop"),
            asyncio.create_task(self._log_status_summary(), name="brawler-status-log"),
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
            state.push_cex_mid(quote.mid, quote.ts)
            state.update_sigma()

    async def _consume_local_quotes(self) -> None:
        while not self._stop.is_set():
            quote: LocalQuote = await self.hyperliquid_stream.queue.get()
            cfg = self.config.assets.get(quote.symbol)
            if not cfg:
                continue
            state = self.states[quote.symbol]
            state.push_local_mid(quote.mid, quote.ts)
            cex_cfg = self._config_for_cex_symbol(cfg.cex_symbol.upper())
            if cex_cfg is None:
                continue
            cex_state = self.states[cfg.symbol]
            basis = quote.mid - cex_state.cex_mid_window[-1] if cex_state.cex_mid_window else 0.0
            cex_state.last_basis = basis
            cex_state.update_basis(basis)

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

    async def _update_quotes(self) -> None:
        for symbol, cfg in self.config.assets.items():
            try:
                state = self.states[symbol]
                decision = self._build_quote_decision(state)
                if decision is None:
                    continue
                await self._ensure_orders(symbol, state, decision)
            except Exception as exc:
                logger.error("Failed to update quotes for %s: %s", symbol, exc, exc_info=exc)

    def _build_quote_decision(self, state: AssetState) -> Optional[QuoteDecision]:
        cfg = state.config

        if not state.cex_mid_window:
            return None

        mid_cex = state.cex_mid_window[-1]
        pfair = mid_cex + state.fair_basis
        sigma = state.sigma
        now = time.time()

        feed_reason = self._feed_suspension_reason(state, now)
        if feed_reason:
            if state.suspended_reason != feed_reason:
                logger.warning("%s suspended: %s", cfg.symbol, feed_reason)
            state.mark_suspended(feed_reason)
            return None

        if sigma > cfg.max_volatility:
            if state.suspended_reason != "volatility":
                logger.warning("%s suspended: sigma=%.4f exceeds %.4f", cfg.symbol, sigma, cfg.max_volatility)
            state.mark_suspended("volatility")
            return None

        if abs(state.inventory) > cfg.max_inventory:
            state.mark_suspended("inventory")
            return None

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
            ratio = abs(state.inventory) / cfg.max_inventory
            if ratio <= cfg.reduce_only_ratio:
                state.clear_suspension()
            else:
                return None
        elif state.suspended_reason and time.time() - state.last_suspend_ts > self.config.risk.resume_backoff_seconds:
            logger.info("Auto-resuming %s from %s", cfg.symbol, state.suspended_reason)
            state.clear_suspension()

        if self.portfolio and not self.portfolio.allow_quotes(state):
            return None

        base_spread = pfair * cfg.base_spread
        vol_component = cfg.risk_aversion * sigma * pfair
        total_spread = max(base_spread + vol_component, 0.0)
        half_spread = total_spread / 2.0

        inv_ratio = 0.0
        if cfg.max_inventory > 0:
            inv_ratio = max(-1.0, min(1.0, state.inventory / cfg.max_inventory))
        gamma = cfg.inventory_aversion * inv_ratio

        bid_price = self._normalize_price(cfg, pfair - half_spread - gamma)
        ask_price = self._normalize_price(cfg, pfair + half_spread - gamma)
        # Order Sizing
        if cfg.vol_sizing_risk_dollars > 0:
            effective_sigma = max(sigma, 0.0005)  # Floor at 5bps vol to prevent explosions
            # risk_dollars = size_units * price * 2 * sigma
            # size_units = risk_dollars / (price * 2 * sigma)
            raw_size = cfg.vol_sizing_risk_dollars / (pfair * 2.0 * effective_sigma)
            # Clamp to safe limits (e.g. not exceeding max inventory in one clip, or reasonable bounds)
            # Using max_inventory as a sanity ceiling for single order
            order_size = min(raw_size, cfg.max_inventory)
        else:
            order_size = state.config.order_size

        if self.portfolio:
            order_size = self.portfolio.scale_order_size(order_size)
        if order_size <= 0:
            return None

        return QuoteDecision(
            bid_price=bid_price,
            ask_price=ask_price,
            half_spread=half_spread,
            fair_value=pfair,
            sigma=sigma,
            gamma=gamma,
            order_size=order_size,
        )

    async def _ensure_orders(self, symbol: str, state: AssetState, decision: QuoteDecision) -> None:
        """Cancel/replace logic; only one resting order per side."""
        if not self.executor:
            return

        now = time.time()
        min_interval = max(0.1, state.config.min_quote_interval_ms / 1000.0)
        if now - state.last_quote_ts < min_interval:
            return

        side_offset = self._next_side_delay(symbol)
        await self._maybe_replace_order(
            symbol,
            state,
            decision.bid_price,
            HyperliquidOrderSide.BUY,
            decision.order_size,
            side_delay=side_offset,
        )
        await self._maybe_replace_order(
            symbol,
            state,
            decision.ask_price,
            HyperliquidOrderSide.SELL,
            decision.order_size,
            side_delay=side_offset,
        )
        state.last_quote_ts = now

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
        tick = max(state.config.tick_size, 1e-12)
        tolerance = state.config.quote_reprice_tolerance_ticks * tick
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
        min_interval = max(0.0, self.config.risk.min_cancel_interval_ms / 1000.0)
        last = self._last_cancel_ts.get(symbol, 0.0)
        now = time.time()
        delay = (last + min_interval) - now
        if delay > 0:
            await asyncio.sleep(delay)
        await self.executor.cancel_order(symbol, snapshot.order_id)
        self._last_cancel_ts[symbol] = time.time()

    def _next_side_delay(self, symbol: str) -> float:
        jitter_ms = max(0.0, self.config.risk.side_jitter_ms)
        if jitter_ms <= 0:
            return 0.0
        return random.uniform(0.0, jitter_ms / 1000.0)
