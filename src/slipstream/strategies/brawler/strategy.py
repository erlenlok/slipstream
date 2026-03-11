from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from .alpha_engine import AlphaEngine
from .config import BrawlerEconomicsConfig
from .economics import RequestPurse, ToleranceController
from .state import AssetState, QuoteDecision

logger = logging.getLogger(__name__)


class BrawlerStrategy(ABC):
    """
    Abstract Base Class for Brawler market making strategies.
    
    A Strategy is responsible for taking the current AssetState and producing a QuoteDecision.
    It should be pure logic (no side effects, no async) and fast.
    """
    
    @abstractmethod
    def calculate_quote(
        self, 
        state: AssetState, 
        alpha_engine: AlphaEngine,
        purse: RequestPurse,
        controller: ToleranceController,
        economics: BrawlerEconomicsConfig
    ) -> Optional[QuoteDecision]:
        """
        Calculate the next quote for a single asset.
        
        Args:
            state: Current state of the asset (prices, inventory, config)
            alpha_engine: Access to alpha signals
            purse: Access to budget/economics
            controller: Access to penalty calculations
            
        Returns:
            QuoteDecision with params for the next order, or None to suspend/cancel.
        """
        pass


class StandardBrawlerStrategy(BrawlerStrategy):
    """
    The standard Brawler strategy implementation.
    
    Features:
    - BPS-based spread model (Base + Sigma * Mult)
    - Inventory Skew (Gamma)
    - Economic Penalties (widen spreads on low budget)
    - Alpha Overrides (Fear signals)
    - Dollar-based Sizing
    - Reduce-Only Clamping
    """
    
    def calculate_quote(
        self, 
        state: AssetState, 
        alpha_engine: AlphaEngine,
        purse: RequestPurse,
        controller: ToleranceController,
        economics: BrawlerEconomicsConfig
    ) -> Optional[QuoteDecision]:
        cfg = state.config

        # 1. Price Source
        mid_cex = getattr(state, "latest_cex_price", 0.0)
        if mid_cex <= 0:
            if state.cex_mid_window:
                mid_cex = state.cex_mid_window[-1]
            else:
                return None

        # 2. Fair Value (BPS Basis)
        # Fair Value = CEX * (1 + FairBasisBPS/10000)
        fair_basis_bps = state.fair_basis_bps
        pfair = mid_cex * (1 + (fair_basis_bps / 10000.0))
        
        sigma = state.sigma
        now = time.time()

        # 3. Suspension Checks
        # (Using internal helper or replicating logic? Replicating for purity)
        # Logic from engine._feed_suspension_reason seems tied to state management but check logic is pure.
        # Ideally engine handles "Is feed alive?" but strategy decides "Do I trade?"
        # For now, let's trust state.suspended_reason is managed by engine for infrastructure (API down etc)
        # But STRATEGY handles Alpha/Risk suspensions.
        
        # [ALPHA] Check Fear Signal
        alpha_state = alpha_engine.states.get(cfg.symbol)
        fear_side = alpha_state.fear_side if alpha_state else None
        
        if fear_side == 'both':
            state.suspended_reason = "alpha_fear_both"
            return None
        elif state.suspended_reason == "alpha_fear_both":
             state.suspended_reason = None

        if state.suspended_reason:
             # Check auto-resume
             if now - state.last_suspend_ts > cfg.risk.resume_backoff_seconds:
                  state.clear_suspension()
             else:
                  return None

        if sigma > cfg.max_volatility:
             state.suspended_reason = "volatility"
             return None

        # [FIX] Basis Guard using BPS
        if cfg.max_basis_deviation > 0:
            current_basis_bps = getattr(state, 'last_basis_bps', 0.0)
            if abs(current_basis_bps - fair_basis_bps) > 500: # 5% decoupling hardcap
                 state.suspended_reason = "basis_decouple"
                 return None

        # 4. Spread Calculation
        # dynamic_bps: Base (BPS) + (Sigma (decimal) * Multiplier (scaled to BPS))
        dynamic_bps = cfg.base_spread + (sigma * cfg.vol_spread_multiplier)
        
        # Economics Penalty
        budget_penalty_bps = controller.calculate_spread_penalty(purse.request_budget) * 10000.0
        
        total_spread_bps = dynamic_bps + budget_penalty_bps
        max_spread_cap_bps = economics.max_spread_bps
        
        if total_spread_bps > max_spread_cap_bps:
             total_spread_bps = max_spread_cap_bps
            
        half_spread_val = (total_spread_bps / 10000.0 * pfair) / 2.0
        
        # 5. Inventory Skew (Gamma)
        inv_ratio = 0.0
        if cfg.max_inventory > 0:
            inv_ratio = max(-1.0, min(1.0, state.inventory / cfg.max_inventory))
        gamma = cfg.inventory_aversion * inv_ratio * pfair 

        bid_price = self._normalize_price(cfg, pfair - half_spread_val - gamma)
        ask_price = self._normalize_price(cfg, pfair + half_spread_val - gamma)
        
        if fear_side == 'bid': bid_price = 0.0
        elif fear_side == 'ask': ask_price = 0.0

        # 6. Order Sizing
        order_size = cfg.order_size
        
        # Target USD Sizing
        if hasattr(cfg, "target_size_usd") and cfg.target_size_usd > 0:
            if pfair > 0:
                raw_usd_size = cfg.target_size_usd / pfair
                order_size = min(raw_usd_size, cfg.max_inventory)
                logger.info("TargetUSD: %.2f Pfair: %.2f -> RawSize: %.4f", 
                            cfg.target_size_usd, pfair, order_size)
        # Brawler Legacy Vol Sizing
        elif hasattr(cfg, "vol_sizing_risk_dollars") and cfg.vol_sizing_risk_dollars > 0:
            effective_sigma = max(sigma, 0.01)
            raw_size = cfg.vol_sizing_risk_dollars / (pfair * 2.0 * effective_sigma)
            order_size = min(raw_size, cfg.max_inventory)
            logger.info("VolSizing: risk_dollars=%.2f sigma=%.4f pfair=%.2f raw=%.4f final=%.4f",
                        cfg.vol_sizing_risk_dollars, effective_sigma, pfair, raw_size, order_size)
    
        # [DEBUG] Detailed Quote Log
        logger.info(
            "QUOTE %s | Pfair: %.2f | Spread: %.2f bps | Gamma: %.4f | Bid: %.2f | Ask: %.2f | Inv: %.4f",
            cfg.symbol, pfair, (half_spread_val * 2 / pfair * 10000), gamma, bid_price, ask_price, state.inventory
        )

        # Portfolio Scaling (Optional, if we passed portfolio in. Engine handles it usually? 
        # Engine calls strategy. If Engine logic says "scale", it should happen after strategy or inside?
        # Let's keep portfolio scaling OUTSIDE strategy for now to preserve separation of concerns?
        # Or pass PortfolioController in? The plan didn't specify. 
        # The Engine code had `if self.portfolio: order_size = scale(order_size)`.
        # Strategy returns "desired size". Engine can clamp/scale it.
        # But wait, Strategy needs to know final size for some things? No, usually not.
        # Let's assume Strategy returns "Ideal Strategy Size". Engine applies Portfolio constraints.
        
        # Rounding
        order_size = self._round_size(order_size, 0.001)
        
        if order_size <= 0:
            return None

        # 7. Reduce Only Flags
        is_reduce_only_bid = (state.inventory < 0)
        is_reduce_only_ask = (state.inventory > 0)

        # Reduce-Only Sizing Clamp
        if is_reduce_only_bid or is_reduce_only_ask:
            inventory_size = abs(state.inventory)
            if order_size > inventory_size:
                order_size = inventory_size
                order_size = self._round_size(order_size, 0.001)

        return QuoteDecision(
            bid_price=bid_price,
            ask_price=ask_price,
            half_spread=half_spread_val,
            fair_value=pfair,
            sigma=sigma,
            gamma=gamma,
            order_size=order_size,
            is_reduce_only_bid=is_reduce_only_bid,
            is_reduce_only_ask=is_reduce_only_ask,
            cex_event_ts=state.latest_cex_ts,
            cex_recv_ts=state.latest_cex_recv_ts
        )

    def _normalize_price(self, config, price: float) -> float:
        """Helper to round price to tick size."""
        # Using built-in logic or duplicating? Duplicating safe for pure class.
        if price <= 0: return 0.0
        tick = config.tick_size
        return round(price / tick) * tick

    def _round_size(self, size: float, step: float = 0.001) -> float:
        """Helper to round size to step."""
        if size <= 0: return 0.0
        steps = round(size / step)
        return round(steps * step, 9)
