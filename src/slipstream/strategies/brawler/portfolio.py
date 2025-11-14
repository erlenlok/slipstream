"""Portfolio-level risk controller for Brawler."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .config import BrawlerPortfolioConfig
from .state import AssetState

logger = logging.getLogger(__name__)


class PortfolioController:
    """Tracks aggregate exposure and adjusts quoting/risk budgets."""

    def __init__(self, config: BrawlerPortfolioConfig) -> None:
        self.config = config
        self.gross_inventory: float = 0.0
        self.net_inventory: float = 0.0
        self.ratio: float = 0.0
        self._halted: bool = False
        self.reduce_only: bool = False
        self._reduce_side: Optional[str] = None
        self._last_halt_state: bool = False
        self._last_reduce_state: bool = False

    def update_metrics(self, states: Dict[str, AssetState]) -> None:
        self.gross_inventory = sum(abs(state.inventory) for state in states.values())
        self.net_inventory = sum(state.inventory for state in states.values())

        cap = max(self.config.max_gross_inventory, 0.0)
        self.ratio = (self.gross_inventory / cap) if cap > 0 else 0.0

        halt_ratio = max(self.config.halt_ratio, 0.0)
        self._halted = cap > 0 and halt_ratio > 0 and self.ratio >= halt_ratio

        reduce_ratio = self.config.reduce_only_ratio
        self.reduce_only = False
        self._reduce_side = None
        if (
            cap > 0
            and 0.0 < reduce_ratio < halt_ratio
            and self.ratio >= reduce_ratio
            and abs(self.net_inventory) > 0
            and not self._halted
        ):
            self.reduce_only = True
            self._reduce_side = "sell" if self.net_inventory > 0 else "buy"

        self._log_state_transitions()

    def allow_quotes(self, state: AssetState) -> bool:
        if self.config.max_gross_inventory <= 0:
            return True

        if self._halted:
            if state.suspended_reason != "portfolio":
                state.mark_suspended("portfolio")
            return False

        if state.suspended_reason == "portfolio":
            resume_ratio = min(self.config.resume_ratio, self.config.halt_ratio)
            if resume_ratio <= 0 or self.ratio <= resume_ratio:
                state.clear_suspension()
            else:
                return False

        return True

    def allow_order(self, side: str) -> bool:
        if not self.reduce_only:
            return True
        return side == self._reduce_side

    def scale_order_size(self, base_size: float) -> float:
        if base_size <= 0 or self.config.max_gross_inventory <= 0:
            return base_size

        start = self.config.taper_start_ratio
        if start <= 0 or self.ratio <= start:
            return base_size

        min_ratio = max(0.0, min(1.0, self.config.min_order_size_ratio))
        progress = min(1.0, (self.ratio - start) / max(1e-9, 1.0 - start))
        scale = max(min_ratio, 1.0 - (1.0 - min_ratio) * progress)
        return base_size * scale

    def _log_state_transitions(self) -> None:
        if self._halted and not self._last_halt_state:
            logger.error(
                "Portfolio halt triggered: gross=%.4f ratio=%.2f (limit=%.2f)",
                self.gross_inventory,
                self.ratio,
                self.config.max_gross_inventory,
            )
        if not self._halted and self._last_halt_state:
            logger.info("Portfolio halt cleared: ratio=%.2f", self.ratio)

        if self.reduce_only and not self._last_reduce_state:
            logger.warning(
                "Portfolio reduce-only mode: net=%.4f ratio=%.2f side=%s",
                self.net_inventory,
                self.ratio,
                self._reduce_side,
            )
        if not self.reduce_only and self._last_reduce_state:
            logger.info("Portfolio reduce-only cleared: ratio=%.2f", self.ratio)

        self._last_halt_state = self._halted
        self._last_reduce_state = self.reduce_only


__all__ = ["PortfolioController"]
