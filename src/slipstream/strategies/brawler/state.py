"""State containers used by the Brawler strategy."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Mapping, Optional

from .config import BrawlerAssetConfig


@dataclass
class OrderSnapshot:
    """Tracks a resting order so we can cancel/replace efficiently."""

    order_id: Optional[str]
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    placed_ts: float = field(default_factory=time.time)


@dataclass
class AssetState:
    """All mutable, per-asset state (basis estimates, vol window, inventory, etc.)."""

    config: BrawlerAssetConfig
    fair_basis: float = 0.0
    cex_mid_window: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    sigma: float = 0.0
    last_cex_mid_ts: float = 0.0
    last_local_mid_ts: float = 0.0
    last_basis: float = 0.0
    last_cex_latency: float = 0.0
    last_local_latency: float = 0.0
    inventory: float = 0.0
    active_bid: Optional[OrderSnapshot] = None
    active_ask: Optional[OrderSnapshot] = None
    suspended_reason: Optional[str] = None
    last_suspend_ts: float = 0.0
    last_quote_ts: float = 0.0

    def update_basis(self, instantaneous_basis: float) -> float:
        """EMA-update for the anchoring basis."""
        alpha = self.config.basis_alpha
        if self.fair_basis == 0.0:
            self.fair_basis = instantaneous_basis
        else:
            self.fair_basis = alpha * instantaneous_basis + (1.0 - alpha) * self.fair_basis
        return self.fair_basis

    def push_cex_mid(self, price: float, timestamp: Optional[float] = None) -> None:
        """Track recent CEX mids for volatility estimation."""
        ts = timestamp or time.time()
        now = time.time()
        self.cex_mid_window.append(price)
        self.last_cex_mid_ts = ts
        self.last_cex_latency = max(0.0, now - ts)

    def push_local_mid(self, timestamp: Optional[float] = None) -> None:
        """Mark the time of the latest Hyperliquid mid-price update."""
        ts = timestamp or time.time()
        now = time.time()
        self.last_local_mid_ts = ts
        self.last_local_latency = max(0.0, now - ts)

    def update_sigma(self) -> float:
        """Recompute rolling sigma using stored CEX mids."""
        window = self.cex_mid_window
        if len(window) < 2:
            self.sigma = 0.0
            return self.sigma

        import math

        values = list(window)
        log_returns = []
        for prev, current in zip(values, values[1:]):
            if current > 0 and prev > 0:
                log_returns.append(math.log(current / prev))

        if len(log_returns) < 2:
            self.sigma = 0.0
            return self.sigma

        mean = sum(log_returns) / len(log_returns)
        variance = sum((x - mean) ** 2 for x in log_returns) / (len(log_returns) - 1)
        self.sigma = math.sqrt(variance)
        return self.sigma

    def mark_suspended(self, reason: str) -> None:
        self.suspended_reason = reason
        self.last_suspend_ts = time.time()

    def clear_suspension(self) -> None:
        self.suspended_reason = None


def build_initial_states(configs: Dict[str, BrawlerAssetConfig]) -> Dict[str, AssetState]:
    """Helper to instantiate state objects for every configured asset."""
    return {symbol: AssetState(config=cfg) for symbol, cfg in configs.items()}


@dataclass
class AssetSnapshot:
    symbol: str
    fair_basis: float
    last_basis: float
    inventory: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "fair_basis": self.fair_basis,
            "last_basis": self.last_basis,
            "inventory": self.inventory,
        }

    @classmethod
    def from_mapping(cls, symbol: str, payload: Mapping[str, float]) -> "AssetSnapshot":
        return cls(
            symbol=symbol,
            fair_basis=float(payload.get("fair_basis") or 0.0),
            last_basis=float(payload.get("last_basis") or 0.0),
            inventory=float(payload.get("inventory") or 0.0),
        )


def capture_state(states: Dict[str, AssetState]) -> Dict[str, AssetSnapshot]:
    return {
        symbol: AssetSnapshot(
            symbol=symbol,
            fair_basis=state.fair_basis,
            last_basis=state.last_basis,
            inventory=state.inventory,
        )
        for symbol, state in states.items()
    }


def restore_state(states: Dict[str, AssetState], snapshots: Mapping[str, AssetSnapshot]) -> None:
    for symbol, snapshot in snapshots.items():
        state = states.get(symbol)
        if not state:
            continue
        state.fair_basis = snapshot.fair_basis
        state.last_basis = snapshot.last_basis
        state.inventory = snapshot.inventory


__all__ = [
    "AssetSnapshot",
    "AssetState",
    "OrderSnapshot",
    "build_initial_states",
    "capture_state",
    "restore_state",
]
