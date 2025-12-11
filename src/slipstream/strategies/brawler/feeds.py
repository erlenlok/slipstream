"""Data feed models shared by the Brawler connectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CexQuote:
    """Represents a Binance best-bid-offer snapshot."""

    symbol: str
    bid: float
    ask: float
    ts: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class LocalQuote:
    """Represents Hyperliquid best-bid-offer snapshot."""

    symbol: str
    bid: float
    ask: float
    ts: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class FillEvent:
    """Individual fill for inventory tracking."""

    symbol: str
    size: float
    price: float
    side: str  # 'buy' if we bought, 'sell' if we sold
    ts: float
    order_id: Optional[str] = None
    fee: float = 0.0
    fee_token: str = "USDC"
    liquidity_type: str = "maker"


__all__ = ["CexQuote", "LocalQuote", "FillEvent"]
