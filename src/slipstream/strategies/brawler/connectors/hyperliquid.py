"""Hyperliquid market data and execution helpers for Brawler."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, Optional

import websockets

from ..feeds import FillEvent, LocalQuote

logger = logging.getLogger(__name__)


class HyperliquidQuoteStream:
    """Subscribe to Hyperliquid BBO updates for a collection of symbols."""

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        ws_url: str = "wss://api.hyperliquid.xyz/ws",
        reconnect_backoff: float = 2.0,
        queue_maxsize: int = 2000,
    ) -> None:
        self.symbols = list(symbols)
        self.ws_url = ws_url
        self.reconnect_backoff = reconnect_backoff
        self.queue_maxsize = max(1, queue_maxsize)
        self.queue: asyncio.Queue[LocalQuote] = asyncio.Queue(maxsize=self.queue_maxsize)
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()
        self._ws = None

    async def subscribe(self, symbol: str) -> None:
        """Subscribe to a new symbol at runtime."""
        if symbol in self.symbols:
            return
        self.symbols.append(symbol)
        if self._ws:
            try:
                msg = {
                    "method": "subscribe",
                    "subscription": {"type": "l2Book", "coin": symbol}
                }
                await self._ws.send(json.dumps(msg))
                logger.info("Dynamically subscribed to HL %s", symbol)
            except Exception as exc:
                logger.error("Failed to dynamic subscribe HL %s: %s", symbol, exc)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="hyperliquid-bbo-stream")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10) as ws:
                    self._ws = ws
                    for symbol in self.symbols:
                        msg = {
                            "method": "subscribe",
                            "subscription": {"type": "l2Book", "coin": symbol}
                        }
                        await ws.send(json.dumps(msg))
                    
                    logger.info("Subscribed to Hyperliquid books: %s", ", ".join(self.symbols))

                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        
                        channel = data.get("channel")
                        update = data.get("data")
                        if channel == "l2Book" and update:
                            quote = self._parse(update)
                            if quote:
                                self._publish_quote(quote)
            except Exception as exc:
                logger.warning("Hyperliquid book stream error: %s", exc, exc_info=exc)
                await asyncio.sleep(self.reconnect_backoff)

    def _parse(self, update) -> Optional[LocalQuote]:
        if isinstance(update, str):
            return None
        
        symbol = update.get("coin")
        levels = update.get("levels")
        if not symbol or not levels:
            return None

        # levels is [[bids], [asks]]
        # Each item is {"px": "...", "sz": "...", "n": ...}
        try:
            bids = levels[0]
            asks = levels[1]
            if not bids or not asks:
                return None
            
            # Helper to get price from the level item
            def get_px(item):
                if isinstance(item, dict):
                    return float(item["px"])
                return float(item[0])

            bid_px = get_px(bids[0])
            ask_px = get_px(asks[0])
        except (ValueError, TypeError, IndexError, KeyError):
            return None

        ts = update.get("time") or time.time()
        # Ensure ts is in seconds for the Quote object (API returns millis)
        if ts > 30000000000:  # simplistic check for millis
            ts = ts / 1000.0
            
        return LocalQuote(symbol=symbol, bid=bid_px, ask=ask_px, ts=float(ts))

    def _publish_quote(self, quote: LocalQuote) -> None:
        try:
            self.queue.put_nowait(quote)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.queue.put_nowait(quote)
            except asyncio.QueueFull:
                logger.warning("Hyperliquid quote queue full; dropping %s", quote.symbol)


class HyperliquidUserFillStream:
    """Stream user fills so we can maintain inventory state."""

    def __init__(
        self,
        wallet_address: str,
        *,
        ws_url: str = "wss://api.hyperliquid.xyz/ws",
        reconnect_backoff: float = 2.0,
        queue_maxsize: int = 1000,
    ) -> None:
        self.wallet = wallet_address
        self.ws_url = ws_url
        self.reconnect_backoff = reconnect_backoff
        self.queue_maxsize = max(1, queue_maxsize)
        self.queue: asyncio.Queue[FillEvent] = asyncio.Queue(maxsize=self.queue_maxsize)
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="hyperliquid-user-fills")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        subscribe_msg = json.dumps(
            {
                "type": "subscribe",
                "subscriptions": [
                    {
                        "type": "userFills",
                        "user": self.wallet,
                    },
                ],
            }
        )

        while not self._stop.is_set():
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10) as ws:
                    await ws.send(subscribe_msg)
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        data = payload.get("data")
                        if not data:
                            continue

                        fills = data if isinstance(data, list) else [data]
                        for fill in fills:
                            event = self._parse(fill)
                            if event:
                                self._publish_fill(event)
            except Exception as exc:
                logger.warning("Hyperliquid user fill stream error: %s", exc, exc_info=exc)
                await asyncio.sleep(self.reconnect_backoff)

    def _parse(self, payload) -> Optional[FillEvent]:
        if isinstance(payload, str):
            return None
        symbol = payload.get("coin")
        px = payload.get("px")
        sz = payload.get("sz")
        dir_flag = payload.get("dir")  # 0 buy, 1 sell
        ts = payload.get("time") or time.time()

        if symbol is None or px is None or sz is None or dir_flag is None:
            return None

        try:
            price = float(px)
            size = float(sz)
        except (TypeError, ValueError):
            return None

        side = "buy" if dir_flag == 0 else "sell"
        if side == "buy":
            signed_size = size
        else:
            signed_size = -size

        return FillEvent(
            symbol=symbol,
            size=signed_size,
            price=price,
            side=side,
            ts=float(ts),
            order_id=str(payload.get("oid") or ""),
            fee=float(payload.get("fee") or 0.0),
            fee_token=str(payload.get("feeToken") or "USDC"),
            liquidity_type="taker" if payload.get("crossed") else "maker",
        )

    def _publish_fill(self, event: FillEvent) -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Hyperliquid fill queue full; dropping event for %s", event.symbol)


@dataclass
class HyperliquidOrder:
    symbol: str
    price: float
    size: float
    side: str  # 'buy' for bid, 'sell' for ask
    alo: bool = False


class HyperliquidOrderSide:
    BUY = "buy"
    SELL = "sell"


class HyperliquidOrderType:
    LIMIT = "limit"


@dataclass
class HyperliquidOrderUpdate:
    order_id: Optional[str]
    status: str
    description: Optional[str] = None


def _extract_order_id(response) -> str:
    if isinstance(response, dict):
        status = response.get("status")
        if status == "err":
            msg = response.get("error", "Unknown order error")
            if msg == "Unknown order error":
                logger.error("Hyperliquid raw error response: %s", response)
            raise RuntimeError(msg)
        if "error" in response and status is None:
            raise RuntimeError(response["error"])

        for key in ("response", "data", "resting", "filled", "orders", "statuses", "status"):
            if key in response:
                if isinstance(response[key], str) and key == "error":
                     raise RuntimeError(response[key])
                oid = _extract_order_id(response[key])
                if oid:
                    return oid
        oid = response.get("oid")
        if oid is not None:
            return str(oid)
    elif isinstance(response, list):
        for item in response:
            oid = _extract_order_id(item)
            if oid:
                return oid
    elif isinstance(response, (int, float)):
        return str(response)
    return ""


def _load_hyperliquid_modules():
    try:
        from eth_account import Account  # type: ignore
        from hyperliquid.exchange import Exchange  # type: ignore
        from hyperliquid.info import Info  # type: ignore
        from hyperliquid.utils import constants  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        raise ImportError(
            "hyperliquid package is required for live trading. pip install hyperliquid"
        ) from exc
    return Info, Exchange, constants, Account


class HyperliquidExecutionClient:
    """Thin wrapper around the Hyperliquid SDK for cancel/replace."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = "https://api.hyperliquid.xyz",
        target_wallet: Optional[str] = None,
    ) -> None:
        Info, Exchange, constants, Account = _load_hyperliquid_modules()
        wallet = Account.from_key(api_secret)
        if api_key and wallet.address.lower() != api_key.lower():
            raise ValueError("Hyperliquid API key does not match wallet derived from secret.")
        self.info = Info(base_url=base_url)
        self.exchange = Exchange(
            wallet=wallet,
            base_url=base_url,
            vault_address=target_wallet or wallet.address,
        )
        self.base_url = base_url
        self._lock = asyncio.Lock()

    async def place_limit_order(self, order: HyperliquidOrder) -> HyperliquidOrderUpdate:
        tif = "Alo" if order.alo else "Gtc"
        kwargs = {
            "name": order.symbol,
            "is_buy": order.side == HyperliquidOrderSide.BUY,
            "sz": order.size,
            "limit_px": order.price,
            "order_type": {"limit": {"tif": tif}},
            "reduce_only": False,
        }
        async with self._lock:
            response = await asyncio.to_thread(self.exchange.order, **kwargs)
        order_id = _extract_order_id(response)
        return HyperliquidOrderUpdate(order_id=order_id or None, status="ok")

    async def cancel_order(self, symbol: str, order_id: Optional[str]) -> None:
        if not order_id:
            return
        try:
            oid_int = int(float(order_id))
        except (TypeError, ValueError):
            return
        async with self._lock:
            await asyncio.to_thread(self.exchange.cancel, symbol, oid_int)


class HyperliquidInfoClient:
    """Read-only client for fetching Hyperliquid state."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.hyperliquid.xyz",
        skip_ws: bool = True
    ) -> None:
        Info, _, constants, _ = _load_hyperliquid_modules()
        self.info = Info(base_url=base_url, skip_ws=skip_ws)

    async def get_user_rate_limit(self, wallet_address: str) -> dict:
        """Fetch user rate limit and volume stats."""
        # The SDK user_rate_limit call is synchronous (blocking HTTP)
        return await asyncio.to_thread(self.info.user_rate_limit, wallet_address)



__all__ = [
    "HyperliquidExecutionClient",
    "HyperliquidInfoClient",
    "HyperliquidOrder",
    "HyperliquidOrderUpdate",
    "HyperliquidOrderSide",
    "HyperliquidOrderType",
    "HyperliquidQuoteStream",
    "HyperliquidUserFillStream",
]
