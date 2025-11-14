"""Binance futures best-bid-offer streaming client."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Iterable, List, Optional

import websockets

from ..feeds import CexQuote

logger = logging.getLogger(__name__)


class BinanceTickerStream:
    """
    Subscribe to Binance bookTicker streams and publish `CexQuote` events.

    The connector exposes an `asyncio.Queue` so the engine can `await queue.get()`
    without caring about WebSocket plumbing.
    """

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        ws_url: str = "wss://fstream.binance.com",
        reconnect_backoff: float = 2.0,
        queue_maxsize: int = 2000,
    ) -> None:
        self.symbols = [symbol.lower() for symbol in symbols]
        self.ws_url = ws_url.rstrip("/")
        self.reconnect_backoff = reconnect_backoff
        self.queue_maxsize = max(1, queue_maxsize)
        self.queue: asyncio.Queue[CexQuote] = asyncio.Queue(maxsize=self.queue_maxsize)
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="binance-bbo-stream")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        stream = "/".join(f"{symbol}@bookTicker" for symbol in self.symbols)
        target = f"{self.ws_url}/stream?streams={stream}"

        while not self._stop.is_set():
            try:
                async with websockets.connect(target, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("Binance bookTicker stream connected: %s", stream)
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        data = payload.get("data") or payload
                        quote = self._parse_quote(data)
                        if quote:
                            self._publish_quote(quote)
            except Exception as exc:
                logger.warning("Binance stream error: %s", exc, exc_info=exc)
                await asyncio.sleep(self.reconnect_backoff)

    def _parse_quote(self, data) -> Optional[CexQuote]:
        symbol = data.get("s")
        if not symbol:
            return None
        try:
            bid = float(data["b"])
            ask = float(data["a"])
        except (KeyError, TypeError, ValueError):
            return None

        bid_qty = float(data.get("B") or 0)
        ask_qty = float(data.get("A") or 0)
        if bid <= 0 or ask <= 0:
            return None

        ts = data.get("T") or data.get("E") or int(time.time() * 1000)
        return CexQuote(symbol=symbol.upper(), bid=bid, ask=ask, ts=ts / 1000.0)

    def _publish_quote(self, quote: CexQuote) -> None:
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
                logger.warning("Binance queue full; dropping quote for %s", quote.symbol)


__all__ = ["BinanceTickerStream"]
