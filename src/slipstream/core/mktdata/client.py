"""Python client for slipstream-mktdata daemon."""

import socket
import struct
from dataclasses import dataclass
from typing import Optional, List

import msgpack


@dataclass
class OrderbookLevel:
    """Single orderbook level (price-size pair)."""
    price: float
    size: float


@dataclass
class OrderbookSnapshot:
    """L2 orderbook snapshot."""
    venue: str
    symbol: str
    timestamp_us: int
    sequence: int
    bids: List[OrderbookLevel]
    asks: List[OrderbookLevel]

    @property
    def best_bid(self) -> Optional[OrderbookLevel]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderbookLevel]:
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        bid = self.best_bid
        ask = self.best_ask
        if bid and ask:
            return (bid.price + ask.price) / 2.0
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        bid = self.best_bid
        ask = self.best_ask
        mid = self.mid_price
        if bid and ask and mid:
            return 10000.0 * (ask.price - bid.price) / mid
        return None


@dataclass
class Candle:
    """OHLCV candle."""
    venue: str
    symbol: str
    timestamp: int  # Unix timestamp (seconds)
    interval: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    num_trades: int


class MarketDataClient:
    """Client for connecting to slipstream-mktdata daemon via Unix socket."""

    def __init__(self, socket_path: str = "/tmp/slipstream-mktdata.sock"):
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None

    def connect(self):
        """Establish connection to daemon."""
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.socket_path)

    def disconnect(self):
        """Close connection to daemon."""
        if self._sock:
            self._sock.close()
            self._sock = None

    def get_candles(
        self,
        venue: str,
        symbol: str,
        interval: str = "4h",
        count: int = 100,
    ) -> List[Candle]:
        """
        Get recent candles for a symbol.

        Args:
            venue: Venue name ("hyperliquid", "binance")
            symbol: Symbol/coin (e.g., "BTC", "ETH")
            interval: Candle interval ("4h", "1h", "15m")
            count: Number of candles to retrieve

        Returns:
            List of Candle objects (oldest first)
        """
        request = {
            "type": "get_candles",
            "venue": venue,
            "symbol": symbol,
            "interval": interval,
            "count": count,
        }

        response = self._send_request(request)

        if response.get("type") == "error":
            raise RuntimeError(f"Error from daemon: {response.get('message')}")

        candles_data = response.get("data", [])
        return [
            Candle(
                venue=c["venue"],
                symbol=c["symbol"],
                timestamp=c["timestamp"],
                interval=c["interval"],
                open=c["open"],
                high=c["high"],
                low=c["low"],
                close=c["close"],
                volume=c["volume"],
                num_trades=c["num_trades"],
            )
            for c in candles_data
        ]

    def get_candles_batch(
        self,
        venue: str,
        symbols: List[str],
        interval: str = "4h",
        count: int = 100,
    ) -> dict[str, List[Candle]]:
        """
        Get candles for multiple symbols in batch.

        Args:
            venue: Venue name
            symbols: List of symbols
            interval: Candle interval
            count: Number of candles per symbol

        Returns:
            Dictionary mapping symbol -> List[Candle]
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_candles(venue, symbol, interval, count)
            except Exception as e:
                print(f"Warning: Failed to fetch candles for {symbol}: {e}")
                result[symbol] = []
        return result

    def get_orderbook(
        self,
        venue: str,
        symbol: str,
    ) -> Optional[OrderbookSnapshot]:
        """
        Get latest orderbook snapshot for a symbol.

        Args:
            venue: Venue name
            symbol: Symbol

        Returns:
            OrderbookSnapshot or None if not available
        """
        request = {
            "type": "get_orderbook",
            "venue": venue,
            "symbol": symbol,
        }

        response = self._send_request(request)

        if response.get("type") == "error":
            return None

        data = response.get("data", {})
        bids = [OrderbookLevel(**level) for level in data.get("bids", [])]
        asks = [OrderbookLevel(**level) for level in data.get("asks", [])]

        return OrderbookSnapshot(
            venue=data["venue"],
            symbol=data["symbol"],
            timestamp_us=data["timestamp_us"],
            sequence=data["sequence"],
            bids=bids,
            asks=asks,
        )

    def _send_request(self, request: dict) -> dict:
        """Send request and receive response (synchronous)."""
        if not self._sock:
            raise RuntimeError("Not connected. Call connect() first.")

        # Encode request with MessagePack
        encoded = msgpack.packb(request)

        # Send length prefix (4 bytes, big-endian)
        length = struct.pack(">I", len(encoded))
        self._sock.sendall(length + encoded)

        # Receive response length
        length_data = self._recv_exactly(4)
        response_length = struct.unpack(">I", length_data)[0]

        # Receive response body
        response_data = self._recv_exactly(response_length)

        # Decode MessagePack
        response = msgpack.unpackb(response_data, raw=False)
        return response

    def _recv_exactly(self, n: int) -> bytes:
        """Receive exactly n bytes from socket."""
        data = b""
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Socket closed by server")
            data += chunk
        return data

    def __enter__(self):
        """Context manager support."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.disconnect()
