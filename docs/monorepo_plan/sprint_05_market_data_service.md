# Sprint 05 — Centralized Market Data Service

**Duration:** ~3 days
**Objective:** Build a high-performance, shared market data daemon in Rust that eliminates redundant connections and provides sub-millisecond data access for multiple live strategies.

## Problem Statement

**Current Architecture (Inefficient):**
- Each strategy maintains its own websocket connections and HTTP polling
- Gradient: REST API polling every 4h + optional Redis cache
- Brawler: Dual websocket feeds (Hyperliquid BBO + Binance BBO/funding)
- Future strategies: More redundant connections
- **Critical Gap:** No order flow data (trades, aggressor side, order book dynamics) for feature engineering
- **Issues:**
  - Duplicated network overhead (multiple connections to same venues)
  - Inconsistent data (strategies see different snapshots)
  - Rate limit risk (multiple clients hammering same endpoints)
  - Slow cold starts (each strategy fetches full history)
  - No historical order flow data for backtesting with realistic market impact

**Proposed Architecture (Efficient):**
- Single Rust daemon (`slipstream-mktdata`) runs as systemd service
- Maintains persistent websocket connections to all venues
- Keeps in-memory orderbook/candle/trade state with configurable retention
- Exposes **four critical data types:**
  1. **Candles** (4h, 1h, 15m) - for trend strategies
  2. **L2 Orderbook** (snapshots + deltas) - for market making, fair value
  3. **Trades** (price, size, aggressor side) - for flow toxicity features
  4. **Quotes** (BBO updates) - for spread dynamics, latency arbitrage
- Python strategies connect via IPC for ultra-low latency data access
- Separate recorder daemon (`slipstream-recorder`) subscribes and persists all data to Parquet
- Daemon handles reconnection, rate limiting, and data quality centrally

## Outcomes

- Rust daemon serving **four data types** from Hyperliquid + Binance:
  1. **Candles** (4h, 1h, 15m) - OHLCV aggregates
  2. **L2 Orderbook** - Full depth snapshots + incremental deltas
  3. **Trades** - Tick data with aggressor side
  4. **Quotes** - BBO updates (bid/ask price/size)
- Python client library for strategies to subscribe to real-time feeds
- **Recorder daemon** persisting all data to Parquet for backtesting/ML
- Systemd service configuration with automatic restart/monitoring
- 10x+ latency reduction vs current HTTP polling (target: <1ms for local reads)
- **Order flow feature engineering** - toxicity, imbalance, depth decay, etc.

## Technical Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     slipstream-mktdata (Rust)                           │
│                                                                         │
│  ┌──────────────────────┐         ┌──────────────────────┐             │
│  │  Hyperliquid WS      │         │  Binance Futures WS  │             │
│  │  ├─ allMids (4h/1h)  │         │  ├─ kline (4h/1h)    │             │
│  │  ├─ l2Book (snapshots)│        │  ├─ depth@100ms      │             │
│  │  ├─ trades           │         │  ├─ aggTrade         │             │
│  │  └─ user (fills)     │         │  └─ bookTicker (BBO) │             │
│  └──────────┬───────────┘         └──────────┬───────────┘             │
│             │                                 │                         │
│             └─────────────────┬───────────────┘                         │
│                               │                                         │
│                      ┌────────▼────────┐                                │
│                      │   Aggregator    │                                │
│                      │   (normalize)   │                                │
│                      └────────┬────────┘                                │
│                               │                                         │
│       ┌───────────────────────┼───────────────────────┐                 │
│       │                       │                       │                 │
│  ┌────▼─────┐  ┌──────────────▼──────┐  ┌────────────▼────────┐       │
│  │ Candles  │  │   L2 Orderbook      │  │  Trades & Quotes    │       │
│  │ RingBuf  │  │   ├─ Snapshot       │  │  ├─ Recent trades   │       │
│  │ per      │  │   ├─ Deltas         │  │  ├─ Aggressor side  │       │
│  │ interval │  │   └─ BBO cache      │  │  └─ BBO history     │       │
│  └──────────┘  └─────────────────────┘  └─────────────────────┘       │
│                                                                         │
│                      ┌─────────────────────┐                            │
│                      │   IPC Server        │                            │
│                      │   (Unix Socket)     │                            │
│                      └──────────┬──────────┘                            │
└─────────────────────────────────┼────────────────────────────────────┘
                                  │
                 ┌────────────────┼────────────────┬─────────────┐
                 │                │                │             │
          ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐  ┌──▼──────────┐
          │  Gradient   │  │  Brawler    │  │ Strategy N │  │  Recorder   │
          │  (candles)  │  │  (L2 + BBO) │  │  (trades)  │  │  (all data) │
          └─────────────┘  └─────────────┘  └────────────┘  └──────┬──────┘
                                                                    │
                                                            ┌───────▼────────┐
                                                            │  Parquet Files │
                                                            │  /data/live/   │
                                                            │  ├─ candles/   │
                                                            │  ├─ l2book/    │
                                                            │  ├─ trades/    │
                                                            │  └─ quotes/    │
                                                            └────────────────┘
```

### Core Components

#### 1. Rust Daemon (`src-rs/mktdata-daemon/`)

**Crates:**
- `tokio` - Async runtime
- `tokio-tungstenite` - WebSocket client
- `serde` / `serde_json` - JSON serialization
- `tokio-serde` + `bincode` - Binary protocol for IPC
- `dashmap` - Concurrent hashmap for orderbook state
- `crossbeam-channel` - Lock-free channels
- `tracing` + `tracing-subscriber` - Structured logging
- `systemd` - Systemd integration (watchdog, notifications)

**Modules:**
```rust
// src-rs/mktdata-daemon/src/
├── main.rs              // Entry point, systemd setup
├── config.rs            // TOML config parsing
├── venues/
│   ├── hyperliquid.rs   // HL websocket client
│   ├── binance.rs       // Binance futures websocket
│   └── trait.rs         // Common VenueClient trait
├── state/
│   ├── candles.rs       // RingBuffer for OHLCV
│   ├── orderbook.rs     // L2 orderbook (bids/asks)
│   └── trades.rs        // Recent trade history
├── ipc/
│   ├── server.rs        // Unix socket server
│   ├── protocol.rs      // MessagePack/bincode wire format
│   └── subscription.rs  // Client subscription management
└── health.rs            // Health checks, metrics export
```

**Key Data Structures:**

```rust
// === L2 Orderbook ===
#[derive(Serialize, Deserialize, Clone)]
pub struct OrderbookLevel {
    pub price: f64,
    pub size: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OrderbookSnapshot {
    pub venue: String,
    pub symbol: String,
    pub timestamp_us: u64,
    pub sequence: u64,  // For delta reconciliation
    pub bids: Vec<OrderbookLevel>,  // Sorted desc by price
    pub asks: Vec<OrderbookLevel>,  // Sorted asc by price
}

impl OrderbookSnapshot {
    pub fn best_bid(&self) -> Option<&OrderbookLevel> { self.bids.first() }
    pub fn best_ask(&self) -> Option<&OrderbookLevel> { self.asks.first() }
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / 2.0),
            _ => None,
        }
    }
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                Some(10000.0 * (ask.price - bid.price) / self.mid_price().unwrap())
            }
            _ => None,
        }
    }
}

// === Trade ===
#[derive(Serialize, Deserialize, Clone)]
pub struct Trade {
    pub venue: String,
    pub symbol: String,
    pub timestamp_us: u64,
    pub price: f64,
    pub size: f64,
    pub side: TradeSide,  // Which side was aggressor
    pub trade_id: String, // Venue-specific ID
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum TradeSide {
    Buy,   // Aggressor was buyer (lifted offer)
    Sell,  // Aggressor was seller (hit bid)
}

// === Quote (BBO) ===
#[derive(Serialize, Deserialize, Clone)]
pub struct Quote {
    pub venue: String,
    pub symbol: String,
    pub timestamp_us: u64,
    pub bid_price: f64,
    pub bid_size: f64,
    pub ask_price: f64,
    pub ask_size: f64,
}

// === Candle ===
#[derive(Serialize, Deserialize, Clone)]
pub struct Candle {
    pub venue: String,
    pub symbol: String,
    pub timestamp: u64,  // Unix timestamp (start of period)
    pub interval: String,  // "4h", "1h", "15m"
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub num_trades: u64,  // Number of trades in period
}

// === Subscription Requests ===
#[derive(Serialize, Deserialize)]
pub enum SubscriptionRequest {
    // L2 orderbook (full depth or top N levels)
    Orderbook { venue: String, symbol: String, depth: usize },

    // BBO only (faster, lower bandwidth)
    Quote { venue: String, symbol: String },

    // Trade feed
    Trades { venue: String, symbol: String },

    // Candles (specific interval)
    Candles { venue: String, symbol: String, interval: String },

    // Bulk subscriptions
    AllCandles { venue: String, interval: String },  // All symbols
    AllTrades { venue: String },  // All symbols
}

// === Messages sent to clients ===
#[derive(Serialize, Deserialize)]
pub enum MarketDataMessage {
    OrderbookSnapshot(OrderbookSnapshot),
    OrderbookDelta(OrderbookDelta),  // Incremental updates
    Quote(Quote),
    Trade(Trade),
    Candle(Candle),
    Heartbeat { timestamp_us: u64 },
    Error { code: String, message: String },
}

// Incremental orderbook update (for bandwidth optimization)
#[derive(Serialize, Deserialize, Clone)]
pub struct OrderbookDelta {
    pub venue: String,
    pub symbol: String,
    pub timestamp_us: u64,
    pub sequence: u64,
    pub bid_changes: Vec<OrderbookLevel>,  // price=0 means delete
    pub ask_changes: Vec<OrderbookLevel>,
}
```

#### 2. Python Client Library (`src/slipstream/core/mktdata/`)

```python
# src/slipstream/core/mktdata/client.py

import socket
import msgpack
from typing import Iterator, Callable, Optional
from dataclasses import dataclass

@dataclass
class OrderbookSnapshot:
    venue: str
    symbol: str
    timestamp_us: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]

    @property
    def mid_price(self) -> float:
        return (self.bids[0][0] + self.asks[0][0]) / 2.0

class MarketDataClient:
    """Client for connecting to slipstream-mktdata daemon."""

    def __init__(self, socket_path: str = "/var/run/slipstream/mktdata.sock"):
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None
        self._unpacker = msgpack.Unpacker(raw=False)

    def connect(self):
        """Establish connection to daemon."""
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.socket_path)

    def subscribe_orderbook(
        self,
        venue: str,
        symbol: str,
        depth: int = 10
    ) -> None:
        """Subscribe to orderbook updates."""
        req = {
            "type": "subscribe_orderbook",
            "venue": venue,
            "symbol": symbol,
            "depth": depth,
        }
        self._send(req)

    def subscribe_candles(
        self,
        venue: str,
        symbols: Optional[list[str]] = None,
        interval: str = "4h"
    ) -> None:
        """Subscribe to candle updates."""
        if symbols is None:
            # Subscribe to all symbols
            req = {
                "type": "subscribe_all_candles",
                "venue": venue,
                "interval": interval,
            }
        else:
            for symbol in symbols:
                req = {
                    "type": "subscribe_candles",
                    "venue": venue,
                    "symbol": symbol,
                    "interval": interval,
                }
                self._send(req)

    def stream(self) -> Iterator[dict]:
        """Stream market data updates."""
        while True:
            data = self._sock.recv(4096)
            if not data:
                raise ConnectionError("Socket closed")
            self._unpacker.feed(data)
            for msg in self._unpacker:
                yield msg

    def get_latest_orderbook(self, venue: str, symbol: str) -> Optional[OrderbookSnapshot]:
        """Synchronous request for latest orderbook (no subscription)."""
        req = {"type": "get_orderbook", "venue": venue, "symbol": symbol}
        self._send(req)
        resp = next(self.stream())
        if resp["type"] == "orderbook_snapshot":
            return OrderbookSnapshot(**resp["data"])
        return None

    def _send(self, msg: dict):
        packed = msgpack.packb(msg)
        self._sock.sendall(packed)
```

#### 3. Configuration (`/etc/slipstream/mktdata.toml`)

```toml
[daemon]
socket_path = "/var/run/slipstream/mktdata.sock"
log_level = "info"
log_path = "/var/log/slipstream/mktdata.log"

[retention]
# How much history to keep in memory
candles_4h_count = 1200  # ~200 days
candles_1h_count = 2000  # ~83 days
orderbook_snapshots_per_symbol = 100
trades_per_symbol = 1000

[venues.hyperliquid]
enabled = true
websocket_url = "wss://api.hyperliquid.xyz/ws"
reconnect_delay_ms = 1000
max_reconnect_delay_ms = 30000

[venues.binance]
enabled = true
websocket_url = "wss://fstream.binance.com/ws"
reconnect_delay_ms = 1000
max_reconnect_delay_ms = 30000

# Optional: Rate limiting per venue
[venues.hyperliquid.rate_limits]
max_subscriptions = 100
max_messages_per_second = 50

[health]
# Expose Prometheus metrics on this port
metrics_port = 9090
# Systemd watchdog interval
watchdog_interval_ms = 5000
```

#### 4. Systemd Service (`/etc/systemd/system/slipstream-mktdata.service`)

```ini
[Unit]
Description=Slipstream Market Data Service
After=network.target
Wants=network-online.target

[Service]
Type=notify
User=slipstream
Group=slipstream
RuntimeDirectory=slipstream
LogsDirectory=slipstream

ExecStart=/usr/local/bin/slipstream-mktdata --config /etc/slipstream/mktdata.toml
Restart=always
RestartSec=5

# Systemd watchdog (daemon must notify within 10s)
WatchdogSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/run/slipstream /var/log/slipstream

# Resource limits
MemoryMax=2G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
```

#### 5. Recorder Daemon (`src-rs/recorder-daemon/` or Python)

**Purpose:** Subscribe to mktdata daemon and persist all market data to disk for:
- Backtesting with realistic order flow
- ML feature engineering (order imbalance, toxicity, etc.)
- Compliance/audit trail
- Post-trade analysis

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│         slipstream-recorder (Python)            │
│                                                 │
│  ┌────────────────────────────────────────┐    │
│  │   MarketDataClient (subscriber)        │    │
│  │   ├─ Subscribe to all candles          │    │
│  │   ├─ Subscribe to all trades           │    │
│  │   ├─ Subscribe to L2 snapshots (1s)    │    │
│  │   └─ Subscribe to quotes (BBO)         │    │
│  └─────────────────┬──────────────────────┘    │
│                    │                            │
│  ┌─────────────────▼──────────────────────┐    │
│  │   Writers (async batching)             │    │
│  │   ├─ CandleWriter  → Parquet           │    │
│  │   ├─ TradeWriter   → Parquet           │    │
│  │   ├─ L2BookWriter  → Parquet (1s snap) │    │
│  │   └─ QuoteWriter   → Parquet (tick)    │    │
│  └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
                     │
             ┌───────▼────────┐
             │  /data/live/   │
             │  ├─ candles/   │
             │  │  ├─ HL/     │
             │  │  └─ BN/     │
             │  ├─ trades/    │
             │  ├─ l2book/    │
             │  └─ quotes/    │
             └────────────────┘
```

**Implementation (Python):**

```python
# scripts/daemons/recorder.py

import asyncio
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from slipstream.core.mktdata import MarketDataClient

class ParquetWriter:
    """Batched Parquet writer for market data."""

    def __init__(self, base_path: Path, data_type: str):
        self.base_path = base_path / data_type
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.batch_size = 1000
        self.flush_interval = 60  # seconds

    async def write(self, record: dict):
        """Add record to buffer, flush if needed."""
        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size:
            await self.flush()

    async def flush(self):
        """Write buffer to Parquet file."""
        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer)

        # Partition by venue and date
        for (venue, date), group in df.groupby(['venue', df['timestamp'].dt.date]):
            path = self.base_path / venue / f"{date}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)

            # Append to existing file or create new
            if path.exists():
                existing = pq.read_table(path)
                combined = pa.concat_tables([existing, pa.Table.from_pandas(group)])
                pq.write_table(combined, path)
            else:
                pq.write_table(pa.Table.from_pandas(group), path)

        self.buffer.clear()

class MarketDataRecorder:
    """Main recorder daemon."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.client = MarketDataClient()
        self.writers = {
            'candles': ParquetWriter(Path(self.config['output_path']), 'candles'),
            'trades': ParquetWriter(Path(self.config['output_path']), 'trades'),
            'l2book': ParquetWriter(Path(self.config['output_path']), 'l2book'),
            'quotes': ParquetWriter(Path(self.config['output_path']), 'quotes'),
        }

    async def run(self):
        """Main loop: subscribe and record."""
        self.client.connect()

        # Subscribe to all data types
        for venue in self.config['venues']:
            # Candles
            for interval in ['4h', '1h', '15m']:
                self.client.subscribe_candles(venue, symbols=None, interval=interval)

            # Trades (all symbols)
            self.client.subscribe({"type": "subscribe_all_trades", "venue": venue})

            # L2 snapshots (every 1s, top 20 levels)
            for symbol in self.config.get('record_orderbook_symbols', []):
                self.client.subscribe_orderbook(venue, symbol, depth=20)

            # BBO quotes (all symbols, tick-by-tick)
            for symbol in self.config.get('record_quote_symbols', []):
                self.client.subscribe_quote(venue, symbol)

        # Process incoming messages
        print("Recorder started. Listening for market data...")
        for msg in self.client.stream():
            await self._handle_message(msg)

    async def _handle_message(self, msg: dict):
        """Route message to appropriate writer."""
        msg_type = msg.get('type')

        if msg_type == 'candle':
            await self.writers['candles'].write({
                'timestamp': pd.Timestamp(msg['data']['timestamp'], unit='s'),
                'venue': msg['data']['venue'],
                'symbol': msg['data']['symbol'],
                'interval': msg['data']['interval'],
                'open': msg['data']['open'],
                'high': msg['data']['high'],
                'low': msg['data']['low'],
                'close': msg['data']['close'],
                'volume': msg['data']['volume'],
                'num_trades': msg['data']['num_trades'],
            })

        elif msg_type == 'trade':
            await self.writers['trades'].write({
                'timestamp': pd.Timestamp(msg['data']['timestamp_us'], unit='us'),
                'venue': msg['data']['venue'],
                'symbol': msg['data']['symbol'],
                'price': msg['data']['price'],
                'size': msg['data']['size'],
                'side': msg['data']['side'],
                'trade_id': msg['data']['trade_id'],
            })

        elif msg_type == 'orderbook_snapshot':
            # Only record top 20 levels to save space
            ob = msg['data']
            await self.writers['l2book'].write({
                'timestamp': pd.Timestamp(ob['timestamp_us'], unit='us'),
                'venue': ob['venue'],
                'symbol': ob['symbol'],
                'bids': ob['bids'][:20],  # Top 20
                'asks': ob['asks'][:20],
                'mid': (ob['bids'][0]['price'] + ob['asks'][0]['price']) / 2 if ob['bids'] and ob['asks'] else None,
                'spread_bps': self._calc_spread_bps(ob),
            })

        elif msg_type == 'quote':
            await self.writers['quotes'].write({
                'timestamp': pd.Timestamp(msg['data']['timestamp_us'], unit='us'),
                'venue': msg['data']['venue'],
                'symbol': msg['data']['symbol'],
                'bid_price': msg['data']['bid_price'],
                'bid_size': msg['data']['bid_size'],
                'ask_price': msg['data']['ask_price'],
                'ask_size': msg['data']['ask_size'],
            })

    def _calc_spread_bps(self, ob: dict) -> float:
        if not ob['bids'] or not ob['asks']:
            return None
        mid = (ob['bids'][0]['price'] + ob['asks'][0]['price']) / 2.0
        return 10000.0 * (ob['asks'][0]['price'] - ob['bids'][0]['price']) / mid

if __name__ == '__main__':
    import sys
    recorder = MarketDataRecorder(sys.argv[1] if len(sys.argv) > 1 else '/etc/slipstream/recorder.toml')
    asyncio.run(recorder.run())
```

**Configuration (`/etc/slipstream/recorder.toml`):**

```toml
output_path = "/data/live"

venues = ["hyperliquid", "binance"]

# Record L2 orderbook for these symbols (expensive, selective)
record_orderbook_symbols = ["BTC", "ETH", "SOL"]

# Record BBO quotes for all traded symbols (cheaper)
record_quote_symbols = ["BTC", "ETH", "SOL", "ATOM", "DOGE"]

[batching]
batch_size = 1000  # Records per flush
flush_interval_seconds = 60
```

**Systemd Service (`/etc/systemd/system/slipstream-recorder.service`):**

```ini
[Unit]
Description=Slipstream Market Data Recorder
After=slipstream-mktdata.service
Requires=slipstream-mktdata.service

[Service]
Type=simple
User=slipstream
Group=slipstream

ExecStart=/usr/bin/python3 /opt/slipstream/scripts/daemons/recorder.py /etc/slipstream/recorder.toml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Data Layout:**

```
/data/live/
├── candles/
│   ├── hyperliquid/
│   │   ├── 2025-01-15.parquet  # All symbols, all intervals for this date
│   │   ├── 2025-01-16.parquet
│   │   └── ...
│   └── binance/
│       └── 2025-01-15.parquet
├── trades/
│   ├── hyperliquid/
│   │   ├── 2025-01-15.parquet  # Partitioned by venue + date
│   │   └── ...
│   └── binance/
├── l2book/
│   └── hyperliquid/
│       ├── BTC/
│       │   ├── 2025-01-15.parquet  # 1s snapshots
│       │   └── ...
│       └── ETH/
└── quotes/
    └── hyperliquid/
        ├── BTC/
        │   └── 2025-01-15.parquet  # Tick-by-tick BBO
        └── ...
```

**Usage in Backtests:**

```python
# Load recorded trades for impact analysis
trades = pd.read_parquet('/data/live/trades/hyperliquid/2025-01-15.parquet')
btc_trades = trades[trades['symbol'] == 'BTC']

# Compute order flow imbalance feature
def compute_imbalance(trades_df, window='1min'):
    trades_df['signed_volume'] = trades_df.apply(
        lambda x: x['size'] if x['side'] == 'Buy' else -x['size'],
        axis=1
    )
    return trades_df.set_index('timestamp')['signed_volume'].resample(window).sum()

# Load L2 snapshots for liquidity analysis
l2 = pd.read_parquet('/data/live/l2book/hyperliquid/BTC/2025-01-15.parquet')
depth_at_10bps = l2.apply(lambda row: sum_depth_within_bps(row['bids'], row['asks'], 10), axis=1)
```

#### Order Flow Features (Examples)

With recorded trades, orderbook, and quotes, you can build sophisticated features:

**1. Flow Toxicity (Aggressor Imbalance)**
```python
def compute_toxicity(trades: pd.DataFrame, window='5min') -> pd.Series:
    """Measure how aggressive recent flow is (buy pressure vs sell pressure)."""
    trades['signed_notional'] = trades.apply(
        lambda x: x['price'] * x['size'] if x['side'] == 'Buy' else -x['price'] * x['size'],
        axis=1
    )
    return trades.set_index('timestamp')['signed_notional'].resample(window).sum()
```

**2. Depth Imbalance (L2)**
```python
def compute_depth_imbalance(l2_snapshot: dict, depth_bps=10) -> float:
    """Measure bid/ask depth imbalance within N bps of mid."""
    mid = l2_snapshot['mid']
    bid_depth = sum(
        level['size'] for level in l2_snapshot['bids']
        if level['price'] >= mid * (1 - depth_bps/10000)
    )
    ask_depth = sum(
        level['size'] for level in l2_snapshot['asks']
        if level['price'] <= mid * (1 + depth_bps/10000)
    )
    return (bid_depth - ask_depth) / (bid_depth + ask_depth)
```

**3. Spread Decay (Quote Velocity)**
```python
def compute_spread_decay(quotes: pd.DataFrame) -> pd.Series:
    """How fast is the spread widening/tightening?"""
    quotes['spread'] = quotes['ask_price'] - quotes['bid_price']
    return quotes['spread'].diff() / quotes['timestamp'].diff().dt.total_seconds()
```

**4. Order Arrival Rate**
```python
def compute_arrival_rate(trades: pd.DataFrame, window='10s') -> pd.Series:
    """Trades per second (proxy for order flow intensity)."""
    return trades.set_index('timestamp').resample(window).size() / window_seconds
```

**5. Volume-Weighted Average Price (VWAP) Deviation**
```python
def compute_vwap_deviation(trades: pd.DataFrame, window='1h') -> pd.Series:
    """How far is current price from VWAP?"""
    trades['notional'] = trades['price'] * trades['size']
    resampled = trades.set_index('timestamp').resample(window).agg({
        'notional': 'sum',
        'size': 'sum',
        'price': 'last',
    })
    resampled['vwap'] = resampled['notional'] / resampled['size']
    return (resampled['price'] - resampled['vwap']) / resampled['vwap']
```

These features are critical for:
- **Brawler:** Detect toxic flow, widen spreads when aggressor imbalance is high
- **Aggressive takers:** Time entries when depth imbalance favors your side
- **ML models:** Predict short-term price moves from microstructure

### Integration with Strategies

#### Gradient Strategy Integration

**Before (current):**
```python
# src/slipstream/strategies/gradient/live/data.py
async def fetch_candles_for_asset(asset: str, endpoint: str, ...):
    # HTTP request to Hyperliquid API
    response = await client.post(f"{endpoint}/info", ...)
```

**After (with daemon):**
```python
# src/slipstream/strategies/gradient/live/data.py
from slipstream.core.mktdata import MarketDataClient

def fetch_live_data(config) -> Dict[str, Any]:
    """Fetch latest 4h candle data using mktdata daemon."""
    client = MarketDataClient()
    client.connect()

    # Get all available symbols
    symbols = client.get_symbols(venue="hyperliquid")

    # Request latest 1100 candles for each symbol (synchronous batch request)
    candles_by_symbol = client.get_candles_batch(
        venue="hyperliquid",
        symbols=symbols,
        interval="4h",
        count=1100,
    )

    # Convert to panel format
    panel = _candles_to_panel(candles_by_symbol)
    return {"panel": panel, "assets": symbols}
```

#### Brawler Strategy Integration

**Before (planned):**
```python
# Brawler needs real-time BBO from both venues
while True:
    hl_orderbook = await fetch_hl_orderbook(symbol)
    cex_orderbook = await fetch_binance_orderbook(symbol)
    # ... compute fair value, quote ...
```

**After (with daemon):**
```python
from slipstream.core.mktdata import MarketDataClient

def run_brawler(config):
    client = MarketDataClient()
    client.connect()

    # Subscribe to both orderbooks
    client.subscribe_orderbook(venue="hyperliquid", symbol="BTC", depth=5)
    client.subscribe_orderbook(venue="binance", symbol="BTCUSDT", depth=5)

    # Stream updates (sub-millisecond latency)
    for msg in client.stream():
        if msg["type"] == "orderbook_update":
            ob = OrderbookSnapshot(**msg["data"])
            if ob.venue == "hyperliquid":
                hl_mid = ob.mid_price
            else:
                cex_mid = ob.mid_price

            # Compute basis, update quotes
            basis = hl_mid - cex_mid
            update_quotes(basis, hl_mid)
```

## Implementation Plan

### Phase 1: Core Daemon (Day 1)

**Goal:** Rust daemon connects to Hyperliquid websocket, stores candles

1. **Setup Rust workspace** (`src-rs/`)
   ```bash
   cargo new --bin mktdata-daemon
   cd mktdata-daemon
   cargo add tokio tokio-tungstenite serde serde_json tracing
   ```

2. **Implement Hyperliquid websocket client**
   - Subscribe to `allMids` channel for candle data
   - Parse JSON messages into `Candle` structs
   - Store in `DashMap<(String, String), RingBuffer<Candle>>`

3. **Basic Unix socket IPC server**
   - Accept connections on `/var/run/slipstream/mktdata.sock`
   - Handle `get_candles` request
   - Return MessagePack-encoded response

4. **Configuration loader**
   - Parse `/etc/slipstream/mktdata.toml`
   - Support environment variable overrides

**Deliverable:** Daemon runs, accepts connections, serves candle data

### Phase 2: Python Client + Gradient Integration (Day 1-2)

**Goal:** Python strategies can query daemon for candles

1. **Python client library**
   - `MarketDataClient` class with Unix socket connection
   - Synchronous request/response methods
   - MessagePack encoding/decoding

2. **Gradient data.py refactor**
   - Replace HTTP polling with daemon queries
   - Maintain same interface (`fetch_live_data()` signature unchanged)
   - Add fallback to HTTP if daemon unavailable

3. **Testing**
   - Unit tests for client library
   - Integration test: start daemon, run Gradient dry-run

**Deliverable:** Gradient works with daemon, 10x faster cold start

### Phase 3: Orderbook Streaming + Brawler (Day 2-3)

**Goal:** Real-time orderbook updates for market making

1. **Orderbook state management**
   - L2 orderbook reconstruction from websocket deltas
   - Maintain snapshot per symbol in `DashMap`

2. **Streaming subscriptions**
   - Clients can subscribe to orderbook updates
   - Server pushes updates via async channel
   - Handle backpressure (drop old updates if client slow)

3. **Binance websocket client**
   - Subscribe to `depth@100ms` stream
   - Parse and normalize to common `OrderbookSnapshot` format

4. **Brawler integration**
   - Refactor Brawler to use streaming client
   - Benchmark latency (target: <1ms for local updates)

**Deliverable:** Brawler runs with sub-millisecond data access

### Phase 4: Production Hardening (Day 3)

**Goal:** Service is production-ready

1. **Systemd integration**
   - Implement `sd_notify` watchdog
   - Graceful shutdown on SIGTERM
   - Automatic restart on crash

2. **Monitoring**
   - Prometheus metrics exporter
   - Key metrics: connection status, message rate, subscriber count, memory usage
   - Health check endpoint

3. **Reconnection logic**
   - Exponential backoff on websocket disconnect
   - Replay missed messages if possible
   - Alert on prolonged disconnection

4. **Documentation**
   - Deployment guide
   - Client library API reference
   - Troubleshooting runbook

**Deliverable:** Service runs reliably, monitored, auto-recovers

## Testing Strategy

### Unit Tests (Rust)

```rust
#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_orderbook_update() {
        let mut ob = Orderbook::new("BTC");
        ob.apply_delta(bids, asks);
        assert_eq!(ob.best_bid(), Some(50000.0));
    }

    #[tokio::test]
    async fn test_candle_ringbuffer() {
        let mut store = CandleStore::new(10);
        store.push(candle);
        assert_eq!(store.len(), 1);
    }
}
```

### Integration Tests (Python)

```python
def test_daemon_candle_fetch(daemon_fixture):
    """Test fetching candles from running daemon."""
    client = MarketDataClient()
    client.connect()

    candles = client.get_candles(
        venue="hyperliquid",
        symbol="BTC",
        interval="4h",
        count=100,
    )

    assert len(candles) == 100
    assert candles[0].symbol == "BTC"

def test_orderbook_streaming(daemon_fixture):
    """Test real-time orderbook subscription."""
    client = MarketDataClient()
    client.connect()
    client.subscribe_orderbook(venue="hyperliquid", symbol="BTC")

    updates = []
    for msg in client.stream():
        if msg["type"] == "orderbook_update":
            updates.append(msg)
            if len(updates) >= 10:
                break

    assert len(updates) == 10
    assert all(u["data"]["symbol"] == "BTC" for u in updates)
```

### Load Testing

```python
# Simulate 10 concurrent strategies subscribing
async def load_test():
    clients = [MarketDataClient() for _ in range(10)]
    for client in clients:
        client.connect()
        client.subscribe_orderbook("hyperliquid", "BTC")

    # Measure latency distribution
    latencies = []
    for _ in range(1000):
        start = time.perf_counter()
        msg = await client.recv()
        latency = time.perf_counter() - start
        latencies.append(latency)

    print(f"p50: {np.percentile(latencies, 50)*1000:.2f}ms")
    print(f"p99: {np.percentile(latencies, 99)*1000:.2f}ms")
```

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Rust learning curve slows development | Medium | Medium | Pair with experienced Rust dev, use well-documented crates |
| Daemon crash disrupts all strategies | High | Low | Systemd auto-restart, watchdog monitoring, fallback to HTTP |
| Memory leak from unbounded growth | High | Medium | Strict retention limits, periodic cleanup, memory profiling |
| Websocket disconnect causes stale data | High | Medium | Reconnection with exponential backoff, staleness detection |
| IPC socket permission issues | Medium | Medium | Proper systemd user/group config, filesystem ACLs |
| Performance worse than expected | Low | Low | Benchmark early (Phase 2), optimize hot paths with `perf` |

## Success Criteria

1. **Performance:**
   - Cold start time for Gradient: <5 seconds (down from ~60s)
   - Orderbook update latency: p99 < 1ms (local IPC)
   - Trade feed latency: p99 < 500μs (microseconds)
   - Memory footprint: <500MB for 100 symbols with full retention (candles + L2 + trades)

2. **Reliability:**
   - Uptime: 99.9% (max 43 minutes downtime/month)
   - Automatic recovery from websocket disconnect within 30s
   - No data loss during daemon restart (graceful shutdown)
   - Recorder writes data with <1s latency, zero message loss

3. **Usability:**
   - Strategies can integrate with <10 lines of code
   - Clear error messages and logging
   - Health endpoint responds in <10ms

4. **Data Quality:**
   - Recorded orderbook snapshots match live daemon state
   - Trade aggressor side correctly identified (>99% accuracy)
   - No duplicate or missing candles in Parquet files

## Future Enhancements (Post-Sprint)

1. **Historical replay mode** - Replay recorded Parquet data at 1x/10x/100x speed for backtesting
2. **Cross-venue arbitrage detection** - Built-in basis monitoring, alert on >X bps deviation
3. **Smart order routing hints** - Recommend best venue based on current liquidity/spread
4. **Order flow ML features** - Pre-computed toxicity, imbalance, depth metrics (served via IPC)
5. **L3 Orderbook** - Full order-by-order data (Hyperliquid supports this)
6. **Execution quality analytics** - Track slippage vs VWAP, adverse selection
7. **gRPC API** - Alternative to Unix sockets for remote clients (multi-server)
8. **Compressed snapshots** - Use zstd compression for L2 snapshots to save memory
9. **Multi-node support** - Run multiple daemons with client-side load balancing

## Dependencies & Coordination

- **DevOps:** Systemd service deployment, monitoring integration
- **Quant:** Validate data quality matches current HTTP polling
- **Trading Ops:** Schedule deployment window, test failover to HTTP fallback

## Metrics

- Daemon uptime tracked via systemd
- Message throughput: Prometheus `mktdata_messages_received_total`
- Active subscriptions: Prometheus `mktdata_active_subscriptions`
- Latency: `mktdata_ipc_latency_seconds` histogram

## Open Questions

1. **IPC protocol choice:** Unix socket + MessagePack vs shared memory vs Redis pub/sub?
   - **Recommendation:** Start with Unix socket (simple, fast), add shared memory if needed
2. **Data persistence:** Should daemon persist state to disk on shutdown?
   - **Recommendation:** No for MVP, strategies handle persistence. Add in future if needed.
3. **Authentication:** Do we need to authenticate client connections?
   - **Recommendation:** No for MVP (local-only), filesystem permissions sufficient

## Summary

This sprint delivers a **production-grade market data infrastructure** that fundamentally changes how Slipstream strategies access market data:

### Key Capabilities

**Data Types (All Four):**
- ✅ **Candles** - OHLCV at multiple intervals (4h, 1h, 15m)
- ✅ **L2 Orderbook** - Full depth, snapshots + deltas, sub-ms latency
- ✅ **Trades** - Tick data with aggressor side identification
- ✅ **Quotes** - BBO updates for spread dynamics

**Architecture:**
- ✅ **Single source of truth** - One daemon, all strategies share data
- ✅ **Ultra-low latency** - <1ms for orderbook, <500μs for trades (local IPC)
- ✅ **Persistent recording** - All data saved to Parquet for backtesting/ML
- ✅ **Production ready** - Systemd integration, auto-restart, monitoring

**Value Proposition:**
1. **10x faster** - Gradient cold start: 60s → 5s
2. **Enables Brawler** - Real-time orderbook needed for market making
3. **Order flow features** - Toxicity, imbalance, depth for ML models
4. **Realistic backtests** - Use actual L2/trade data instead of synthetic
5. **Cost reduction** - One connection vs N connections (rate limit safety)

### Use Cases Unlocked

- **Passive market making** (Brawler) - Fair value from CEX anchor + local L2
- **Aggressive taking** - Time entries when depth imbalance favors you
- **Toxicity detection** - Widen spreads when flow is toxic
- **Microstructure ML** - Predict short-term moves from order flow
- **Execution TCA** - Measure slippage vs VWAP, adverse selection

---

**Status:** Planning
**Owner:** TBD
**Target Start:** After Sprint 04 completion
**Dependencies:** None (can run in parallel with other strategy work)
