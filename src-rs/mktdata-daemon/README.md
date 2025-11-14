# Slipstream Market Data Daemon

High-performance Rust daemon for aggregating market data from multiple venues (Hyperliquid, Binance) and serving it to Python trading strategies via Unix domain sockets.

## Status: MVP Complete (Candles Working)

✅ **Implemented:**
- Rust daemon with Tokio async runtime
- Hyperliquid websocket client (candles: 4h, 1h)
- In-memory storage (DashMap ring buffers)
- IPC server (Unix socket + MessagePack protocol)
- Python client library
- Configuration management (TOML)
- Automatic reconnection logic

🔨 **In Progress:**
- TLS/websocket connection debugging
- L2 orderbook support
- Trade feed support
- BBO quote feed

⏳ **Planned:**
- Binance websocket client
- Recorder daemon (Parquet persistence)
- Systemd integration
- Prometheus metrics

## Architecture

```
┌─────────────────────────────────┐
│   Rust Daemon (mktdata-daemon)  │
│                                 │
│  ┌──────────────────────────┐  │
│  │  Hyperliquid WS Client   │  │
│  │  ├─ allMids (4h/1h)      │  │
│  │  ├─ l2Book (TODO)        │  │
│  │  └─ trades (TODO)        │  │
│  └──────────┬───────────────┘  │
│             │                   │
│  ┌──────────▼───────────────┐  │
│  │  In-Memory Storage       │  │
│  │  (DashMap + RingBuffer)  │  │
│  └──────────┬───────────────┘  │
│             │                   │
│  ┌──────────▼───────────────┐  │
│  │  IPC Server              │  │
│  │  (Unix Socket)           │  │
│  └──────────┬───────────────┘  │
└─────────────┼───────────────────┘
              │
      ┌───────▼────────┐
      │ Python Client  │
      │  (Strategies)  │
      └────────────────┘
```

## Building

```bash
cd src-rs/mktdata-daemon
cargo build --release
```

Binary location: `target/release/mktdata-daemon`

## Running

```bash
# With default config
MKTDATA_CONFIG=/root/slipstream/config/mktdata.toml \
./target/release/mktdata-daemon

# Or specify config via environment
export MKTDATA_CONFIG=/etc/slipstream/mktdata.toml
./target/release/mktdata-daemon
```

## Configuration

See `config/mktdata.toml`:

```toml
[daemon]
socket_path = "/tmp/slipstream-mktdata.sock"
log_level = "info"

[retention]
candles_4h_count = 1200  # ~200 days

[venues.hyperliquid]
enabled = true
websocket_url = "wss://api.hyperliquid.xyz/ws"
```

## Python Client Usage

```python
from slipstream.core.mktdata import MarketDataClient

# Connect to daemon
with MarketDataClient() as client:
    # Get recent candles
    candles = client.get_candles(
        venue="hyperliquid",
        symbol="BTC",
        interval="4h",
        count=100,
    )

    for candle in candles:
        print(f"{candle.timestamp}: ${candle.close:,.2f}")

    # Batch fetch multiple symbols
    batch = client.get_candles_batch(
        venue="hyperliquid",
        symbols=["BTC", "ETH", "SOL"],
        interval="4h",
        count=50,
    )
```

## Data Types

### Candle
```rust
pub struct Candle {
    pub venue: String,
    pub symbol: String,
    pub timestamp: u64,      // Unix timestamp (seconds)
    pub interval: String,    // "4h", "1h", "15m"
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub num_trades: u64,
}
```

### OrderbookSnapshot (TODO)
```rust
pub struct OrderbookSnapshot {
    pub venue: String,
    pub symbol: String,
    pub timestamp_us: u64,
    pub sequence: u64,
    pub bids: Vec<OrderbookLevel>,  // Sorted desc
    pub asks: Vec<OrderbookLevel>,  // Sorted asc
}
```

### Trade (TODO)
```rust
pub struct Trade {
    pub venue: String,
    pub symbol: String,
    pub timestamp_us: u64,
    pub price: f64,
    pub size: f64,
    pub side: TradeSide,  // Buy/Sell (aggressor)
    pub trade_id: String,
}
```

## IPC Protocol

**Wire Format:** Length-prefixed MessagePack

```
┌───────────────┬──────────────────────┐
│ Length (4B)   │  MessagePack Body    │
│ (big-endian)  │                      │
└───────────────┴──────────────────────┘
```

**Request Types:**
- `get_candles` - Fetch historical candles
- `get_orderbook` - Fetch latest L2 snapshot
- `subscribe_candles` - Stream candle updates (TODO)
- `subscribe_orderbook` - Stream L2 updates (TODO)

## Performance Targets

- **Latency**: p99 < 1ms for IPC requests
- **Memory**: <500MB for 100 symbols (1200 candles each)
- **Throughput**: 10K messages/sec (when streaming)

## Testing

```bash
# Start daemon
MKTDATA_CONFIG=config/mktdata.toml ./target/release/mktdata-daemon &

# Run test script
python3 scripts/test_mktdata.py
```

## Known Issues

1. **TLS Connection**: Websocket connection to Hyperliquid fails on some servers
   - Likely missing system CA certificates
   - Workaround: Install `ca-certificates` package

2. **Streaming Subscriptions**: Not yet implemented
   - Currently only supports sync queries (`get_candles`, `get_orderbook`)
   - Need to implement pub/sub with tokio channels

## Next Steps

1. **Fix TLS/WS connection**
   - Install ca-certificates
   - Test on server with proper networking

2. **Add L2 Orderbook Support**
   - Subscribe to `l2Book` channel
   - Implement snapshot + delta reconciliation
   - Store in DashMap

3. **Add Trade Feed**
   - Subscribe to `trades` channel
   - Parse aggressor side
   - Store recent trades (ring buffer)

4. **Add Binance Client**
   - `depth@100ms` for L2
   - `aggTrade` for trades
   - `kline` for candles

5. **Build Recorder Daemon**
   - Python daemon that subscribes to all feeds
   - Writes Parquet files partitioned by date/venue
   - Enables realistic backtesting

6. **Production Hardening**
   - Systemd service file
   - Prometheus metrics
   - Graceful shutdown
   - Health checks

## Files

```
src-rs/mktdata-daemon/
├── Cargo.toml                 # Dependencies
├── src/
│   ├── main.rs                # Entry point
│   ├── config.rs              # TOML configuration
│   ├── types.rs               # Core data types
│   ├── storage.rs             # In-memory storage (DashMap)
│   ├── ipc.rs                 # Unix socket IPC server
│   └── venues/
│       ├── mod.rs
│       ├── hyperliquid.rs     # Hyperliquid websocket client
│       └── binance.rs         # TODO: Binance client
└── target/release/
    └── mktdata-daemon         # Compiled binary

src/slipstream/core/mktdata/
├── __init__.py
└── client.py                   # Python client library

scripts/
└── test_mktdata.py             # Integration test script

config/
└── mktdata.toml                # Daemon configuration
```

## Contributing

When adding new features:
1. Add data type to `types.rs`
2. Implement storage in `storage.rs`
3. Add venue-specific parsing in `venues/`
4. Update IPC protocol in `ipc.rs`
5. Update Python client in `client.py`
6. Write tests

## License

Part of the Slipstream trading framework.
