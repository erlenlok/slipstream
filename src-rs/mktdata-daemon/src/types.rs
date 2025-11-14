//! Core market data types.

use serde::{Deserialize, Serialize};

/// OHLCV candle data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub venue: String,
    pub symbol: String,
    pub timestamp: u64,      // Unix timestamp (start of period)
    pub interval: String,    // "4h", "1h", "15m"
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub num_trades: u64,
}

/// Single orderbook level (price-size pair)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookLevel {
    pub price: f64,
    pub size: f64,
}

/// L2 orderbook snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookSnapshot {
    pub venue: String,
    pub symbol: String,
    pub timestamp_us: u64,
    pub sequence: u64, // For delta reconciliation
    pub bids: Vec<OrderbookLevel>, // Sorted desc by price
    pub asks: Vec<OrderbookLevel>, // Sorted asc by price
}

impl OrderbookSnapshot {
    pub fn best_bid(&self) -> Option<&OrderbookLevel> {
        self.bids.first()
    }

    pub fn best_ask(&self) -> Option<&OrderbookLevel> {
        self.asks.first()
    }

    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / 2.0),
            _ => None,
        }
    }

    pub fn spread_bps(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask(), self.mid_price()) {
            (Some(bid), Some(ask), Some(mid)) => {
                Some(10000.0 * (ask.price - bid.price) / mid)
            }
            _ => None,
        }
    }
}

/// Trade with aggressor side
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub venue: String,
    pub symbol: String,
    pub timestamp_us: u64,
    pub price: f64,
    pub size: f64,
    pub side: TradeSide,
    pub trade_id: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TradeSide {
    Buy,  // Aggressor was buyer (lifted offer)
    Sell, // Aggressor was seller (hit bid)
}

/// BBO quote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    pub venue: String,
    pub symbol: String,
    pub timestamp_us: u64,
    pub bid_price: f64,
    pub bid_size: f64,
    pub ask_price: f64,
    pub ask_size: f64,
}

impl Quote {
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }

    pub fn spread_bps(&self) -> f64 {
        10000.0 * (self.ask_price - self.bid_price) / self.mid_price()
    }
}

/// Messages sent from daemon to clients
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MarketDataMessage {
    CandleUpdate { data: Candle },
    OrderbookSnapshot { data: OrderbookSnapshot },
    Trade { data: Trade },
    Quote { data: Quote },
    Heartbeat { timestamp_us: u64 },
    Error { code: String, message: String },
}

/// Subscription requests from clients
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SubscriptionRequest {
    SubscribeCandles {
        venue: String,
        symbol: String,
        interval: String,
    },
    SubscribeAllCandles {
        venue: String,
        interval: String,
    },
    SubscribeOrderbook {
        venue: String,
        symbol: String,
        depth: usize,
    },
    SubscribeQuote {
        venue: String,
        symbol: String,
    },
    SubscribeTrades {
        venue: String,
        symbol: String,
    },
    SubscribeAllTrades {
        venue: String,
    },
    GetCandles {
        venue: String,
        symbol: String,
        interval: String,
        count: usize,
    },
    GetOrderbook {
        venue: String,
        symbol: String,
    },
}

/// Response to sync queries (GetCandles, GetOrderbook)
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueryResponse {
    Candles { data: Vec<Candle> },
    Orderbook { data: OrderbookSnapshot },
    Error { code: String, message: String },
}
