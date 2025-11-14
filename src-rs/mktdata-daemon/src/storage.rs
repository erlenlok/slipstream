//! In-memory storage for market data.

use crate::types::{Candle, OrderbookSnapshot, Quote, Trade};
use dashmap::DashMap;
use std::collections::VecDeque;
use std::sync::Arc;

/// Key for candle storage: (venue, symbol, interval)
type CandleKey = (String, String, String);

/// Key for orderbook/trade/quote storage: (venue, symbol)
type AssetKey = (String, String);

/// Thread-safe market data store
#[derive(Clone)]
pub struct MarketDataStore {
    candles: Arc<DashMap<CandleKey, VecDeque<Candle>>>,
    orderbooks: Arc<DashMap<AssetKey, VecDeque<OrderbookSnapshot>>>,
    trades: Arc<DashMap<AssetKey, VecDeque<Trade>>>,
    quotes: Arc<DashMap<AssetKey, VecDeque<Quote>>>,
    max_candles: usize,
    max_orderbooks: usize,
    max_trades: usize,
}

impl MarketDataStore {
    pub fn new(max_candles: usize, max_orderbooks: usize, max_trades: usize) -> Self {
        Self {
            candles: Arc::new(DashMap::new()),
            orderbooks: Arc::new(DashMap::new()),
            trades: Arc::new(DashMap::new()),
            quotes: Arc::new(DashMap::new()),
            max_candles,
            max_orderbooks,
            max_trades,
        }
    }

    /// Insert a candle, maintaining ring buffer
    pub fn insert_candle(&self, candle: Candle) {
        let key = (
            candle.venue.clone(),
            candle.symbol.clone(),
            candle.interval.clone(),
        );

        let mut entry = self.candles.entry(key).or_insert_with(VecDeque::new);

        // Avoid duplicates (same timestamp)
        if let Some(last) = entry.back() {
            if last.timestamp == candle.timestamp {
                // Update existing candle
                entry.pop_back();
            }
        }

        entry.push_back(candle);

        // Maintain max size
        while entry.len() > self.max_candles {
            entry.pop_front();
        }
    }

    /// Get recent candles
    pub fn get_candles(
        &self,
        venue: &str,
        symbol: &str,
        interval: &str,
        count: usize,
    ) -> Vec<Candle> {
        let key = (venue.to_string(), symbol.to_string(), interval.to_string());

        self.candles
            .get(&key)
            .map(|deque| {
                let start = deque.len().saturating_sub(count);
                deque.iter().skip(start).cloned().collect()
            })
            .unwrap_or_default()
    }

    /// Insert orderbook snapshot
    pub fn insert_orderbook(&self, snapshot: OrderbookSnapshot) {
        let key = (snapshot.venue.clone(), snapshot.symbol.clone());

        let mut entry = self.orderbooks.entry(key).or_insert_with(VecDeque::new);
        entry.push_back(snapshot);

        while entry.len() > self.max_orderbooks {
            entry.pop_front();
        }
    }

    /// Get latest orderbook snapshot
    pub fn get_latest_orderbook(&self, venue: &str, symbol: &str) -> Option<OrderbookSnapshot> {
        let key = (venue.to_string(), symbol.to_string());
        self.orderbooks
            .get(&key)
            .and_then(|deque| deque.back().cloned())
    }

    /// Insert trade
    pub fn insert_trade(&self, trade: Trade) {
        let key = (trade.venue.clone(), trade.symbol.clone());

        let mut entry = self.trades.entry(key).or_insert_with(VecDeque::new);
        entry.push_back(trade);

        while entry.len() > self.max_trades {
            entry.pop_front();
        }
    }

    /// Get recent trades
    pub fn get_trades(&self, venue: &str, symbol: &str, count: usize) -> Vec<Trade> {
        let key = (venue.to_string(), symbol.to_string());

        self.trades
            .get(&key)
            .map(|deque| {
                let start = deque.len().saturating_sub(count);
                deque.iter().skip(start).cloned().collect()
            })
            .unwrap_or_default()
    }

    /// Insert quote (BBO)
    pub fn insert_quote(&self, quote: Quote) {
        let key = (quote.venue.clone(), quote.symbol.clone());

        // For quotes, we only keep the latest (no history)
        self.quotes.insert(key, {
            let mut deque = VecDeque::new();
            deque.push_back(quote);
            deque
        });
    }

    /// Get latest quote
    pub fn get_latest_quote(&self, venue: &str, symbol: &str) -> Option<Quote> {
        let key = (venue.to_string(), symbol.to_string());
        self.quotes
            .get(&key)
            .and_then(|deque| deque.back().cloned())
    }

    /// Get all symbols with candles for a given venue/interval
    pub fn get_symbols_with_candles(&self, venue: &str, interval: &str) -> Vec<String> {
        self.candles
            .iter()
            .filter_map(|entry| {
                let (v, sym, intv) = entry.key();
                if v == venue && intv == interval {
                    Some(sym.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_storage() {
        let store = MarketDataStore::new(10, 100, 1000);

        let candle = Candle {
            venue: "hyperliquid".to_string(),
            symbol: "BTC".to_string(),
            timestamp: 1700000000,
            interval: "4h".to_string(),
            open: 50000.0,
            high: 51000.0,
            low: 49000.0,
            close: 50500.0,
            volume: 100.0,
            num_trades: 50,
        };

        store.insert_candle(candle.clone());

        let retrieved = store.get_candles("hyperliquid", "BTC", "4h", 10);
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].close, 50500.0);
    }

    #[test]
    fn test_candle_ring_buffer() {
        let store = MarketDataStore::new(3, 100, 1000);

        for i in 0..5 {
            let candle = Candle {
                venue: "hyperliquid".to_string(),
                symbol: "BTC".to_string(),
                timestamp: 1700000000 + i * 14400, // 4h apart
                interval: "4h".to_string(),
                open: 50000.0,
                high: 51000.0,
                low: 49000.0,
                close: 50000.0 + i as f64 * 100.0,
                volume: 100.0,
                num_trades: 50,
            };
            store.insert_candle(candle);
        }

        let retrieved = store.get_candles("hyperliquid", "BTC", "4h", 10);
        // Should only have last 3 candles
        assert_eq!(retrieved.len(), 3);
        assert_eq!(retrieved[0].close, 50200.0); // i=2
        assert_eq!(retrieved[2].close, 50400.0); // i=4
    }
}
