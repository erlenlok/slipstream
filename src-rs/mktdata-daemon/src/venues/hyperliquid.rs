//! Hyperliquid websocket client.

use crate::types::Candle;
use anyhow::{anyhow, Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

const WS_URL: &str = "wss://api.hyperliquid.xyz/ws";

#[derive(Debug, Clone, Serialize)]
struct SubscriptionMessage {
    method: String,
    subscription: Subscription,
}

#[derive(Debug, Clone, Serialize)]
struct Subscription {
    #[serde(rename = "type")]
    sub_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    interval: Option<String>,
}

/// Hyperliquid candle data (from allMids channel)
#[derive(Debug, Deserialize)]
struct HyperliquidCandle {
    #[serde(rename = "coin")]
    symbol: String,
    #[serde(rename = "s")]
    interval: String,
    #[serde(rename = "t")]
    timestamp: u64, // milliseconds
    #[serde(rename = "o")]
    open: String,
    #[serde(rename = "h")]
    high: String,
    #[serde(rename = "l")]
    low: String,
    #[serde(rename = "c")]
    close: String,
    #[serde(rename = "v")]
    volume: String,
    #[serde(rename = "n")]
    num_trades: u64,
}

impl HyperliquidCandle {
    fn to_candle(self) -> Result<Candle> {
        Ok(Candle {
            venue: "hyperliquid".to_string(),
            symbol: self.symbol,
            timestamp: self.timestamp / 1000, // Convert ms to seconds
            interval: self.interval,
            open: self.open.parse().context("Failed to parse open")?,
            high: self.high.parse().context("Failed to parse high")?,
            low: self.low.parse().context("Failed to parse low")?,
            close: self.close.parse().context("Failed to parse close")?,
            volume: self.volume.parse().context("Failed to parse volume")?,
            num_trades: self.num_trades,
        })
    }
}

pub struct HyperliquidClient {
    candle_tx: mpsc::UnboundedSender<Candle>,
}

impl HyperliquidClient {
    pub fn new(candle_tx: mpsc::UnboundedSender<Candle>) -> Self {
        Self { candle_tx }
    }

    pub async fn run(&self) -> Result<()> {
        info!("Connecting to Hyperliquid websocket: {}", WS_URL);

        let (ws_stream, response) = connect_async(WS_URL)
            .await
            .context("Failed to connect to Hyperliquid")?;

        info!("Connected to Hyperliquid (status: {:?})", response.status());

        let (mut write, mut read) = ws_stream.split();

        // Subscribe to allMids (4h candles)
        let subscribe_msg = SubscriptionMessage {
            method: "subscribe".to_string(),
            subscription: Subscription {
                sub_type: "allMids".to_string(),
                interval: Some("4h".to_string()),
            },
        };

        let msg_json = serde_json::to_string(&subscribe_msg)?;
        write.send(Message::Text(msg_json)).await?;
        info!("Subscribed to Hyperliquid allMids (4h)");

        // Also subscribe to 1h
        let subscribe_1h = SubscriptionMessage {
            method: "subscribe".to_string(),
            subscription: Subscription {
                sub_type: "allMids".to_string(),
                interval: Some("1h".to_string()),
            },
        };
        write.send(Message::Text(serde_json::to_string(&subscribe_1h)?)).await?;
        info!("Subscribed to Hyperliquid allMids (1h)");

        // Process messages
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = self.handle_message(&text) {
                        warn!("Failed to handle message: {}", e);
                        debug!("Raw message: {}", text);
                    }
                }
                Ok(Message::Ping(data)) => {
                    debug!("Received ping");
                    write.send(Message::Pong(data)).await?;
                }
                Ok(Message::Pong(_)) => {
                    debug!("Received pong");
                }
                Ok(Message::Close(frame)) => {
                    info!("Websocket closed: {:?}", frame);
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    error!("Websocket error: {}", e);
                    return Err(e.into());
                }
            }
        }

        Ok(())
    }

    fn handle_message(&self, text: &str) -> Result<()> {
        let value: Value = serde_json::from_str(text)?;

        // Check if this is a candle update
        if let Some(channel) = value.get("channel").and_then(|c| c.as_str()) {
            if channel == "allMids" {
                if let Some(data) = value.get("data") {
                    let hl_candle: HyperliquidCandle = serde_json::from_value(data.clone())
                        .context("Failed to parse Hyperliquid candle")?;

                    let candle = hl_candle.to_candle()?;

                    // Send to storage
                    self.candle_tx.send(candle).map_err(|e| anyhow!("Failed to send candle: {}", e))?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_candle() {
        let json = r#"{
            "channel": "allMids",
            "data": {
                "coin": "BTC",
                "s": "4h",
                "t": 1700000000000,
                "o": "50000.5",
                "h": "51000.0",
                "l": "49500.0",
                "c": "50500.0",
                "v": "123.45",
                "n": 100
            }
        }"#;

        let value: Value = serde_json::from_str(json).unwrap();
        let data = value.get("data").unwrap();
        let hl_candle: HyperliquidCandle = serde_json::from_value(data.clone()).unwrap();
        let candle = hl_candle.to_candle().unwrap();

        assert_eq!(candle.symbol, "BTC");
        assert_eq!(candle.interval, "4h");
        assert_eq!(candle.open, 50000.5);
        assert_eq!(candle.close, 50500.0);
        assert_eq!(candle.num_trades, 100);
    }
}
