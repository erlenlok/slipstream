//! Slipstream Market Data Daemon
//!
//! High-performance market data aggregator for crypto trading strategies.

mod config;
mod ipc;
mod storage;
mod types;
mod venues;

use anyhow::{Context, Result};
use config::Config;
use ipc::IpcServer;
use storage::MarketDataStore;
use tokio::sync::mpsc;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use venues::HyperliquidClient;

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = Config::load().context("Failed to load configuration")?;

    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| config.daemon.log_level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting Slipstream Market Data Daemon");
    info!("Socket path: {}", config.daemon.socket_path);

    // Create market data store
    let store = MarketDataStore::new(
        config.retention.candles_4h_count,
        config.retention.orderbook_snapshots_per_symbol,
        config.retention.trades_per_symbol,
    );

    // Create channels for data ingestion
    let (candle_tx, mut candle_rx) = mpsc::unbounded_channel();

    // Spawn storage task (receives data from venue clients and stores it)
    let store_clone = store.clone();
    tokio::spawn(async move {
        while let Some(candle) = candle_rx.recv().await {
            store_clone.insert_candle(candle);
        }
    });

    // Start venue clients
    if let Some(hl_config) = config.venues.hyperliquid {
        if hl_config.enabled {
            let client = HyperliquidClient::new(candle_tx.clone());
            tokio::spawn(async move {
                loop {
                    if let Err(e) = client.run().await {
                        error!("Hyperliquid client error: {:?}", e);
                        error!("Error chain: {:#}", e);
                        info!("Reconnecting in 5s...");
                        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                    }
                }
            });
        }
    }

    // Start IPC server
    let ipc_server = IpcServer::new(store, config.daemon.socket_path.clone());
    ipc_server.run().await?;

    Ok(())
}
