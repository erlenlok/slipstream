//! Configuration management for the market data daemon.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub daemon: DaemonConfig,
    pub retention: RetentionConfig,
    pub venues: VenuesConfig,
    #[serde(default)]
    pub health: HealthConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DaemonConfig {
    pub socket_path: String,
    pub log_level: String,
    #[serde(default = "default_log_path")]
    pub log_path: String,
}

fn default_log_path() -> String {
    "/var/log/slipstream/mktdata.log".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RetentionConfig {
    #[serde(default = "default_candles_4h_count")]
    pub candles_4h_count: usize,
    #[serde(default = "default_candles_1h_count")]
    pub candles_1h_count: usize,
    #[serde(default = "default_orderbook_snapshots")]
    pub orderbook_snapshots_per_symbol: usize,
    #[serde(default = "default_trades_per_symbol")]
    pub trades_per_symbol: usize,
}

fn default_candles_4h_count() -> usize {
    1200
}
fn default_candles_1h_count() -> usize {
    2000
}
fn default_orderbook_snapshots() -> usize {
    100
}
fn default_trades_per_symbol() -> usize {
    1000
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VenuesConfig {
    pub hyperliquid: Option<VenueConfig>,
    pub binance: Option<VenueConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VenueConfig {
    pub enabled: bool,
    pub websocket_url: String,
    #[serde(default = "default_reconnect_delay")]
    pub reconnect_delay_ms: u64,
    #[serde(default = "default_max_reconnect_delay")]
    pub max_reconnect_delay_ms: u64,
}

fn default_reconnect_delay() -> u64 {
    1000
}
fn default_max_reconnect_delay() -> u64 {
    30000
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HealthConfig {
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,
    #[serde(default = "default_watchdog_interval")]
    pub watchdog_interval_ms: u64,
}

fn default_metrics_port() -> u16 {
    9090
}
fn default_watchdog_interval() -> u64 {
    5000
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            metrics_port: default_metrics_port(),
            watchdog_interval_ms: default_watchdog_interval(),
        }
    }
}

impl Config {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = std::fs::read_to_string(path.as_ref())
            .context("Failed to read config file")?;
        let config: Config =
            toml::from_str(&contents).context("Failed to parse config TOML")?;
        Ok(config)
    }

    /// Load configuration with environment variable overrides
    pub fn load() -> Result<Self> {
        let config_path = std::env::var("MKTDATA_CONFIG")
            .unwrap_or_else(|_| "/etc/slipstream/mktdata.toml".to_string());

        Self::from_file(config_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let toml = r#"
            [daemon]
            socket_path = "/var/run/slipstream/mktdata.sock"
            log_level = "info"

            [retention]
            candles_4h_count = 1200

            [venues.hyperliquid]
            enabled = true
            websocket_url = "wss://api.hyperliquid.xyz/ws"
        "#;

        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.daemon.socket_path, "/var/run/slipstream/mktdata.sock");
        assert_eq!(config.retention.candles_4h_count, 1200);
        assert!(config.venues.hyperliquid.is_some());
    }
}
