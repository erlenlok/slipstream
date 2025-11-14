//! IPC server using Unix domain sockets and MessagePack protocol.

use crate::storage::MarketDataStore;
use crate::types::{QueryResponse, SubscriptionRequest};
use anyhow::{Context, Result};
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tracing::{debug, error, info, warn};

pub struct IpcServer {
    store: MarketDataStore,
    socket_path: String,
}

impl IpcServer {
    pub fn new(store: MarketDataStore, socket_path: String) -> Self {
        Self { store, socket_path }
    }

    pub async fn run(&self) -> Result<()> {
        // Remove existing socket file if it exists
        if Path::new(&self.socket_path).exists() {
            std::fs::remove_file(&self.socket_path)
                .context("Failed to remove existing socket")?;
        }

        // Create parent directory if needed
        if let Some(parent) = Path::new(&self.socket_path).parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create socket directory")?;
        }

        let listener = UnixListener::bind(&self.socket_path)
            .context("Failed to bind Unix socket")?;

        info!("IPC server listening on {}", self.socket_path);

        // Set permissions (readable/writable by owner and group)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&self.socket_path)?.permissions();
            perms.set_mode(0o660);
            std::fs::set_permissions(&self.socket_path, perms)?;
        }

        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    let store = self.store.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_client(stream, store).await {
                            error!("Client handler error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                }
            }
        }
    }
}

async fn handle_client(mut stream: UnixStream, store: MarketDataStore) -> Result<()> {
    debug!("New client connected");

    let mut buffer = Vec::new();

    loop {
        // Read message length (4 bytes, big-endian)
        let mut len_buf = [0u8; 4];
        match stream.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                debug!("Client disconnected");
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        }

        let msg_len = u32::from_be_bytes(len_buf) as usize;

        // Read message body
        buffer.resize(msg_len, 0);
        stream.read_exact(&mut buffer).await?;

        // Decode MessagePack
        let request: SubscriptionRequest = rmp_serde::from_slice(&buffer)
            .context("Failed to decode request")?;

        debug!("Received request: {:?}", request);

        // Handle request
        match request {
            SubscriptionRequest::GetCandles {
                venue,
                symbol,
                interval,
                count,
            } => {
                let candles = store.get_candles(&venue, &symbol, &interval, count);
                let response = QueryResponse::Candles { data: candles };

                send_response(&mut stream, &response).await?;
            }

            SubscriptionRequest::GetOrderbook { venue, symbol } => {
                match store.get_latest_orderbook(&venue, &symbol) {
                    Some(ob) => {
                        let response = QueryResponse::Orderbook { data: ob };
                        send_response(&mut stream, &response).await?;
                    }
                    None => {
                        let response = QueryResponse::Error {
                            code: "NOT_FOUND".to_string(),
                            message: format!("No orderbook found for {}/{}", venue, symbol),
                        };
                        send_response(&mut stream, &response).await?;
                    }
                }
            }

            // TODO: Handle streaming subscriptions (need tokio::select! with broadcast channels)
            _ => {
                warn!("Streaming subscriptions not yet implemented: {:?}", request);
                let response = QueryResponse::Error {
                    code: "NOT_IMPLEMENTED".to_string(),
                    message: "Streaming subscriptions not yet supported".to_string(),
                };
                send_response(&mut stream, &response).await?;
            }
        }
    }
}

async fn send_response<T: serde::Serialize>(
    stream: &mut UnixStream,
    response: &T,
) -> Result<()> {
    // Encode MessagePack
    let encoded = rmp_serde::to_vec(response).context("Failed to encode response")?;

    // Send length prefix
    let len = encoded.len() as u32;
    stream.write_all(&len.to_be_bytes()).await?;

    // Send body
    stream.write_all(&encoded).await?;

    Ok(())
}
