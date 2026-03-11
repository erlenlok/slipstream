import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict
import websockets
from collections import deque
import statistics

# --- Configuration ---
# Target Asset (Must exist on both Futures venues)
ASSET: str = "BTC" 

# Websocket Endpoints
BINANCE_WS: str = "wss://fstream.binance.com/ws" # Futures
HL_WS: str = "wss://api.hyperliquid.xyz/ws"      # Mainnet

# Thresholds
PRICE_CHANGE_THRESHOLD: float = 0.0005  # 0.05% move to trigger "Lead/Lag" check

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ArbMonitor")

try:
    import uvloop
    uvloop.install()
    logger.info("uvloop installed for maximum performance.")
except ImportError:
    logger.info("uvloop not found, using standard asyncio.")

@dataclass
class MarketState:
    # Binance State
    bin_price: float = 0.0
    bin_update_time_local: float = 0.0
    bin_event_time_server: int = 0  # ms
    
    # Hyperliquid State
    hl_price: float = 0.0
    hl_update_time_local: float = 0.0
    hl_event_time_server: int = 0   # ms
    
    # Metrics
    lead_lag_samples: list[float] = field(default_factory=list)

async def binance_consumer(state: MarketState) -> None:
    """Consumes Binance BookTicker (Fastest Update)."""
    stream_url = f"{BINANCE_WS}/{ASSET.lower()}usdt@bookTicker"
    
    async for websocket in websockets.connect(stream_url):
        try:
            logger.info(f"Connected to Binance: {ASSET}")
            async for message in websocket:
                local_ts = time.time() * 1000  # ms
                data = json.loads(message)
                
                # Update State
                best_bid = float(data['b'])
                best_ask = float(data['a'])
                mid_price = (best_bid + best_ask) / 2
                event_ts = data['E'] # Event time
                
                # Check for significant price move (The "Signal")
                if state.bin_price > 0:
                    delta = abs(mid_price - state.bin_price) / state.bin_price
                    if delta > PRICE_CHANGE_THRESHOLD:
                         logger.info(f"⚡ BINANCE SURGE: {state.bin_price:.4f} -> {mid_price:.4f} ({delta:.2%})")
                
                state.bin_price = mid_price
                state.bin_update_time_local = local_ts
                state.bin_event_time_server = event_ts

        except websockets.ConnectionClosed:
            logger.warning("Binance connection closed. Reconnecting...")
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Binance Error: {e}")

async def hyperliquid_consumer(state: MarketState) -> None:
    """Consumes Hyperliquid L2Book (Block Time Truth)."""
    
    sub_msg = {
        "method": "subscribe",
        "subscription": {"type": "l2Book", "coin": ASSET}
    }

    async for websocket in websockets.connect(HL_WS):
        try:
            await websocket.send(json.dumps(sub_msg))
            logger.info(f"Connected to Hyperliquid: {ASSET}")
            
            async for message in websocket:
                local_ts = time.time() * 1000 # ms
                msg = json.loads(message)
                
                if msg.get("channel") == "l2Book":
                    data = msg.get("data", {})
                    server_ts = data.get("time", 0)
                    
                    # Calculate Mid Price from L2
                    levels = data.get("levels", [[], []])
                    if levels[0] and levels[1]:
                        bid = float(levels[0][0]['px'])
                        ask = float(levels[1][0]['px'])
                        mid = (bid + ask) / 2
                        
                        # Capture the "Lag"
                        # If Binance updated RECENTLY (e.g. < 200ms ago) and HL just updated now:
                        # This delta effectively visualizes the "Arbitrage Window"
                        time_since_binance = local_ts - state.bin_update_time_local
                        
                        state.hl_price = mid
                        state.hl_update_time_local = local_ts
                        state.hl_event_time_server = server_ts
                        
        except websockets.ConnectionClosed:
            logger.warning("Hyperliquid connection closed. Reconnecting...")
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Hyperliquid Error: {e}")

async def monitor_loop(state: MarketState) -> None:
    """The 'Referee': Prints the side-by-side comparison with Rolling Stats."""
    logger.info("Starting Latency Monitor... (Wait for streams to warm up)")
    await asyncio.sleep(2) # Warmup
    
    # Rolling Windows (N=1000 samples, approx 50s at 20 updates/sec)
    WINDOW_SIZE = 1000
    edge_history = deque(maxlen=WINDOW_SIZE)
    spread_history = deque(maxlen=WINDOW_SIZE)

    # Header
    print(f"{'TYPE':<10} | {'PRICE (USD)':<12} | {'LATENCY (ms)':<15} | {'AGE (ms)':<10}")
    print("-" * 60)

    while True:
        now = time.time() * 1000
        
        # 1. Transport Latency
        bin_network_lag = state.bin_update_time_local - state.bin_event_time_server
        hl_network_lag = state.hl_update_time_local - state.hl_event_time_server
        
        # 2. Information Age
        bin_age = now - state.bin_update_time_local
        hl_age = now - state.hl_update_time_local
        
        # 3. The "Arb Gap" (Price Diff)
        if state.bin_price > 0 and state.hl_price > 0:
            price_diff_bps = ((state.hl_price - state.bin_price) / state.bin_price) * 10000
        else:
            price_diff_bps = 0.0

        # --- Statistics Calculation ---
        current_edge = hl_age - bin_age
        
        # Only add to history if both streams are alive (prices > 0)
        if state.bin_price > 0 and state.hl_price > 0:
            edge_history.append(current_edge)
            spread_history.append(price_diff_bps)

        def get_stats(data: deque):
            if not data:
                return 0.0, 0.0, 0.0
            return (
                statistics.mean(data),
                statistics.stdev(data) if len(data) > 1 else 0.0,
                statistics.quantiles(data, n=100)[-1] if len(data) >= 100 else 0.0 # P99 approx
            )

        edge_mean, edge_std, edge_p99 = get_stats(edge_history)
        spread_mean, spread_std, spread_p99 = get_stats(spread_history)
        
        # Visualizing the Output
        status = f"""
Binance   | {state.bin_price:<12.4f} | {bin_network_lag:>5.1f} ms        | {bin_age:>5.0f} ms ago
Hyperliq  | {state.hl_price:<12.4f} | {hl_network_lag:>5.1f} ms        | {hl_age:>5.0f} ms ago
------------------------------------------------------------
>> CURRENT SPREAD: {price_diff_bps:+.2f} bps  |  >> CURRENT EDGE: {current_edge:.1f} ms
------------------------------------------------------------
ROLLING WINDOW ({len(edge_history)} samples)
>> SPREAD STATS | Mean: {spread_mean:+.2f} | Std: {spread_std:.2f} | P99: {spread_p99:.2f}
>> EDGE STATS   | Mean: {edge_mean:.1f}  | Std: {edge_std:.1f} | P99: {edge_p99:.1f}
"""
        # Clear screen roughly or just print chunks
        print(f"\033[H\033[J{status}") 
        
        await asyncio.sleep(0.05) # 20fps refresh

if __name__ == "__main__":
    state = MarketState()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(asyncio.gather(
            binance_consumer(state),
            hyperliquid_consumer(state),
            monitor_loop(state)
        ))
    except KeyboardInterrupt:
        print("Stopping...")