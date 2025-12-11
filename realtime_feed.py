import json
import time
import requests
import threading
import queue
from websocket import WebSocketApp

# Constants
WS_URL = "wss://api.hyperliquid.xyz/ws"
# WS_URL = "wss://api-testnet.hyperliquid.xyz/ws" # Uncomment for Testnet
REST_URL = "https://api.hyperliquid.xyz/info"
# REST_URL = "https://api-testnet.hyperliquid.xyz/info" # Uncomment for Testnet

class HyperliquidRealTimeFeed:
    def __init__(self, coin="BTC"):
        self.coin = coin
        self.ws = None
        self.latest_block_height = 0
        self.lock = threading.Lock()
        
        # Queue for unified event processing (optional, for main thread consumption)
        self.event_queue = queue.Queue()

    def on_message(self, ws, message):
        """
        Handles incoming WebSocket messages.
        """
        try:
            data = json.loads(message)
        except Exception as e:
            print(f"[WS] Error decoding JSON: {e}")
            return

        if isinstance(data, list):
            # Some messages might be lists? Print to debug.
            # print(f"[WS-Debug] Received List: {data}")
            return

        channel = data.get("channel")
        msg_data = data.get("data")


        if channel == "trades":
            # Real-time trade stream (Individual trades as they happen)
            # data structure: [{"coin": "BTC", "side": "B", "px": "...", "sz": "...", "time": ...}, ...]
            for trade in msg_data:
                print(f"[WS-Trade] {trade['coin']} {trade['side']} {trade['sz']} @ {trade['px']}")
                
        elif channel == "explorerBlock":
            # New block produced
            print(f"[WS-Debug] explorerBlock data: {msg_data}")
            
            # data structure: [{"height": 12345, ...}] ? Or just object? 
            # Looking at docs/usage, explorerBlock usually returns block metadata
            # Let's assume it has 'height'
            if isinstance(msg_data, list) and len(msg_data) > 0:
                block_info = msg_data[0]
            elif isinstance(msg_data, dict):
                block_info = msg_data
            else:
                return

            height = block_info.get("height")
            if height:
                with self.lock:
                    if height > self.latest_block_height:
                        self.latest_block_height = height
                        print(f"[WS-Block] New Block height: {height}")
                        # Trigger REST fetch for deep data
                        threading.Thread(target=self.fetch_block_details, args=(height,)).start()

        elif channel == "subscriptionResponse":
            print(f"[WS] Subscribed: {msg_data}")

    def on_error(self, ws, error):
        print(f"[WS] Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[WS] Closed: {close_status_code} - {close_msg}")

    def on_open(self, ws):
        print("[WS] Connected.")
        
        # 1. Subscribe to real-time Trades
        trades_sub = {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": self.coin
            }
        }
        ws.send(json.dumps(trades_sub))
        
        # 2. Subscribe to Block updates
        # 'explorerBlock' gives us the tick when a new block is born
        block_sub = {
            "method": "subscribe",
            "subscription": {
                "type": "explorerBlock"
            }
        }
        ws.send(json.dumps(block_sub))

    def fetch_block_details(self, height):
        """
        Fetches the full block details (transactions, orders, statuses) via REST.
        This approximates the 'StreamBlockTrades' gRPC behavior.
        """
        # We might need to wait a split second for the block to be available via REST 
        # API nodes might lag slightly behind the WS feed
        time.sleep(0.5) 
        
        url = REST_URL
        payload = {
            "type": "blockDetails",
            "height": height
        }
        
        try:
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=3.0)
            if resp.status_code == 200:
                block_data = resp.json()
                self.process_block_data(height, block_data)
            else:
                print(f"[REST] Failed to fetch block {height}: {resp.status_code}")
        except Exception as e:
            print(f"[REST] Exception fetching block {height}: {e}")

    def process_block_data(self, height, data):
        """
        Parse and print the deep block data.
        """
        block_time = data.get("blockTime")
        txs_len = len(data.get("txs", []))
        print(f"[REST-Block] Fetched Details for {height} (Time: {block_time}, Txs: {txs_len})")
        
        # Extract trade/order info if needed
        # Example: hunt for CLOIDs or Fill events
        # This matches the logic user wanted from 'investigate_l1.py'
        # ... logic to parse 'signed_action_bundles' could go here ...

    def start(self):
        self.ws = WebSocketApp(
            WS_URL,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever()

if __name__ == "__main__":
    print("Starting Hyperliquid Real-Time Hybrid Feed...")
    print("Press Ctrl+C to stop.")
    
    feed = HyperliquidRealTimeFeed(coin="BTC")
    feed.start()
