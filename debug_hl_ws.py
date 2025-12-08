
import asyncio
import json
import logging
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_ws")

async def test_ws():
    uri = "wss://api.hyperliquid.xyz/ws"
    async with websockets.connect(uri) as ws:
        # Subscribe
        msg = {
            "method": "subscribe",
            "subscription": {"type": "l2Book", "coin": "kPEPE"}
        }
        await ws.send(json.dumps(msg))
        logger.info(f"Sent: {json.dumps(msg)}")

        # Read loop
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < 10:
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(resp)
                
                # Check structure
                channel = data.get("channel")
                payload = data.get("data")
                
                logger.info(f"Received: {json.dumps(data)[:200]}...")  # Truncate
                
                if channel == "l2Book":
                    # verify parsing logic
                    coin = payload.get("coin")
                    levels = payload.get("levels")
                    if not levels:
                        logger.warning("No levels!")
                        continue
                        
                    bids = levels[0]
                    asks = levels[1]
                    
                    if bids and asks:
                        bid_px = float(bids[0]['px']) if isinstance(bids[0], dict) else float(bids[0][0])
                        logger.info(f"Parsed BBO: Bid={bid_px} for {coin}")
                    else:
                        logger.warning("Bids/asks empty")
            except asyncio.TimeoutError:
                logger.info("Ping...")

if __name__ == "__main__":
    asyncio.run(test_ws())
