import asyncio
import os
import sys
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from slipstream.strategies.brawler.connectors.hyperliquid import (
    HyperliquidExecutionClient, HyperliquidOrder, HyperliquidOrderSide, _load_hyperliquid_modules
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ForceReload")

async def main():
    # Load .env.brawler
    env_path = Path(__file__).parent.parent.parent.parent / ".env.brawler"
    if env_path.exists():
        logger.info(f"Loading env from {env_path}")
        load_dotenv(env_path)
    else:
        logger.warning(f"{env_path} not found, relying on environment variables")

    api_key = os.getenv("HYPERLIQUID_API_KEY")
    api_secret = os.getenv("HYPERLIQUID_API_SECRET")
    wallet = os.getenv("HYPERLIQUID_BRAWLER_WALLET") or os.getenv("HYPERLIQUID_MAIN_WALLET")
    
    if not api_key or not api_secret:
        logger.error("Missing API Key/Secret")
        return

    client = HyperliquidExecutionClient(
        api_key=api_key,
        api_secret=api_secret,
        base_url="https://api.hyperliquid.xyz",
        target_wallet=wallet
    )
    
    # 1. Get info to determine price
    logger.info("Fetching BTC price...")
    info_client = client.info
    meta = await asyncio.to_thread(info_client.meta)
    mids = await asyncio.to_thread(info_client.all_mids)
    btc_price = float(mids["BTC"])
    logger.info(f"BTC Price: {btc_price}")
    
    # 2. Calculate Size for 5k USD
    target_usd = 5000.0
    size = round(target_usd / btc_price, 4)
    logger.info(f"Target Size: {size} BTC (~${target_usd})")
    
    # Rate limit pause
    logger.info("Waiting 12s for rate limit...")
    await asyncio.sleep(12) 
    
    # 3. Buy Loop (ALO)
    logger.info("--- Starting BUY Leg ---")
    while True:
        # Get BBO to place passive
        logger.info("Fetching L2 snapshot...")
        l2 = await asyncio.to_thread(info_client.l2_snapshot, "BTC")
        best_bid = float(l2["levels"][0][0]["px"])
        buy_price = best_bid
        
        logger.info(f"Placing ALO Buy: {size} BTC @ {buy_price}")
        order = HyperliquidOrder(
            symbol="BTC",
            price=buy_price,
            size=size,
            side=HyperliquidOrderSide.BUY,
            alo=True
        )
        try:
            res = await client.place_limit_order(order)
            logger.info(f"Order sent: {res}")
            # Wait a bit then check fill
        except Exception as e:
            logger.error(f"Order failed: {e}")
            await asyncio.sleep(12)
            continue
            
        logger.info("Waiting 20s for fill or rate limit...")
        await asyncio.sleep(20)
        
        # Check filled
        # We can check open orders, if gone assumed filled or cancelled (since it's ALO, it might cancel if crossing)
        open_orders = await asyncio.to_thread(info_client.open_orders, client.exchange.account_address)
        is_open = any(o["oid"] == int(res.order_id) for o in open_orders) if res.order_id else False
        
        if is_open:
            logger.info("Order still open. Cancelling to reprice...")
            await client.cancel_order("BTC", res.order_id)
            await asyncio.sleep(12)
        else:
            # Assume filled! (simplification)
            logger.info("Order no longer open! Assuming BUY filled (or ALO cancelled immediately). Checking inventory...")
            # Ideally checking user state, but let's assume success to proceed to sell leg to be neutral
            # user said "get in and out", so we must sell what we bought.
            # safe approach: check position
            user_state = await asyncio.to_thread(info_client.user_state, client.exchange.account_address)
            btc_pos = next((p for p in user_state["assetPositions"] if p["position"]["coin"] == "BTC"), None)
            current_inv = float(btc_pos["position"]["szi"]) if btc_pos else 0.0
            logger.info(f"Current BTC Inventory: {current_inv}")
            
            if current_inv > 0.05: # roughly 5k
                logger.info("Inventory confirmed! Proceeding to SELL.")
                break
            else:
                logger.warning("Inventory not sufficient. ALO likely cancelled due to crossing. Retrying...")
                await asyncio.sleep(12)

    # 4. Sell Loop (ALO)
    logger.info("--- Starting SELL Leg ---")
    # We sell the size we have to be flat
    # Or just sell the size we intended
    sell_size = size 
    
    while True:
        logger.info("Fetching L2 snapshot...")
        l2 = await asyncio.to_thread(info_client.l2_snapshot, "BTC")
        best_ask = float(l2["levels"][1][0]["px"])
        sell_price = best_ask
        
        logger.info(f"Placing ALO Sell: {sell_size} BTC @ {sell_price}")
        order = HyperliquidOrder(
            symbol="BTC",
            price=sell_price,
            size=sell_size,
            side=HyperliquidOrderSide.SELL,
            alo=True
        )
        try:
            res = await client.place_limit_order(order)
            logger.info(f"Order sent: {res}")
        except Exception as e:
            logger.error(f"Order failed: {e}")
            await asyncio.sleep(12)
            continue
            
        logger.info("Waiting 20s for fill...")
        await asyncio.sleep(20)
        
        open_orders = await asyncio.to_thread(info_client.open_orders, client.exchange.account_address)
        is_open = any(o["oid"] == int(res.order_id) for o in open_orders) if res.order_id else False

        if is_open:
            logger.info("Order still open. Cancelling to reprice...")
            await client.cancel_order("BTC", res.order_id)
            await asyncio.sleep(12)
        else:
            logger.info("Order closed. Checking inventory...")
            user_state = await asyncio.to_thread(info_client.user_state, client.exchange.account_address)
            btc_pos = next((p for p in user_state["assetPositions"] if p["position"]["coin"] == "BTC"), None)
            current_inv = float(btc_pos["position"]["szi"]) if btc_pos else 0.0
            logger.info(f"Current BTC Inventory: {current_inv}")
            
            if abs(current_inv) < 0.01: # Close to 0
                logger.info("Inventory neutral! Mission Complete.")
                break
            else:
                 logger.warning("Still holding inventory. Retrying sell...")
                 await asyncio.sleep(12)

if __name__ == "__main__":
    asyncio.run(main())
