import asyncio
import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.slipstream.strategies.brawler.connectors.hyperliquid import (
    HyperliquidExecutionClient,
    HyperliquidOrder,
    HyperliquidOrderSide,
    HyperliquidInfoClient,
)
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmergencyClose")

async def main():
    parser = argparse.ArgumentParser(description="Emergency Close Position")
    parser.add_argument("symbol", type=str, help="Symbol to close (e.g. BTC, ETH)")
    args = parser.parse_args()
    
    symbol = args.symbol.upper()
    env_path = Path(__file__).parent.parent / ".env.brawler"
    load_dotenv(env_path)
    
    api_key = os.getenv("HYPERLIQUID_API_KEY")
    api_secret = os.getenv("HYPERLIQUID_API_SECRET")
    base_url = "https://api.hyperliquid.xyz"

    client = HyperliquidExecutionClient(api_key, api_secret, base_url=base_url)
    info_client = HyperliquidInfoClient(base_url=base_url)

    target_wallet = os.getenv("HYPERLIQUID_BRAWLER_WALLET") or client.exchange.account_address
    logger.info("Checking %s position for %s...", symbol, target_wallet)

    # Fetch user state via info client
    # Need to run in thread as SDK is sync
    user_state = await asyncio.to_thread(info_client.info.user_state, target_wallet)
    
    target_pos = None
    for item in user_state.get("assetPositions", []):
        pos = item["position"]
        if pos["coin"] == symbol:
            target_pos = pos
            break
            
    if not target_pos:
        logger.info("No position found for %s.", symbol)
        return

    size = float(target_pos["szi"])
    val = float(target_pos["positionValue"])
    entry = float(target_pos["entryPx"] or 0)
    
    logger.info("Found Position: %s size=%.4f value=$%.2f entry=%.2f", symbol, size, val, entry)

    if size == 0:
        logger.info("Position is zero. Nothing to close.")
        return

    # Determine Close Side
    close_side = HyperliquidOrderSide.SELL if size > 0 else HyperliquidOrderSide.BUY
    close_size = abs(size)
    
    # Get Mark Price for aggressive limit
    mids = await asyncio.to_thread(info_client.info.all_mids)
    mark_px = float(mids.get(symbol, 0))
    if mark_px == 0:
        logger.error("Could not fetch mark price for %s", symbol)
        return

    # Aggressive Price (5% slippage to guarantee fill)
    slippage = 0.05
    if close_side == HyperliquidOrderSide.BUY:
        limit_px = mark_px * (1 + slippage)
    else:
        limit_px = mark_px * (1 - slippage)

    # Round price to valid tick? 
    # For now relying on float format (Hyperliquid SDK usually handles standard precision if passed as float, 
    # but strictly we should round. Assuming 5 decimals or integer for major usage).
    # ETH/BTC are usually fine with 1-2 decimals.
    if symbol in ["BTC", "ETH"]:
        limit_px = round(limit_px, 1)
    else:
        limit_px = round(limit_px, 4)

    logger.info("Placing EMERGENCY %s order: %.4f %s @ %.2f", close_side, close_size, symbol, limit_px)
    
    order = HyperliquidOrder(
        symbol=symbol,
        price=limit_px,
        size=close_size,
        side=close_side,
        alo=False # Taker allowed
    )

    res = await client.place_limit_order(order)
    logger.info("Order Result: %s", res)

if __name__ == "__main__":
    asyncio.run(main())
