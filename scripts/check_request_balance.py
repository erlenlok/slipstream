import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.slipstream.strategies.brawler.connectors.hyperliquid import HyperliquidInfoClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CheckRequestBalance")

async def main():
    # Load .env.brawler
    env_path = Path(__file__).parent.parent / ".env.brawler"
    if env_path.exists():
        logger.info(f"Loading env from {env_path}")
        load_dotenv(env_path)
    
    wallet = os.getenv("HYPERLIQUID_BRAWLER_WALLET") or os.getenv("HYPERLIQUID_MAIN_WALLET")
    
    if not wallet:
        logger.error("No wallet address found in environment variables.")
        return

    client = HyperliquidInfoClient(base_url="https://api.hyperliquid.xyz")
    
    logger.info(f"Fetching Rate Limit Stats for {wallet}...")
    try:
        info = await client.get_user_rate_limit(wallet)
        requests_used = int(info.get("nRequestsUsed", 0))
        cum_vol = float(info.get("cumVlm", "0"))
        
        print("\n--- Request Balance ---")
        print(f"Requests Used: {requests_used}")
        print(f"Cumulative Volume: ${cum_vol:,.2f}")
        
    except Exception as e:
        logger.error(f"Failed to fetch rate limits: {e}")

if __name__ == "__main__":
    asyncio.run(main())
