import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from slipstream.strategies.brawler.connectors.hyperliquid import HyperliquidExecutionClient

async def main():
    # Load .env.brawler
    # scripts/check_positions.py -> scripts -> slipstream
    env_path = Path(__file__).parent.parent / ".env.brawler"
    if env_path.exists():
        print(f"Loading env from {env_path}")
        load_dotenv(env_path)
    
    api_key = os.getenv("HYPERLIQUID_API_KEY")
    api_secret = os.getenv("HYPERLIQUID_API_SECRET")
    
    client = HyperliquidExecutionClient(
        api_key=api_key,
        api_secret=api_secret,
        base_url="https://api.hyperliquid.xyz",
        target_wallet=os.getenv("HYPERLIQUID_BRAWLER_WALLET")
    )
    
    target = os.getenv("HYPERLIQUID_BRAWLER_WALLET")
    # Use the target wallet if provided, otherwise use the client's default account address
    address = target if target else client.exchange.account_address
    print("Fetching User State...")
    # Assuming client.get_user_state is a new method or a wrapper around client.info.user_state
    # and it expects the address.
    user_state = await client.get_user_state(address)
    
    print("\n--- Positions ---")
    found_any = False
    for item in user_state["assetPositions"]:
        pos = item["position"]
        symbol = pos["coin"]
        size = float(pos["szi"])
        val = float(pos["positionValue"]) # Changed from 'val' to 'positionValue' to match Hyperliquid API
        entry = float(pos["entryPx"] or 0)
        
        if size != 0:
            found_any = True
            print(f"{symbol}: {size} (Value: ${val:.2f}, Entry: {entry})")
            
    if not found_any:
        print("No open positions.")

if __name__ == "__main__":
    asyncio.run(main())
