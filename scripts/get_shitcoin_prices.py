import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from slipstream.strategies.brawler.connectors.hyperliquid import HyperliquidInfoClient

async def main():
    client = HyperliquidInfoClient(base_url="https://api.hyperliquid.xyz")
    print("Fetching mids...")
    mids = await asyncio.to_thread(client.info.all_mids)
    
    for sym in ["WIF", "kPEPE"]:
        if sym in mids:
            price = float(mids[sym])
            size_200 = 200.0 / price
            print(f"{sym}: ${price:.6f} -> Order Size (200 USD): {size_200:.4f}")
        else:
            print(f"{sym}: Not found")

if __name__ == "__main__":
    asyncio.run(main())
