import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from slipstream.strategies.brawler.connectors.hyperliquid import HyperliquidInfoClient

async def main():
    client = HyperliquidInfoClient(base_url="https://api.hyperliquid.xyz")
    meta = await asyncio.to_thread(client.info.meta)
    universe = meta["universe"]
    
    targets = ["PEPE", "BONK", "SHIB", "DOGE", "WIF", "POPCAT", "BRETT"]
    found = []
    
    for item in universe:
        name = item["name"]
        for t in targets:
            if t in name:
                found.append(name)
    
    print("Found symbols:", found)

if __name__ == "__main__":
    asyncio.run(main())
