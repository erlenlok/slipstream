import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.slipstream.strategies.brawler.connectors.hyperliquid import HyperliquidInfoClient

async def main():
    client = HyperliquidInfoClient(base_url="https://api.hyperliquid.xyz")
    print("Fetching ETH price...")
    try:
        mids = await asyncio.to_thread(client.info.all_mids)
        price = float(mids.get("ETH", 0.0))
        if price == 0.0:
            print("ETH price not found in all_mids")
            return
            
        print(f"ETH Price: {price}")
        print(f"Size for $500: {500.0 / price}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
