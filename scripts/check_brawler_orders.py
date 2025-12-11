
import asyncio
import os
import aiohttp
import json

async def main():
    api_key = os.getenv("HYPERLIQUID_API_KEY")
    wallet = os.getenv("HYPERLIQUID_MAIN_WALLET")
    
    if not wallet:
        print("No wallet found")
        return

    print(f"Checking open orders for {wallet}...")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.hyperliquid.xyz/info", 
            json={"type": "frontendOpenOrders", "user": wallet}
        ) as resp:
            data = await resp.json()
            
    print(f"Found {len(data)} open orders.")
    for o in data:
        print(f" - {o['coin']} {o['side']} {o['sz']} @ {o['limitPx']}")
        if o['coin'] == "FARTCOIN":
            print("   >>> FARTCOIN ORDER FOUND <<<")

if __name__ == "__main__":
    asyncio.run(main())
