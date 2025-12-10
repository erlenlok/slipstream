
import json
from hyperliquid.info import Info
from hyperliquid.utils import constants

def main():
    info = Info(base_url=constants.MAINNET_API_URL, skip_ws=True)
    wallet = "0xFd5cf66Cf037140A477419B89656E5F735fa82f4"
    
    print(f"Fetching rate limit for {wallet}...")
    try:
        limits = info.user_rate_limit(wallet)
        print(json.dumps(limits, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
