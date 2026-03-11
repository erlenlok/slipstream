
import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from slipstream.strategies.brawler.connectors.hyperliquid import HyperliquidExecutionClient, _load_hyperliquid_modules
from slipstream.strategies.brawler.cli import _load_env_files

async def verify_client_initialization(api_key, api_secret, target_wallet):
    print(f"\n--- Verifying HyperliquidExecutionClient Initialization ---")
    try:
        executor = HyperliquidExecutionClient(
            api_key=api_key,
            api_secret=api_secret,
            target_wallet=target_wallet
        )
        print("✅ Client initialized successfully.")
        print(f"  - Signing Wallet (API Key): {executor.exchange.wallet.address}")
        print(f"  - Target/Vault Wallet:      {executor.exchange.vault_address}")
        
        if executor.exchange.wallet.address != executor.exchange.vault_address:
            print("  - Mode: AGENT (Signing wallet differs from target wallet)")
        else:
            print("  - Mode: DIRECT (Signing wallet matches target wallet)")
            
        return True
    except Exception as e:
        print(f"❌ Client initialization FAILED: {e}")
        return False

def main():
    print("Loading environment...")
    _load_env_files()
    
    api_key = os.getenv("HYPERLIQUID_API_KEY")
    api_secret = os.getenv("HYPERLIQUID_API_SECRET")
    # This logic matches cli.py's fallback
    wallet_addr = os.getenv("HYPERLIQUID_BRAWLER_WALLET") or os.getenv("HYPERLIQUID_MAIN_WALLET")

    if not api_key:
        print("ERROR: HYPERLIQUID_API_KEY not found in environment.")
        return
    if not api_secret:
        print("ERROR: HYPERLIQUID_API_SECRET not found in environment.")
        return
    if not wallet_addr:
        print("ERROR: HYPERLIQUID_BRAWLER_WALLET (or MAIN_WALLET) not found.")
        return

    print(f"Configured Target Wallet: {wallet_addr}")
    
    # Run the async check
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(verify_client_initialization(api_key, api_secret, wallet_addr))
    
    if success:
        print("\n✅ Verification PASSED. You are ready to run Brawler.")
    else:
        print("\n❌ Verification FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
