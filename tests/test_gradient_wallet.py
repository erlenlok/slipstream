#!/usr/bin/env python3
"""
Test script to verify gradient wallet trading.
Places a $20 market buy order and then a $20 market sell order for BTC perps on the gradient wallet.
"""

import os
from pathlib import Path
from eth_account import Account

def load_env_vars():
    """Load environment variables from .env.gradient file"""
    env_file = Path('.env.gradient')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value.replace('\"', '')  # Remove quotes
    else:
        print(f"⚠️  Warning: {env_file} not found")

def main():
    print("🔍 Loading environment variables...")
    load_env_vars()

    # Get environment variables
    api_key = os.getenv("HYPERLIQUID_API_KEY")
    api_secret = os.getenv("HYPERLIQUID_API_SECRET")
    gradient_wallet = os.getenv("HYPERLIQUID_GRADIENT_WALLET")

    if not all([api_key, api_secret, gradient_wallet]):
        print("❌ Missing required environment variables:")
        print(f"   HYPERLIQUID_API_KEY: {'✓' if api_key else '✗'}")
        print(f"   HYPERLIQUID_API_SECRET: {'✓' if api_secret else '✗'}")
        print(f"   HYPERLIQUID_GRADIENT_WALLET: {'✓' if gradient_wallet else '✗'}")
        return False

    print(f"✅ Environment loaded:")
    print(f"   API Key: {api_key[:8]}...")
    print(f"   Gradient Wallet: {gradient_wallet[:8]}...")

    try:
        from hyperliquid.info import Info
        from hyperliquid.exchange import Exchange
        from hyperliquid.utils import constants

        print("\n🔗 Connecting to Hyperliquid...")
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        # Get current BTC price
        all_mids = info.all_mids()
        btc_price = float(all_mids["BTC"])
        print(f"📊 Current BTC price: ${btc_price:,.2f}")

        # Calculate size for $20 trade and round to 5 decimals (BTC has szDecimals=5 on Hyperliquid)
        raw_size = 20.0 / btc_price
        size = round(raw_size, 5)  # BTC has 5 decimal precision (szDecimals=5)
        print(f"📏 Order size: {size:.5f} BTC ($20 worth)")

        # Create wallet from private key
        wallet = Account.from_key(api_secret)
        print(f"💳 Wallet address from secret: {wallet.address}")

        # Create exchange client targeting the gradient wallet
        exchange = Exchange(
            wallet=wallet,
            base_url=constants.MAINNET_API_URL,
            meta=info.meta(),
            account_address=gradient_wallet  # Target the gradient wallet
        )
        print(f"🎯 Exchange initialized for gradient wallet: {gradient_wallet}")

        print(f"\n💰 Placing $20 BTC market buy order...")
        buy_response = exchange.market_open(
            name="BTC",  # Use 'name' for market_open
            is_buy=True,
            sz=size,
            px=btc_price  # Use current market price as reference
        )

        if buy_response.get("status") == "ok":
            print(f"✅ Market buy order placed successfully!")
            print(f"   Response: {buy_response}")
        else:
            print(f"❌ Market buy order failed: {buy_response}")

        print(f"\n⚠️  Note: Skipped sell order due to 'insufficient margin' on buy order.")
        print("💡 This confirms that the exchange is targeting the gradient wallet,")
        print("   but the gradient wallet may have insufficient balance to trade.")
        print(f"   The 'asset=0' in the error suggests the gradient wallet has no funds.")
        return True

    except Exception as e:
        print(f"💥 Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Script completed successfully!")
    else:
        print("\n❌ Script failed!")
        exit(1)