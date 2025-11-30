#!/usr/bin/env python3
"""
Debug script to test wallet selection in Hyperliquid SDK.
This script will help identify why trades are being placed on the main wallet instead of gradient wallet.
"""

import os
import sys
from pathlib import Path
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange


def load_env_vars():
    """Load environment variables from .env.gradient file."""
    env_file = '/home/ubuntu/slipstream/.env.gradient'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"\'')
    else:
        print(f"Warning: {env_file} not found")


def main():
    print("Loading environment variables...")
    load_env_vars()

    # Get environment variables
    api_key = os.getenv('HYPERLIQUID_API_KEY')
    api_secret = os.getenv('HYPERLIQUID_API_SECRET')
    main_wallet = os.getenv('HYPERLIQUID_MAIN_WALLET')
    gradient_wallet = os.getenv('HYPERLIQUID_GRADIENT_WALLET')
    brawler_wallet = os.getenv('HYPERLIQUID_BRAWLER_WALLET')

    print(f"Environment variables:")
    print(f"  HYPERLIQUID_API_KEY: {api_key[:8] if api_key else 'None'}...")
    print(f"  HYPERLIQUID_API_SECRET: {'✓' if api_secret else '✗'}")
    print(f"  HYPERLIQUID_MAIN_WALLET: {main_wallet}")
    print(f"  HYPERLIQUID_GRADIENT_WALLET: {gradient_wallet}")
    print(f"  HYPERLIQUID_BRAWLER_WALLET: {brawler_wallet}")

    # Verify API credentials
    if not all([api_key, api_secret]):
        print("Error: Missing required API credentials")
        return

    # Create wallet from API secret
    wallet = Account.from_key(api_secret)
    print(f"\nWallet derived from API secret:")
    print(f"  Derived address: {wallet.address}")
    print(f"  API_KEY matches secret: {wallet.address.lower() == api_key.lower()}")

    # Check if derived wallet matches any configured wallets
    if main_wallet and wallet.address.lower() == main_wallet.lower():
        print(f"  ✓ Derived wallet matches HYPERLIQUID_MAIN_WALLET")
    elif gradient_wallet and wallet.address.lower() == gradient_wallet.lower():
        print(f"  ✓ Derived wallet matches HYPERLIQUID_GRADIENT_WALLET")
    elif brawler_wallet and wallet.address.lower() == brawler_wallet.lower():
        print(f"  ✓ Derived wallet matches HYPERLIQUID_BRAWLER_WALLET")
    else:
        print(f"  ⚠️  Derived wallet doesn't match any specific wallet variable")
        print(f"    Derived: {wallet.address}")
        print(f"    Expected MAIN: {main_wallet}")
        print(f"    Expected GRADIENT: {gradient_wallet}")
        print(f"    Expected BRAWLER: {brawler_wallet}")

    print(f"\nInitializing Hyperliquid clients...")
    
    # Initialize Info client (read-only)
    info = Info("https://api.hyperliquid.xyz")
    
    # Get account metadata to check current positions
    print(f"\nChecking account positions for {wallet.address}...")
    try:
        account_summary = info.account_summary(wallet.address, type="spot")
        print(f"  Account summary: {account_summary}")
    except Exception as e:
        print(f"  Error getting account summary: {e}")

    # Get current BTC price
    all_mids = info.all_mids()
    btc_price = float(all_mids.get('BTC', 0))
    print(f"  Current BTC price: ${btc_price}")

    # Get L2 book data for BTC
    try:
        book_data = info.l2_snapshot('BTC')
        if book_data and 'levels' in book_data:
            levels = book_data['levels']
            best_bid = float(levels[0][0]['px']) if levels[0] else None
            best_ask = float(levels[1][0]['px']) if levels[1] else None
            print(f"  Market: BID ${best_bid}, ASK ${best_ask}")
        else:
            print(f"  Could not get L2 book data")
    except Exception as e:
        print(f"  Error getting market data: {e}")

    # Get metadata for exchange initialization
    try:
        response = info.meta_and_asset_ctxs()
        meta = response[0] if isinstance(response, list) and len(response) >= 2 else response
        print(f"  Got metadata for exchange initialization")

        # Initialize Exchange client (for placing orders)
        exchange = Exchange(
            wallet=wallet,
            base_url="https://api.hyperliquid.xyz",
            meta=meta,
            account_address=wallet.address,  # This is the key - it should match the wallet
        )
        print(f"  Initialized exchange client for {wallet.address}")

        # Test placing a small order (should be placed on the wallet derived from the API secret)
        print(f"\n--- Test Order Placement ---")
        print(f"This order will be placed on the account corresponding to the API key/secret provided")
        print(f"Wallet address: {wallet.address}")
        
        # Small test trade: $15 USD worth of BTC
        trade_size_usd = 15.0
        size_coin = trade_size_usd / btc_price
        size_coin = round(size_coin, 5)  # Appropriate decimal precision for BTC
        print(f"Target trade size: ${trade_size_usd} USD = {size_coin} BTC")

        # Get current market prices
        best_bid = best_ask = btc_price  # Fallback to mid price if we couldn't get L2
        if 'best_bid' in locals() and best_bid:
            pass  # We already have best_bid and best_ask
        else:
            # Calculate approximate bid/ask with small spread
            spread = btc_price * 0.001  # 10 bps spread
            best_bid = btc_price - spread/2
            best_ask = btc_price + spread/2

        # Place a buy order at bid (to test)
        print(f"Attempting to place buy order: {size_coin} BTC at ${best_bid}")
        
        try:
            response = exchange.order(
                name='BTC',
                is_buy=True,
                sz=size_coin,
                limit_px=best_bid,
                order_type={"limit": {"tif": "Ioc"}},  # Immediate or Cancel
                reduce_only=False,
            )
            print(f"  Order response: {response}")
            
            # Extract order ID and check status
            if isinstance(response, dict) and 'response' in response:
                data = response['response'].get('data', {})
                statuses = data.get('statuses', [])
                if statuses:
                    status = statuses[0]
                    if 'filled' in status:
                        filled_info = status['filled']
                        filled_size = float(filled_info.get('totalSz', 0))
                        avg_price = float(filled_info.get('avgPx', 0))
                        print(f"  ✓ Order filled: {filled_size} BTC at ${avg_price}")
                        print(f"  Amount: ${filled_size * avg_price:.2f}")
                        
                        # Now try to close the position - place sell order for the same amount
                        if filled_size > 0:
                            print(f"Closing position: selling {filled_size} BTC")
                            close_response = exchange.order(
                                name='BTC',
                                is_buy=False,
                                sz=filled_size,
                                limit_px=best_ask,
                                order_type={"limit": {"tif": "Ioc"}},  # Immediate or Cancel
                                reduce_only=False,
                            )
                            print(f"  Close order response: {close_response}")
                            
                    elif 'resting' in status:
                        print(f"  ⚠️  Order resting (not filled immediately)")
                        # Cancel the order if it's resting
                        oid = status['resting'].get('oid')
                        if oid:
                            print(f"  Canceling order {oid}")
                            cancel_response = exchange.cancel('BTC', int(oid))
                            print(f"  Cancel response: {cancel_response}")
                    else:
                        print(f"  Order status: {status}")
                else:
                    print(f"  No status in response data")
            else:
                print(f"  Unexpected response format: {response}")
                
        except Exception as e:
            print(f"  Error placing order: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"  Error initializing exchange: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()