#!/usr/bin/env python3
"""
Test script to properly use Hyperliquid subaccounts for the Gradient strategy.
Based on the guide you provided, this shows the correct way to set up trading on subaccounts.
"""

import os
import sys
import time
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

    # Create wallet from API secret (this is the SIGNER)
    signer_wallet = Account.from_key(api_secret)
    print(f"\nAPI Signer wallet (does the signing): {signer_wallet.address}")
    print(f"  API_KEY matches signer: {signer_wallet.address.lower() == api_key.lower()}")

    # Determine which account to trade on (the vault address)
    target_wallet = gradient_wallet  # Use gradient wallet as target
    print(f"Target trading account (vault address): {target_wallet}")

    if not target_wallet:
        print("Error: No target wallet specified")
        return

    print(f"\nSetting up exchange to trade on behalf of: {target_wallet}")
    print(f"Transaction signing will be done by: {signer_wallet.address}")

    print(f"\nInitializing Hyperliquid clients...")
    
    # Initialize Info client (read-only)
    info = Info("https://api.hyperliquid.xyz", skip_ws=True)
    
    # Get metadata for exchange initialization
    try:
        response = info.meta_and_asset_ctxs()
        meta = response[0] if isinstance(response, list) and len(response) >= 2 else response
        print(f"  Got metadata for exchange initialization")

        # Initialize Exchange client with vault address (THIS is the key!)
        exchange = Exchange(
            wallet=signer_wallet,  # The API wallet that signs transactions
            base_url="https://api.hyperliquid.xyz",  # Mainnet
            meta=meta,
            account_address=target_wallet,  # TRADE ON BEHALF OF THIS ACCOUNT
        )
        print(f"  ✓ Initialized exchange client")
        print(f"  ✓ Signer: {signer_wallet.address}")
        print(f"  ✓ Trading on behalf of: {target_wallet}")

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

        # Test placing a small order on the target wallet (not the signer wallet)
        print(f"\n--- Small Test Order ---")
        print(f"Order will be placed FOR account: {target_wallet}")
        print(f"Order will be signed BY account: {signer_wallet.address}")
        
        # Small test trade: $15 USD worth of BTC
        trade_size_usd = 15.0
        size_coin = trade_size_usd / btc_price
        size_coin = round(size_coin, 5)  # Appropriate decimal precision for BTC
        print(f"Target trade size: ${trade_size_usd} USD = {size_coin} BTC")

        # Check account balance before the trade
        try:
            # Use raw HTTP request like in volume_generator
            import requests
            body = {"type": "clearinghouseState", "user": target_wallet}
            response = requests.post(f"https://api.hyperliquid.xyz/info", json=body, timeout=10)
            response.raise_for_status()
            account_state = response.json()
            account_value = float(account_state['marginSummary']['accountValue'])
            print(f"Account value before trade: ${account_value}")

            if account_value < 50:  # Need minimum balance for trading
                print("⚠️  Account balance is very low (< $50), this might fail!")
        except Exception as e:
            print(f"Could not check account balance: {e}")

        # Use best bid for buying (to test) - adjust price to tick size requirements
        if best_bid:
            # Calculate appropriate price rounding based on tick size
            # BTC has sz_decimals=5, which means price should be rounded to 1 decimal (6-5=1)
            import math
            tick_size = 0.1  # For BTC
            adjusted_price = math.floor(best_bid * 10) / 10  # Round down to 1 decimal
            print(f"Using adjusted price: ${adjusted_price} (from ${best_bid})")
        else:
            # Fallback to mid price with small adjustment
            adjusted_price = btc_price * 0.999  # 0.1% below market
            print(f"Using fallback price: ${adjusted_price}")

        # Place a small buy order
        print(f"Attempting to place buy order: {size_coin} BTC at ${adjusted_price}")

        try:
            response = exchange.order(
                name='BTC',  # Use 'name' not 'coin' based on volume_generator code
                is_buy=True,
                sz=size_coin,
                limit_px=adjusted_price,
                order_type={"limit": {"tif": "Ioc"}},  # Immediate or Cancel
                reduce_only=False,
            )
            print(f"  Order response: {response}")
            
            # Check if order was successful
            if isinstance(response, dict) and response.get('status') == 'ok':
                data = response.get('response', {}).get('data', {})
                statuses = data.get('statuses', [])
                if statuses:
                    status = statuses[0]
                    if 'filled' in status:
                        filled_info = status['filled']
                        filled_size = float(filled_info.get('totalSz', 0))
                        avg_price = float(filled_info.get('avgPx', 0))
                        print(f"  ✅ Order FILLED on account {target_wallet}: {filled_size} BTC at ${avg_price}")
                        print(f"  Amount: ${filled_size * avg_price:.2f}")
                        
                        # Now close the position immediately - sell the same amount
                        if filled_size > 0:
                            # Get current sell price
                            sell_price = avg_price * 1.001  # Slightly above purchase price
                            if best_ask:
                                sell_price = best_ask  # Use current market price if available
                            
                            print(f"Closing position: selling {filled_size} BTC at ${sell_price}")
                            close_response = exchange.order(
                                name='BTC',  # Use 'name' not 'coin' based on volume_generator code
                                is_buy=False,
                                sz=filled_size,
                                limit_px=sell_price,
                                order_type={"limit": {"tif": "Ioc"}},  # Immediate or Cancel
                                reduce_only=False,
                            )
                            print(f"  Close order response: {close_response}")

                            # Check if close order was successful
                            close_data = close_response.get('response', {}).get('data', {})
                            close_statuses = close_data.get('statuses', [])
                            if close_statuses and 'filled' in close_statuses[0]:
                                close_filled = close_statuses[0]['filled']
                                close_filled_size = float(close_filled.get('totalSz', 0))
                                close_avg_price = float(close_filled.get('avgPx', 0))
                                print(f"  ✅ Close order FILLED: {close_filled_size} BTC at ${close_avg_price}")
                                print(f"  Close amount: ${close_filled_size * close_avg_price:.2f}")
                            else:
                                print(f"  ❌ Close order not filled immediately, remaining open")
                    elif 'resting' in status:
                        print(f"  ⚠️  Order placed but resting (not filled immediately)")
                        # Cancel the order if it's resting
                        oid = status['resting'].get('oid')
                        if oid:
                            print(f"  Canceling resting order {oid}")
                            cancel_response = exchange.cancel('BTC', int(oid))  # This one looks correct based on volume_generator
                            print(f"  Cancel response: {cancel_response}")
                    else:
                        print(f"  Order status: {status}")
                else:
                    print(f"  No status in response data")
            else:
                print(f"  ❌ Order failed: {response}")
                
        except Exception as e:
            print(f"  ❌ Error placing order: {e}")
            import traceback
            traceback.print_exc()

        # Final check: get account state after potential trade
        try:
            # Use raw HTTP request like in volume_generator
            import requests
            body = {"type": "clearinghouseState", "user": target_wallet}
            response = requests.post(f"https://api.hyperliquid.xyz/info", json=body, timeout=10)
            response.raise_for_status()
            new_account_state = response.json()
            new_account_value = float(new_account_state['marginSummary']['accountValue'])
            print(f"Account value after trade: ${new_account_value}")
        except Exception as e:
            print(f"Could not check account balance after trade: {e}")

    except Exception as e:
        print(f"  ❌ Error initializing exchange: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nThis error likely occurs because the API wallet {signer_wallet.address} is not authorized to trade on behalf of {target_wallet}")
        print(f"Solution: Ensure the API wallet is authorized to trade for the target wallet.")


if __name__ == "__main__":
    main()