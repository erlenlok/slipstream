#!/usr/bin/env python3
"""
Test to find the RIGHT approach to trade on subaccounts.
"""

import os
import sys
import time
from pathlib import Path
import requests
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

    print(f"Setup:")
    print(f"  API Signer (signs): {api_key[:8]}...")
    print(f"  Main Wallet: {main_wallet[:8]}... (master)")
    print(f"  Gradient Wallet: {gradient_wallet[:8]}... (subaccount)")

    # Create wallet from API secret
    signer_wallet = Account.from_key(api_secret)

    # Get market data
    info = Info("https://api.hyperliquid.xyz", skip_ws=True)
    btc_price = float(info.all_mids().get('BTC', 0))
    size_coin = round(15.0 / btc_price, 5)  # $15 worth
    book_data = info.l2_snapshot('BTC')
    best_bid = float(book_data['levels'][0][0]['px']) if book_data and 'levels' in book_data else btc_price * 0.999

    print(f"\nMarket: BTC @ ${btc_price}, buying {size_coin} BTC @ ${best_bid}")

    # According to the Hyperliquid subaccount guide, there might be a more explicit way
    # Let's check the metadata first to understand the subaccount relationship
    print(f"\n--- ATTEMPTING TWO DIFFERENT APPROACHES ---")

    # APPROACH 1: Using the account_address parameter (current method)
    print(f"\nAPPROACH 1: Using account_address parameter")
    response = info.meta_and_asset_ctxs()
    meta = response[0] if isinstance(response, list) and len(response) >= 2 else response

    exchange1 = Exchange(
        wallet=signer_wallet,
        base_url="https://api.hyperliquid.xyz",
        meta=meta,
        account_address=gradient_wallet,  # Should trade on gradient wallet
    )

    print(f"  Placing order via exchange1 (with account_address=gradient_wallet)...")
    try:
        response1 = exchange1.order(
            name='BTC',
            is_buy=True,
            sz=size_coin,
            limit_px=best_bid,
            order_type={"limit": {"tif": "Gtc"}},
            reduce_only=False,
        )
        print(f"  Response 1: {response1}")
        
        order_id_1 = None
        if isinstance(response1, dict) and response1.get('status') == 'ok':
            data = response1.get('response', {}).get('data', {})
            statuses = data.get('statuses', [])
            if statuses and 'resting' in statuses[0]:
                order_id_1 = statuses[0]['resting']['oid']
                print(f"  Order 1 placed with ID: {order_id_1}")
        
    except Exception as e:
        print(f"  Error with approach 1: {e}")

    # APPROACH 2: Maybe we should use a different method - let's try creating a new exchange instance for each target
    # Actually, let me check if there's an issue with how the account_address is supposed to be used
    print(f"\nAPPROACH 2: Let's check if account_address needs special handling")
    
    # Let me look at actual recent fills for both accounts to see where recent trades went
    print(f"\n--- CHECKING RECENT FILLS ---")
    
    # Check main wallet recent fills
    try:
        main_fills_response = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "userFills", "user": main_wallet},
            timeout=10
        )
        main_fills = main_fills_response.json()
        print(f"Main wallet recent fills: {len(main_fills) if isinstance(main_fills, list) else 'Error'}")
        if isinstance(main_fills, list) and len(main_fills) > 0:
            latest_main_fill = main_fills[0] if main_fills else None
            if latest_main_fill:
                print(f"  Latest main fill: {latest_main_fill.get('coin', 'N/A')} {latest_main_fill.get('side', 'N/A')} {latest_main_fill.get('sz', 'N/A')} @ ${latest_main_fill.get('px', 'N/A')}")
    except Exception as e:
        print(f"Error checking main wallet fills: {e}")
    
    # Check gradient wallet recent fills
    try:
        grad_fills_response = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "userFills", "user": gradient_wallet},
            timeout=10
        )
        grad_fills = grad_fills_response.json()
        print(f"Gradient wallet recent fills: {len(grad_fills) if isinstance(grad_fills, list) else 'Error'}")
        if isinstance(grad_fills, list) and len(grad_fills) > 0:
            latest_grad_fill = grad_fills[0] if grad_fills else None
            if latest_grad_fill:
                print(f"  Latest grad fill: {latest_grad_fill.get('coin', 'N/A')} {latest_grad_fill.get('side', 'N/A')} {latest_grad_fill.get('sz', 'N/A')} @ ${latest_grad_fill.get('px', 'N/A')}")
        else:
            print("  Gradient wallet has NO recent fills")
    except Exception as e:
        print(f"Error checking gradient wallet fills: {e}")

    print(f"\n--- THE REAL PROBLEM ---")
    print("Based on your observation that the gradient wallet has empty order history,")
    print("even though we can place orders, the actual fills are not going to the gradient wallet.")
    print("This suggests the account_address parameter in Exchange() constructor is not working as expected,")
    print("or there's a setup issue with the API wallet permissions.")
    
    print(f"\n--- POSSIBLE ROOT CAUSES ---")
    print("1. The API wallet (0x998c0B...) may not be properly authorized to trade on behalf of the gradient subaccount")
    print("2. The account_address parameter may require additional setup steps")
    print("3. There might be a version issue with the Hyperliquid SDK")
    print("4. The subaccount may need to be activated in a specific way")
    
    print(f"\n--- RECOMMENDED SOLUTION ---")
    print("You need to verify the API wallet permissions in the Hyperliquid UI:")
    print("- Go to app.hyperliquid.xyz/API")
    print("- Check that the API wallet is properly authorized for the subaccounts")
    print("- The API wallet needs explicit permission to trade on behalf of the gradient subaccount")
    
    # Cancel the order we placed
    if order_id_1:
        print(f"\nCancelling order {order_id_1}...")
        try:
            cancel_response = exchange1.cancel('BTC', int(order_id_1))
            print(f"  Cancel result: {cancel_response}")
        except Exception as e:
            print(f"  Cancel error: {e}")


if __name__ == "__main__":
    main()