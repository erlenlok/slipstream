#!/usr/bin/env python3
"""
Test to understand how the Hyperliquid Exchange class actually works with account_address.
"""

import os
import sys
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

def get_positions(wallet_address):
    """Get current positions for a wallet."""
    body = {"type": "clearinghouseState", "user": wallet_address}
    response = requests.post(f"https://api.hyperliquid.xyz/info", json=body, timeout=10)
    response.raise_for_status()
    state = response.json()
    
    positions = []
    asset_positions = state.get("assetPositions", [])
    for ap in asset_positions:
        pos = ap.get("position", {})
        if float(pos.get("szi", 0)) != 0:  # Only non-zero positions
            positions.append({
                'coin': pos.get("coin"),
                'size': float(pos.get("szi", 0)),
                'side': 'LONG' if float(pos.get("szi", 0)) > 0 else 'SHORT',
                'entryPx': pos.get("entryPx"),
                'positionValue': pos.get("positionValue")
            })
    
    return positions

def main():
    print("Loading environment variables...")
    load_env_vars()

    # Get environment variables
    api_key = os.getenv('HYPERLIQUID_API_KEY')
    api_secret = os.getenv('HYPERLIQUID_API_SECRET')
    main_wallet = os.getenv('HYPERLIQUID_MAIN_WALLET')
    gradient_wallet = os.getenv('HYPERLIQUID_GRADIENT_WALLET')

    print(f"API Key: {api_key[:8]}...")
    print(f"Main Wallet: {main_wallet}")
    print(f"Gradient Wallet: {gradient_wallet}")

    # Create wallet from API secret
    signer_wallet = Account.from_key(api_secret)
    print(f"Signer Wallet: {signer_wallet.address}")

    # Test 1: Initialize Exchange WITHOUT account_address (should default to signer wallet)
    print(f"\n--- TEST 1: Exchange without account_address ---")
    info = Info("https://api.hyperliquid.xyz", skip_ws=True)
    response = info.meta_and_asset_ctxs()
    meta = response[0] if isinstance(response, list) and len(response) >= 2 else response

    try:
        exchange_default = Exchange(
            wallet=signer_wallet,
            base_url="https://api.hyperliquid.xyz",
            meta=meta,
            # NO account_address parameter - should use signer address directly
        )
        print(f"Exchange initialized WITHOUT account_address")
        print(f"This should trade on the signer wallet: {signer_wallet.address}")
    except Exception as e:
        print(f"Error initializing exchange without account_address: {e}")

    # Test 2: Initialize Exchange WITH account_address (current approach)
    print(f"\n--- TEST 2: Exchange WITH account_address ---")
    try:
        exchange_with_account = Exchange(
            wallet=signer_wallet,
            base_url="https://api.hyperliquid.xyz",
            meta=meta,
            account_address=gradient_wallet,  # This is how the code does it
        )
        print(f"Exchange initialized WITH account_address={gradient_wallet}")
        print(f"Transactions signed by: {signer_wallet.address}")
        print(f"Should trade on behalf of: {gradient_wallet}")
    except Exception as e:
        print(f"Error initializing exchange with account_address: {e}")
        print(f"Error: {e}")
        return

    # Get initial positions to compare
    print(f"\n--- INITIAL POSITIONS ---")
    main_pos = get_positions(main_wallet)
    grad_pos = get_positions(gradient_wallet)
    
    print(f"Main wallet positions: {len(main_pos)}")
    for p in main_pos:
        print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']} (on {main_wallet[:8]}...)")
    
    print(f"Gradient wallet positions: {len(grad_pos)}")
    for p in grad_pos:
        print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']} (on {gradient_wallet[:8]}...)")

    # Check the all_mids to place a tiny test order
    all_mids = info.all_mids()
    btc_price = float(all_mids.get('BTC', 0))
    print(f"\nBTC price: ${btc_price}")

    # Get market data
    book_data = info.l2_snapshot('BTC')
    if book_data and 'levels' in book_data:
        levels = book_data['levels']
        best_bid = float(levels[0][0]['px']) if levels[0] else 0
        best_ask = float(levels[1][0]['px']) if levels[1] else 0
        print(f"Market: BID ${best_bid}, ASK ${best_ask}")
    else:
        print("Could not get market data")
        return

    # Test a very small order using the exchange_with_account
    size_coin = 0.00001  # Tiny size
    print(f"\n--- PLACING TINY TEST ORDER via exchange_with_account ---")
    print(f"Attempting to place buy order for {size_coin} BTC on gradient wallet via signer")
    
    try:
        response = exchange_with_account.order(
            name='BTC',
            is_buy=True,
            sz=size_coin,
            limit_px=best_bid,
            order_type={"limit": {"tif": "Gtc"}},
            reduce_only=False,
        )
        print(f"Order response: {response}")
        
        # Check if the order was resting or filled
        if isinstance(response, dict) and response.get('status') == 'ok':
            data = response.get('response', {}).get('data', {})
            statuses = data.get('statuses', [])
            if statuses:
                status = statuses[0]
                if 'resting' in status:
                    print("Order placed as RESTING")
                    order_id = status['resting'].get('oid')
                    print(f"Order ID: {order_id}")
                elif 'filled' in status:
                    filled = status['filled']
                    print(f"Order FILLED: {filled.get('totalSz')} BTC at ${filled.get('avgPx')}")
        
        # Wait and check positions again
        import time
        time.sleep(3)
        
        print(f"\n--- POSITIONS AFTER TEST ORDER ---")
        main_pos_after = get_positions(main_wallet)
        grad_pos_after = get_positions(gradient_wallet)
        
        print(f"Main wallet positions: {len(main_pos_after)}")
        for p in main_pos_after:
            print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']} (on {main_wallet[:8]}...)")
        
        print(f"Gradient wallet positions: {len(grad_pos_after)}")
        for p in grad_pos_after:
            print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']} (on {gradient_wallet[:8]}...)")

        # Check if positions changed
        main_changed = len(main_pos_after) != len(main_pos)
        grad_changed = len(grad_pos_after) != len(grad_pos)
        
        if main_changed and not grad_changed:
            print(f"\n❌ CONFIRMED: Positions still going to MAIN wallet!")
            old_coins = {p['coin'] for p in main_pos}
            new_coins = {p['coin'] for p in main_pos_after}
            if new_coins - old_coins:
                print(f"  New positions appeared on main wallet: {new_coins - old_coins}")
        elif grad_changed and not main_changed:
            print(f"\n✅ CORRECT: Positions going to gradient wallet!")
        elif main_changed and grad_changed:
            print(f"\n❓ BOTH wallets changed - investigate further")
        else:
            print(f"\nℹ️  No new positions - order may still be open")
        
    except Exception as e:
        print(f"Error placing order: {e}")
        import traceback
        traceback.print_exc()

    # Let's also try to understand how the account_address is supposed to work
    print(f"\n--- RESEARCHING THE ISSUE ---")
    print("The account_address parameter in Exchange constructor may not work as expected.")
    print("Let me check if there's an alternative approach...")
    print("\nAccording to the Hyperliquid docs and subaccount guide you shared:")
    print("- The account_address parameter should allow trading on behalf of another wallet")
    print("- But maybe it requires special permissions or setup")
    print("- Or maybe there's an issue with how the API wallet is authorized")

    # Check if the signer wallet is properly authorized to trade on the gradient wallet
    print(f"\nCurrent setup:")
    print(f"  Signer (does signing): {signer_wallet.address}")
    print(f"  Target (where trades should go): {gradient_wallet}")
    print(f"  Signer and target are {'different' if signer_wallet.address != gradient_wallet else 'the same'}")

    print(f"\nFor this to work, the signer wallet needs to be authorized to trade on behalf of the target wallet.")
    print(f"This might require special permissions or the target wallet to be a subaccount of the signer wallet.")


if __name__ == "__main__":
    main()