#!/usr/bin/env python3
"""
Test the correct approach using vault_address instead of account_address
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

    print(f"Setup:")
    print(f"  API Signer: {api_key[:8]}...")
    print(f"  Main Wallet: {main_wallet[:8]}...")
    print(f"  Gradient Wallet: {gradient_wallet[:8]}...")

    # Create wallet from API secret
    signer_wallet = Account.from_key(api_secret)

    # Check positions before
    print(f"\n--- POSITIONS BEFORE ---")
    main_pos_before = get_positions(main_wallet)
    grad_pos_before = get_positions(gradient_wallet)
    
    print(f"Main wallet: {len(main_pos_before)} positions")
    print(f"Gradient wallet: {len(grad_pos_before)} positions")

    # Get market data
    info = Info("https://api.hyperliquid.xyz", skip_ws=True)
    btc_price = float(info.all_mids().get('BTC', 0))
    size_coin = round(15.0 / btc_price, 5)  # $15 worth
    book_data = info.l2_snapshot('BTC')
    best_bid = float(book_data['levels'][0][0]['px']) if book_data and 'levels' in book_data else btc_price * 0.999

    print(f"\nMarket: BTC @ ${btc_price}, buying {size_coin} BTC @ ${best_bid}")

    # APPROACH 1: Using vault_address parameter (CORRECT for subaccounts)
    print(f"\n--- APPROACH 1: Using vault_address (CORRECT for subaccounts) ---")
    response = info.meta_and_asset_ctxs()
    meta = response[0] if isinstance(response, list) and len(response) >= 2 else response

    exchange_vault = Exchange(
        wallet=signer_wallet,
        base_url="https://api.hyperliquid.xyz",
        meta=meta,
        vault_address=gradient_wallet,  # CORRECT parameter for subaccounts
    )

    print(f"  Exchange created with vault_address={gradient_wallet[:8]}...")
    print(f"  Transactions signed by: {signer_wallet.address[:8]}...")

    print(f"  Placing order via exchange_vault (with vault_address=gradient_wallet)...")
    try:
        response_vault = exchange_vault.order(
            name='BTC',
            is_buy=True,
            sz=size_coin,
            limit_px=best_bid,
            order_type={"limit": {"tif": "Gtc"}},
            reduce_only=False,
        )
        print(f"  Response vault: {response_vault}")
        
        order_id_vault = None
        if isinstance(response_vault, dict) and response_vault.get('status') == 'ok':
            data = response_vault.get('response', {}).get('data', {})
            statuses = data.get('statuses', [])
            if statuses:
                status = statuses[0]
                if 'resting' in status:
                    order_id_vault = status['resting']['oid']
                    print(f"  Vault order placed with ID: {order_id_vault}")
                elif 'filled' in status:
                    filled = status['filled']
                    print(f"  Vault order FILLED: {filled.get('totalSz')} BTC @ ${filled.get('avgPx')}")
        
    except Exception as e:
        print(f"  Error with vault approach: {e}")
        import traceback
        traceback.print_exc()

    # Wait and check positions after
    time.sleep(3)
    print(f"\n--- POSITIONS AFTER VAULT APPROACH ---")
    main_pos_after_vault = get_positions(main_wallet)
    grad_pos_after_vault = get_positions(gradient_wallet)
    
    print(f"Main wallet: {len(main_pos_after_vault)} positions")
    print(f"Gradient wallet: {len(grad_pos_after_vault)} positions")

    # Check what changed
    main_new_vault = len(main_pos_after_vault) - len(main_pos_before)
    grad_new_vault = len(grad_pos_after_vault) - len(grad_pos_before)
    
    print(f"Main wallet changes: {main_new_vault} new positions")
    print(f"Gradient wallet changes: {grad_new_vault} new positions")

    if grad_new_vault > 0:
        print(f"  ✅ SUCCESS: Positions went to gradient wallet (vault_address worked!)")
    elif main_new_vault > 0:
        print(f"  ❌ FAILED: Positions went to main wallet (vault_address didn't work)")
    else:
        print(f"  ℹ️  No new positions - order likely resting")

    # APPROACH 2: For comparison, also try account_address (WRONG for subaccounts, but let's see difference)
    print(f"\n--- APPROACH 2: Using account_address (WRONG for subaccounts) ---")
    exchange_account = Exchange(
        wallet=signer_wallet,
        base_url="https://api.hyperliquid.xyz",
        meta=meta,
        account_address=gradient_wallet,  # WRONG parameter for subaccounts
    )

    print(f"  Exchange created with account_address={gradient_wallet[:8]}...")

    # Place a second order with the old approach to see the difference
    size_coin2 = round(12.0 / btc_price, 5)  # slightly different size
    best_bid2 = best_bid * 0.9999  # slightly different price
    print(f"  Placing second order via exchange_account (with account_address=gradient_wallet)...")
    
    try:
        response_account = exchange_account.order(
            name='BTC',
            is_buy=True,
            sz=size_coin2,
            limit_px=best_bid2,
            order_type={"limit": {"tif": "Gtc"}},
            reduce_only=False,
        )
        print(f"  Response account: {response_account}")
        
        order_id_account = None
        if isinstance(response_account, dict) and response_account.get('status') == 'ok':
            data = response_account.get('response', {}).get('data', {})
            statuses = data.get('statuses', [])
            if statuses:
                status = statuses[0]
                if 'resting' in status:
                    order_id_account = status['resting']['oid']
                    print(f"  Account order placed with ID: {order_id_account}")
                elif 'filled' in status:
                    filled = status['filled']
                    print(f"  Account order FILLED: {filled.get('totalSz')} BTC @ ${filled.get('avgPx')}")
        
    except Exception as e:
        print(f"  Error with account approach: {e}")

    # Wait and check final positions
    time.sleep(3)
    print(f"\n--- FINAL POSITIONS ---")
    main_pos_final = get_positions(main_wallet)
    grad_pos_final = get_positions(gradient_wallet)
    
    print(f"Main wallet final: {len(main_pos_final)} positions")
    for p in main_pos_final:
        print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']}")
    
    print(f"Gradient wallet final: {len(grad_pos_final)} positions")
    for p in grad_pos_final:
        print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']}")

    # Analyze final results
    main_final_changes = len(main_pos_final) - len(main_pos_before)
    grad_final_changes = len(grad_pos_final) - len(grad_pos_before)
    
    print(f"\n--- COMPARISON ---")
    print(f"Total new positions on main wallet: {main_final_changes}")
    print(f"Total new positions on gradient wallet: {grad_final_changes}")
    
    if grad_final_changes > 0:
        print(f"✅ GOOD: Positions are going to gradient wallet!")
    else:
        print(f"❌ PROBLEM: No positions went to gradient wallet")
        
    # Clean up - cancel any resting orders
    if order_id_vault:
        print(f"\nCancelling vault order {order_id_vault}...")
        try:
            cancel_resp = exchange_vault.cancel('BTC', int(order_id_vault))
            print(f"  Vault cancel result: {cancel_resp}")
        except Exception as e:
            print(f"  Vault cancel error: {e}")
    
    if order_id_account:
        print(f"Cancelling account order {order_id_account}...")
        try:
            cancel_resp = exchange_account.cancel('BTC', int(order_id_account))
            print(f"  Account cancel result: {cancel_resp}")
        except Exception as e:
            print(f"  Account cancel error: {e}")

    print(f"\n--- CONCLUSION ---")
    print(f"The vault_address parameter should be used for subaccount trading")
    print(f"instead of account_address, according to Hyperliquid documentation.")


if __name__ == "__main__":
    main()