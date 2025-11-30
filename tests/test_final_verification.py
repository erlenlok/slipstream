#!/usr/bin/env python3
"""
Final comprehensive test to understand the exact flow of trades.
"""

import os
import sys
import time
import requests
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

def get_account_state(wallet_address):
    """Get account state using raw HTTP request."""
    body = {"type": "clearinghouseState", "user": wallet_address}
    response = requests.post(f"https://api.hyperliquid.xyz/info", json=body, timeout=10)
    response.raise_for_status()
    return response.json()

def get_positions(wallet_address):
    """Get current positions for a wallet."""
    state = get_account_state(wallet_address)
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
    print(f"  API Signer: {api_key[:8]}... (signs transactions)")
    print(f"  Main Wallet: {main_wallet[:8]}... (master account)")
    print(f"  Gradient Wallet: {gradient_wallet[:8]}... (subaccount)")
    
    # Create wallet from API secret
    signer_wallet = Account.from_key(api_secret)
    print(f"  Derived signer: {signer_wallet.address[:8]}... (from API secret)")
    print(f"  Signer matches API key: {signer_wallet.address.lower() == api_key.lower()}")

    # Check initial states
    print(f"\n--- INITIAL STATE ---")
    main_pos_before = get_positions(main_wallet)
    grad_pos_before = get_positions(gradient_wallet)
    
    print(f"Main wallet positions before: {len(main_pos_before)}")
    for p in main_pos_before:
        print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']}")
    
    print(f"Gradient wallet positions before: {len(grad_pos_before)}")
    for p in grad_pos_before:
        print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']}")

    # Initialize Exchange with account_address (the way gradient code does it)
    info = Info("https://api.hyperliquid.xyz", skip_ws=True)
    response = info.meta_and_asset_ctxs()
    meta = response[0] if isinstance(response, list) and len(response) >= 2 else response

    exchange = Exchange(
        wallet=signer_wallet,
        base_url="https://api.hyperliquid.xyz",
        meta=meta,
        account_address=gradient_wallet,  # Trade on behalf of gradient wallet
    )
    print(f"\nExchange initialized:")
    print(f"  Signer: {signer_wallet.address[:8]}...")
    print(f"  Trading on behalf of: {gradient_wallet[:8]}...")

    # Get market data
    all_mids = info.all_mids()
    btc_price = float(all_mids.get('BTC', 0))
    book_data = info.l2_snapshot('BTC')
    if book_data and 'levels' in book_data:
        levels = book_data['levels']
        best_bid = float(levels[0][0]['px']) if levels[0] else btc_price * 0.999
        best_ask = float(levels[1][0]['px']) if levels[1] else btc_price * 1.001
    else:
        best_bid = btc_price * 0.999
        best_ask = btc_price * 1.001
    
    print(f"BTC: ${btc_price}, BID: ${best_bid}, ASK: ${best_ask}")

    # Place a minimal order that should go to gradient wallet
    size_coin = 0.0001  # Small but above minimum
    print(f"\n--- PLACING ORDER TO GRADIENT WALLET ---")
    print(f"Placing buy order: {size_coin} BTC at ${best_bid}")
    
    try:
        response = exchange.order(
            name='BTC',
            is_buy=True,
            sz=size_coin,
            limit_px=best_bid,
            order_type={"limit": {"tif": "Gtc"}},
            reduce_only=False,
        )
        print(f"Order response: {response}")
        
        # Check if order was filled immediately
        if isinstance(response, dict) and response.get('status') == 'ok':
            data = response.get('response', {}).get('data', {})
            statuses = data.get('statuses', [])
            if statuses:
                status = statuses[0]
                if 'filled' in status:
                    filled = status['filled']
                    filled_size = float(filled.get('totalSz', 0))
                    avg_px = float(filled.get('avgPx', 0))
                    print(f"  ✅ Order FILLED immediately: {filled_size} BTC @ ${avg_px}")
                    
                    # Place closing order (sell) to clear the position
                    print(f"Placing closing sell order: {filled_size} BTC at ${best_ask}")
                    close_response = exchange.order(
                        name='BTC',
                        is_buy=False,
                        sz=filled_size,
                        limit_px=best_ask,
                        order_type={"limit": {"tif": "Gtc"}},
                        reduce_only=False,
                    )
                    print(f"  Close order response: {close_response}")
                    
                elif 'resting' in status:
                    order_id = status['resting'].get('oid')
                    print(f"  Order placed as RESTING with ID: {order_id}")
                    
    except Exception as e:
        print(f"Error placing order: {e}")
        import traceback
        traceback.print_exc()
        return

    # Wait for processing
    time.sleep(3)

    # Check positions after the order
    print(f"\n--- POSITIONS AFTER ORDER ---")
    main_pos_after = get_positions(main_wallet)
    grad_pos_after = get_positions(gradient_wallet)
    
    print(f"Main wallet positions after: {len(main_pos_after)}")
    for p in main_pos_after:
        print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']}")
    
    print(f"Gradient wallet positions after: {len(grad_pos_after)}")
    for p in grad_pos_after:
        print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']}")

    # Analyze the result
    print(f"\n--- ANALYSIS ---")
    
    # Check if main wallet gained positions
    main_new_positions = []
    for p in main_pos_after:
        is_new = True
        for old_p in main_pos_before:
            if old_p['coin'] == p['coin'] and abs(old_p['size'] - p['size']) < 0.00001:
                is_new = False
                break
        if is_new:
            main_new_positions.append(p)
    
    # Check if gradient wallet gained positions
    grad_new_positions = []
    for p in grad_pos_after:
        is_new = True
        for old_p in grad_pos_before:
            if old_p['coin'] == p['coin'] and abs(old_p['size'] - p['size']) < 0.00001:
                is_new = False
                break
        if is_new:
            grad_new_positions.append(p)
    
    print(f"New positions on main wallet: {len(main_new_positions)}")
    for p in main_new_positions:
        print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']}")
    
    print(f"New positions on gradient wallet: {len(grad_new_positions)}")
    for p in grad_new_positions:
        print(f"  {p['coin']}: {p['size']} {p['side']} @ ${p['entryPx']}")
    
    if len(grad_new_positions) > 0:
        print(f"\n✅ SUCCESS: New positions on gradient wallet!")
    elif len(main_new_positions) > 0:
        print(f"\n❌ PROBLEM: New positions on main wallet instead of gradient!")
    else:
        print(f"\nℹ️  No new positions detected - order may still be open or was cancelled")

    # Let's also check the detailed account states
    print(f"\n--- DETAILED ACCOUNT VALUES ---")
    main_state = get_account_state(main_wallet)
    grad_state = get_account_state(gradient_wallet)
    
    main_value = float(main_state['marginSummary']['accountValue'])
    grad_value = float(grad_state['marginSummary']['accountValue'])
    
    print(f"Main wallet value: ${main_value}")
    print(f"Gradient wallet value: ${grad_value}")
    
    # Compare with before values
    main_state_before = get_account_state(main_wallet)
    grad_state_before = get_account_state(gradient_wallet)
    
    main_value_before = float(main_state_before['marginSummary']['accountValue'])
    grad_value_before = float(grad_state_before['marginSummary']['accountValue'])
    
    print(f"Main value change: ${main_value - main_value_before:.6f}")
    print(f"Gradient value change: ${grad_value - grad_value_before:.6f}")


if __name__ == "__main__":
    main()