#!/usr/bin/env python3
"""
Comprehensive test to verify exactly which wallet holds positions after trading.
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

    print(f"\n--- BEFORE TRADING ---")
    print(f"Checking positions on MAIN wallet: {main_wallet}")
    main_positions = get_positions(main_wallet)
    print(f"  Main wallet positions: {len(main_positions)} assets")
    for pos in main_positions:
        print(f"    {pos['coin']}: {pos['size']} {pos['side']} (entry: ${pos['entryPx']})")

    print(f"Checking positions on GRADIENT wallet: {gradient_wallet}")
    grad_positions = get_positions(gradient_wallet)
    print(f"  Gradient wallet positions: {len(grad_positions)} assets")
    for pos in grad_positions:
        print(f"    {pos['coin']}: {pos['size']} {pos['side']} (entry: ${pos['entryPx']})")

    print(f"\nInitializing Hyperliquid clients to trade on gradient wallet...")
    
    # Initialize Info client (read-only)
    info = Info("https://api.hyperliquid.xyz", skip_ws=True)

    # Get metadata for exchange initialization
    response = info.meta_and_asset_ctxs()
    meta = response[0] if isinstance(response, list) and len(response) >= 2 else response

    # Initialize Exchange client with vault address
    exchange = Exchange(
        wallet=signer_wallet,  # The API wallet that signs transactions
        base_url="https://api.hyperliquid.xyz",  # Mainnet
        meta=meta,
        account_address=target_wallet,  # TRADE ON BEHALF OF THIS ACCOUNT
    )
    print(f"✓ Exchange initialized to trade on behalf of: {target_wallet}")

    # Get current market data
    all_mids = info.all_mids()
    btc_price = float(all_mids.get('BTC', 0))
    print(f"Current BTC price: ${btc_price}")

    # Get L2 book data for BTC
    book_data = info.l2_snapshot('BTC')
    if book_data and 'levels' in book_data:
        levels = book_data['levels']
        best_bid = float(levels[0][0]['px']) if levels[0] else 0
        best_ask = float(levels[1][0]['px']) if levels[1] else 0
        print(f"Market: BID ${best_bid}, ASK ${best_ask}")
    else:
        print(f"Could not get L2 book data")
        return

    # Place a small buy order (we'll cancel it rather than trying to close with a sell)
    print(f"\n--- PLACING TEST BUY ORDER ---")
    trade_size_usd = 15.0
    size_coin = trade_size_usd / btc_price
    size_coin = round(size_coin, 5)  # Appropriate decimal precision for BTC
    print(f"Target trade size: ${trade_size_usd} USD = {size_coin} BTC")
    
    print(f"Placing buy order: {size_coin} BTC at ${best_bid} (best bid)")
    
    try:
        response = exchange.order(
            name='BTC',  # Use 'name' not 'coin' based on volume_generator code
            is_buy=True,
            sz=size_coin,
            limit_px=best_bid,
            order_type={"limit": {"tif": "Gtc"}},  # Good til cancelled (so we can cancel later)
            reduce_only=False,
        )
        print(f"Order response: {response}")
        
        order_id = None
        if isinstance(response, dict) and response.get('status') == 'ok':
            data = response.get('response', {}).get('data', {})
            statuses = data.get('statuses', [])
            if statuses and 'filled' not in statuses[0]:
                # Order is resting (not filled immediately)
                status = statuses[0]
                if 'resting' in status:
                    order_id = status['resting'].get('oid')
                    print(f"Order placed as RESTING with ID: {order_id}")
            elif statuses and 'filled' in statuses[0]:
                filled_info = statuses[0]['filled']
                filled_size = float(filled_info.get('totalSz', 0))
                avg_price = float(filled_info.get('avgPx', 0))
                print(f"Order FILLED immediately: {filled_size} BTC at ${avg_price}")
                # Since it filled immediately, we need to place a sell order to close
                print("Since order filled immediately, placing closing sell order...")
                sell_response = exchange.order(
                    name='BTC',
                    is_buy=False,
                    sz=filled_size,
                    limit_px=best_ask,
                    order_type={"limit": {"tif": "Gtc"}},
                    reduce_only=False,
                )
                print(f"Closing order response: {sell_response}")
        
    except Exception as e:
        print(f"Error placing buy order: {e}")
        import traceback
        traceback.print_exc()
        return

    # Wait a moment to ensure the order is processed
    time.sleep(2)

    print(f"\n--- AFTER PLACING ORDER ---")
    print(f"Checking positions on MAIN wallet: {main_wallet}")
    main_positions_after = get_positions(main_wallet)
    print(f"  Main wallet positions: {len(main_positions_after)} assets")
    for pos in main_positions_after:
        print(f"    {pos['coin']}: {pos['size']} {pos['side']} (entry: ${pos['entryPx']})")

    print(f"Checking positions on GRADIENT wallet: {gradient_wallet}")
    grad_positions_after = get_positions(gradient_wallet)
    print(f"  Gradient wallet positions: {len(grad_positions_after)} assets")
    for pos in grad_positions_after:
        print(f"    {pos['coin']}: {pos['size']} {pos['side']} (entry: ${pos['entryPx']})")

    # Check if positions changed
    main_changed = len(main_positions_after) != len(main_positions)
    grad_changed = len(grad_positions_after) != len(grad_positions)
    
    if main_changed:
        print(f"\n⚠️  MAIN WALLET POSITIONS CHANGED - THIS IS THE PROBLEM!")
        old_coins = {p['coin'] for p in main_positions}
        new_coins = {p['coin'] for p in main_positions_after}
        if new_coins - old_coins:
            print(f"  New positions on main wallet: {new_coins - old_coins}")
    elif grad_changed:
        print(f"\n✅ GRADIENT WALLET POSITIONS CHANGED - This is correct!")
        old_coins = {p['coin'] for p in grad_positions}
        new_coins = {p['coin'] for p in grad_positions_after}
        if new_coins - old_coins:
            print(f"  New positions on gradient wallet: {new_coins - old_coins}")
    else:
        print(f"\nℹ️  No positions changed on either wallet - order may still be open")
        
    # Calculate account values
    main_account_state = get_account_state(main_wallet) 
    main_account_value = float(main_account_state['marginSummary']['accountValue'])
    
    grad_account_state = get_account_state(gradient_wallet)
    grad_account_value = float(grad_account_state['marginSummary']['accountValue'])
    
    print(f"\nAccount Values:")
    print(f"  Main wallet: ${main_account_value}")
    print(f"  Gradient wallet: ${grad_account_value}")
    
    # If we have an order ID to cancel, cancel it
    if order_id:
        print(f"\nCancelling order {order_id}...")
        try:
            cancel_response = exchange.cancel('BTC', int(order_id))
            print(f"Cancel response: {cancel_response}")
        except Exception as e:
            print(f"Error cancelling order: {e}")
    
    print(f"\n--- CONCLUSION ---")
    if len(grad_positions_after) > len(grad_positions):
        print("✅ POSITIONS ARE ON GRADIENT WALLET - Trading working correctly")
    elif len(main_positions_after) > len(main_positions):
        print("❌ POSITIONS ARE ON MAIN WALLET - This is the problem!")
    else:
        print("ℹ️  No new positions detected - order may be resting or cancelled")


if __name__ == "__main__":
    main()