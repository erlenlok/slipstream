#!/usr/bin/env python3
"""
Test script to place a small BTC trade and then read positions.
This helps diagnose authentication issues between write and read operations.
"""

import os
import sys
import time
from decimal import Decimal

# Add src to path
sys.path.insert(0, "/root/slipstream/src")

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account


def main():
    print("=" * 70)
    print("Test: Place BTC trade → Read position")
    print("=" * 70)

    # Get credentials
    api_secret = os.getenv("HYPERLIQUID_API_SECRET")
    api_key = os.getenv("HYPERLIQUID_API_KEY")

    if not api_secret or not api_key:
        print("ERROR: HYPERLIQUID_API_SECRET and HYPERLIQUID_API_KEY must be set")
        sys.exit(1)

    # Derive wallet from secret
    wallet = Account.from_key(api_secret)
    print(f"Wallet address: {wallet.address}")
    print(f"API_KEY matches: {wallet.address.lower() == api_key.lower()}")

    if wallet.address.lower() != api_key.lower():
        print("ERROR: API_KEY does not match wallet address from secret!")
        sys.exit(1)

    print()

    # Initialize clients
    print("Initializing Hyperliquid clients...")
    base_url = constants.MAINNET_API_URL
    info = Info(base_url)

    # Get meta for Exchange client
    response = info.meta_and_asset_ctxs()
    if isinstance(response, list) and len(response) >= 2:
        meta = response[0]
    else:
        meta = response

    exchange = Exchange(
        wallet=wallet,
        base_url=base_url,
        meta=meta,
        account_address=wallet.address,
    )
    print("✓ Clients initialized")
    print()

    # Step 1: Check current positions BEFORE trade
    print("Step 1: Fetching current positions (before trade)...")
    try:
        user_state = info.user_state(api_key)
        positions_before = user_state.get("assetPositions", [])
        print(f"✓ Current positions: {len(positions_before)}")

        # Check if we already have a BTC position
        btc_pos = None
        for pos_data in positions_before:
            pos = pos_data.get("position", {})
            if pos.get("coin") == "BTC":
                btc_pos = pos
                size = float(pos.get("szi", 0))
                value = float(pos.get("positionValue", 0))
                print(f"  WARNING: Already have BTC position: {size} BTC (${value:.2f})")
                break

        if btc_pos is None:
            print("  No existing BTC position")
    except Exception as e:
        print(f"✗ ERROR fetching positions: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()

    # Step 2: Get BTC price
    print("Step 2: Getting BTC price...")
    try:
        all_mids = info.all_mids()
        print(f"  Debug: all_mids is dict with {len(all_mids)} coins")

        if "BTC" in all_mids:
            btc_price = float(all_mids["BTC"])
        else:
            print(f"  BTC not in all_mids. Available: {list(all_mids.keys())[:10]}")
            sys.exit(1)

        print(f"✓ BTC price: ${btc_price:,.2f}")
    except Exception as e:
        print(f"✗ ERROR getting price: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()

    # Step 3: Place a small BTC long position ($20 worth)
    trade_size_usd = 20.0
    size_btc = trade_size_usd / btc_price
    size_btc = round(size_btc, 5)  # BTC has 5 decimals on Hyperliquid

    print(f"Step 3: Placing BTC long order (${trade_size_usd} = {size_btc} BTC)...")
    print(f"  Direction: BUY (long)")
    print(f"  Size: {size_btc} BTC")
    print(f"  Type: Market order")

    try:
        response = exchange.market_open(
            name="BTC",  # Use 'name' not 'coin'
            is_buy=True,
            sz=size_btc,
            px=btc_price * 1.01,  # Slippage tolerance
        )
        print(f"✓ Order placed!")
        print(f"  Response: {response}")
    except Exception as e:
        print(f"✗ ERROR placing order: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()

    # Step 4: Wait a moment for order to fill
    print("Step 4: Waiting 3 seconds for order to fill...")
    time.sleep(3)
    print()

    # Step 5: Read positions AFTER trade
    print("Step 5: Fetching current positions (after trade)...")
    try:
        user_state = info.user_state(api_key)
        positions_after = user_state.get("assetPositions", [])
        print(f"✓ Fetched user_state")
        print(f"  Total asset positions: {len(positions_after)}")

        # Debug: show margin summary
        margin = user_state.get("marginSummary", {})
        print(f"  Account value: ${margin.get('accountValue', 'N/A')}")
        print(f"  Total notional: ${margin.get('totalNtlPos', 'N/A')}")
        print(f"  Total margin used: ${margin.get('totalMarginUsed', 'N/A')}")

        # Look for BTC position
        btc_found = False
        for pos_data in positions_after:
            pos = pos_data.get("position", {})
            coin = pos.get("coin")
            size = float(pos.get("szi", 0))
            value = float(pos.get("positionValue", 0))
            entry_px = float(pos.get("entryPx", 0))

            print(f"  Position found: {coin}, size={size}, value=${value:.2f}")

            if coin == "BTC":
                btc_found = True
                print(f"  ✓ BTC Position:")
                print(f"    Size: {size} BTC")
                print(f"    Value: ${value:.2f}")
                print(f"    Entry: ${entry_px:,.2f}")

        if not btc_found:
            print("  ✗ WARNING: No BTC position found in assetPositions!")
            print(f"  Debug: Full user_state keys: {list(user_state.keys())}")
    except Exception as e:
        print(f"✗ ERROR fetching positions after trade: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("=" * 70)
    print("Test complete!")
    print()
    print("To close the BTC position, run:")
    print("  python test_trade_and_read.py --close")
    print("=" * 70)


def close_btc_position():
    """Close any open BTC position."""
    print("=" * 70)
    print("Closing BTC position...")
    print("=" * 70)

    api_secret = os.getenv("HYPERLIQUID_API_SECRET")
    api_key = os.getenv("HYPERLIQUID_API_KEY")

    wallet = Account.from_key(api_secret)
    base_url = constants.MAINNET_API_URL
    info = Info(base_url)

    # Get meta
    response = info.meta_and_asset_ctxs()
    if isinstance(response, list) and len(response) >= 2:
        meta = response[0]
    else:
        meta = response

    exchange = Exchange(
        wallet=wallet,
        base_url=base_url,
        meta=meta,
        account_address=wallet.address,
    )

    # Get current BTC position
    user_state = info.user_state(api_key)
    positions = user_state.get("assetPositions", [])

    btc_size = None
    for pos_data in positions:
        pos = pos_data.get("position", {})
        if pos.get("coin") == "BTC":
            btc_size = float(pos.get("szi", 0))
            break

    if btc_size is None or abs(btc_size) < 1e-6:
        print("No BTC position to close")
        return

    print(f"Found BTC position: {btc_size} BTC")

    # Close position (opposite side)
    is_buy = btc_size < 0  # If short, buy to close. If long, sell to close
    size = abs(btc_size)

    print(f"Placing closing order: {'BUY' if is_buy else 'SELL'} {size} BTC")

    # Get current price for slippage
    all_mids = info.all_mids()
    btc_price = float(all_mids["BTC"])
    slippage_px = btc_price * (1.01 if is_buy else 0.99)

    response = exchange.market_close(
        name="BTC",
        sz=size,
        px=slippage_px,
    )

    print(f"✓ Closing order placed: {response}")
    print("Position should be closed in a few seconds")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--close":
        close_btc_position()
    else:
        main()
