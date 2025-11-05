#!/usr/bin/env python3
"""
Full end-to-end test: Place trade and fetch positions from correct wallet.

This test demonstrates the fix for the wallet confusion issue:
- HYPERLIQUID_API_KEY = API vault (0x998c...) - used for signing trades
- HYPERLIQUID_MAIN_WALLET = Main wallet (0xFd5c...) - where positions live
"""

import os
import sys
import time

sys.path.insert(0, "/root/slipstream/src")

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account
from slipstream.strategies.gradient.live.config import load_config
from slipstream.strategies.gradient.live.execution import get_current_positions


def main():
    print("=" * 80)
    print("FULL WORKFLOW TEST: Trade Placement + Position Fetching")
    print("=" * 80)
    print()

    # Verify environment variables
    api_secret = os.getenv("HYPERLIQUID_API_SECRET")
    api_key = os.getenv("HYPERLIQUID_API_KEY")  # API vault address
    main_wallet = os.getenv("HYPERLIQUID_MAIN_WALLET")  # Main wallet address

    if not all([api_secret, api_key, main_wallet]):
        print("ERROR: Missing required environment variables:")
        print(f"  HYPERLIQUID_API_SECRET: {'✓' if api_secret else '✗'}")
        print(f"  HYPERLIQUID_API_KEY: {'✓' if api_key else '✗'}")
        print(f"  HYPERLIQUID_MAIN_WALLET: {'✓' if main_wallet else '✗'}")
        sys.exit(1)

    print("Environment Variables:")
    print(f"  API Vault (signs trades): {api_key}")
    print(f"  Main Wallet (holds positions): {main_wallet}")
    print()

    # Verify API key matches derived wallet
    wallet = Account.from_key(api_secret)
    if wallet.address.lower() != api_key.lower():
        print(f"ERROR: API_KEY ({api_key}) doesn't match derived wallet ({wallet.address})")
        sys.exit(1)
    print("✓ API key verification passed")
    print()

    # Step 1: Check positions BEFORE trade
    print("Step 1: Fetching current positions (BEFORE trade)...")
    config = load_config()
    positions_before = get_current_positions(config)
    print(f"  Positions: {len(positions_before)}")
    if positions_before:
        for asset, value in list(positions_before.items())[:3]:
            direction = "LONG" if value > 0 else "SHORT"
            print(f"    {asset}: {direction} ${abs(value):.2f}")
    print()

    # Step 2: Initialize Hyperliquid clients
    print("Step 2: Initializing Hyperliquid clients...")
    base_url = constants.MAINNET_API_URL
    info = Info(base_url)

    # Get meta for Exchange
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
    print("  ✓ Clients initialized")
    print()

    # Step 3: Get BTC price
    print("Step 3: Getting BTC price...")
    all_mids = info.all_mids()
    btc_price = float(all_mids["BTC"])
    print(f"  BTC price: ${btc_price:,.2f}")
    print()

    # Step 4: Place a $20 BTC long order
    trade_size_usd = 20.0
    size_btc = trade_size_usd / btc_price
    size_btc = round(size_btc, 5)  # BTC has 5 decimals

    print(f"Step 4: Placing BTC long order...")
    print(f"  Trade size: ${trade_size_usd} = {size_btc} BTC")
    print(f"  Direction: BUY (long)")
    print(f"  Order type: Market")

    try:
        response = exchange.market_open(
            name="BTC",
            is_buy=True,
            sz=size_btc,
            px=btc_price * 1.01,  # Slippage tolerance
        )

        # Check if order was filled
        if isinstance(response, dict):
            resp_data = response.get("response", {})
            if isinstance(resp_data, dict):
                data = resp_data.get("data", {})
                statuses = data.get("statuses", [])
                if statuses and isinstance(statuses, list):
                    first_status = statuses[0]
                    if "filled" in first_status:
                        filled_info = first_status["filled"]
                        print(f"  ✓ Order FILLED!")
                        print(f"    Size: {filled_info.get('totalSz')} BTC")
                        print(f"    Avg Price: ${filled_info.get('avgPx')}")
                        print(f"    Order ID: {filled_info.get('oid')}")
                    elif "error" in first_status:
                        print(f"  ✗ Order ERROR: {first_status['error']}")
                        sys.exit(1)

        print(f"  Response: {response}")
    except Exception as e:
        print(f"  ✗ ERROR placing order: {e}")
        sys.exit(1)

    print()

    # Step 5: Wait for position to settle
    print("Step 5: Waiting 3 seconds for position to settle...")
    time.sleep(3)
    print()

    # Step 6: Fetch positions AFTER trade (from MAIN wallet)
    print("Step 6: Fetching current positions (AFTER trade from MAIN wallet)...")
    positions_after = get_current_positions(config)
    print(f"  Total positions: {len(positions_after)}")
    print()

    if positions_after:
        print("  Current positions:")
        for asset, value in positions_after.items():
            direction = "LONG" if value > 0 else "SHORT"
            print(f"    {asset}: {direction} ${abs(value):.2f}")
        print()

        # Check if BTC position exists
        if "BTC" in positions_after:
            btc_value = positions_after["BTC"]
            print(f"  ✓ SUCCESS! BTC position found: ${abs(btc_value):.2f}")
            print()
            print("  🎉 Full workflow test PASSED!")
            print("     - Trade placement: ✓")
            print("     - Position fetching from main wallet: ✓")
        else:
            print("  ⚠️  BTC position not found immediately (may take a moment)")
    else:
        print("  ⚠️  No positions found (position may take a moment to appear)")

    print()
    print("=" * 80)
    print("Test complete!")
    print()
    print("To close the BTC position:")
    print("  python test_full_workflow.py --close")
    print("=" * 80)


def close_position():
    """Close BTC position for cleanup."""
    print("=" * 80)
    print("Closing BTC position...")
    print("=" * 80)
    print()

    api_secret = os.getenv("HYPERLIQUID_API_SECRET")
    api_key = os.getenv("HYPERLIQUID_API_KEY")
    main_wallet = os.getenv("HYPERLIQUID_MAIN_WALLET")

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

    # Get BTC position from main wallet
    import requests
    body = {"type": "clearinghouseState", "user": main_wallet}
    resp = requests.post(f"{base_url}/info", json=body, timeout=10)
    resp.raise_for_status()
    state = resp.json()

    asset_positions = state.get("assetPositions", [])

    btc_size = None
    for ap in asset_positions:
        pos = ap.get("position", {})
        if pos.get("coin") == "BTC":
            btc_size = float(pos.get("szi", 0))
            break

    if btc_size is None or abs(btc_size) < 1e-6:
        print("No BTC position to close")
        return

    print(f"Found BTC position: {btc_size} BTC")

    # Close position
    is_buy = btc_size < 0  # If short, buy to close
    size = abs(btc_size)

    print(f"Placing closing order: {'BUY' if is_buy else 'SELL'} {size} BTC")

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
        close_position()
    else:
        main()
