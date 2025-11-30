#!/usr/bin/env python3
"""
Script to generate $50k volume in max $2k clips using IOC orders.
This ensures no lingering positions by using IOC (Immediate-or-Cancel) orders.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any
import time

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from eth_account import Account


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


class VolumeGenerator:
    """Volume generator that creates volume with IOC orders to avoid lingering positions."""
    
    def __init__(self, api_key, api_secret, main_wallet, base_url="https://api.hyperliquid.xyz"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.main_wallet = main_wallet
        self.base_url = base_url
        self.info = None
        self.exchange = None
        self.wallet = None
        self._initialized = False

    async def initialize(self):
        """Initialize Hyperliquid clients."""
        if not all([self.api_secret, self.api_key, self.main_wallet]):
            missing = []
            if not self.api_secret: missing.append("api_secret")
            if not self.api_key: missing.append("api_key")
            if not self.main_wallet: missing.append("main_wallet")
            raise ValueError(f"Missing required credentials: {', '.join(missing)}")

        self.wallet = Account.from_key(self.api_secret)
        if self.wallet.address.lower() != self.api_key.lower():
            raise ValueError("API_KEY does not match wallet derived from secret")

        self.info = Info(self.base_url)

        # Get metadata for exchange initialization
        response = self.info.meta_and_asset_ctxs()
        meta = response[0] if isinstance(response, list) and len(response) >= 2 else response

        self.exchange = Exchange(
            wallet=self.wallet,
            base_url=self.base_url,
            meta=meta,
            account_address=self.wallet.address,
        )

        self._initialized = True
        print(f"✓ Volume Generator initialized for wallet: {self.main_wallet[:8]}...")

    async def get_current_price(self) -> float:
        """Get current BTC price from Hyperliquid."""
        if not self.info:
            raise RuntimeError("Bot not initialized")

        all_mids = self.info.all_mids()
        price_str = all_mids.get("BTC")
        if not price_str:
            raise ValueError("Could not get price for BTC")

        return float(price_str)

    async def place_ioc_order(self, is_buy: bool, size_coin: float, price: float) -> Dict[str, Any]:
        """Place an IOC order that either fills immediately or gets cancelled."""
        if not self.exchange:
            raise RuntimeError("Bot not initialized")

        print(f"  Placing IOC {'BUY' if is_buy else 'SELL'} order: {size_coin} BTC @ ${price}")

        # Place IOC (Immediate or Cancel) order - either fills immediately or gets cancelled
        response = self.exchange.order(
            name="BTC",
            is_buy=is_buy,
            sz=size_coin,
            limit_px=price,
            order_type={"limit": {"tif": "Ioc"}},  # Immediate or Cancel
            reduce_only=False,
        )

        return response

    def extract_filled_size(self, response: Dict[str, Any]) -> float:
        """Extract the filled size from an order response."""
        if isinstance(response, dict):
            resp_data = response.get("response", {})
            if isinstance(resp_data, dict):
                data = resp_data.get("data", {})
                statuses = data.get("statuses", [])
                if statuses and isinstance(statuses, list):
                    first_status = statuses[0]
                    if "filled" in first_status:
                        filled_info = first_status["filled"]
                        size_str = filled_info.get('totalSz')
                        return float(size_str) if size_str else 0.0
        return 0.0

    def extract_avg_price(self, response: Dict[str, Any]) -> float:
        """Extract the average fill price from a response."""
        if isinstance(response, dict):
            resp_data = response.get("response", {})
            if isinstance(resp_data, dict):
                data = resp_data.get("data", {})
                statuses = data.get("statuses", [])
                if statuses and isinstance(statuses, list):
                    first_status = statuses[0]
                    if "filled" in first_status:
                        filled_info = first_status["filled"]
                        avg_px = filled_info.get('avgPx')
                        return float(avg_px) if avg_px else 0.0
        return 0.0

    async def get_best_bid_ask(self) -> tuple[float, float]:
        """Get best bid and ask prices for BTC."""
        if not self.info:
            raise RuntimeError("Bot not initialized")

        # Get L2 book data
        book_data = self.info.l2_snapshot("BTC")
        if not book_data or 'levels' not in book_data:
            current_price = await self.get_current_price()
            # If book data unavailable, estimate bid/ask based on current price
            spread = current_price * 0.001  # 10 bps spread estimate
            best_bid = current_price - spread/2
            best_ask = current_price + spread/2
            return best_bid, best_ask

        levels = book_data['levels']
        if len(levels) < 2 or len(levels[0]) == 0 or len(levels[1]) == 0:
            current_price = await self.get_current_price()
            spread = current_price * 0.001
            best_bid = current_price - spread/2
            best_ask = current_price + spread/2
            return best_bid, best_ask

        # [0] is bids, [1] is asks
        best_bid = float(levels[0][0]['px']) if levels[0] else 0
        best_ask = float(levels[1][0]['px']) if levels[1] else 0

        if not best_bid or not best_ask:
            current_price = await self.get_current_price()
            spread = current_price * 0.001
            best_bid = current_price - spread/2
            best_ask = current_price + spread/2

        return best_bid, best_ask

    async def generate_volume(self, target_volume_usd: float = 50000.0, max_clip_usd: float = 2000.0):
        """Generate volume with IOC orders to avoid lingering positions."""
        if not self._initialized:
            await self.initialize()

        print(f"Starting volume generation: ${target_volume_usd:,.2f} total using max ${max_clip_usd} clips")
        
        current_price = await self.get_current_price()
        max_clip_coin = max_clip_usd / current_price
        max_clip_coin = round(max_clip_coin, 5)

        print(f"Max clip size: {max_clip_coin} BTC (approx ${max_clip_usd})")
        print(f"Current price: ${current_price}")

        generated_volume = 0.0
        trade_count = 0
        total_volume = 0.0

        while generated_volume < target_volume_usd:
            trade_count += 1
            print(f"\n--- Volume Clip {trade_count} ---")
            print(f"Target: ${target_volume_usd:,.2f}, Generated: ${generated_volume:,.2f}, Remaining: ${target_volume_usd - generated_volume:,.2f}")
            
            # Get current market prices
            best_bid, best_ask = await self.get_best_bid_ask()
            print(f"Market: BID ${best_bid}, ASK ${best_ask}")

            # First, try to buy at the bid (provide liquidity to get filled)
            buy_response = await self.place_ioc_order(is_buy=True, size_coin=max_clip_coin, price=best_bid)
            buy_filled = self.extract_filled_size(buy_response)
            buy_price = self.extract_avg_price(buy_response)
            
            if buy_filled > 0:
                print(f"  ✓ Buy order filled: {buy_filled} BTC @ ${buy_price}")
                total_volume += buy_filled * buy_price
                
                # Exit by selling at the bid (to avoid creating position)
                sell_response = await self.place_ioc_order(is_buy=False, size_coin=buy_filled, price=best_bid)
                sell_filled = self.extract_filled_size(sell_response)
                sell_price = self.extract_avg_price(sell_response)
                
                if sell_filled > 0:
                    print(f"  ✓ Sell order filled: {sell_filled} BTC @ ${sell_price}")
                    total_volume += sell_filled * sell_price
                else:
                    print(f"  ! Sell order not filled")
            else:
                print(f"  ! Buy order not filled, trying to sell at ask...")

                # If buy didn't fill, try to sell at the ask
                sell_response = await self.place_ioc_order(is_buy=False, size_coin=max_clip_coin, price=best_ask)
                sell_filled = self.extract_filled_size(sell_response)
                sell_price = self.extract_avg_price(sell_response)
                
                if sell_filled > 0:
                    print(f"  ✓ Sell order filled: {sell_filled} BTC @ ${sell_price}")
                    total_volume += sell_filled * sell_price
                    
                    # Cover by buying at the ask
                    cover_response = await self.place_ioc_order(is_buy=True, size_coin=sell_filled, price=best_ask)
                    cover_filled = self.extract_filled_size(cover_response)
                    cover_price = self.extract_avg_price(cover_response)
                    
                    if cover_filled > 0:
                        print(f"  ✓ Cover order filled: {cover_filled} BTC @ ${cover_price}")
                        total_volume += cover_filled * cover_price
                    else:
                        print(f"  ! Cover order not filled")
                else:
                    print(f"  ! Sell order not filled, no volume generated in this cycle")

            # Update the volume generated
            generated_volume = total_volume

            # Small delay between clips
            await asyncio.sleep(1.0)

        print(f"\n--- Volume Generation Complete ---")
        print(f"Target volume: ${target_volume_usd:,.2f}")
        print(f"Actual volume: ${total_volume:,.2f}")
        print(f"Clips executed: {trade_count}")
        print(f"Difference: ${target_volume_usd - total_volume:,.2f}")


def main():
    print("Loading environment variables...")
    load_env_vars()
    
    api_key = os.getenv('HYPERLIQUID_API_KEY')
    api_secret = os.getenv('HYPERLIQUID_API_SECRET')
    main_wallet = os.getenv('HYPERLIQUID_MAIN_WALLET')

    if not all([api_key, api_secret, main_wallet]):
        print("Error: Missing required environment variables:")
        print(f"  HYPERLIQUID_API_KEY: {'✓' if api_key else '✗'}")
        print(f"  HYPERLIQUID_API_SECRET: {'✓' if api_secret else '✗'}")
        print(f"  HYPERLIQUID_MAIN_WALLET: {'✓' if main_wallet else '✗'}")
        return

    print(f"Using wallet: {main_wallet[:8]}...")
    print(f"API key: {api_key[:8]}...")

    print("\nConfiguring volume generator...")
    
    # Create volume generator instance
    vg = VolumeGenerator(
        api_key=api_key,
        api_secret=api_secret,
        main_wallet=main_wallet
    )

    async def run():
        try:
            print("\nInitializing bot...")
            await vg.initialize()

            print("\nStarting volume generation...")
            await vg.generate_volume(target_volume_usd=50000.0, max_clip_usd=2000.0)

            print("\n✓ Volume generation completed successfully!")

        except KeyboardInterrupt:
            print("\nVolume generation interrupted by user")
        except Exception as e:
            print(f"✗ Error running volume generator: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(run())


if __name__ == "__main__":
    main()