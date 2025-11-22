#!/usr/bin/env python3
"""
Script to run the volume generator with $1000 trades.
This ensures the environment variables are properly loaded.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any
import time

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from slipstream.strategies.volume_generator import VolumeBotConfig
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


class ImprovedVolumeGeneratorBot:
    """Improved bot that properly tracks fills to ensure flat positions."""

    def __init__(self, config: VolumeBotConfig):
        self.config = config
        self.info = None
        self.exchange = None
        self.wallet = None
        self._initialized = False
        # Track positions based on fills, not API position queries
        self.net_position = 0.0
        self.active_orders = {}  # Track order IDs and their intended sizes

    async def initialize(self):
        """Initialize Hyperliquid clients and validate credentials."""
        api_secret = self.config.api_secret
        api_key = self.config.api_key
        main_wallet = self.config.main_wallet

        if self.config.dry_run:
            print("⚠️  DRY RUN MODE: No real orders will be placed")
            # In dry run mode, we still need to initialize the info client
            self.info = Info(self.config.base_url)
            self._initialized = True
            return

        if not all([api_secret, api_key, main_wallet]):
            missing = []
            if not api_secret: missing.append("api_secret")
            if not api_key: missing.append("api_key")
            if not main_wallet: missing.append("main_wallet")
            raise ValueError(f"Missing required credentials: {', '.join(missing)}")

        self.wallet = Account.from_key(api_secret)
        if self.wallet.address.lower() != api_key.lower():
            raise ValueError("API_KEY does not match wallet derived from secret")

        self.info = Info(self.config.base_url)

        # Get metadata for exchange initialization
        response = self.info.meta_and_asset_ctxs()
        meta = response[0] if isinstance(response, list) and len(response) >= 2 else response

        self.exchange = Exchange(
            wallet=self.wallet,
            base_url=self.config.base_url,
            meta=meta,
            account_address=self.wallet.address,
        )

        self._initialized = True
        print(f"✓ Improved Volume Generator Bot initialized for wallet: {main_wallet[:8]}...")

    async def get_current_price(self) -> float:
        """Get current BTC price from Hyperliquid."""
        if not self.info:
            raise RuntimeError("Bot not initialized")

        all_mids = self.info.all_mids()
        price_str = all_mids.get(self.config.symbol)
        if not price_str:
            raise ValueError(f"Could not get price for {self.config.symbol}")

        return float(price_str)

    async def place_order(self, is_buy: bool, size_coin: float, price: float) -> Dict[str, Any]:
        """Place an IOC order that either fills immediately or gets cancelled."""
        if self.config.dry_run:
            # Simulate a successful order response in dry run mode
            print(f"  [DRY RUN] Would place IOC {'BUY' if is_buy else 'SELL'} order: {size_coin} BTC @ ${price}")
            # Simulated response structure - if it would have filled
            import random
            if random.random() > 0.5:  # Simulate partial fill sometimes
                filled_size = size_coin * random.uniform(0.5, 1.0)
                return {
                    "response": {
                        "data": {
                            "statuses": [
                                {
                                    "filled": {
                                        "oid": str(int(time.time() * 1000000)),  # Simulate an order ID
                                        "status": "filled",
                                        "totalSz": str(filled_size),
                                        "avgPx": str(price)
                                    }
                                }
                            ]
                        }
                    }
                }
            else:  # Simulate no fill
                return {
                    "response": {
                        "data": {
                            "statuses": [
                                {
                                    "resting": {
                                        "oid": str(int(time.time() * 1000000)),  # Simulate an order ID
                                        "status": "resting"
                                    }
                                }
                            ]
                        }
                    }
                }

        if not self.exchange:
            raise RuntimeError("Bot not initialized")

        print(f"  Placing IOC {'BUY' if is_buy else 'SELL'} order: {size_coin} BTC @ ${price}")

        # Place IOC (Immediate or Cancel) order - either fills immediately or gets cancelled
        response = self.exchange.order(
            name=self.config.symbol,
            is_buy=is_buy,
            sz=size_coin,
            limit_px=price,
            order_type={"limit": {"tif": "Ioc"}},  # Immediate or Cancel
            reduce_only=False,
        )

        return response

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order."""
        if self.config.dry_run:
            print(f"  [DRY RUN] Would cancel order: {order_id}")
            return {"status": "ok"}

        if not self.exchange:
            raise RuntimeError("Bot not initialized")

        print(f"  Cancelling order: {order_id}")

        try:
            # Convert order ID to integer for cancellation
            oid_int = int(float(order_id))
            response = self.exchange.cancel(self.config.symbol, oid_int)
            return response
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid order ID for cancellation: {order_id}") from e

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
                    elif "resting" in first_status:
                        # Order is resting (not filled)
                        return 0.0
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

    def extract_order_id(self, response: Dict[str, Any]) -> str:
        """Extract order ID from the response."""
        if isinstance(response, dict):
            resp_data = response.get("response", {})
            if isinstance(resp_data, dict):
                data = resp_data.get("data", {})
                statuses = data.get("statuses", [])
                if statuses and isinstance(statuses, list):
                    for status in statuses:
                        if "filled" in status:
                            return str(status["filled"].get("oid", ""))
                        elif "resting" in status:
                            return str(status["resting"].get("oid", ""))
        return ""

    async def run_volume_generation(self):
        """Run the volume generation with proper position tracking."""
        if not self._initialized:
            await self.initialize()

        print(f"Starting volume generation: {self.config.trade_count} in-and-out trades")
        print(f"Symbol: {self.config.symbol}, Size per trade: ${self.config.trade_size_usd}")

        current_price = await self.get_current_price()
        size_coin = self.config.trade_size_usd / current_price
        size_coin = round(size_coin, 5)  # Use appropriate decimal precision

        # Ensure minimum trade size (10 USD minimum as mentioned)
        min_size_usd = 10.0
        min_size_coin = min_size_usd / current_price
        if size_coin < min_size_coin:
            size_coin = min_size_coin
            print(f"Adjusted trade size to minimum: {size_coin} BTC (approx ${size_coin * current_price:.2f})")

        print(f"Trade size: {size_coin} BTC (approx ${size_coin * current_price:.2f})")

        # Ensure we start with a flat position
        initial_position = 0.0  # We track position through fills
        print(f"Starting with net position: {initial_position} BTC")

        completed_pairs = 0
        total_volume = 0.0

        for i in range(self.config.trade_count):
            print(f"\n--- Trade {i+1}/{self.config.trade_count} ---")

            # Get current market prices
            best_bid, best_ask = await self.get_best_bid_ask()
            print(f"Market: BID ${best_bid}, ASK ${best_ask}")

            # Enter position with IOC order at the current best bid to get filled or cancelled immediately
            print("Entering position with IOC buy order at best bid...")
            entry_price = best_bid  # Place at bid - will fill if there's a market order against it
            entry_response = await self.place_order(is_buy=True, size_coin=size_coin, price=entry_price)

            # Get the actual filled size for entry
            entry_filled = self.extract_filled_size(entry_response)
            avg_entry_price = self.extract_avg_price(entry_response)
            if avg_entry_price == 0:
                avg_entry_price = entry_price

            if entry_filled > 0:
                self.net_position += entry_filled  # Track position based on fills
                total_volume += entry_filled * avg_entry_price
                print(f"  ✓ Entry order filled: {entry_filled} BTC @ ${avg_entry_price}")

                # Exit the position by placing an IOC sell order at the current best bid for the exact filled amount
                exit_price = best_bid  # Place at bid - will fill if there's a market order against it
                print(f"Exiting position with IOC sell order: SELL {entry_filled} BTC @ ${exit_price}")
                exit_response = await self.place_order(is_buy=False, size_coin=entry_filled, price=exit_price)

                # Get the actual filled size for exit
                exit_filled = self.extract_filled_size(exit_response)
                avg_exit_price = self.extract_avg_price(exit_response)
                if avg_exit_price == 0:
                    avg_exit_price = exit_price

                if exit_filled > 0:
                    self.net_position -= exit_filled  # Track position based on fills
                    total_volume += exit_filled * avg_exit_price
                    print(f"  ✓ Exit order filled: {exit_filled} BTC @ ${avg_exit_price}")

                    if abs(exit_filled - entry_filled) < 0.00001:  # Fully matched
                        completed_pairs += 1
                        print(f"  ✓ Complete in-and-out pair: {entry_filled} BTC")
                    else:
                        print(f"  ⚠️  Partial fill mismatch: entry={entry_filled}, exit={exit_filled}")
                else:
                    print(f"  ⚠️  Exit order not filled, current net position: {self.net_position} BTC")
            else:
                print(f"  ! Buy order not filled, trying to enter with a short position...")

                # Try to enter by placing an IOC short order at the ask
                print("Entering short position with IOC sell order at best ask...")
                entry_price = best_ask  # Place at ask - will fill if there's a market order against it
                entry_response = await self.place_order(is_buy=False, size_coin=size_coin, price=entry_price)

                # Get the actual filled size for short entry
                entry_filled = self.extract_filled_size(entry_response)
                avg_entry_price = self.extract_avg_price(entry_response)
                if avg_entry_price == 0:
                    avg_entry_price = entry_price

                if entry_filled > 0:
                    self.net_position -= entry_filled  # Negative means short position
                    total_volume += entry_filled * avg_entry_price
                    print(f"  ✓ Short entry order filled: {entry_filled} BTC @ ${avg_entry_price}")

                    # Cover the short by placing an IOC buy order at the current best ask for the exact filled amount
                    exit_price = best_ask  # Place at ask - will fill if there's a market order against it
                    print(f"Covering short with IOC buy order: BUY {entry_filled} BTC @ ${exit_price}")
                    exit_response = await self.place_order(is_buy=True, size_coin=entry_filled, price=exit_price)

                    # Get the actual filled size for short cover
                    exit_filled = self.extract_filled_size(exit_response)
                    avg_exit_price = self.extract_avg_price(exit_response)
                    if avg_exit_price == 0:
                        avg_exit_price = exit_price

                    if exit_filled > 0:
                        self.net_position += exit_filled  # Reduce short position
                        total_volume += exit_filled * avg_exit_price
                        print(f"  ✓ Cover order filled: {exit_filled} BTC @ ${avg_exit_price}")

                        if abs(exit_filled - entry_filled) < 0.00001:  # Fully matched
                            completed_pairs += 1
                            print(f"  ✓ Complete short in-and-out pair: {entry_filled} BTC")
                        else:
                            print(f"  ⚠️  Partial fill mismatch: short={entry_filled}, cover={exit_filled}")
                    else:
                        print(f"  ⚠️  Cover order not filled, current net position: {self.net_position} BTC")
                else:
                    print(f"  ! No orders filled in this cycle, net position remains: {self.net_position} BTC")

            # Wait before next trade pair
            if i < self.config.trade_count - 1:
                print(f"Waiting {self.config.delay_between_trades}s before next trade pair...")
                await asyncio.sleep(self.config.delay_between_trades)

        print(f"\n--- Volume Generation Complete ---")
        print(f"Completed {completed_pairs} in-and-out pairs out of {self.config.trade_count} attempts")
        print(f"Total volume generated: ${total_volume:,.2f}")
        print(f"Final net position tracked: {self.net_position} BTC")

        # Check final position
        if abs(self.net_position) > 0.001:  # Small tolerance for floating point
            print(f"⚠️  Warning: Final position is not flat: {self.net_position} BTC")
            print("  This may require manual position closure!")
        else:
            print("✓ Position is flat as expected")

    async def get_best_bid_ask(self) -> tuple[float, float]:
        """Get best bid and ask prices for the symbol."""
        if not self.info:
            raise RuntimeError("Bot not initialized")

        # Get L2 book data using the correct method
        book_data = self.info.l2_snapshot(self.config.symbol)
        if not book_data or 'levels' not in book_data:
            current_price = await self.get_current_price()
            # If book data unavailable, estimate bid/ask based on current price
            spread = current_price * 0.001  # 10 bps spread estimate
            return current_price - spread/2, current_price + spread/2

        levels = book_data['levels']
        if len(levels) < 2 or len(levels[0]) == 0 or len(levels[1]) == 0:
            current_price = await self.get_current_price()
            spread = current_price * 0.001
            return current_price - spread/2, current_price + spread/2

        # [0] is bids, [1] is asks
        best_bid = float(levels[0][0]['px']) if levels[0] else None
        best_ask = float(levels[1][0]['px']) if levels[1] else None

        if not best_bid or not best_ask:
            current_price = await self.get_current_price()
            spread = current_price * 0.001
            return current_price - spread/2, current_price + spread/2

        return best_bid, best_ask


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

    print("\nConfiguring volume generator with $1000 trades...")

    # Use the config loader to properly load environment variables
    from slipstream.strategies.volume_generator.config import load_volume_generator_config

    # First, create a config object using the loader which will pick up environment variables
    config = load_volume_generator_config(path=None)  # Load defaults which should pick up env vars

    # Then override the specific parameters we want
    config.trade_count = 10  # 10 in-and-out trades for volume generation
    config.trade_size_usd = 1000.0  # $1000 per trade
    config.delay_between_trades = 2.0  # 2 seconds between trades for faster volume generation
    config.slippage_tolerance_bps = 100  # 1% slippage tolerance
    config.symbol = "BTC"  # Trading BTC
    config.dry_run = False  # Actually place real orders

    print(f"Trade configuration: {config.trade_count} trade(s) of ${config.trade_size_usd} USD each")
    print(f"Delay between trades: {config.delay_between_trades}s")
    print(f"Dry run mode: {config.dry_run}")

    # Debug print the config values
    print(f"Config API key: {config.api_key[:8] if config.api_key else 'None'}...")
    print(f"Config API secret: {config.api_secret[-4] if config.api_secret else 'None'}...")  # Last 4 chars
    print(f"Config wallet: {config.main_wallet[:8] if config.main_wallet else 'None'}...")

    bot = ImprovedVolumeGeneratorBot(config)

    async def run():
        try:
            print("\nInitializing bot...")
            await bot.initialize()

            print("\nStarting volume generation...")
            await bot.run_volume_generation()

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