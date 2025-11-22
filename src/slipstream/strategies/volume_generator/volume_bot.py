"""Volume Generator Bot: Makes in-and-out trades to generate volume."""

import asyncio
import os
import time
from typing import Dict, Optional, Any

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account

from .config import VolumeBotConfig




class VolumeGeneratorBot:
    """Bot that places market orders to generate volume by making 42 in-and-out trades."""
    
    def __init__(self, config: VolumeBotConfig):
        self.config = config
        self.info: Optional[Info] = None
        self.exchange: Optional[Exchange] = None
        self.wallet: Optional[Account] = None
        self._initialized = False
        
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
        print(f"✓ Volume Generator Bot initialized for wallet: {main_wallet[:8]}...")
    
    async def get_current_price(self) -> float:
        """Get current BTC price from Hyperliquid."""
        if not self.info:
            raise RuntimeError("Bot not initialized")
        
        all_mids = self.info.all_mids()
        price_str = all_mids.get(self.config.symbol)
        if not price_str:
            raise ValueError(f"Could not get price for {self.config.symbol}")
        
        return float(price_str)
    
    async def place_passive_order(self, is_buy: bool, size_coin: float, price: float) -> Dict[str, Any]:
        """Place a passive limit order on the top of the book and return the response."""
        if self.config.dry_run:
            # Simulate a successful order response in dry run mode
            print(f"  [DRY RUN] Would place passive {'BUY' if is_buy else 'SELL'} order: {size_coin} BTC @ ${price}")
            # Simulated response structure
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

        print(f"  Placing passive {'BUY' if is_buy else 'SELL'} order: {size_coin} BTC @ ${price}")

        # Place limit order using the exchange (passive order)
        response = self.exchange.order(
            name=self.config.symbol,
            is_buy=is_buy,
            sz=size_coin,
            limit_px=price,
            order_type={"limit": {"tif": "Gtc"}},  # Good till cancelled
            reduce_only=False,
        )

        return response

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order."""
        if self.config.dry_run:
            print(f"  [DRY RUN] Would cancel order: {order_id}")
            return {"status": "ok", "cancelled": True}

        if not self.exchange:
            raise RuntimeError("Bot not initialized")

        try:
            # Convert order ID to integer for cancellation
            oid_int = int(float(order_id))
            response = self.exchange.cancel(self.config.symbol, oid_int)
            return response
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid order ID for cancellation: {order_id}") from e
    
    async def get_current_position(self) -> float:
        """Get current BTC position size."""
        if self.config.dry_run:
            # In dry run mode, always return 0 position
            return 0.0

        if not self.info:
            raise RuntimeError("Bot not initialized")

        main_wallet = self.config.main_wallet or os.getenv("HYPERLIQUID_VOLUME_WALLET") or os.getenv("HYPERLIQUID_MAIN_WALLET")
        if not main_wallet:
            raise ValueError("Main wallet not configured")

        # Fetch user state to get positions
        body = {"type": "clearinghouseState", "user": main_wallet}
        import requests
        response = requests.post(f"{self.config.base_url}/info", json=body, timeout=10)
        response.raise_for_status()
        state = response.json()

        asset_positions = state.get("assetPositions", [])

        for ap in asset_positions:
            pos = ap.get("position", {})
            if pos.get("coin") == self.config.symbol:
                return float(pos.get("szi", 0))

        return 0.0
    
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

    async def run_volume_generation(self):
        """Run the volume generation by making 42 in-and-out trades with passive orders."""
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

        # Track completed in-and-out pairs
        completed_pairs = 0
        total_volume = 0.0

        for i in range(self.config.trade_count):
            print(f"\n--- Trade {i+1}/{self.config.trade_count} ---")

            # Get current market prices
            best_bid, best_ask = await self.get_best_bid_ask()
            print(f"Market: BID ${best_bid}, ASK ${best_ask}")

            # Enter position with a passive buy order at the current best bid
            print("Entering position with passive bid order on top of book...")
            buy_price = best_bid
            buy_response = await self.place_passive_order(is_buy=True, size_coin=size_coin, price=buy_price)
            buy_order_id = self._extract_order_id(buy_response)

            # Wait for a short time to see if the buy order gets filled
            await asyncio.sleep(1.0)

            # Check how much of the buy order was filled
            filled_size = self._extract_filled_size(buy_response)
            if filled_size == 0:
                # If not filled, check if it's still in the book and get position
                position_after_buy = await self.get_current_position()
                buy_filled = abs(position_after_buy) > 0.0001
                filled_size = position_after_buy if buy_filled else 0
            else:
                buy_filled = True

            if buy_filled and filled_size > 0:
                avg_price = self._extract_avg_price(buy_response)
                if avg_price is None:
                    avg_price = buy_price
                total_volume += filled_size * avg_price
                print(f"  ✓ Buy order filled: {filled_size} BTC @ ${avg_price}")

                # Now exit the position by placing a passive sell order with the actual filled size
                final_bid, final_ask = await self.get_best_bid_ask()
                print(f"Position filled, exiting with passive ask order: ASK ${final_ask}")
                sell_response = await self.place_passive_order(is_buy=False, size_coin=filled_size, price=final_ask)

                # Wait a bit for the sell order to fill
                await asyncio.sleep(1.0)

                # Check how much of the sell order was filled
                sell_filled_size = self._extract_filled_size(sell_response)
                if sell_filled_size == 0:
                    # If not filled, check position
                    position_after_sell = await self.get_current_position()
                    sell_filled = abs(position_after_sell) < 0.0001  # Close to zero means position closed
                else:
                    sell_filled = True

                if sell_filled:
                    avg_sell_price = self._extract_avg_price(sell_response)
                    if avg_sell_price is None:
                        avg_sell_price = final_ask
                    total_volume += sell_filled_size * avg_sell_price
                    print(f"  ✓ Sell order filled: {sell_filled_size} BTC @ ${avg_sell_price}")
                    completed_pairs += 1  # Completed a full in-and-out
                else:
                    print(f"  ! Sell order not filled, position remains: {await self.get_current_position()} BTC")
            else:
                # The buy order was not filled, so we try the other direction
                print(f"  ! Buy order not filled, trying to place passive ask order...")

                # Place a passive sell order at the best ask instead
                sell_price = best_ask
                sell_response = await self.place_passive_order(is_buy=False, size_coin=size_coin, price=sell_price)

                # Wait to see if sell order gets filled
                await asyncio.sleep(1.0)

                # Check how much of the sell order was filled
                short_filled_size = self._extract_filled_size(sell_response)
                if short_filled_size == 0:
                    position_after_sell = await self.get_current_position()
                    sell_filled = abs(position_after_sell) > 0.0001 and position_after_sell < 0  # Negative position means short
                    short_filled_size = abs(position_after_sell) if sell_filled else 0
                else:
                    sell_filled = True

                if sell_filled:
                    avg_price = self._extract_avg_price(sell_response)
                    if avg_price is None:
                        avg_price = sell_price
                    total_volume += short_filled_size * avg_price
                    print(f"  ✓ Sell order filled: {short_filled_size} BTC @ ${avg_price} (short position)")

                    # Now cover the short by placing a passive buy order with the actual filled size
                    final_bid, final_ask = await self.get_best_bid_ask()
                    print(f"Short position filled, covering with passive bid order: BID ${final_bid}")
                    cover_response = await self.place_passive_order(is_buy=True, size_coin=short_filled_size, price=final_bid)

                    # Wait a bit for the cover order to fill
                    await asyncio.sleep(1.0)

                    # Check if the cover order was filled
                    final_position = await self.get_current_position()
                    cover_filled = abs(final_position) < 0.0001

                    if cover_filled:
                        avg_cover_price = self._extract_avg_price(cover_response)
                        if avg_cover_price is None:
                            avg_cover_price = final_bid
                        total_volume += short_filled_size * avg_cover_price
                        print(f"  ✓ Cover order filled: {short_filled_size} BTC @ ${avg_cover_price}")
                        completed_pairs += 1  # Completed a full in-and-out
                    else:
                        print(f"  ! Cover order not filled, position remains: {final_position} BTC")
                else:
                    print(f"  ! Neither buy nor sell orders filled in this cycle")

            # Wait before next trade pair
            if i < self.config.trade_count - 1:
                print(f"Waiting {self.config.delay_between_trades}s before next trade pair...")
                await asyncio.sleep(self.config.delay_between_trades)

        print(f"\n--- Volume Generation Complete ---")
        print(f"Completed {completed_pairs} in-and-out pairs out of {self.config.trade_count} attempts")
        print(f"Total volume generated: ${total_volume:,.2f}")

        # Check final position
        final_position = await self.get_current_position()
        if abs(final_position) > 0.001:  # Small tolerance for floating point
            print(f"⚠️  Warning: Final position is not flat: {final_position} BTC")
        else:
            print("✓ Position is flat as expected")
    
    def _check_fill_status(self, response: Dict[str, Any], side: str) -> bool:
        """Check if an order was filled successfully."""
        if isinstance(response, dict):
            resp_data = response.get("response", {})
            if isinstance(resp_data, dict):
                data = resp_data.get("data", {})
                statuses = data.get("statuses", [])
                if statuses and isinstance(statuses, list):
                    first_status = statuses[0]
                    if "filled" in first_status:
                        filled_info = first_status["filled"]
                        print(f"    Order filled! Size: {filled_info.get('totalSz')} BTC, Avg Price: ${filled_info.get('avgPx')}")
                        return True
                    elif "error" in first_status:
                        print(f"    {side.title()} order error: {first_status['error']}")
                        return False
        print(f"    {side.title()} order response structure unexpected: {response}")
        return False
    
    def _extract_filled_size(self, response: Dict[str, Any]) -> float:
        """Extract the filled size from a response."""
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

    def _extract_avg_price(self, response: Dict[str, Any]) -> Optional[float]:
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
                        return float(avg_px) if avg_px else None
        return None

    def _extract_order_id(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract order ID from the response for potential cancellation."""
        if isinstance(response, dict):
            resp_data = response.get("response", {})
            if isinstance(resp_data, dict):
                data = resp_data.get("data", {})
                # For new orders, the order ID might be in different locations
                if "statuses" in data and isinstance(data["statuses"], list):
                    for status in data["statuses"]:
                        if "resting" in status:
                            resting_info = status["resting"]
                            oid = resting_info.get("oid")
                            if oid is not None:
                                return str(oid)
        return None


async def main():
    """Main function to run the volume generator."""
    config = VolumeBotConfig(
        trade_count=42,
        trade_size_usd=20.0,  # $20 per trade should be sufficient for volume
        delay_between_trades=1.5  # 1.5 seconds between trades
    )
    
    bot = VolumeGeneratorBot(config)
    try:
        await bot.run_volume_generation()
    except Exception as e:
        print(f"Error running volume generator: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())