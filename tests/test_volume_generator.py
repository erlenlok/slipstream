#!/usr/bin/env python3
"""
Test script for volume generator with tiny amounts.
This runs a single in-and-out trade with minimal size to verify functionality.
"""

import asyncio
import os
from slipstream.strategies.volume_generator import VolumeGeneratorBot, VolumeBotConfig


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


async def test_volume_generator():
    """Test the volume generator with tiny amounts."""

    print("Testing Volume Generator with tiny amounts...")
    print("This will perform 1 in-and-out trade with $2 to test functionality.")
    print()

    # Load environment variables
    load_env_vars()

    # Check if required environment variables are set
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

    # Configure with minimum amounts for testing
    config = VolumeBotConfig(
        trade_count=1,  # Just 1 trade for testing
        trade_size_usd=10.0,  # $10 per trade (minimum)
        delay_between_trades=2.0,
        slippage_tolerance_bps=200,  # 2% slippage tolerance
    )

    bot = VolumeGeneratorBot(config)

    try:
        print("Initializing bot...")
        await bot.initialize()

        current_price = await bot.get_current_price()
        print(f"Current BTC price: ${current_price}")

        print("\nRunning a single test in-and-out trade...")
        await bot.run_volume_generation()

        print("\n✓ Test completed successfully!")

    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_volume_generator())