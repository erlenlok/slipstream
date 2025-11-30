#!/usr/bin/env python3
"""Test script to verify the volume generator config loading works properly."""

import tempfile
import os
from pathlib import Path
from slipstream.strategies.volume_generator.config import load_volume_generator_config, VolumeBotConfig


def test_config_loading():
    """Test that config loading works from file and defaults."""
    
    print("Testing Volume Generator Config Loading...")
    
    # Test 1: Default config without file
    print("\n1. Testing default config (no file)...")
    default_config = load_volume_generator_config()
    print(f"   Default config: trade_count={default_config.trade_count}, trade_size_usd={default_config.trade_size_usd}")
    assert default_config.trade_count == 42
    assert default_config.trade_size_usd == 20.0
    print("   ✓ Default config loaded correctly")
    
    # Test 2: Create a temporary config file and test loading
    print("\n2. Testing config from file...")
    config_content = """
volume_generation:
  trade_count: 10
  trade_size_usd: 100.0
  delay_between_trades: 2.0
  symbol: "ETH"
  slippage_tolerance_bps: 50
  dry_run: true

hyperliquid:
  api_key: "test_key"
  api_secret: "test_secret"
  main_wallet: "test_wallet"
  base_url: "https://api.test.hyperliquid.xyz"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write(config_content)
        temp_config_path = f.name
    
    try:
        file_config = load_volume_generator_config(temp_config_path)
        print(f"   File config: trade_count={file_config.trade_count}, trade_size_usd={file_config.trade_size_usd}")
        print(f"   File config: symbol={file_config.symbol}, dry_run={file_config.dry_run}")
        assert file_config.trade_count == 10
        assert file_config.trade_size_usd == 100.0
        assert file_config.symbol == "ETH"
        assert file_config.dry_run == True
        assert file_config.api_key == "test_key"
        print("   ✓ Config from file loaded correctly")
    finally:
        os.unlink(temp_config_path)
    
    # Test 3: Test environment variable substitution
    print("\n3. Testing environment variable substitution...")
    os.environ['TEST_HYPERLIQUID_API_KEY'] = 'env_key_value'
    os.environ['TEST_HYPERLIQUID_SECRET'] = 'env_secret_value'
    
    env_config_content = """
volume_generation:
  trade_count: 5

hyperliquid:
  api_key: "${TEST_HYPERLIQUID_API_KEY}"
  api_secret: "${TEST_HYPERLIQUID_SECRET}"
  main_wallet: "${NONEXISTENT_VAR}"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write(env_config_content)
        temp_env_config_path = f.name
    
    try:
        env_config = load_volume_generator_config(temp_env_config_path)
        print(f"   Env config: api_key={env_config.api_key}, api_secret={env_config.api_secret}")
        print(f"   Env config: main_wallet={env_config.main_wallet}")
        assert env_config.api_key == 'env_key_value'
        assert env_config.api_secret == 'env_secret_value'
        # Should remain None since env var doesn't exist
        # When env var doesn't exist, the placeholder remains as string
        assert env_config.main_wallet == "${NONEXISTENT_VAR}"
        print("   ✓ Environment variable substitution works correctly")
    finally:
        os.unlink(temp_env_config_path)
        del os.environ['TEST_HYPERLIQUID_API_KEY']
        del os.environ['TEST_HYPERLIQUID_SECRET']

    print("\n✓ All config loading tests passed!")


if __name__ == "__main__":
    test_config_loading()