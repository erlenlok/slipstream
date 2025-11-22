#!/usr/bin/env python3
"""Test script to verify the volume generator strategy integration with the Slipstream framework."""

from slipstream.strategies import get_strategy_info, list_strategies, load_strategy
from slipstream.strategies.volume_generator import VolumeGeneratorBot, VolumeBotConfig


def test_strategy_integration():
    """Test that volume_generator is properly integrated with the framework."""
    
    print("Testing Volume Generator Strategy Integration...")
    
    # Test 1: Check if volume_generator is registered in the strategy registry
    print("\n1. Testing strategy registration...")
    strategies = list_strategies()
    strategy_names = [s.key for s in strategies]
    print(f"   Registered strategies: {strategy_names}")
    assert "volume_generator" in strategy_names, "volume_generator not found in registry"
    print("   ✓ Volume generator is registered in the strategy registry")
    
    # Test 2: Get strategy info
    print("\n2. Testing strategy info retrieval...")
    vg_info = get_strategy_info("volume_generator")
    print(f"   Strategy key: {vg_info.key}")
    print(f"   Strategy title: {vg_info.title}")
    print(f"   Module: {vg_info.module}")
    print(f"   Description: {vg_info.description}")
    print(f"   CLI entrypoints: {list(vg_info.cli_entrypoints.keys())}")
    assert vg_info.key == "volume_generator"
    assert vg_info.title == "Volume Generator Bot"
    assert vg_info.module == "slipstream.strategies.volume_generator"
    assert "run_volume_gen" in vg_info.cli_entrypoints
    print("   ✓ Strategy info is correct")
    
    # Test 3: Load strategy module
    print("\n3. Testing strategy module loading...")
    vg_module = load_strategy("volume_generator")
    print(f"   Module loaded: {vg_module.__name__}")
    assert hasattr(vg_module, 'VolumeGeneratorBot')
    assert hasattr(vg_module, 'VolumeBotConfig')
    print("   ✓ Strategy module loaded successfully")
    
    # Test 4: Create bot instances
    print("\n4. Testing bot instantiation...")
    config = VolumeBotConfig(dry_run=True, trade_count=1, trade_size_usd=10.0)
    bot = VolumeGeneratorBot(config)
    print(f"   Bot created with config: trade_count={bot.config.trade_count}, dry_run={bot.config.dry_run}")
    assert bot.config.dry_run == True
    assert bot.config.trade_count == 1
    print("   ✓ Bot instantiated successfully")
    
    # Test 5: Check CLI entrypoint availability
    print("\n5. Testing CLI entrypoint availability...")
    handler = vg_info.load_cli_handler("run_volume_gen")
    print(f"   CLI handler loaded: {handler.__name__}")
    assert handler is not None
    print("   ✓ CLI entrypoint is available and accessible")
    
    print("\n🎉 All integration tests passed! The Volume Generator strategy is fully integrated with the Slipstream framework.")


if __name__ == "__main__":
    test_strategy_integration()