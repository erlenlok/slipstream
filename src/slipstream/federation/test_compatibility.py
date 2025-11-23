"""
Test implementations to verify API wrapper compatibility with existing strategies.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Any


class MockExistingStrategy:
    """
    Mock class representing an existing strategy to test backward compatibility.
    This simulates the structure of existing strategies without breaking changes.
    """
    
    def __init__(self, name: str = "mock_strategy", initial_capital: float = 10000.0):
        self.name = name
        self.capital = initial_capital
        self.exposure = 0.0
        self.open_orders = 0
        self.pnl = 0.0
        self.max_position = 1000.0
        self.volatility_target = 0.02
        self._running = True
        
    async def get_exposure(self) -> float:
        """Simulate getting current net exposure."""
        # In a real strategy, this would calculate actual exposure
        return self.exposure
    
    async def get_open_orders(self) -> int:
        """Simulate getting current number of open orders."""
        return self.open_orders
    
    async def get_pnl(self) -> float:
        """Simulate getting current PnL."""
        return self.pnl
    
    async def update_config(self, **kwargs) -> bool:
        """Simulate updating strategy configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return True
    
    async def stop_gracefully(self) -> bool:
        """Simulate stopping the strategy gracefully."""
        self._running = False
        return True


class MockGradientStrategy:
    """
    Mock class representing a gradient strategy to test API compatibility.
    """
    
    def __init__(self):
        self.exposure = 500.0
        self.open_orders = 3
        self.pnl = 25.50
        self.max_position = 2000.0
        self.volatility_target = 0.015
        self._running = True

    # Note: This strategy doesn't have the enhanced methods, testing fallback behavior


class MockBrawlerStrategy:
    """
    Mock class representing a brawler strategy to test API compatibility.
    """
    
    def __init__(self):
        self.current_exposure = 1200.0
        self.active_orders = 5
        self.daily_pnl = -15.30
        self.max_inventory = 10.0
        self._running = True

    # Note: This strategy has different attribute names, testing attribute fallback


async def test_api_wrapper_compatibility():
    """
    Test that the API wrapper works with different types of existing strategies.
    """
    from .api import wrap_strategy_for_api
    
    print("Testing API wrapper compatibility with existing strategies...")
    
    # Test 1: Strategy with enhanced methods
    mock_strategy = MockExistingStrategy()
    wrapped_api = wrap_strategy_for_api(mock_strategy)
    
    print(f"Test 1 - Wrapped Mock Strategy:")
    status = await wrapped_api.get_status()
    print(f"  Status: {status}")
    
    # Test configuration
    from .api import ConfigurationUpdate
    config_update = ConfigurationUpdate(max_position=1500.0, volatility_target=0.025)
    config_result = await wrapped_api.configure(config_update)
    print(f"  Config Result: {config_result}")
    
    # Verify configuration was applied
    assert mock_strategy.max_position == 1500.0
    assert mock_strategy.volatility_target == 0.025
    print("  ✓ Configuration applied correctly")
    
    # Test 2: Strategy without enhanced methods (fallback behavior)
    gradient_strategy = MockGradientStrategy()
    wrapped_gradient = wrap_strategy_for_api(gradient_strategy)
    
    print(f"\nTest 2 - Wrapped Gradient Strategy (fallback behavior):")
    status = await wrapped_gradient.get_status()
    print(f"  Status: {status}")
    
    # Should use fallback values since no get_exposure/get_open_orders/get_pnl methods
    assert status.net_exposure == 0.0  # fallback value
    assert status.open_orders == 0     # fallback value
    assert status.pnl == 0.0          # fallback value
    print("  ✓ Fallback behavior works correctly")
    
    # Test 3: Strategy with different attribute names (direct attribute access)
    brawler_strategy = MockBrawlerStrategy()
    wrapped_brawler = wrap_strategy_for_api(brawler_strategy)
    
    print(f"\nTest 3 - Wrapped Brawler Strategy:")
    status = await wrapped_brawler.get_status()
    print(f"  Status: {status}")
    
    print("  ✓ All compatibility tests passed!")


async def test_federation_concepts():
    """
    Test the core federation concepts implemented in our API layer.
    """
    print("\nTesting Federation Concepts:")

    # Test that we maintain backward compatibility
    original_strategy = MockExistingStrategy()

    # The original strategy should continue to work exactly as before
    original_exposure = await original_strategy.get_exposure()
    print(f"✓ Original strategy works: exposure = {original_exposure}")

    # API wrapper should not interfere with original functionality
    from .api import wrap_strategy_for_api
    wrapped_api = wrap_strategy_for_api(original_strategy)
    original_exposure2 = await original_strategy.get_exposure()
    print(f"✓ Original functionality preserved after wrapping: exposure = {original_exposure2}")

    # Both should return same value
    assert original_exposure == original_exposure2
    print("✓ No interference between wrapper and original strategy")


if __name__ == "__main__":
    asyncio.run(test_api_wrapper_compatibility())
    asyncio.run(test_federation_concepts())
    print("\n✓ All federation API tests passed!")