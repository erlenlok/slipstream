
import pytest
from slipstream.strategies.brawler.economics import ToleranceController

class TestToleranceController:
    def test_tolerance_rich(self):
        """When budget is high, tolerance should be minimal."""
        ctrl = ToleranceController(
            min_tolerance_ticks=1.0,
            dilation_k=1000.0,
            survival_tolerance_ticks=100.0
        )
        # Budget = 10,000 -> K/B = 0.1 -> Max(1, 0.1) = 1
        assert ctrl.calculate_tolerance(10000.0) == 1.0

    def test_tolerance_poor(self):
        """When budget is low, tolerance should dilate."""
        ctrl = ToleranceController(
            min_tolerance_ticks=1.0,
            dilation_k=1000.0,
            survival_tolerance_ticks=100.0
        )
        # Budget = 100 -> K/B = 10 -> Max(1, 10) = 10
        assert ctrl.calculate_tolerance(100.0) == 10.0

    def test_tolerance_bankrupt(self):
        """When budget is negative, return survival tolerance."""
        ctrl = ToleranceController(
            min_tolerance_ticks=1.0,
            dilation_k=1000.0, 
            survival_tolerance_ticks=55.0
        )
        assert ctrl.calculate_tolerance(0.0) == 55.0
        assert ctrl.calculate_tolerance(-500.0) == 55.0

    def test_min_tolerance_prevails(self):
        """Ensure we never go below min_tolerance even if super rich."""
        ctrl = ToleranceController(min_tolerance_ticks=5.0, dilation_k=1000.0)
        # Budget 1M -> K/B = 0.001 -> Max(5, 0.001) = 5
        assert ctrl.calculate_tolerance(1_000_000.0) == 5.0
