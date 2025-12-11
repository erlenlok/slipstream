"""Economics module for Brawler strategy.

Handles request budgeting, shadow pricing, and rate limit synchronization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PurseState:
    """Snapshot of the economic state."""
    request_count: int
    cumulative_volume: float
    # Derived metrics could go here or be calculated on the fly
    budget_usd: float = 0.0


class RequestPurse:
    """Tracks economic 'credits' to spend on API requests.

    The 'Budget' is defined as:
        Budget = (Volume * CreditPerVol) - (Requests * CostPerRequest)
    
    However, for Sprint 1, we primarily focus on accurate tracking and syncing with the exchange.
    """

    def __init__(self, cost_per_request: float = 0.00035) -> None:
        self.cost_per_request = cost_per_request
        self._request_count: int = 0
        self._cumulative_volume: float = 0.0
        
        # We might track a local offset if the exchange data is delayed or cumulative
        self._request_offset: int = 0
        self._volume_offset: float = 0.0

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def cumulative_volume(self) -> float:
        return self._cumulative_volume

    def deduct_request(self) -> None:
        """Record that a request has been sent."""
        self._request_count += 1

    def add_fill_credit(self, volume_usd: float) -> None:
        """Record trade volume (which earns credits)."""
        self._cumulative_volume += abs(volume_usd)

    def sync(
        self, 
        exchange_requests: int, 
        exchange_volume: float, 
        reset_offsets: bool = False
    ) -> None:
        """Synchronize with truthful exchange data.

        If we are tracking local state that resets (like session-based), 
        we might need logic to handle the fact that exchange counters might be lifetime or monthly.
        For now, we assume the exchange returns growing counters and we adopt them.
        """
        # Simple adoption for now -> "The Meter" reads what the exchange says.
        # But we might want to log if our local tracking drifted significantly
        
        requests_drift = exchange_requests - self._request_count
        volume_drift = exchange_volume - self._cumulative_volume
        
        if abs(requests_drift) > 10 or abs(volume_drift) > 100.0:
            logger.debug(
                "Purse sync large drift: req_drift=%d, vol_drift=%.2f", 
                requests_drift, volume_drift
            )

        self._request_count = exchange_requests
        self._cumulative_volume = exchange_volume

    @property
    def request_budget(self) -> float:
        """Estimate of remaining 'economic' budget for requests.
        
        This is a synthetic metric for Brawler, separate from the hard API rate limits.
        We treat volume as income and requests as cost.
        
        Assumptions (can be tuned):
        - 1 USD of volume earns ~0.1 'credits'? 
        - Or simpler: We just track net value relative to a starting point?
        
        For Sprint 2, let's use a simple linearized model:
           Budget = (Vol * 0.1) - (Requests * CostPerRequest)
           
        Note: This means we start at 0 and go negative until we trade.
        """
        # Hyperliquid Rate Limit Rule: 1 USDC Volume frees up 1 Request.
        # We track "Surplus Requests" as the budget.
        # Budget = Volume - Requests
        return self._cumulative_volume - self._request_count


class ToleranceController:
    """Adjusts quoting tolerance based on economic budget."""

    def __init__(
        self, 
        min_tolerance_ticks: float,
        dilation_k: float = 1000.0,
        survival_tolerance_ticks: float = 100.0
    ) -> None:
        self.min_tolerance_ticks = min_tolerance_ticks
        self.dilation_k = dilation_k
        self.survival_tolerance_ticks = survival_tolerance_ticks

    def calculate_tolerance(self, budget: float) -> float:
        """Calculate required tolerance in ticks.
        
        Formula: T = max(T_min, K / Budget)
        If Budget <= 0, returns survival tolerance.
        """
        if budget <= 0:
            return self.survival_tolerance_ticks
        
        # Hyperbolic dilation
        # If Budget is large (rich), K/Budget -> small -> T_min prevails.
        # If Budget is small (poor), K/Budget -> large.
        dynamic = self.dilation_k / budget
        return max(self.min_tolerance_ticks, dynamic)

    def calculate_spread_penalty(self, budget: float) -> float:
        """Calculate additional spread penalty (raw float) based on budget."""
        if budget >= 0:
            return 0.0
        # 1 bp per 200 unit deficit
        unit_deficit = abs(budget) / 200.0
        return min(unit_deficit * 0.0001, 0.01) # Cap at 100bps penalty

