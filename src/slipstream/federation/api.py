"""
API wrapper classes for federated strategy pods.

This module provides standardized API endpoints for all strategy pods
to implement the federated architecture while maintaining backward compatibility.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any, Union
from enum import Enum


@dataclass
class StrategyStatus:
    """Standardized status response for strategy pods."""
    net_exposure: float
    open_orders: int
    pnl: float
    health_status: str
    uptime: float
    strategy_name: str


@dataclass
class ConfigurationUpdate:
    """Configuration update request for strategy pods."""
    max_position: Optional[float] = None
    volatility_target: Optional[float] = None
    risk_limits: Optional[Dict[str, Any]] = None
    other_params: Optional[Dict[str, Any]] = None


class HaltReason(Enum):
    """Reasons for halting a strategy."""
    MANUAL = "manual"
    RISK_VIOLATION = "risk_violation"
    PERFORMANCE_FAILURE = "performance_failure"
    SYSTEM_MAINTENANCE = "system_maintenance"


class StrategyAPI(ABC):
    """
    Abstract base class for strategy API endpoints that provides standardized
    federation interfaces while maintaining backward compatibility with existing strategies.
    """

    def __init__(self, strategy_instance: Any):
        """
        Initialize the API wrapper with a strategy instance.
        
        Args:
            strategy_instance: The existing strategy instance to wrap
        """
        self.strategy = strategy_instance
        self._halt_requested = False
        self._uptime = 0.0
        self._start_time = None

    @abstractmethod
    async def get_status(self) -> StrategyStatus:
        """
        GET /status: Returns Net Exposure, Open Orders, and PnL.
        
        This is the primary endpoint for monitoring strategy health.
        
        Returns:
            StrategyStatus: Current status of the strategy including exposure metrics
        """
        pass

    @abstractmethod
    async def configure(self, config_update: ConfigurationUpdate) -> Dict[str, Any]:
        """
        POST /configure: Accepts dynamic limits (Max Position, Volatility Target).
        
        Allows the meta-optimizer to dynamically adjust strategy parameters.
        
        Args:
            config_update: Configuration parameters to update
            
        Returns:
            Dict with confirmation of applied changes
        """
        pass

    @abstractmethod
    async def halt(self, reason: Optional[HaltReason] = None, message: Optional[str] = None) -> Dict[str, Any]:
        """
        POST /halt: A hard kill-switch for emergency shutdown.
        
        Allows the central system to stop a strategy immediately.
        
        Args:
            reason: Reason for the halt
            message: Optional message explaining the halt
            
        Returns:
            Dict with halt confirmation
        """
        pass

    async def heartbeat(self) -> Dict[str, Any]:
        """
        GET /health: Basic health check endpoint.
        
        Returns basic health status of the strategy.
        
        Returns:
            Dict with health status
        """
        return {
            "status": "healthy" if not self._halt_requested else "halted",
            "strategy": getattr(self.strategy, '__class__', type(self.strategy)).__name__,
            "uptime": self._uptime,
            "timestamp": asyncio.get_event_loop().time()
        }

    def is_halted(self) -> bool:
        """Check if the strategy has been requested to halt."""
        return self._halt_requested

    def start_uptime_tracking(self):
        """Start tracking uptime for the strategy."""
        self._start_time = asyncio.get_event_loop().time()

    async def update_uptime(self):
        """Update uptime counter."""
        if self._start_time:
            self._uptime = asyncio.get_event_loop().time() - self._start_time


class StrategyPod(Protocol):
    """
    Protocol that defines the expected interface for a strategy pod.
    This helps ensure that existing strategies can be wrapped properly.
    """
    async def get_exposure(self) -> float:
        """Get current net exposure."""
        ...

    async def get_open_orders(self) -> int:
        """Get current number of open orders."""
        ...

    async def get_pnl(self) -> float:
        """Get current PnL."""
        ...

    async def update_config(self, **kwargs) -> bool:
        """Update strategy configuration."""
        ...

    async def stop_gracefully(self) -> bool:
        """Stop the strategy gracefully."""
        ...


def wrap_strategy_for_api(strategy_instance: Union[StrategyPod, Any]) -> StrategyAPI:
    """
    Factory function to wrap an existing strategy with API functionality.
    
    This maintains backward compatibility by preserving the original strategy
    while adding the required API endpoints.
    
    Args:
        strategy_instance: An existing strategy instance
        
    Returns:
        A StrategyAPI wrapper for the strategy
    """
    # Create a concrete implementation that wraps the existing strategy
    class WrappedStrategy(StrategyAPI):
        def __init__(self, strategy_instance):
            super().__init__(strategy_instance)
            self.start_uptime_tracking()

        async def get_status(self) -> StrategyStatus:
            # Try to use existing methods if available, otherwise provide defaults
            try:
                net_exposure = await strategy_instance.get_exposure() if hasattr(strategy_instance, 'get_exposure') else 0.0
                open_orders = await strategy_instance.get_open_orders() if hasattr(strategy_instance, 'get_open_orders') else 0
                pnl = await strategy_instance.get_pnl() if hasattr(strategy_instance, 'get_pnl') else 0.0
            except:
                # Fallback to defaults if methods don't exist
                net_exposure = 0.0
                open_orders = 0
                pnl = 0.0

            return StrategyStatus(
                net_exposure=net_exposure,
                open_orders=open_orders,
                pnl=pnl,
                health_status="healthy" if not self._halt_requested else "halted",
                uptime=self._uptime,
                strategy_name=getattr(strategy_instance, '__name__', type(strategy_instance).__name__)
            )

        async def configure(self, config_update: ConfigurationUpdate) -> Dict[str, Any]:
            try:
                # Update the strategy's configuration
                update_kwargs = {}
                if config_update.max_position is not None:
                    update_kwargs['max_position'] = config_update.max_position
                if config_update.volatility_target is not None:
                    update_kwargs['volatility_target'] = config_update.volatility_target
                if config_update.risk_limits is not None:
                    update_kwargs['risk_limits'] = config_update.risk_limits
                if config_update.other_params is not None:
                    update_kwargs.update(config_update.other_params)

                # Try to call the strategy's update_config method if it exists
                if hasattr(self.strategy, 'update_config'):
                    success = await self.strategy.update_config(**update_kwargs)
                else:
                    # If update_config doesn't exist, try to update attributes directly
                    for key, value in update_kwargs.items():
                        if hasattr(self.strategy, key):
                            setattr(self.strategy, key, value)
                    success = True

                return {
                    "status": "success",
                    "applied_updates": update_kwargs,
                    "timestamp": asyncio.get_event_loop().time()
                }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": asyncio.get_event_loop().time()
                }

        async def halt(self, reason: Optional[HaltReason] = None, message: Optional[str] = None) -> Dict[str, Any]:
            self._halt_requested = True
            
            # Try to stop the strategy gracefully if possible
            if hasattr(self.strategy, 'stop_gracefully'):
                try:
                    success = await self.strategy.stop_gracefully()
                except:
                    success = False
            else:
                success = True  # If no stop method, mark as halted anyway

            return {
                "status": "halted" if success else "halt_requested",
                "reason": reason.value if reason else "manual",
                "message": message or "Manual halt requested",
                "timestamp": asyncio.get_event_loop().time()
            }

    return WrappedStrategy(strategy_instance)


__all__ = [
    "StrategyAPI",
    "StrategyStatus",
    "ConfigurationUpdate",
    "HaltReason",
    "StrategyPod",
    "wrap_strategy_for_api",
]