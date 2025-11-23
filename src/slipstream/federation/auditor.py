"""
Risk Auditor for federated strategy monitoring.

This module implements an independent process that listens to exchange via read-only keys
to verify strategies aren't lying about their exposure, as specified in the federated vision.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Protocol, Any
from datetime import datetime, timedelta
from enum import Enum
import time


@dataclass
class ExchangeEvent:
    """Base class for exchange events that the auditor monitors."""
    timestamp: datetime
    event_type: str
    strategy_id: Optional[str] = None
    account_id: Optional[str] = None


@dataclass
class TradeEvent:
    """Represents a trade execution event."""
    timestamp: datetime
    event_type: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_id: str
    strategy_id: Optional[str] = None
    account_id: Optional[str] = None
    fees: float = 0.0


@dataclass
class PositionEvent:
    """Represents a position update event."""
    timestamp: datetime
    event_type: str
    symbol: str
    position_size: float
    entry_price: float
    unrealized_pnl: float
    strategy_id: Optional[str] = None
    account_id: Optional[str] = None


@dataclass
class OrderEvent:
    """Represents an order event."""
    timestamp: datetime
    event_type: str
    symbol: str
    order_id: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str
    status: str  # 'open', 'filled', 'cancelled', etc.
    strategy_id: Optional[str] = None
    account_id: Optional[str] = None


@dataclass
class StrategyReport:
    """Report from a strategy pod about its positions/exposures."""
    strategy_id: str
    timestamp: datetime
    net_exposure: float
    open_orders: int
    reported_pnl: float
    positions: Dict[str, float]  # symbol -> position_size
    total_capital: Optional[float] = None


@dataclass
class AuditResult:
    """Result of an audit comparison between reported and actual."""
    strategy_id: str
    timestamp: datetime
    reported_exposure: float
    actual_exposure: float
    exposure_difference: float
    reported_pnl: float
    actual_pnl: float
    pnl_difference: float
    discrepancies: List[str]
    overall_status: str  # 'PASS', 'WARN', 'FAIL'


class ExchangeConnector(Protocol):
    """
    Protocol for exchange connectors that the auditor can use.
    This allows the auditor to work with any exchange without tight coupling.
    """
    async def connect_readonly(self) -> bool:
        """Connect to exchange in read-only mode."""
        ...

    async def subscribe_to_trades(self, account_ids: List[str]) -> None:
        """Subscribe to trade events for specific accounts."""
        ...

    async def get_account_positions(self, account_id: str) -> Dict[str, PositionEvent]:
        """Get current positions for an account (read-only)."""
        ...

    async def get_account_balance(self, account_id: str) -> Dict[str, float]:
        """Get account balance information (read-only)."""
        ...


class RiskAuditor:
    """
    Independent process that listens to exchange via read-only keys
    to verify strategies aren't lying about their exposure.
    
    This component operates completely independently from strategies
    and maintains its own view of actual positions/exposures.
    """

    def __init__(self, exchange_connector: Optional[ExchangeConnector] = None):
        """
        Initialize the Risk Auditor.
        
        Args:
            exchange_connector: Optional exchange connector for live monitoring
        """
        self.exchange_connector = exchange_connector
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._audit_results: List[AuditResult] = []
        
        # Track actual exposures per account/strategy
        self._actual_positions: Dict[str, Dict[str, float]] = {}  # strategy_id -> {symbol: position}
        self._actual_pnls: Dict[str, float] = {}  # strategy_id -> actual_pnl
        self._actual_exposures: Dict[str, float] = {}  # strategy_id -> actual_exposure
        
        # Track reported exposures for comparison
        self._reported_positions: Dict[str, Dict[str, float]] = {}  # strategy_id -> {symbol: position}
        self._reported_pnls: Dict[str, float] = {}  # strategy_id -> reported_pnl
        self._reported_exposures: Dict[str, float] = {}  # strategy_id -> reported_exposure
        
        # Track trade history for PnL calculation
        self._trade_history: Dict[str, List[TradeEvent]] = {}  # strategy_id -> [trades]

    async def start(self):
        """Start the risk auditor monitoring process."""
        self._running = True
        self.logger.info("Risk Auditor started")
        
        # If we have an exchange connector, start listening to events
        if self.exchange_connector:
            await self.exchange_connector.connect_readonly()
            # Additional setup for live monitoring would go here
            
        self.logger.info("Risk Auditor monitoring started")

    async def stop(self):
        """Stop the risk auditor."""
        self._running = False
        self.logger.info("Risk Auditor stopped")

    async def add_strategy(self, strategy_id: str):
        """
        Register a strategy to be monitored by the auditor.
        
        Args:
            strategy_id: Unique identifier for the strategy
        """
        if strategy_id not in self._actual_positions:
            self._actual_positions[strategy_id] = {}
            self._actual_pnls[strategy_id] = 0.0
            self._actual_exposures[strategy_id] = 0.0
            self._reported_positions[strategy_id] = {}
            self._reported_pnls[strategy_id] = 0.0
            self._reported_exposures[strategy_id] = 0.0
            self._trade_history[strategy_id] = []
            
        self.logger.info(f"Strategy {strategy_id} added to audit monitoring")

    async def record_exchange_event(self, event: ExchangeEvent):
        """
        Record an exchange event for audit purposes.
        
        This method processes exchange events to maintain the auditor's
        independent view of actual positions/exposures.
        
        Args:
            event: The exchange event to record
        """
        if not self._running:
            return
            
        if isinstance(event, TradeEvent):
            await self._process_trade_event(event)
        elif isinstance(event, PositionEvent):
            await self._process_position_event(event)
        elif isinstance(event, OrderEvent):
            await self._process_order_event(event)

    async def _process_trade_event(self, event: TradeEvent):
        """Process a trade event to update actual positions."""
        strategy_id = event.strategy_id
        if not strategy_id:
            return  # Skip if no strategy ID associated

        # Initialize strategy tracking if needed
        if strategy_id not in self._actual_positions:
            await self.add_strategy(strategy_id)

        # Update positions based on trade
        current_position = self._actual_positions[strategy_id].get(event.symbol, 0.0)
        trade_size = event.quantity if event.side.lower() == 'buy' else -event.quantity
        new_position = current_position + trade_size

        self._actual_positions[strategy_id][event.symbol] = new_position

        # Update trade history for PnL calculation
        if strategy_id not in self._trade_history:
            self._trade_history[strategy_id] = []
        self._trade_history[strategy_id].append(event)

        # Calculate exposure (absolute value of positions)
        exposure = sum(abs(pos) for pos in self._actual_positions[strategy_id].values())
        self._actual_exposures[strategy_id] = exposure

        self.logger.debug(f"Processed trade for {strategy_id}: {event.symbol} {event.side} {event.quantity} @ {event.price}")

    async def _process_position_event(self, event: PositionEvent):
        """Process a position event to update actual positions."""
        strategy_id = event.strategy_id
        if not strategy_id:
            return  # Skip if no strategy ID associated
            
        # Initialize strategy tracking if needed
        if strategy_id not in self._actual_positions:
            await self.add_strategy(strategy_id)
            
        # Update position directly
        self._actual_positions[strategy_id][event.symbol] = event.position_size
        
        # Calculate exposure
        exposure = sum(abs(pos) for pos in self._actual_positions[strategy_id].values())
        self._actual_exposures[strategy_id] = exposure
        
        self.logger.debug(f"Updated position for {strategy_id}: {event.symbol} = {event.position_size}")

    async def _process_order_event(self, event: OrderEvent):
        """Process an order event."""
        # Currently just logs for future enhancement
        strategy_id = event.strategy_id
        if strategy_id:
            self.logger.debug(f"Order event for {strategy_id}: {event.order_type} {event.side} {event.quantity} {event.symbol}")

    async def record_strategy_report(self, report: StrategyReport):
        """
        Record a strategy's self-reported positions/exposures.
        
        This method stores the strategy's self-reported data for later comparison
        with the auditor's independent view.
        
        Args:
            report: The strategy's self-report
        """
        strategy_id = report.strategy_id
        
        # Initialize strategy tracking if needed
        if strategy_id not in self._reported_positions:
            await self.add_strategy(strategy_id)
            
        # Store reported values
        self._reported_positions[strategy_id] = report.positions.copy()
        self._reported_pnls[strategy_id] = report.reported_pnl
        self._reported_exposures[strategy_id] = report.net_exposure
        
        self.logger.debug(f"Recorded strategy report for {strategy_id}: exposure={report.net_exposure}, PnL={report.reported_pnl}")

    async def perform_audit(self, strategy_id: str) -> AuditResult:
        """
        Perform an audit comparison between reported and actual values.
        
        Args:
            strategy_id: The strategy to audit
            
        Returns:
            AuditResult with comparison details
        """
        if strategy_id not in self._reported_exposures:
            return AuditResult(
                strategy_id=strategy_id,
                timestamp=datetime.now(),
                reported_exposure=0,
                actual_exposure=0,
                exposure_difference=0,
                reported_pnl=0,
                actual_pnl=0,
                pnl_difference=0,
                discrepancies=["Strategy not found in reports"],
                overall_status="FAIL"
            )
        
        # Get reported values
        reported_exposure = self._reported_exposures.get(strategy_id, 0.0)
        reported_pnl = self._reported_pnls.get(strategy_id, 0.0)
        
        # Get actual values (maintained by auditor)
        actual_exposure = self._actual_exposures.get(strategy_id, 0.0)
        actual_pnl = self._actual_pnls.get(strategy_id, 0.0)
        
        # Calculate differences
        exposure_diff = abs(reported_exposure - actual_exposure)
        pnl_diff = abs(reported_pnl - actual_pnl)
        
        # Determine discrepancies
        discrepancies = []
        tolerance = 1e-6  # Small tolerance for floating point comparisons
        
        if exposure_diff > tolerance:
            discrepancies.append(f"Exposure mismatch: reported={reported_exposure}, actual={actual_exposure}")
            
        if pnl_diff > tolerance:
            discrepancies.append(f"PnL mismatch: reported={reported_pnl}, actual={actual_pnl}")
        
        # Calculate overall status based on discrepancies
        if len(discrepancies) == 0:
            overall_status = "PASS"
        elif len(discrepancies) == 1 and (exposure_diff < 1.0 and pnl_diff < 1.0):  # Small differences
            overall_status = "WARN"
        else:
            overall_status = "FAIL"
        
        result = AuditResult(
            strategy_id=strategy_id,
            timestamp=datetime.now(),
            reported_exposure=reported_exposure,
            actual_exposure=actual_exposure,
            exposure_difference=exposure_diff,
            reported_pnl=reported_pnl,
            actual_pnl=actual_pnl,
            pnl_difference=pnl_diff,
            discrepancies=discrepancies,
            overall_status=overall_status
        )
        
        # Store the result
        self._audit_results.append(result)
        
        return result

    async def perform_all_audits(self) -> List[AuditResult]:
        """
        Perform audits for all registered strategies.
        
        Returns:
            List of audit results for all strategies
        """
        results = []
        for strategy_id in set(self._reported_exposures.keys()) | set(self._actual_exposures.keys()):
            result = await self.perform_audit(strategy_id)
            results.append(result)
        return results

    async def get_unified_exposure_view(self) -> Dict[str, Dict[str, float]]:
        """
        Get a unified view of all strategy exposures from the auditor's perspective.
        
        Returns:
            Dict mapping strategy_id to exposure information
        """
        unified_view = {}
        all_strategy_ids = set(self._actual_exposures.keys()) | set(self._reported_exposures.keys())
        
        for strategy_id in all_strategy_ids:
            unified_view[strategy_id] = {
                'actual_exposure': self._actual_exposures.get(strategy_id, 0.0),
                'reported_exposure': self._reported_exposures.get(strategy_id, 0.0),
                'position_breakdown': self._actual_positions.get(strategy_id, {}),
                'last_audit_result': self._audit_results[-1].overall_status if self._audit_results else 'NO_AUDIT',
                'active': strategy_id in self._actual_exposures or strategy_id in self._reported_exposures
            }
        
        return unified_view

    def get_audit_history(self, strategy_id: Optional[str] = None) -> List[AuditResult]:
        """
        Get audit history, optionally filtered by strategy.
        
        Args:
            strategy_id: Optional strategy ID to filter results
            
        Returns:
            List of audit results
        """
        if strategy_id:
            return [r for r in self._audit_results if r.strategy_id == strategy_id]
        return self._audit_results.copy()


class HyperliquidRiskAuditor(RiskAuditor):
    """
    Specific implementation for Hyperliquid exchange.
    
    This is an example of how the generic RiskAuditor can be extended
    to work with a specific exchange while maintaining the same interface.
    """
    
    def __init__(self, api_key: Optional[str] = None, read_only: bool = True):
        """
        Initialize Hyperliquid-specific risk auditor.
        
        Args:
            api_key: Hyperliquid API key (should be read-only)
            read_only: Whether to enforce read-only operations
        """
        super().__init__()
        self.api_key = api_key
        self.read_only = read_only
        self.logger = logging.getLogger(f"{__name__}.Hyperliquid")
        
        # Hyperliquid-specific tracking
        self._user_fills_stream = None
        self._account_info = None

    async def connect_readonly(self):
        """Connect to Hyperliquid API in read-only mode."""
        if not self.read_only:
            raise ValueError("Attempting to connect non-read-only auditor")
        
        # In a real implementation, this would connect to Hyperliquid
        # using the read-only API key to monitor events
        self.logger.info("Connected to Hyperliquid in read-only mode")
        return True


__all__ = [
    "RiskAuditor",
    "HyperliquidRiskAuditor",
    "ExchangeEvent",
    "TradeEvent", 
    "PositionEvent",
    "OrderEvent",
    "StrategyReport",
    "AuditResult",
    "ExchangeConnector"
]