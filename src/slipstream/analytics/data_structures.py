"""
Core data structures for Brawler performance tracking.

This module defines the fundamental data classes needed for
tracking and analyzing Brawler's market making performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any
import json
import numpy as np


class TradeType(Enum):
    """Type of trade: MAKER for passive fills, TAKER for aggressive fills"""
    MAKER = "maker"  # Passive fill (liquidity provision)
    TAKER = "taker"  # Aggressive fill (liquidity taking)


@dataclass
class TradeEvent:
    """Represents a single trade execution event for performance tracking."""
    
    # Core trade information
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    trade_type: TradeType
    
    # Reference information for markout calculation
    reference_price: Optional[float] = None  # Price to compare against for markout
    next_trade_price: Optional[float] = None  # Price of next trade in same direction
    
    # Fee and cost information
    fees_paid: float = 0.0
    funding_paid: float = 0.0  # Funding costs/earnings
    
    # Position information
    position_before: float = 0.0  # Position size before trade
    position_after: float = 0.0   # Position size after trade
    
    # Additional metadata
    order_id: Optional[str] = None
    quote_id: Optional[str] = None  # ID of the quote that resulted in this trade
    spread_at_quote: Optional[float] = None  # Spread at time of quote placement
    
    def calculate_pnl(self, exit_price: Optional[float] = None) -> float:
        """
        Calculate PnL for this trade.
        
        Args:
            exit_price: Optional exit price; if None, uses current position value
            
        Returns:
            PnL value accounting for fees and funding
        """
        if exit_price is None:
            # If we don't have an exit price, PnL is just the fees and funding impact
            return -(self.fees_paid + self.funding_paid)
        
        # Calculate PnL based on trade direction and exit
        if self.side.lower() == 'buy':
            pnl = (exit_price - self.price) * self.quantity
        else:  # sell
            pnl = (self.price - exit_price) * self.quantity
            
        return pnl - self.fees_paid - self.funding_paid
    
    def calculate_markout(self) -> Optional[float]:
        """
        Calculate the markout for this trade against reference price.
        
        Returns:
            Markout value or None if reference price unavailable
        """
        if self.reference_price is None:
            return None
            
        if self.side.lower() == 'buy':
            markout = (self.reference_price - self.price) * self.quantity
        else:  # sell
            markout = (self.price - self.reference_price) * self.quantity
            
        return markout - self.fees_paid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class MarkoutAnalysis:
    """Analysis of markout performance for a set of trades."""
    
    # Basic markout statistics
    avg_markout_in: float = 0.0      # Average markout for maker (passive) trades
    avg_markout_out: float = 0.0     # Average markout for taker (aggressive) trades
    total_markout: float = 0.0       # Net markout across all trades
    markout_count: int = 0           # Number of trades with markout data
    
    # Markout distribution
    markout_std: float = 0.0         # Standard deviation of markouts
    markout_min: float = 0.0         # Minimum markout observed
    markout_max: float = 0.0         # Maximum markout observed
    markout_percentiles: Dict[str, float] = field(default_factory=dict)  # 25th, 50th, 75th percentiles
    
    # Markout by trade type
    maker_markout_count: int = 0     # Number of maker trades with markout
    taker_markout_count: int = 0     # Number of taker trades with markout
    
    def update(self, trade: TradeEvent) -> None:
        """Update markout analysis with a new trade."""
        markout = trade.calculate_markout()
        if markout is None:
            return
            
        # Add to running statistics
        self.total_markout += markout
        self.markout_count += 1
        
        # Update min/max
        if self.markout_count == 1:
            self.markout_min = self.markout_max = markout
        else:
            self.markout_min = min(self.markout_min, markout)
            self.markout_max = max(self.markout_max, markout)
        
        # Update average
        self.avg_markout_in = self.total_markout / self.markout_count  # Simplified for now
        
        # Count by trade type
        if trade.trade_type == TradeType.MAKER:
            self.maker_markout_count += 1
            # This is a simplified approach - in real implementation we'd track maker vs taker separately
        else:
            self.taker_markout_count += 1


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics over a time period."""
    
    # Time window information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    window_hours: int = 24  # Default to 24-hour window
    
    # PnL metrics
    total_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    
    # Hit rate metrics
    total_quotes: int = 0
    total_fills: int = 0
    hit_rate: float = 0.0  # Percentage of quotes that resulted in fills
    fill_rate: float = 0.0  # Percentage of quotes or orders that resulted in fills
    
    # Volume metrics
    total_volume: float = 0.0  # Total notional value traded
    total_trades: int = 0      # Total number of trades executed
    
    # Inventory metrics
    avg_inventory: float = 0.0    # Average absolute inventory held
    max_inventory: float = 0.0    # Maximum inventory exposure
    inventory_turnover: float = 0.0  # How frequently inventory is cycled
    
    # Markout metrics
    markout_analysis: MarkoutAnalysis = field(default_factory=MarkoutAnalysis)
    
    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.0
    
    # Operational metrics
    cancellation_rate: float = 0.0  # Percentage of orders cancelled
    total_cancellations: int = 0
    
    # Per-asset breakdown (symbol -> metrics)
    per_asset_metrics: Dict[str, 'PerformanceMetrics'] = field(default_factory=dict)
    
    def calculate_hit_rate(self) -> float:
        """Calculate hit rate as percentage of quotes that resulted in fills."""
        if self.total_quotes == 0:
            return 0.0
        return (self.total_fills / self.total_quotes) * 100
    
    def calculate_fill_rate(self) -> float:
        """Calculate fill rate as percentage of orders placed that resulted in fills."""
        total_attempts = self.total_quotes + self.total_trades  # Quotes + direct orders
        if total_attempts == 0:
            return 0.0
        return (self.total_fills / total_attempts) * 100
    
    def calculate_pnl_per_quote(self) -> float:
        """Calculate average PnL per quote placed."""
        if self.total_quotes == 0:
            return 0.0
        return self.total_pnl / self.total_quotes
    
    def update_from_trade(self, trade: TradeEvent) -> None:
        """Update metrics with a new trade event."""
        # Update basic trade metrics
        self.total_trades += 1
        self.total_volume += trade.price * trade.quantity
        self.total_pnl += trade.calculate_pnl()  # This will be 0 for now since exit_price not provided
        self.fees_paid += trade.fees_paid
        self.funding_paid += trade.funding_paid

        # Update hit rate if this is from a quote
        if trade.quote_id is not None:
            self.total_fills += 1

        # Update markout analysis
        self.markout_analysis.update(trade)

        # Initialize per-asset metrics if needed
        if trade.symbol not in self.per_asset_metrics:
            self.per_asset_metrics[trade.symbol] = PerformanceMetrics(start_time=self.start_time, end_time=self.end_time)

        # Update per-asset metrics
        # Only update the specific asset metrics, not recursively call self again
        asset_metrics = self.per_asset_metrics[trade.symbol]
        # Update the asset metrics with the same trade characteristics but without recursion
        asset_metrics.total_trades += 1
        asset_metrics.total_volume += trade.price * trade.quantity
        asset_metrics.total_pnl += trade.calculate_pnl()
        asset_metrics.fees_paid += trade.fees_paid
        asset_metrics.funding_paid += trade.funding_paid
        if trade.quote_id is not None:
            asset_metrics.total_fills += 1
        asset_metrics.markout_analysis.update(trade)
    
    def update_from_quote(self, quote_event: Dict[str, Any]) -> None:
        """Update metrics with a new quote event (placed or cancelled)."""
        self.total_quotes += 1
        # We only increment total_fills when a fill happens, not when quotes are placed
        
    def update_from_cancellation(self) -> None:
        """Update metrics when an order is cancelled."""
        self.total_cancellations += 1
    
    def finalize_calculations(self) -> None:
        """Perform final calculations after all data is collected."""
        # Calculate hit rate and fill rate
        self.hit_rate = self.calculate_hit_rate()
        self.fill_rate = self.calculate_fill_rate()
        
        # Calculate cancellation rate
        if self.total_quotes > 0:
            self.cancellation_rate = (self.total_cancellations / self.total_quotes) * 100
        
        # Finalize per-asset metrics
        for asset_metrics in self.per_asset_metrics.values():
            asset_metrics.finalize_calculations()