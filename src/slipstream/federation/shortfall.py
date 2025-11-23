"""
Implementation Shortfall Analysis for trading execution quality.

This module implements the post-trade analysis capability to separate "Bad Luck" from "Bad Execution"
by tracking the difference between decision price and realized fill price, as specified in the federated vision.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd


@dataclass
class TradeDecision:
    """Represents a trading decision made by a strategy."""
    strategy_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    decision_price: float
    quantity: float
    decision_reason: str
    order_id: str
    signal_strength: Optional[float] = None
    expected_pnl: Optional[float] = None


@dataclass
class FillEvent:
    """Represents an actual fill event from the exchange."""
    strategy_id: str
    timestamp: datetime  
    symbol: str
    side: str  # 'buy' or 'sell'
    fill_price: float
    fill_quantity: float
    order_id: str
    fees: float = 0.0
    slippage: Optional[float] = None  # Optional exchange-reported slippage


@dataclass
class ShortfallReport:
    """Report on implementation shortfall for a trade or period."""
    strategy_id: str
    timestamp: datetime
    trade_id: str
    decision_price: float
    realized_price: float
    shortfall: float  # |Decision - Realized|
    quantity: float
    side: str
    execution_quality: str  # 'excellent', 'good', 'poor', 'failure'
    analysis: Dict[str, Any]


@dataclass
class AggregateShortfallMetrics:
    """Aggregate metrics for implementation shortfall analysis."""
    strategy_id: str
    period_start: datetime
    period_end: datetime
    total_trades: int
    avg_shortfall: float
    median_shortfall: float
    max_shortfall: float
    min_shortfall: float
    shortfall_std: float
    percentage_of_trades_with_high_shortfall: float
    total_cost_due_to_shortfall: float
    quality_score: float  # 0-100 scale based on shortfall


class ExecutionDataProvider(Protocol):
    """
    Protocol for data providers that the shortfall analyzer can use to
    collect execution data.
    """
    async def get_trade_decisions(self, strategy_id: str, 
                                  start_time: datetime, 
                                  end_time: datetime) -> List[TradeDecision]:
        """Get trade decisions made by a strategy."""
        ...

    async def get_fill_events(self, strategy_id: str, 
                              start_time: datetime, 
                              end_time: datetime) -> List[FillEvent]:
        """Get actual fill events for a strategy."""
        ...

    async def get_strategy_symbols(self, strategy_id: str) -> List[str]:
        """Get symbols traded by a strategy."""
        ...


class ImplementationShortfallAnalyzer:
    """
    Analyzes implementation shortfall to separate 'Bad Luck' from 'Bad Execution'.
    
    This component tracks Decision Price vs. Realized Price for each trade and
    calculates shortfall = |Decision - Realized| to identify when execution logic
    needs optimization rather than alpha signal improvements.
    """
    
    def __init__(self, 
                 data_provider: Optional[ExecutionDataProvider] = None,
                 high_shortfall_threshold: float = 0.10,  # 10 bps
                 analysis_interval: timedelta = timedelta(hours=1)):
        """
        Initialize the shortfall analyzer.
        
        Args:
            data_provider: Optional data provider for execution data
            high_shortfall_threshold: Threshold above which shortfall is considered high
            analysis_interval: How often to run analysis
        """
        self.data_provider = data_provider
        self.high_shortfall_threshold = high_shortfall_threshold
        self.analysis_interval = analysis_interval
        self.logger = logging.getLogger(__name__)
        self._running = False
        
        # Store trade decisions and fills for analysis
        self._trade_decisions: Dict[str, List[TradeDecision]] = {}  # strategy_id -> decisions
        self._fill_events: Dict[str, List[FillEvent]] = {}  # strategy_id -> fills
        self._shortfall_reports: Dict[str, List[ShortfallReport]] = {}  # strategy_id -> reports
        self._aggregate_metrics: Dict[str, List[AggregateShortfallMetrics]] = {}  # strategy_id -> metrics
        
        # Thresholds for execution quality classification
        self._excellent_threshold = 0.0005   # 5 bps
        self._good_threshold = 0.0010       # 10 bps
        self._poor_threshold = 0.0050       # 50 bps

    async def start(self):
        """Start the shortfall analyzer."""
        self._running = True
        self.logger.info("Implementation Shortfall Analyzer started")
        
        # Initial data loading if provider available
        if self.data_provider:
            # Initial data collection would happen here in a real system
            pass
        
        self.logger.info("Shortfall analyzer monitoring started")

    async def stop(self):
        """Stop the shortfall analyzer."""
        self._running = False
        self.logger.info("Implementation Shortfall Analyzer stopped")

    async def record_trade_decision(self, decision: TradeDecision):
        """
        Record a trade decision made by a strategy.
        
        Args:
            decision: The trade decision to record
        """
        strategy_id = decision.strategy_id
        
        if strategy_id not in self._trade_decisions:
            self._trade_decisions[strategy_id] = []
            
        self._trade_decisions[strategy_id].append(decision)
        self.logger.debug(f"Recorded trade decision for {strategy_id}: {decision.symbol} {decision.side} @ {decision.decision_price}")

    async def record_fill_event(self, fill: FillEvent):
        """
        Record an actual fill event from the exchange.
        
        Args:
            fill: The fill event to record
        """
        strategy_id = fill.strategy_id
        
        if strategy_id not in self._fill_events:
            self._fill_events[strategy_id] = []
            
        self._fill_events[strategy_id].append(fill)
        self.logger.debug(f"Recorded fill event for {strategy_id}: {fill.symbol} {fill.side} @ {fill.fill_price}")

    async def match_decisions_and_fills(self, strategy_id: str) -> List[tuple[TradeDecision, FillEvent]]:
        """
        Match trade decisions with their corresponding fill events.
        
        Args:
            strategy_id: The strategy ID to match events for
            
        Returns:
            List of (TradeDecision, FillEvent) tuples that match
        """
        decisions = self._trade_decisions.get(strategy_id, [])
        fills = self._fill_events.get(strategy_id, [])
        
        # Create lookup for fills by order_id
        fill_lookup = {fill.order_id: fill for fill in fills}
        
        matched_pairs = []
        
        for decision in decisions:
            fill = fill_lookup.get(decision.order_id)
            if fill:
                matched_pairs.append((decision, fill))
                
        return matched_pairs

    async def calculate_shortfall(self, decision: TradeDecision, fill: FillEvent) -> ShortfallReport:
        """
        Calculate implementation shortfall for a single trade.
        
        Args:
            decision: The trade decision
            fill: The actual fill event
            
        Returns:
            Shortfall report with analysis
        """
        # Calculate shortfall: |Decision - Realized| 
        # For consistency, we'll use the average fill price weighted by quantity
        shortfall = abs(decision.decision_price - fill.fill_price)
        
        # Calculate percentage shortfall relative to decision price
        if decision.decision_price != 0:
            pct_shortfall = shortfall / abs(decision.decision_price)
        else:
            pct_shortfall = 0.0
            
        # Classify execution quality based on shortfall
        if pct_shortfall <= self._excellent_threshold:
            quality = 'excellent'
        elif pct_shortfall <= self._good_threshold:
            quality = 'good'
        elif pct_shortfall <= self._poor_threshold:
            quality = 'poor'
        else:
            quality = 'failure'
        
        # Analyze potential causes
        analysis = {
            'decision_price': decision.decision_price,
            'realized_price': fill.fill_price,
            'shortfall_bps': pct_shortfall * 10000,  # Basis points
            'quantity': fill.fill_quantity,
            'side': fill.side,
            'execution_quality': quality,
            'signal_strength': decision.signal_strength,
            'expected_pnl_impact': decision.expected_pnl,
            'potential_causes': []
        }
        
        # Identify potential causes based on shortfall
        if pct_shortfall > self.high_shortfall_threshold:
            if fill.side == 'buy' and fill.fill_price > decision.decision_price:
                analysis['potential_causes'].append('Aggressive buying pushed price up')
            elif fill.side == 'sell' and fill.fill_price < decision.decision_price:
                analysis['potential_causes'].append('Aggressive selling pushed price down')
        
        if decision.signal_strength and abs(decision.signal_strength) < 0.1:
            analysis['potential_causes'].append('Weak signal may have affected execution timing')
        
        # Calculate cost impact
        cost_impact = (
            (fill.fill_price - decision.decision_price) * fill.fill_quantity
            if fill.side == 'buy'
            else (decision.decision_price - fill.fill_price) * fill.fill_quantity
        )
        
        analysis['cost_impact'] = cost_impact
        
        report = ShortfallReport(
            strategy_id=decision.strategy_id,
            timestamp=fill.timestamp,
            trade_id=fill.order_id,
            decision_price=decision.decision_price,
            realized_price=fill.fill_price,
            shortfall=shortfall,
            quantity=fill.fill_quantity,
            side=fill.side,
            execution_quality=quality,
            analysis=analysis
        )
        
        # Store the report
        if decision.strategy_id not in self._shortfall_reports:
            self._shortfall_reports[decision.strategy_id] = []
        self._shortfall_reports[decision.strategy_id].append(report)
        
        self.logger.debug(f"Calculated shortfall for {decision.strategy_id}: {pct_shortfall*10000:.2f} bps ({quality})")
        
        return report

    async def analyze_strategy_shortfall(self, strategy_id: str) -> List[ShortfallReport]:
        """
        Analyze all matched decisions and fills for a strategy.
        
        Args:
            strategy_id: The strategy ID to analyze
            
        Returns:
            List of shortfall reports for all trades
        """
        matched_pairs = await self.match_decisions_and_fills(strategy_id)
        
        reports = []
        for decision, fill in matched_pairs:
            report = await self.calculate_shortfall(decision, fill)
            reports.append(report)
            
        return reports

    async def calculate_aggregate_metrics(self, strategy_id: str, 
                                        period_start: Optional[datetime] = None,
                                        period_end: Optional[datetime] = None) -> AggregateShortfallMetrics:
        """
        Calculate aggregate shortfall metrics for a strategy over a period.
        
        Args:
            strategy_id: The strategy ID to analyze
            period_start: Start of the analysis period (defaults to beginning)
            period_end: End of the analysis period (defaults to now)
            
        Returns:
            Aggregate shortfall metrics
        """
        # Get all shortfall reports for the strategy in the specified period
        all_reports = self._shortfall_reports.get(strategy_id, [])
        
        if period_start or period_end:
            filtered_reports = []
            for report in all_reports:
                if period_start and report.timestamp < period_start:
                    continue
                if period_end and report.timestamp > period_end:
                    continue
                filtered_reports.append(report)
        else:
            filtered_reports = all_reports
            
        if not filtered_reports:
            return AggregateShortfallMetrics(
                strategy_id=strategy_id,
                period_start=period_start or datetime.now() - timedelta(days=1),
                period_end=period_end or datetime.now(),
                total_trades=0,
                avg_shortfall=0.0,
                median_shortfall=0.0,
                max_shortfall=0.0,
                min_shortfall=0.0,
                shortfall_std=0.0,
                percentage_of_trades_with_high_shortfall=0.0,
                total_cost_due_to_shortfall=0.0,
                quality_score=100.0  # Default to perfect when no data
            )
        
        # Extract shortfall values
        shortfalls = [r.shortfall for r in filtered_reports]
        pct_shortfalls = [abs(r.decision_price - r.realized_price) / abs(r.decision_price) 
                         if r.decision_price != 0 else 0.0 for r in filtered_reports]
        costs = [r.analysis.get('cost_impact', 0.0) for r in filtered_reports]
        
        # Calculate metrics
        avg_shortfall = np.mean(pct_shortfalls) if pct_shortfalls else 0.0
        median_shortfall = float(np.median(pct_shortfalls)) if pct_shortfalls else 0.0
        max_shortfall = float(np.max(pct_shortfalls)) if pct_shortfalls else 0.0
        min_shortfall = float(np.min(pct_shortfalls)) if pct_shortfalls else 0.0
        shortfall_std = float(np.std(pct_shortfalls)) if pct_shortfalls else 0.0
        
        # Calculate percentage of trades with high shortfall
        high_shortfall_count = sum(1 for sf in pct_shortfalls if sf > self.high_shortfall_threshold)
        pct_high_shortfall = high_shortfall_count / len(pct_shortfalls) if pct_shortfalls else 0.0
        
        total_cost = sum(costs) if costs else 0.0
        
        # Calculate quality score (0-100, higher is better)
        # Invert the shortfall and scale: lower shortfall = higher score
        if max(pct_shortfalls) > 0:
            # Normalize shortfall to 0-1 scale, then invert
            normalized_shortfall = avg_shortfall / max(pct_shortfalls)
            quality_score = max(0, 100 * (1.0 - normalized_shortfall))
        else:
            quality_score = 100.0  # Perfect if no shortfall
            
        # Cap quality score
        quality_score = min(100.0, max(0.0, quality_score))
        
        metrics = AggregateShortfallMetrics(
            strategy_id=strategy_id,
            period_start=period_start or min(r.timestamp for r in filtered_reports),
            period_end=period_end or max(r.timestamp for r in filtered_reports),
            total_trades=len(filtered_reports),
            avg_shortfall=avg_shortfall,
            median_shortfall=median_shortfall,
            max_shortfall=max_shortfall,
            min_shortfall=min_shortfall,
            shortfall_std=shortfall_std,
            percentage_of_trades_with_high_shortfall=pct_high_shortfall,
            total_cost_due_to_shortfall=total_cost,
            quality_score=quality_score
        )
        
        # Store metrics
        if strategy_id not in self._aggregate_metrics:
            self._aggregate_metrics[strategy_id] = []
        self._aggregate_metrics[strategy_id].append(metrics)
        
        self.logger.info(f"Calculated aggregate metrics for {strategy_id}: {len(filtered_reports)} trades, avg_shortfall={avg_shortfall*10000:.2f} bps, quality_score={quality_score:.1f}")
        
        return metrics

    async def get_strategy_execution_quality_summary(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get a summary of execution quality for a strategy.
        
        Args:
            strategy_id: The strategy ID to summarize
            
        Returns:
            Execution quality summary
        """
        if strategy_id not in self._aggregate_metrics:
            # Calculate metrics if not available
            await self.calculate_aggregate_metrics(strategy_id)
            
        # Get the most recent metrics
        metrics_list = self._aggregate_metrics.get(strategy_id, [])
        if not metrics_list:
            return {
                'strategy_id': strategy_id,
                'message': 'No execution data available',
                'has_data': False
            }
            
        latest_metrics = metrics_list[-1]
        
        return {
            'strategy_id': strategy_id,
            'has_data': True,
            'total_trades_analyzed': latest_metrics.total_trades,
            'avg_shortfall_bps': latest_metrics.avg_shortfall * 10000,
            'execution_quality_score': latest_metrics.quality_score,
            'pct_high_shortfall_trades': latest_metrics.percentage_of_trades_with_high_shortfall * 100,
            'total_cost_impact': latest_metrics.total_cost_due_to_shortfall,
            'recommendation': self._get_recommendation(latest_metrics)
        }

    def _get_recommendation(self, metrics: AggregateShortfallMetrics) -> str:
        """
        Get a recommendation based on the aggregate metrics.
        
        Args:
            metrics: The aggregate metrics to evaluate
            
        Returns:
            Recommendation string
        """
        if metrics.quality_score >= 80:
            return "Execution quality excellent - no changes needed"
        elif metrics.quality_score >= 60:
            return "Execution quality good - monitor for improvements"
        elif metrics.quality_score >= 40:
            return "Execution quality fair - consider execution strategy optimization"
        else:
            return "Execution quality poor - immediate execution strategy optimization required"

    async def get_trades_requiring_execution_review(self, strategy_id: str) -> List[ShortfallReport]:
        """
        Get trades with high implementation shortfall that require execution review.

        Args:
            strategy_id: The strategy ID to check

        Returns:
            List of shortfall reports for trades requiring review
        """
        all_reports = self._shortfall_reports.get(strategy_id, [])

        # Filter for high shortfall trades
        high_shortfall_reports = []
        for report in all_reports:
            pct_shortfall = abs(report.decision_price - report.realized_price) / abs(report.decision_price) if report.decision_price != 0 else 0.0
            self.logger.debug(f"Trade {report.trade_id}: decision={report.decision_price}, realized={report.realized_price}, pct_shortfall={pct_shortfall}, threshold={self.high_shortfall_threshold}")

            if pct_shortfall > self.high_shortfall_threshold:
                high_shortfall_reports.append(report)
                self.logger.debug(f"  -> Marked for review (pct_shortfall {pct_shortfall*10000:.2f} bps > threshold {self.high_shortfall_threshold*10000:.2f} bps)")
            else:
                self.logger.debug(f"  -> Not marked for review (pct_shortfall {pct_shortfall*10000:.2f} bps <= threshold {self.high_shortfall_threshold*10000:.2f} bps)")

        return high_shortfall_reports


class MockExecutionDataProvider:
    """
    Mock data provider for testing purposes.
    """
    
    def __init__(self):
        self._mock_decisions = {}
        self._mock_fills = {}
        
    async def get_trade_decisions(self, strategy_id: str, 
                                  start_time: datetime, 
                                  end_time: datetime) -> List[TradeDecision]:
        if strategy_id not in self._mock_decisions:
            # Create some mock decisions
            decisions = []
            base_time = start_time
            for i in range(20):
                decisions.append(TradeDecision(
                    strategy_id=strategy_id,
                    timestamp=base_time + timedelta(minutes=i*5),
                    symbol=f"SYM{i%3}",
                    side="buy" if i % 2 == 0 else "sell",
                    decision_price=100.0 + np.random.normal(0, 2),
                    quantity=100 + np.random.randint(0, 50),
                    decision_reason="test signal",
                    order_id=f"order_{i}",
                    signal_strength=np.random.uniform(-1, 1),
                    expected_pnl=np.random.uniform(-100, 100)
                ))
            self._mock_decisions[strategy_id] = decisions
            
        return self._mock_decisions[strategy_id]

    async def get_fill_events(self, strategy_id: str, 
                              start_time: datetime, 
                              end_time: datetime) -> List[FillEvent]:
        if strategy_id not in self._mock_fills:
            # Create mock fills that match some decisions with varying shortfall
            fills = []
            decisions = await self.get_trade_decisions(strategy_id, start_time, end_time)
            
            for i, decision in enumerate(decisions):
                # Add some slippage to the fill price
                slippage_factor = np.random.uniform(-0.005, 0.005)  # ±50 bps
                fill_price = decision.decision_price * (1 + slippage_factor)
                
                fills.append(FillEvent(
                    strategy_id=strategy_id,
                    timestamp=decision.timestamp + timedelta(seconds=10),  # Fill slightly after decision
                    symbol=decision.symbol,
                    side=decision.side,
                    fill_price=fill_price,
                    fill_quantity=decision.quantity,
                    order_id=decision.order_id,
                    fees=np.random.uniform(0.1, 1.0),
                    slippage=slippage_factor
                ))
            self._mock_fills[strategy_id] = fills
            
        return self._mock_fills[strategy_id]

    async def get_strategy_symbols(self, strategy_id: str) -> List[str]:
        return [f"SYM{i}" for i in range(3)]


__all__ = [
    "ImplementationShortfallAnalyzer",
    "MockExecutionDataProvider",
    "TradeDecision",
    "FillEvent", 
    "ShortfallReport",
    "AggregateShortfallMetrics",
    "ExecutionDataProvider"
]