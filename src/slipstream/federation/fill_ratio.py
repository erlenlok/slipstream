"""
Fill Ratio Analysis for Execution Style Verification.

This module implements the tracking of maker vs. taker volume ratios to verify 
that execution style matches strategy intent, as specified in the federated vision.
It flags process failures where execution style doesn't match strategy intent.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd


@dataclass
class TradeExecution:
    """Represents a single trade execution with execution style information."""
    strategy_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_id: str
    fill_type: str  # 'maker' or 'taker'
    fees_paid: float
    slippage: Optional[float] = None
    time_to_fill: Optional[timedelta] = None  # Time from order placement to fill
    order_type: Optional[str] = None  # 'limit', 'market', 'post_only', etc.


@dataclass
class ExecutionStyleProfile:
    """Profile representing the intended execution style of a strategy."""
    strategy_id: str
    intended_style: str  # 'maker_heavy', 'taker_heavy', 'balanced', 'passive', 'aggressive'
    maker_ratio_target: float  # Target ratio of maker fills (0.0 to 1.0)
    taker_ratio_target: float  # Target ratio of taker fills (0.0 to 1.0)
    max_slippage_tolerance: float  # Maximum acceptable slippage
    time_in_force_preference: str  # 'gtc', 'post_only', 'ioc', 'fok', etc.
    notes: str = ""


@dataclass
class FillRatioReport:
    """Report on fill ratios and execution style verification."""
    strategy_id: str
    period_start: datetime
    period_end: datetime
    total_trades: int
    maker_fills: int
    taker_fills: int
    maker_ratio: float  # Actual maker ratio
    taker_ratio: float  # Actual taker ratio
    intended_maker_ratio: float  # Target maker ratio
    deviation_score: float  # How much actual deviates from intended (0-1 scale)
    style_compliance: str  # 'excellent', 'good', 'poor', 'failure'
    process_failure_detected: bool  # Whether execution doesn't match strategy intent
    avg_slippage: float
    median_time_to_fill: Optional[timedelta]
    analysis_details: Dict[str, Any]


@dataclass
class ProcessFailureAlert:
    """Alert for process failures in execution style."""
    strategy_id: str
    timestamp: datetime
    issue: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    current_maker_ratio: float
    target_maker_ratio: float
    action_required: str  # 'monitor', 'investigate', 'fix_execution'


class ExecutionDataProvider(Protocol):
    """
    Protocol for data providers that the analyzer can use to
    collect execution data.
    """
    async def get_executions(self, strategy_id: str,
                            start_time: datetime,
                            end_time: datetime) -> List[TradeExecution]:
        """Get trade executions for a strategy."""
        ...

    async def get_execution_style_profile(self, strategy_id: str) -> Optional[ExecutionStyleProfile]:
        """Get the intended execution style profile for a strategy."""
        ...

    async def get_active_strategies(self) -> List[str]:
        """Get all active strategy IDs to analyze."""
        ...


class FillRatioAnalyzer:
    """
    Analyzes fill ratios to verify execution style matches strategy intent.
    
    This component tracks maker vs. taker volume ratios and flags when execution
    doesn't match the intended strategy style (e.g., trend strategy executing 100% 
    taker orders, which indicates process failure).
    """
    
    def __init__(self,
                 data_provider: Optional[ExecutionDataProvider] = None,
                 process_failure_threshold: float = 0.2,  # 20% deviation triggers alert
                 analysis_interval: timedelta = timedelta(hours=1)):
        """
        Initialize the fill ratio analyzer.
        
        Args:
            data_provider: Optional data provider for execution data
            process_failure_threshold: Deviation threshold to trigger process failure alerts
            analysis_interval: How often to run analysis
        """
        self.data_provider = data_provider
        self.process_failure_threshold = process_failure_threshold
        self.analysis_interval = analysis_interval
        self.logger = logging.getLogger(__name__)
        self._running = False
        
        # Store execution data
        self._executions: Dict[str, List[TradeExecution]] = {}  # strategy_id -> executions
        self._style_profiles: Dict[str, ExecutionStyleProfile] = {}  # strategy_id -> profile
        self._analysis_history: Dict[str, List[FillRatioReport]] = {}  # strategy_id -> reports
        
        # For tracking trends
        self._execution_trends: Dict[str, List[FillRatioReport]] = {}

    async def start(self):
        """Start the fill ratio analyzer."""
        self._running = True
        self.logger.info("Fill Ratio Analyzer started")
        
        if self.data_provider:
            # Initial data loading would happen here in a real system
            pass
        
        self.logger.info("Fill ratio analysis monitoring started")

    async def stop(self):
        """Stop the fill ratio analyzer."""
        self._running = False
        self.logger.info("Fill Ratio Analyzer stopped")

    async def record_execution(self, execution: TradeExecution):
        """Record a trade execution."""
        strategy_id = execution.strategy_id
        
        if strategy_id not in self._executions:
            self._executions[strategy_id] = []
            
        self._executions[strategy_id].append(execution)
        self.logger.debug(f"Recorded execution for {strategy_id}: {execution.fill_type} {execution.quantity}@{execution.price}")

    async def set_execution_style_profile(self, profile: ExecutionStyleProfile):
        """Set the intended execution style profile for a strategy."""
        self._style_profiles[profile.strategy_id] = profile
        self.logger.info(f"Set execution style profile for {profile.strategy_id}: {profile.intended_style}")

    async def analyze_fill_ratios(self,
                                 strategy_id: str,
                                 period_start: Optional[datetime] = None,
                                 period_end: Optional[datetime] = None) -> Optional[FillRatioReport]:
        """
        Analyze fill ratios for a strategy over a period.
        
        Args:
            strategy_id: The strategy to analyze
            period_start: Start of analysis period (defaults to available data start)
            period_end: End of analysis period (defaults to now)
            
        Returns:
            Fill ratio report or None if insufficient data
        """
        # Get execution data for the strategy
        all_executions = self._executions.get(strategy_id, [])
        
        if not all_executions:
            self.logger.warning(f"No execution data available for {strategy_id}")
            return None
            
        # Filter by time period if specified
        if period_start or period_end:
            filtered_executions = []
            for exec in all_executions:
                if period_start and exec.timestamp < period_start:
                    continue
                if period_end and exec.timestamp > period_end:
                    continue
                filtered_executions.append(exec)
        else:
            filtered_executions = all_executions
            
        if not filtered_executions:
            self.logger.warning(f"No execution data in specified period for {strategy_id}")
            return None
            
        # Count maker vs taker fills
        maker_count = sum(1 for exec in filtered_executions if exec.fill_type.lower() == 'maker')
        taker_count = sum(1 for exec in filtered_executions if exec.fill_type.lower() == 'taker')
        total_count = len(filtered_executions)
        
        if total_count == 0:
            return None
            
        # Calculate ratios
        maker_ratio = maker_count / total_count if total_count > 0 else 0
        taker_ratio = taker_count / total_count if total_count > 0 else 0
        
        # Get intended ratios from profile
        profile = self._style_profiles.get(strategy_id)
        if profile:
            intended_maker_ratio = profile.maker_ratio_target
            intended_taker_ratio = profile.taker_ratio_target
        else:
            # Default to 0.5 for both if no profile (balanced)
            intended_maker_ratio = 0.5
            intended_taker_ratio = 0.5
        
        # Calculate deviation score (0-1, where 0 = perfect match, 1 = complete mismatch)
        maker_deviation = abs(maker_ratio - intended_maker_ratio)
        deviation_score = min(1.0, maker_deviation)  # Use maker deviation as primary measure
        
        # Determine style compliance
        if deviation_score < 0.1:
            compliance = 'excellent'
        elif deviation_score < 0.2:
            compliance = 'good'  
        elif deviation_score < 0.3:
            compliance = 'poor'
        else:
            compliance = 'failure'
        
        # Calculate average slippage (where available)
        slippage_values = [e.slippage for e in filtered_executions if e.slippage is not None]
        avg_slippage = np.mean(slippage_values) if slippage_values else 0.0
        
        # Calculate median time to fill (where available)
        time_to_fill_values = [e.time_to_fill.total_seconds() for e in filtered_executions 
                              if e.time_to_fill is not None]
        median_time_to_fill = timedelta(seconds=np.median(time_to_fill_values)) if time_to_fill_values else None
        
        # Determine if process failure is detected
        process_failure_detected = deviation_score > self.process_failure_threshold
        
        # Create analysis details
        analysis_details = {
            'maker_count': maker_count,
            'taker_count': taker_count,
            'total_count': total_count,
            'maker_ratio': maker_ratio,
            'taker_ratio': taker_ratio,
            'intended_maker_ratio': intended_maker_ratio,
            'intended_taker_ratio': intended_taker_ratio,
            'deviation_score': deviation_score,
            'avg_slippage': avg_slippage,
            'median_time_to_fill_sec': median_time_to_fill.total_seconds() if median_time_to_fill else None,
            'profile_exists': profile is not None,
            'profile_style': profile.intended_style if profile else 'unknown'
        }
        
        report = FillRatioReport(
            strategy_id=strategy_id,
            period_start=period_start or min(e.timestamp for e in filtered_executions),
            period_end=period_end or max(e.timestamp for e in filtered_executions),
            total_trades=total_count,
            maker_fills=maker_count,
            taker_fills=taker_count,
            maker_ratio=maker_ratio,
            taker_ratio=taker_ratio,
            intended_maker_ratio=intended_maker_ratio,
            deviation_score=deviation_score,
            style_compliance=compliance,
            process_failure_detected=process_failure_detected,
            avg_slippage=avg_slippage,
            median_time_to_fill=median_time_to_fill,
            analysis_details=analysis_details
        )
        
        # Store the report
        if strategy_id not in self._analysis_history:
            self._analysis_history[strategy_id] = []
        self._analysis_history[strategy_id].append(report)
        
        # Add to trends
        if strategy_id not in self._execution_trends:
            self._execution_trends[strategy_id] = []
        self._execution_trends[strategy_id].append(report)
        
        self.logger.info(f"Fill ratio analysis for {strategy_id}: "
                        f"maker={maker_ratio:.2%}, intended={intended_maker_ratio:.2%}, "
                        f"deviation={deviation_score:.2f}, "
                        f"compliance={compliance}, "
                        f"failure={process_failure_detected}")
        
        return report

    async def get_strategy_execution_analysis(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get execution analysis for a strategy.
        
        Args:
            strategy_id: The strategy to analyze
            
        Returns:
            Execution analysis information
        """
        if strategy_id not in self._analysis_history:
            # Try to run analysis if not available
            result = await self.analyze_fill_ratios(strategy_id)
            if not result:
                return {
                    'strategy_id': strategy_id,
                    'message': 'Insufficient data for analysis',
                    'has_data': False
                }
        
        # Get the most recent analysis
        analysis_list = self._analysis_history.get(strategy_id, [])
        if not analysis_list:
            return {
                'strategy_id': strategy_id,
                'message': 'No analysis available',
                'has_data': False
            }
        
        latest_analysis = analysis_list[-1]
        
        return {
            'strategy_id': strategy_id,
            'has_data': True,
            'analysis_period': {
                'start': latest_analysis.period_start,
                'end': latest_analysis.period_end
            },
            'total_trades': latest_analysis.total_trades,
            'maker_fills': latest_analysis.maker_fills,
            'taker_fills': latest_analysis.taker_fills,
            'maker_ratio': latest_analysis.maker_ratio,
            'taker_ratio': latest_analysis.taker_ratio,
            'intended_maker_ratio': latest_analysis.intended_maker_ratio,
            'deviation_score': latest_analysis.deviation_score,
            'style_compliance': latest_analysis.style_compliance,
            'process_failure_detected': latest_analysis.process_failure_detected,
            'avg_slippage': latest_analysis.avg_slippage,
            'median_time_to_fill': latest_analysis.median_time_to_fill,
            'recommendation': self._get_recommendation(latest_analysis)
        }

    def _get_recommendation(self, analysis: FillRatioReport) -> str:
        """
        Get a recommendation based on the analysis results.
        
        Args:
            analysis: The analysis result to evaluate
            
        Returns:
            Recommendation string
        """
        if analysis.process_failure_detected:
            return "PROCESS FAILURE: Execution style significantly deviates from intended strategy, immediate investigation required"
        elif analysis.style_compliance == 'failure':
            return "Poor execution style compliance, investigate execution logic"
        elif analysis.style_compliance == 'poor':
            return "Execution style needs improvement, review execution parameters"
        elif analysis.style_compliance == 'good':
            return "Good execution style compliance, continue monitoring"
        else:
            return "Excellent execution style compliance, optimal performance"

    async def detect_process_failures(self) -> List[ProcessFailureAlert]:
        """
        Detect strategies with process failures in execution style.
        
        Returns:
            List of process failure alerts
        """
        alerts = []
        
        for strategy_id, analysis_list in self._analysis_history.items():
            if not analysis_list:
                continue
                
            latest_analysis = analysis_list[-1]
            
            if latest_analysis.process_failure_detected:
                # Determine severity based on deviation magnitude
                if latest_analysis.deviation_score > 0.5:
                    severity = 'critical'
                elif latest_analysis.deviation_score > 0.3:
                    severity = 'high'
                elif latest_analysis.deviation_score > 0.2:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                alert = ProcessFailureAlert(
                    strategy_id=strategy_id,
                    timestamp=latest_analysis.period_end,
                    issue=f"Execution style deviates significantly from intended: "
                          f"actual maker ratio {latest_analysis.maker_ratio:.2%} vs "
                          f"intended {latest_analysis.intended_maker_ratio:.2%}",
                    severity=severity,
                    current_maker_ratio=latest_analysis.maker_ratio,
                    target_maker_ratio=latest_analysis.intended_maker_ratio,
                    action_required='fix_execution'
                )
                alerts.append(alert)
        
        self.logger.info(f"Detected {len(alerts)} process failure alerts")
        return alerts

    async def get_execution_trends(self, strategy_id: str, days: int = 7) -> List[FillRatioReport]:
        """
        Get execution trends for a strategy over recent days.
        
        Args:
            strategy_id: The strategy to analyze
            days: Number of days to look back
            
        Returns:
            List of recent analysis reports
        """
        if strategy_id not in self._execution_trends:
            return []
            
        cutoff_date = datetime.now() - timedelta(days=days)
        trends = [r for r in self._execution_trends[strategy_id] if r.period_end >= cutoff_date]
        
        # Sort by date
        trends.sort(key=lambda x: x.period_end)
        return trends

    async def calculate_execution_efficiency_metrics(self, strategy_id: str) -> Dict[str, float]:
        """
        Calculate execution efficiency metrics for a strategy.
        
        Args:
            strategy_id: The strategy to evaluate
            
        Returns:
            Dictionary of efficiency metrics
        """
        if strategy_id not in self._executions:
            return {}
            
        executions = self._executions[strategy_id]
        if not executions:
            return {}
        
        # Calculate various efficiency metrics
        maker_ratio = sum(1 for e in executions if e.fill_type == 'maker') / len(executions)
        
        # Average fees paid for makers vs takers
        maker_fees = [e.fees_paid for e in executions if e.fill_type == 'maker']
        taker_fees = [e.fees_paid for e in executions if e.fill_type == 'taker']
        
        avg_maker_fees = np.mean(maker_fees) if maker_fees else 0.0
        avg_taker_fees = np.mean(taker_fees) if taker_fees else 0.0
        
        # Slippage analysis where available
        slippage_data = [e.slippage for e in executions if e.slippage is not None]
        avg_slippage = np.mean(slippage_data) if slippage_data else 0.0
        
        # Time to fill where available
        time_to_fill_data = [e.time_to_fill.total_seconds() for e in executions 
                            if e.time_to_fill is not None]
        avg_time_to_fill = np.mean(time_to_fill_data) if time_to_fill_data else 0.0
        median_time_to_fill = np.median(time_to_fill_data) if time_to_fill_data else 0.0
        
        return {
            'maker_ratio': maker_ratio,
            'avg_maker_fees': avg_maker_fees,
            'avg_taker_fees': avg_taker_fees,
            'avg_slippage': avg_slippage,
            'avg_time_to_fill': avg_time_to_fill,
            'median_time_to_fill': median_time_to_fill,
            'total_executions': len(executions)
        }


class MockExecutionDataProvider:
    """
    Mock data provider for testing purposes.
    """
    
    def __init__(self):
        self._executions = {}
        self._profiles = {}
        
    async def get_executions(self, strategy_id: str,
                            start_time: datetime,
                            end_time: datetime) -> List[TradeExecution]:
        if strategy_id not in self._executions:
            # Generate mock execution data
            executions = []
            base_time = start_time
            for i in range(20):
                # Vary execution type based on strategy
                if strategy_id.startswith("maker"):
                    fill_type = "maker" if i % 3 != 0 else "taker"  # 2/3rds maker
                elif strategy_id.startswith("taker"):
                    fill_type = "taker" if i % 3 != 0 else "maker"  # 2/3rds taker
                else:
                    fill_type = "maker" if i % 2 == 0 else "taker"  # 50/50
                
                executions.append(TradeExecution(
                    strategy_id=strategy_id,
                    timestamp=base_time + timedelta(minutes=i*10),
                    symbol="BTC",
                    side="buy" if i % 2 == 0 else "sell",
                    quantity=0.1 + (i * 0.01),
                    price=45000.0 + np.random.uniform(-100, 100),
                    order_id=f"exec_{i}",
                    fill_type=fill_type,
                    fees_paid=np.random.uniform(0.1, 1.0),
                    slippage=np.random.uniform(-0.001, 0.002) if fill_type == "taker" else 0,
                    time_to_fill=timedelta(seconds=np.random.uniform(0.1, 5)),
                    order_type="limit"
                ))
            self._executions[strategy_id] = executions
        
        # Filter by time period
        filtered = [e for e in self._executions[strategy_id] 
                   if start_time <= e.timestamp <= end_time]
        return filtered

    async def get_execution_style_profile(self, strategy_id: str) -> Optional[ExecutionStyleProfile]:
        if strategy_id not in self._profiles:
            # Create profile based on strategy name pattern
            if strategy_id.startswith("maker"):
                profile = ExecutionStyleProfile(
                    strategy_id=strategy_id,
                    intended_style="maker_heavy",
                    maker_ratio_target=0.7,
                    taker_ratio_target=0.3,
                    max_slippage_tolerance=0.001,
                    time_in_force_preference="post_only"
                )
            elif strategy_id.startswith("taker"):
                profile = ExecutionStyleProfile(
                    strategy_id=strategy_id,
                    intended_style="taker_heavy", 
                    maker_ratio_target=0.3,
                    taker_ratio_target=0.7,
                    max_slippage_tolerance=0.005,
                    time_in_force_preference="ioc"
                )
            else:
                profile = ExecutionStyleProfile(
                    strategy_id=strategy_id,
                    intended_style="balanced",
                    maker_ratio_target=0.5,
                    taker_ratio_target=0.5,
                    max_slippage_tolerance=0.002,
                    time_in_force_preference="gtc"
                )
            self._profiles[strategy_id] = profile
            
        return self._profiles[strategy_id]

    async def get_active_strategies(self) -> List[str]:
        return ["maker_strategy_1", "taker_strategy_1", "balanced_strategy_1"]


__all__ = [
    "FillRatioAnalyzer",
    "MockExecutionDataProvider",
    "TradeExecution",
    "ExecutionStyleProfile",
    "FillRatioReport",
    "ProcessFailureAlert",
    "ExecutionDataProvider"
]