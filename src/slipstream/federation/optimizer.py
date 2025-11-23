"""
Meta-Optimizer for federated strategy allocation.

This module implements the central allocator that views strategies as statistical assets,
ingests their performance, calculates covariance, and re-optimizes capital allocation
as specified in the federated vision document.
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
class StrategyPerformance:
    """Performance metrics for a single strategy."""
    strategy_id: str
    timestamp: datetime
    returns: float  # Period return
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    total_capital: float
    net_exposure: float
    gross_exposure: float
    win_rate: float
    avg_win: float
    avg_loss: float
    trades: int
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None


@dataclass
class AllocationRequest:
    """Request for capital allocation to strategies."""
    strategy_id: str
    requested_capital: float
    current_capital: float
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]


@dataclass
class AllocationDecision:
    """Decision from the meta-optimizer about capital allocation."""
    strategy_id: str
    allocated_capital: float
    previous_capital: float
    allocation_change: float
    confidence_score: float
    allocation_reason: str
    timestamp: datetime


@dataclass
class PortfolioMetrics:
    """Metrics for the entire portfolio of strategies."""
    total_capital: float
    portfolio_return: float
    portfolio_volatility: float
    portfolio_sharpe: float
    diversification_ratio: float
    correlation_matrix: Optional[np.ndarray] = None
    strategy_weights: Optional[Dict[str, float]] = None


class StrategyDataProvider(Protocol):
    """
    Protocol for data providers that the meta-optimizer can use to
    collect strategy performance data.
    """
    async def get_strategy_performance(self, strategy_id: str, 
                                      start_time: datetime, 
                                      end_time: datetime) -> List[StrategyPerformance]:
        """Get performance data for a strategy over a time period."""
        ...

    async def get_all_strategy_ids(self) -> List[str]:
        """Get all registered strategy IDs."""
        ...

    async def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        """Get current status of a strategy."""
        ...


class MetaOptimizer:
    """
    The central allocator that treats strategies as statistical assets.
    
    This component operates in observation-only mode initially, collecting
    performance data and calculating optimal allocations without controlling
    any existing strategy capital. It implements the "Meta-Optimizer (The Central Bank)"
    layer from the federated vision document.
    """
    
    def __init__(self, 
                 data_provider: Optional[StrategyDataProvider] = None,
                 rebalance_interval: timedelta = timedelta(hours=4)):
        """
        Initialize the Meta-Optimizer.
        
        Args:
            data_provider: Optional data provider for strategy performance
            rebalance_interval: How often to run allocation optimization
        """
        self.data_provider = data_provider
        self.rebalance_interval = rebalance_interval
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._allocation_history: List[AllocationDecision] = []
        self._performance_history: Dict[str, List[StrategyPerformance]] = {}
        self._current_allocations: Dict[str, float] = {}
        self._default_capital_per_strategy: float = 10000.0  # Default starting capital
        
        # For covariance calculation
        self._returns_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Risk management parameters (initially conservative)
        self._max_allocation_per_strategy = 0.3  # Max 30% to any single strategy
        self._min_allocation = 100.0  # Minimum allocation per strategy
        self._correlation_threshold = 0.7  # Threshold for correlation penalties

    async def start(self):
        """Start the meta-optimizer collection and analysis."""
        self._running = True
        self.logger.info("Meta-Optimizer started in observation mode")
        
        # Initial data collection
        if self.data_provider:
            await self._collect_initial_data()
        
        self.logger.info("Meta-Optimizer data collection started")

    async def stop(self):
        """Stop the meta-optimizer."""
        self._running = False
        self.logger.info("Meta-Optimizer stopped")

    async def _collect_initial_data(self):
        """Collect initial performance data for all strategies."""
        if not self.data_provider:
            return
            
        strategy_ids = await self.data_provider.get_all_strategy_ids()
        
        for strategy_id in strategy_ids:
            # Initialize strategy tracking
            if strategy_id not in self._performance_history:
                self._performance_history[strategy_id] = []
                self._returns_history[strategy_id] = []
                self._current_allocations[strategy_id] = self._default_capital_per_strategy
                
            self.logger.info(f"Registered strategy {strategy_id} for monitoring")

    async def collect_performance_data(self, strategy_id: str, performance: StrategyPerformance):
        """
        Collect performance data for a strategy without affecting capital allocation yet.
        
        Args:
            strategy_id: ID of the strategy
            performance: Performance metrics for the strategy
        """
        if strategy_id not in self._performance_history:
            self._performance_history[strategy_id] = []
            
        self._performance_history[strategy_id].append(performance)
        
        # Also track returns for covariance analysis
        if strategy_id not in self._returns_history:
            self._returns_history[strategy_id] = []
        self._returns_history[strategy_id].append((performance.timestamp, performance.returns))
        
        self.logger.debug(f"Collected performance data for {strategy_id}")

    async def get_performance_summary(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance summary for strategies.
        
        Args:
            strategy_id: Optional strategy ID to get summary for specific strategy
            
        Returns:
            Performance summary data
        """
        if strategy_id:
            performances = self._performance_history.get(strategy_id, [])
            if not performances:
                return {}
            
            # Calculate summary metrics
            returns = [p.returns for p in performances]
            volatilities = [p.volatility for p in performances]
            sharpe_ratios = [p.sharpe_ratio for p in performances if p.sharpe_ratio is not None]
            
            return {
                'strategy_id': strategy_id,
                'total_periods': len(returns),
                'avg_return': np.mean(returns) if returns else 0.0,
                'avg_volatility': np.mean(volatilities) if volatilities else 0.0,
                'avg_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0.0,
                'latest_return': returns[-1] if returns else 0.0,
                'latest_volatility': volatilities[-1] if volatilities else 0.0,
                'latest_sharpe': sharpe_ratios[-1] if sharpe_ratios else 0.0,
            }
        else:
            # Return summary for all strategies
            summary = {}
            for strat_id in self._performance_history:
                summary[strat_id] = await self.get_performance_summary(strat_id)
            return summary

    async def calculate_covariance_matrix(self) -> Optional[np.ndarray]:
        """
        Calculate covariance matrix between all strategy returns.
        
        Returns:
            Covariance matrix or None if insufficient data
        """
        if not self._returns_history or len(self._returns_history) < 2:
            return None
            
        # Get common time periods where all strategies have data
        strategy_ids = list(self._returns_history.keys())
        
        # Create aligned return series
        aligned_returns = {}
        common_timestamps = set()
        
        for strategy_id in strategy_ids:
            timestamps_returns = self._returns_history[strategy_id]
            timestamps = [tr[0] for tr in timestamps_returns]
            returns = [tr[1] for tr in timestamps_returns]
            aligned_returns[strategy_id] = (timestamps, returns)
            
            if not common_timestamps:
                common_timestamps = set(timestamps)
            else:
                common_timestamps = common_timestamps.intersection(set(timestamps))
        
        if len(common_timestamps) < 2:
            return None
            
        # Align returns to common timestamps
        return_matrix = []
        valid_strategy_ids = []
        
        sorted_timestamps = sorted(list(common_timestamps))
        
        for strategy_id in strategy_ids:
            timestamps, returns = aligned_returns[strategy_id]
            # Create mapping from timestamp to return
            ts_return_map = dict(zip(timestamps, returns))
            
            # Extract returns for common timestamps
            aligned_strategy_returns = [ts_return_map[ts] for ts in sorted_timestamps if ts in ts_return_map]
            
            if len(aligned_strategy_returns) == len(sorted_timestamps):
                return_matrix.append(aligned_strategy_returns)
                valid_strategy_ids.append(strategy_id)
        
        if len(return_matrix) < 2:
            return None
            
        # Convert to numpy array and calculate covariance
        return_array = np.array(return_matrix)
        covariance_matrix = np.cov(return_array)
        
        self.logger.debug(f"Calculated covariance matrix for {len(valid_strategy_ids)} strategies")
        return covariance_matrix

    async def calculate_diversification_metrics(self) -> Dict[str, Any]:
        """
        Calculate portfolio diversification metrics.
        
        Returns:
            Diversification metrics dictionary
        """
        covariance_matrix = await self.calculate_covariance_matrix()
        
        if covariance_matrix is None or covariance_matrix.shape[0] < 2:
            return {
                'diversification_ratio': 1.0,  # Fully diversified if no correlation data
                'avg_correlation': 0.0,
                'strategy_ids': list(self._returns_history.keys())
            }
        
        # Calculate diversification ratio: total risk / weighted sum of individual risks
        std_devs = np.sqrt(np.diag(covariance_matrix))
        total_risk = np.sqrt(np.sum(covariance_matrix))  # Portfolio risk with equal weights
        weighted_individual_risk = np.mean(std_devs)  # Average individual risk
        
        if weighted_individual_risk > 0:
            diversification_ratio = total_risk / weighted_individual_risk
        else:
            diversification_ratio = 1.0
            
        # Calculate average correlation
        n = covariance_matrix.shape[0]
        if n > 1:
            # Extract correlations (excluding diagonal)
            correlations = []
            for i in range(n):
                for j in range(i+1, n):
                    if std_devs[i] > 0 and std_devs[j] > 0:
                        corr = covariance_matrix[i, j] / (std_devs[i] * std_devs[j])
                        correlations.append(corr)
            avg_correlation = np.mean(correlations) if correlations else 0.0
        else:
            avg_correlation = 0.0
        
        return {
            'diversification_ratio': diversification_ratio,
            'avg_correlation': avg_correlation,
            'strategy_ids': list(self._returns_history.keys()),
            'correlation_matrix': covariance_matrix.tolist() if covariance_matrix is not None else None
        }

    async def run_allocation_analysis(self) -> Dict[str, Any]:
        """
        Run allocation analysis without making actual allocation changes yet.
        
        This method analyzes strategy performance and calculates what allocations
        would be optimal, but doesn't actually implement them.
        
        Returns:
            Analysis results showing potential allocations
        """
        if not self._performance_history:
            return {
                'status': 'no_data',
                'message': 'No performance data available for analysis',
                'suggested_allocations': {},
                'portfolio_metrics': {}
            }
        
        # Calculate performance scores for each strategy
        strategy_scores = {}
        performance_summaries = await self.get_performance_summary()
        
        for strategy_id, summary in performance_summaries.items():
            if not isinstance(summary, dict) or 'avg_sharpe' not in summary:
                continue
                
            # Calculate composite score based on multiple factors
            avg_sharpe = summary.get('avg_sharpe', 0.0)
            avg_return = summary.get('avg_return', 0.0)
            latest_sharpe = summary.get('latest_sharpe', 0.0)
            
            # Basic scoring: weighted combination of metrics
            # In production, this would be much more sophisticated
            score = (
                0.4 * (avg_sharpe if avg_sharpe > 0 else 0) +
                0.3 * (avg_return if avg_return > 0 else 0) +
                0.3 * (latest_sharpe if latest_sharpe > 0 else 0)
            )
            
            strategy_scores[strategy_id] = max(0.0, score)  # Ensure non-negative
        
        # Calculate what optimal allocations would be based on scores
        total_score = sum(strategy_scores.values())
        if total_score == 0:
            # If all scores are 0, distribute evenly
            equal_allocation = 1.0 / len(strategy_scores) if strategy_scores else 0.0
            suggested_allocations = {sid: equal_allocation for sid in strategy_scores}
        else:
            suggested_allocations = {
                sid: score / total_score 
                for sid, score in strategy_scores.items()
            }
        
        # Calculate current allocation percentages
        current_total = sum(self._current_allocations.values())
        if current_total > 0:
            current_percentages = {
                sid: allocation / current_total 
                for sid, allocation in self._current_allocations.items()
            }
        else:
            current_percentages = {}
        
        # Calculate portfolio-level metrics
        portfolio_metrics = await self.calculate_diversification_metrics()
        avg_sharpe = np.mean([s.get('avg_sharpe', 0) for s in performance_summaries.values() if isinstance(s, dict)]) if performance_summaries else 0.0
        
        result = {
            'status': 'analysis_complete',
            'timestamp': datetime.now(),
            'number_of_strategies': len(strategy_scores),
            'suggested_allocations': suggested_allocations,
            'current_allocations': current_percentages,
            'allocation_differences': {
                sid: suggested_allocations.get(sid, 0) - current_percentages.get(sid, 0)
                for sid in set(suggested_allocations.keys()) | set(current_percentages.keys())
            },
            'portfolio_metrics': {
                'total_capital': current_total,
                'avg_portfolio_sharpe': avg_sharpe,
                **portfolio_metrics
            },
            'strategy_analysis': {
                sid: {
                    'performance_score': strategy_scores.get(sid, 0),
                    'suggested_percentage': suggested_allocations.get(sid, 0),
                    'current_percentage': current_percentages.get(sid, 0),
                    'performance_summary': performance_summaries.get(sid, {})
                }
                for sid in strategy_scores.keys()
            }
        }
        
        self.logger.info(f"Allocation analysis completed for {len(strategy_scores)} strategies")
        return result

    async def get_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Get overall portfolio metrics across all strategies.
        
        Returns:
            PortfolioMetrics object with overall portfolio statistics
        """
        if not self._performance_history:
            return PortfolioMetrics(
                total_capital=0.0,
                portfolio_return=0.0,
                portfolio_volatility=0.0,
                portfolio_sharpe=0.0,
                diversification_ratio=1.0
            )
        
        # Calculate total capital
        total_capital = sum(self._current_allocations.values())
        
        # Calculate portfolio return (weighted average of strategy returns)
        if self._performance_history:
            latest_returns = {}
            for sid, perf_list in self._performance_history.items():
                if perf_list:
                    latest_returns[sid] = perf_list[-1].returns
            
            if latest_returns and self._current_allocations:
                # Weighted portfolio return
                weighted_returns = []
                weights = []
                
                for sid in latest_returns:
                    if sid in self._current_allocations:
                        weight = self._current_allocations[sid] / total_capital if total_capital > 0 else 0.0
                        weighted_returns.append(latest_returns[sid] * weight)
                        weights.append(weight)
                
                portfolio_return = sum(weighted_returns) if weighted_returns else 0.0
            else:
                portfolio_return = 0.0
        else:
            portfolio_return = 0.0
        
        # Get diversification metrics
        diversification_metrics = await self.calculate_diversification_metrics()
        
        # Calculate portfolio Sharpe (simplified)
        # In reality, this would be much more sophisticated
        portfolio_volatility = 0.0  # Placeholder
        portfolio_sharpe = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        return PortfolioMetrics(
            total_capital=total_capital,
            portfolio_return=portfolio_return,
            portfolio_volatility=portfolio_volatility,
            portfolio_sharpe=portfolio_sharpe,
            diversification_ratio=diversification_metrics.get('diversification_ratio', 1.0),
            strategy_weights=self._current_allocations
        )

    async def get_strategies_requiring_attention(self) -> List[Dict[str, Any]]:
        """
        Get a list of strategies that require attention based on performance.
        
        Returns:
            List of strategies with issues or opportunities
        """
        attention_list = []
        performance_summaries = await self.get_performance_summary()
        
        for strategy_id, summary in performance_summaries.items():
            if not isinstance(summary, dict):
                continue
                
            issues = []
            
            # Check for poor performance
            avg_sharpe = summary.get('avg_sharpe', 0.0)
            if avg_sharpe < -0.5:
                issues.append(f"Consistently poor Sharpe ratio: {avg_sharpe:.2f}")
            
            # Check for declining performance
            latest_sharpe = summary.get('latest_sharpe', 0.0)
            if avg_sharpe > 0.5 and latest_sharpe < -0.5:
                issues.append(f"Performance degradation: avg {avg_sharpe:.2f} vs latest {latest_sharpe:.2f}")
            
            # Check for high drawdown
            # Note: we don't have drawdown in the summary, but we could add it
            
            if issues:
                attention_list.append({
                    'strategy_id': strategy_id,
                    'issues': issues,
                    'performance_summary': summary,
                    'action': 'review' if avg_sharpe < 0 else 'monitor'
                })
        
        return attention_list


class MockStrategyDataProvider:
    """
    Mock data provider for testing purposes.
    In a real implementation, this would connect to actual strategy data sources.
    """
    
    def __init__(self):
        self._mock_strategies = ["strategy_alpha", "strategy_beta", "strategy_gamma"]
        
    async def get_strategy_performance(self, strategy_id: str, 
                                      start_time: datetime, 
                                      end_time: datetime) -> List[StrategyPerformance]:
        # Return mock performance data
        return [
            StrategyPerformance(
                strategy_id=strategy_id,
                timestamp=datetime.now(),
                returns=np.random.normal(0.001, 0.02),  # Small positive drift
                volatility=np.random.uniform(0.01, 0.05),
                sharpe_ratio=np.random.uniform(-1.0, 2.0),
                max_drawdown=np.random.uniform(-0.1, 0.0),
                total_return=np.random.uniform(-0.2, 0.5),
                total_capital=10000.0,
                net_exposure=np.random.uniform(-5000, 5000),
                gross_exposure=np.random.uniform(2000, 8000),
                win_rate=np.random.uniform(0.4, 0.7),
                avg_win=np.random.uniform(50, 200),
                avg_loss=np.random.uniform(-200, -50),
                trades=np.random.randint(10, 100),
                alpha=np.random.uniform(-0.01, 0.02),
                beta=np.random.uniform(-0.2, 0.2),
                information_ratio=np.random.uniform(-0.5, 1.5)
            )
        ]

    async def get_all_strategy_ids(self) -> List[str]:
        return self._mock_strategies

    async def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        return {
            "status": "active",
            "uptime": np.random.uniform(0.8, 1.0),
            "last_update": datetime.now()
        }


__all__ = [
    "MetaOptimizer",
    "MockStrategyDataProvider",
    "StrategyPerformance",
    "AllocationRequest",
    "AllocationDecision",
    "PortfolioMetrics",
    "StrategyDataProvider"
]