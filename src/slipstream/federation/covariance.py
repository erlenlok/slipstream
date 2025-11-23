"""
Covariance Stress Testing for Correlation Risk Management.

This module implements covariance stress testing to prevent hidden beta accumulation
by calculating portfolio correlation to benchmarks and reducing limits when correlations
exceed thresholds, as specified in the federated vision document.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


@dataclass
class StrategyReturn:
    """Represents returns for a single strategy at a specific time."""
    strategy_id: str
    timestamp: datetime
    returns: float
    benchmark_returns: Dict[str, float]  # Symbol -> return (BTC, ETH, etc.)
    portfolio_weight: float  # Current portfolio weight


@dataclass
class PortfolioMetrics:
    """Metrics for a portfolio of strategies."""
    timestamp: datetime
    strategy_returns: Dict[str, float]  # strategy_id -> return
    strategy_weights: Dict[str, float]  # strategy_id -> weight
    portfolio_return: float
    portfolio_volatility: float
    benchmark_correlations: Dict[str, float]  # strategy_id -> correlation to benchmark
    portfolio_benchmark_correlation: float  # Correlation of whole portfolio to benchmark
    diversification_ratio: float  # Portfolio volatility / weighted average of individual volatilities


@dataclass
class CovarianceStressResult:
    """Result of covariance stress testing."""
    timestamp: datetime
    portfolio_correlation_to_btc: float
    portfolio_correlation_to_eth: float
    max_allowed_correlation: float  # Threshold (e.g., 0.8)
    stress_test_passed: bool
    strategies_contributing_most: List[Tuple[str, float]]  # (strategy_id, contribution_score)
    recommended_actions: List[Dict[str, Any]]  # Actions to reduce correlation
    overall_risk_score: float  # 0-100 scale
    correlation_matrix: Optional[np.ndarray] = None


@dataclass
class CovarianceAlert:
    """Alert for high correlation situations."""
    timestamp: datetime
    alert_type: str  # 'HIGH_PORTFOLIO_CORRELATION', 'HIGH_STRATEGY_CORRELATION', etc.
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_strategies: List[str]
    correlation_value: float
    threshold: float
    suggested_action: str  # 'reduce_allocation', 'monitor', etc.


@dataclass
class AllocationAdjustment:
    """Recommended adjustment to strategy allocations based on correlation."""
    strategy_id: str
    current_allocation: float
    recommended_allocation: float
    reason: str  # 'high_correlation', 'diversification_needed', etc.
    adjustment_factor: float  # Multiplier to apply to current allocation


class ReturnsDataProvider(Protocol):
    """
    Protocol for data providers that the covariance tester can use to
    collect strategy return data.
    """
    async def get_strategy_returns(self, strategy_ids: List[str],
                                  start_time: datetime,
                                  end_time: datetime) -> Dict[str, List[StrategyReturn]]:
        """Get returns data for specified strategies."""
        ...

    async def get_benchmark_returns(self, symbols: List[str],
                                   start_time: datetime,
                                   end_time: datetime) -> Dict[str, List[Tuple[datetime, float]]]:
        """Get benchmark returns (BTC, ETH, etc.)."""
        ...

    async def get_current_allocations(self) -> Dict[str, float]:
        """Get current strategy allocations."""
        ...


class CovarianceStressTester:
    """
    Performs covariance stress testing to prevent hidden beta accumulation.
    
    This component calculates portfolio correlation to benchmarks (BTC, ETH) and
    recommends reducing limits when correlations exceed thresholds (e.g., 0.8),
    as specified in the federated vision.
    """
    
    def __init__(self,
                 data_provider: Optional[ReturnsDataProvider] = None,
                 correlation_threshold: float = 0.8,  # Max allowed correlation to benchmark
                 stress_test_interval: timedelta = timedelta(hours=4),
                 min_data_points: int = 10):
        """
        Initialize the covariance stress tester.
        
        Args:
            data_provider: Optional data provider for returns data
            correlation_threshold: Max allowed correlation to benchmarks
            stress_test_interval: How often to run stress tests
            min_data_points: Minimum data points needed for correlation calculation
        """
        self.data_provider = data_provider
        self.correlation_threshold = correlation_threshold
        self.stress_test_interval = stress_test_interval
        self.min_data_points = min_data_points
        self.logger = logging.getLogger(__name__)
        self._running = False
        
        # Store return data for correlation calculations
        self._strategy_returns: Dict[str, List[StrategyReturn]] = {}  # strategy_id -> returns
        self._benchmark_returns: Dict[str, List[Tuple[datetime, float]]] = {}  # benchmark_symbol -> (time, return)
        self._stress_test_history: List[CovarianceStressResult] = []  # Historical test results
        self._alert_history: List[CovarianceAlert] = []  # Historical alerts
        
        # Benchmarks to test against
        self._benchmark_symbols = ["BTC", "ETH"]

    async def start(self):
        """Start the covariance stress tester."""
        self._running = True
        self.logger.info("Covariance Stress Tester started")
        
        if self.data_provider:
            # Initial data loading would happen here in a real system
            pass
        
        self.logger.info("Covariance stress testing started")

    async def stop(self):
        """Stop the covariance stress tester."""
        self._running = False
        self.logger.info("Covariance Stress Tester stopped")

    async def record_strategy_returns(self, return_data: StrategyReturn):
        """Record strategy return data for correlation analysis."""
        strategy_id = return_data.strategy_id
        
        if strategy_id not in self._strategy_returns:
            self._strategy_returns[strategy_id] = []
            
        self._strategy_returns[strategy_id].append(return_data)
        self.logger.debug(f"Recorded returns for {strategy_id}: {return_data.returns:.4f}")

    async def record_benchmark_returns(self, symbol: str, timestamp: datetime, return_value: float):
        """Record benchmark return data."""
        if symbol not in self._benchmark_returns:
            self._benchmark_returns[symbol] = []
            
        self._benchmark_returns[symbol].append((timestamp, return_value))
        self.logger.debug(f"Recorded {symbol} benchmark return: {return_value:.4f}")

    async def calculate_portfolio_metrics(self,
                                        strategy_ids: List[str],
                                        timestamp: datetime) -> Optional[PortfolioMetrics]:
        """
        Calculate portfolio metrics for given strategies at a specific time.
        
        Args:
            strategy_ids: List of strategy IDs in the portfolio
            timestamp: Time for the calculation
            
        Returns:
            Portfolio metrics or None if insufficient data
        """
        if not strategy_ids:
            return None
            
        # Get the most recent returns for each strategy around the given timestamp
        strategy_returns = {}
        strategy_weights = {}
        
        for strategy_id in strategy_ids:
            returns_list = self._strategy_returns.get(strategy_id, [])
            if not returns_list:
                continue
                
            # Find the return closest to the given timestamp
            closest_return = self._find_closest_return(returns_list, timestamp)
            if closest_return:
                strategy_returns[strategy_id] = closest_return.returns
                strategy_weights[strategy_id] = closest_return.portfolio_weight
        
        if not strategy_returns:
            return None
            
        # Calculate portfolio return (weighted sum of strategy returns)
        portfolio_return = sum(
            strategy_returns[sid] * strategy_weights.get(sid, 1.0/len(strategy_returns))
            for sid in strategy_returns
        )
        
        # Calculate correlations to benchmarks
        benchmark_correlations = await self._calculate_benchmark_correlations(strategy_ids, timestamp)
        
        # Calculate portfolio correlation to benchmarks (weighted combination)
        portfolio_btc_corr = 0.0
        portfolio_eth_corr = 0.0
        total_weight = sum(strategy_weights.get(sid, 1.0/len(strategy_returns)) for sid in strategy_returns)
        
        for sid in strategy_returns:
            weight = strategy_weights.get(sid, 1.0/len(strategy_returns))
            btc_corr = benchmark_correlations.get(sid, {}).get('BTC', 0.0)
            eth_corr = benchmark_correlations.get(sid, {}).get('ETH', 0.0)
            portfolio_btc_corr += weight * btc_corr
            portfolio_eth_corr += weight * eth_corr
            
        portfolio_btc_corr /= total_weight if total_weight > 0 else 1.0
        portfolio_eth_corr /= total_weight if total_weight > 0 else 1.0
        
        # Calculate portfolio volatility (simplified as weighted average for now)
        individual_volatilities = [abs(ret) for ret in strategy_returns.values()]
        portfolio_volatility = sum(
            vol * strategy_weights.get(list(strategy_returns.keys())[i], 1.0/len(individual_volatilities))
            for i, vol in enumerate(individual_volatilities)
        )
        
        metrics = PortfolioMetrics(
            timestamp=timestamp,
            strategy_returns=strategy_returns,
            strategy_weights=strategy_weights,
            portfolio_return=portfolio_return,
            portfolio_volatility=portfolio_volatility,
            benchmark_correlations=benchmark_correlations,
            portfolio_benchmark_correlation=abs(portfolio_btc_corr),  # Using BTC as primary benchmark
            diversification_ratio=portfolio_volatility / (sum(individual_volatilities) / len(individual_volatilities)) if individual_volatilities else 1.0
        )
        
        return metrics

    def _find_closest_return(self, returns_list: List[StrategyReturn], target_time: datetime) -> Optional[StrategyReturn]:
        """Find the return data point closest to the target time."""
        if not returns_list:
            return None
            
        closest = None
        min_diff = timedelta.max
        
        for ret in returns_list:
            time_diff = abs(ret.timestamp - target_time)
            if time_diff < min_diff:
                min_diff = time_diff
                closest = ret
                
        # Only return if the time difference is within a reasonable threshold
        if min_diff < timedelta(hours=2):
            return closest
        else:
            return None

    async def _calculate_benchmark_correlations(self, 
                                              strategy_ids: List[str], 
                                              timestamp: datetime) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between strategies and benchmarks."""
        correlations = {}
        
        for strategy_id in strategy_ids:
            strategy_returns_list = self._strategy_returns.get(strategy_id, [])
            if not strategy_returns_list:
                continue
                
            strategy_data = {}
            
            for benchmark_symbol in self._benchmark_symbols:
                benchmark_data_list = self._benchmark_returns.get(benchmark_symbol, [])
                if not benchmark_data_list:
                    continue
                    
                # Align strategy and benchmark returns by timestamp
                aligned_strategy_returns = []
                aligned_benchmark_returns = []
                
                for strat_ret in strategy_returns_list:
                    # Find the closest benchmark return to this time
                    closest_benchmark = self._find_closest_benchmark_return(benchmark_data_list, strat_ret.timestamp)
                    if closest_benchmark:
                        aligned_strategy_returns.append(strat_ret.returns)
                        aligned_benchmark_returns.append(closest_benchmark[1])  # Return value
                
                # Calculate correlation if we have enough aligned data
                if len(aligned_strategy_returns) >= self.min_data_points:
                    try:
                        correlation, _ = pearsonr(aligned_strategy_returns, aligned_benchmark_returns)
                        strategy_data[benchmark_symbol] = float(correlation)
                    except:
                        strategy_data[benchmark_symbol] = 0.0
                else:
                    strategy_data[benchmark_symbol] = 0.0
            
            if strategy_data:
                correlations[strategy_id] = strategy_data
        
        return correlations

    def _find_closest_benchmark_return(self, 
                                     benchmark_list: List[Tuple[datetime, float]], 
                                     target_time: datetime) -> Optional[Tuple[datetime, float]]:
        """Find the benchmark return closest to the target time."""
        if not benchmark_list:
            return None
            
        closest = None
        min_diff = timedelta.max
        
        for timestamp, return_val in benchmark_list:
            time_diff = abs(timestamp - target_time)
            if time_diff < min_diff:
                min_diff = time_diff
                closest = (timestamp, return_val)
                
        # Only return if the time difference is within a reasonable threshold
        if min_diff < timedelta(hours=1):
            return closest
        else:
            return None

    async def run_covariance_stress_test(self, strategy_ids: Optional[List[str]] = None) -> CovarianceStressResult:
        """
        Run covariance stress test to check portfolio correlation to benchmarks.
        
        Args:
            strategy_ids: List of strategy IDs to test (defaults to all tracked)
            
        Returns:
            Stress test results
        """
        if strategy_ids is None:
            strategy_ids = list(set(list(self._strategy_returns.keys())))
        
        if not strategy_ids:
            # Return default result if no strategies
            result = CovarianceStressResult(
                timestamp=datetime.now(),
                portfolio_correlation_to_btc=0.0,
                portfolio_correlation_to_eth=0.0,
                max_allowed_correlation=self.correlation_threshold,
                stress_test_passed=True,
                strategies_contributing_most=[],
                recommended_actions=[],
                overall_risk_score=0.0
            )
            self._stress_test_history.append(result)
            return result
        
        # Calculate current portfolio correlations
        current_time = datetime.now()
        portfolio_metrics = await self.calculate_portfolio_metrics(strategy_ids, current_time)
        
        if not portfolio_metrics:
            # Return default result if no metrics could be calculated
            result = CovarianceStressResult(
                timestamp=current_time,
                portfolio_correlation_to_btc=0.0,
                portfolio_correlation_to_eth=0.0,
                max_allowed_correlation=self.correlation_threshold,
                stress_test_passed=True,
                strategies_contributing_most=[],
                recommended_actions=[],
                overall_risk_score=0.0
            )
            self._stress_test_history.append(result)
            return result
        
        # Extract correlations
        portfolio_btc_corr = abs(portfolio_metrics.portfolio_benchmark_correlation)  # Using BTC as primary
        portfolio_eth_corr = abs(max(portfolio_metrics.benchmark_correlations.get(sid, {}).get('ETH', 0.0) 
                                   for sid in strategy_ids if sid in portfolio_metrics.benchmark_correlations) if strategy_ids else 0.0)
        
        # Determine if test passes
        stress_test_passed = portfolio_btc_corr <= self.correlation_threshold and portfolio_eth_corr <= self.correlation_threshold
        
        # Identify strategies contributing most to correlation
        contributing_strategies = []
        for strategy_id in strategy_ids:
            if strategy_id in portfolio_metrics.benchmark_correlations:
                # Use BTC correlation as primary measure
                btc_corr = portfolio_metrics.benchmark_correlations[strategy_id].get('BTC', 0.0)
                contributing_strategies.append((strategy_id, abs(btc_corr)))
        
        # Sort by contribution (highest first)
        contributing_strategies.sort(key=lambda x: x[1], reverse=True)
        
        # Generate recommended actions
        recommended_actions = []
        if not stress_test_passed:
            # If test failed, recommend reducing allocations for highly correlated strategies
            for strategy_id, corr in contributing_strategies[:3]:  # Top 3 contributors
                if abs(corr) > self.correlation_threshold:
                    recommended_actions.append({
                        'strategy_id': strategy_id,
                        'action': 'reduce_allocation',
                        'reason': f'High correlation ({corr:.3f}) to benchmark',
                        'suggested_reduction': min(0.5, abs(corr) - self.correlation_threshold)  # Reduce by correlation excess
                    })
        
        # Calculate overall risk score (0-100, higher means more risk)
        max_observed_corr = max(portfolio_btc_corr, portfolio_eth_corr)
        risk_score = min(100, max(0, (max_observed_corr - self.correlation_threshold) * 500 if max_observed_corr > self.correlation_threshold else 0))
        
        result = CovarianceStressResult(
            timestamp=current_time,
            portfolio_correlation_to_btc=portfolio_btc_corr,
            portfolio_correlation_to_eth=portfolio_eth_corr,
            max_allowed_correlation=self.correlation_threshold,
            stress_test_passed=stress_test_passed,
            strategies_contributing_most=contributing_strategies,
            recommended_actions=recommended_actions,
            overall_risk_score=risk_score
        )
        
        # Store the result
        self._stress_test_history.append(result)
        
        self.logger.info(f"Covariance stress test: BTC_corr={portfolio_btc_corr:.3f}, "
                        f"ETH_corr={portfolio_eth_corr:.3f}, passed={stress_test_passed}, "
                        f"risk_score={risk_score:.1f}")
        
        return result

    async def generate_correlation_alerts(self) -> List[CovarianceAlert]:
        """
        Generate alerts for high correlation situations.
        
        Returns:
            List of correlation alerts
        """
        alerts = []
        
        if not self._stress_test_history:
            return alerts
        
        latest_test = self._stress_test_history[-1]
        
        # Check for high portfolio correlation to BTC
        if latest_test.portfolio_correlation_to_btc > self.correlation_threshold:
            severity = 'critical' if latest_test.portfolio_correlation_to_btc > 0.9 else 'high'
            alerts.append(CovarianceAlert(
                timestamp=latest_test.timestamp,
                alert_type='HIGH_PORTFOLIO_CORRELATION',
                severity=severity,
                affected_strategies=[s[0] for s in latest_test.strategies_contributing_most[:3]],
                correlation_value=latest_test.portfolio_correlation_to_btc,
                threshold=self.correlation_threshold,
                suggested_action='reduce_allocation'
            ))
        
        # Check for high portfolio correlation to ETH
        if latest_test.portfolio_correlation_to_eth > self.correlation_threshold:
            severity = 'critical' if latest_test.portfolio_correlation_to_eth > 0.9 else 'high'
            alerts.append(CovarianceAlert(
                timestamp=latest_test.timestamp,
                alert_type='HIGH_PORTFOLIO_CORRELATION',
                severity=severity,
                affected_strategies=[s[0] for s in latest_test.strategies_contributing_most[:3]],
                correlation_value=latest_test.portfolio_correlation_to_eth,
                threshold=self.correlation_threshold,
                suggested_action='reduce_allocation'
            ))
        
        # Check individual strategy correlations
        for strategy_id, corr in latest_test.strategies_contributing_most:
            if abs(corr) > self.correlation_threshold:
                severity = 'high' if abs(corr) > 0.9 else 'medium'
                alerts.append(CovarianceAlert(
                    timestamp=latest_test.timestamp,
                    alert_type='HIGH_STRATEGY_CORRELATION',
                    severity=severity,
                    affected_strategies=[strategy_id],
                    correlation_value=abs(corr),
                    threshold=self.correlation_threshold,
                    suggested_action='monitor_or_reduce'
                ))
        
        # Store alerts
        self._alert_history.extend(alerts)
        
        self.logger.info(f"Generated {len(alerts)} correlation alerts")
        return alerts

    async def get_allocation_recommendations(self) -> List[AllocationAdjustment]:
        """
        Get allocation adjustments based on correlation analysis.
        
        Returns:
            List of recommended allocation adjustments
        """
        recommendations = []
        
        # Run stress test to get current state
        stress_result = await self.run_covariance_stress_test()
        
        if stress_result.stress_test_passed:
            # If stress test passes, no adjustments needed
            return []
        
        # If stress test fails, recommend adjustments based on contribution
        for strategy_id, contribution in stress_result.strategies_contributing_most:
            if contribution > self.correlation_threshold:
                # Calculate reduction factor based on how much over the threshold
                excess_ratio = (contribution - self.correlation_threshold) / contribution if contribution > 0 else 0
                reduction_factor = max(0.1, 1.0 - excess_ratio)  # Don't reduce below 10% of current allocation
                
                # Get current allocation if available
                current_allocation = 1.0  # Default to 1.0 if not available
                if self.data_provider:
                    try:
                        allocations = await self.data_provider.get_current_allocations()
                        current_allocation = allocations.get(strategy_id, current_allocation)
                    except:
                        pass
                
                recommended_allocation = current_allocation * reduction_factor
                
                recommendations.append(AllocationAdjustment(
                    strategy_id=strategy_id,
                    current_allocation=current_allocation,
                    recommended_allocation=recommended_allocation,
                    reason='high_correlation_to_benchmark',
                    adjustment_factor=reduction_factor
                ))
        
        self.logger.info(f"Generated {len(recommendations)} allocation recommendations")
        return recommendations

    async def get_portfolio_diversification_metrics(self, strategy_ids: List[str]) -> Dict[str, float]:
        """
        Get diversification metrics for a portfolio of strategies.
        
        Args:
            strategy_ids: List of strategy IDs in portfolio
            
        Returns:
            Dictionary of diversification metrics
        """
        if not strategy_ids or len(strategy_ids) < 2:
            return {
                'concentration_ratio': 1.0 if strategy_ids else 0.0,  # If only one strategy, fully concentrated
                'diversification_ratio': 0.0 if strategy_ids else 1.0,
                'average_correlation': 0.0,
                'correlation_volatility': 0.0
            }
        
        # Calculate correlations between all pairs of strategies
        correlations = []
        n = len(strategy_ids)
        
        for i in range(n):
            for j in range(i+1, n):
                corr = await self._calculate_strategy_to_strategy_correlation(
                    strategy_ids[i], strategy_ids[j]
                )
                if corr is not None:
                    correlations.append(corr)
        
        if not correlations:
            return {
                'concentration_ratio': 1.0,
                'diversification_ratio': 0.0,
                'average_correlation': 0.0,
                'correlation_volatility': 0.0
            }
        
        avg_correlation = np.mean(correlations)
        correlation_volatility = np.std(correlations) if len(correlations) > 1 else 0.0
        
        # Concentration ratio: 0 = perfectly diversified, 1 = fully concentrated
        concentration_ratio = avg_correlation if avg_correlation > 0 else 0.0
        
        # Diversification ratio: 1 = perfectly diversified, 0 = not diversified
        diversification_ratio = 1.0 - abs(avg_correlation)
        
        return {
            'concentration_ratio': concentration_ratio,
            'diversification_ratio': max(0.0, diversification_ratio),
            'average_correlation': avg_correlation,
            'correlation_volatility': correlation_volatility
        }

    async def _calculate_strategy_to_strategy_correlation(self, 
                                                        strategy1: str, 
                                                        strategy2: str) -> Optional[float]:
        """Calculate correlation between two strategies."""
        returns1 = self._strategy_returns.get(strategy1, [])
        returns2 = self._strategy_returns.get(strategy2, [])
        
        if not returns1 or not returns2:
            return None
        
        # Align returns by timestamp
        aligned_returns1 = []
        aligned_returns2 = []
        
        # Use strategy1 as the base, find matching times in strategy2
        for ret1 in returns1:
            closest_ret2 = self._find_closest_return([r for r in returns2 if hasattr(r, 'timestamp')], ret1.timestamp)
            if closest_ret2:
                aligned_returns1.append(ret1.returns)
                aligned_returns2.append(closest_ret2.returns)
        
        if len(aligned_returns1) < self.min_data_points:
            return None
        
        try:
            correlation, _ = pearsonr(aligned_returns1, aligned_returns2)
            return float(correlation)
        except:
            return None

    async def get_stress_test_trends(self, days: int = 7) -> List[CovarianceStressResult]:
        """
        Get stress test trends over recent days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of historical stress test results
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        trends = [r for r in self._stress_test_history if r.timestamp >= cutoff_date]
        
        # Sort by date
        trends.sort(key=lambda x: x.timestamp)
        return trends


class MockReturnsDataProvider:
    """
    Mock data provider for testing purposes.
    """
    
    def __init__(self):
        self._strategy_returns = {}
        self._benchmark_returns = {}
        self._allocations = {}
        
    async def get_strategy_returns(self, strategy_ids: List[str],
                                  start_time: datetime,
                                  end_time: datetime) -> Dict[str, List[StrategyReturn]]:
        for strategy_id in strategy_ids:
            if strategy_id not in self._strategy_returns:
                # Generate mock returns with varying correlation to benchmarks
                returns = []
                current_time = start_time
                base_return = 0.001 if strategy_id.startswith("low_corr") else 0.002
                
                for i in range(20):  # 20 data points
                    # Simulate returns with some correlation to BTC
                    btc_component = 0.6 if strategy_id.startswith("high_corr") else 0.1
                    btc_benchmark = 0.01 * (i % 5 - 2) * 0.1  # Simple benchmark pattern
                    strategy_return = base_return + btc_component * btc_benchmark + np.random.normal(0, 0.005)
                    
                    returns.append(StrategyReturn(
                        strategy_id=strategy_id,
                        timestamp=current_time + timedelta(hours=i*24),
                        returns=strategy_return,
                        benchmark_returns={"BTC": btc_benchmark, "ETH": btc_benchmark * 0.8},
                        portfolio_weight=0.1 + (i * 0.005)  # Varying weights
                    ))
                
                self._strategy_returns[strategy_id] = returns
        
        result = {}
        for strategy_id in strategy_ids:
            result[strategy_id] = [r for r in self._strategy_returns[strategy_id] 
                                 if start_time <= r.timestamp <= end_time]
        return result

    async def get_benchmark_returns(self, symbols: List[str],
                                   start_time: datetime,
                                   end_time: datetime) -> Dict[str, List[Tuple[datetime, float]]]:
        for symbol in symbols:
            if symbol not in self._benchmark_returns:
                # Generate mock benchmark returns
                returns = []
                current_time = start_time
                
                for i in range(20):
                    # BTC returns with some pattern
                    if symbol == "BTC":
                        ret = 0.01 * (i % 7 - 3) * 0.1 + np.random.normal(0, 0.008)
                    elif symbol == "ETH":
                        ret = 0.01 * (i % 7 - 3) * 0.08 + np.random.normal(0, 0.012)  # Similar but different
                    else:
                        ret = np.random.normal(0, 0.01)
                    
                    returns.append((current_time + timedelta(hours=i*24), ret))
                
                self._benchmark_returns[symbol] = returns
        
        result = {}
        for symbol in symbols:
            result[symbol] = [r for r in self._benchmark_returns[symbol] 
                             if start_time <= r[0] <= end_time]
        return result

    async def get_current_allocations(self) -> Dict[str, float]:
        # Return some mock allocations
        return {
            "high_corr_strategy_1": 0.4,
            "low_corr_strategy_1": 0.3,
            "medium_corr_strategy_1": 0.3
        }


__all__ = [
    "CovarianceStressTester",
    "MockReturnsDataProvider",
    "StrategyReturn",
    "PortfolioMetrics",
    "CovarianceStressResult",
    "CovarianceAlert",
    "AllocationAdjustment",
    "ReturnsDataProvider"
]