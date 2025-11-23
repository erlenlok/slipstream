"""
Residual Return Analysis for Alpha/Beta Separation.

This module implements the post-trade analysis capability to separate Beta PnL 
(market drift) from Alpha PnL (skill) by regressing daily strategy returns 
against market benchmarks (BTC, ETH), as specified in the federated vision.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


@dataclass
class MarketBenchmark:
    """Market benchmark data for a specific symbol."""
    symbol: str
    timestamp: datetime
    price: float
    returns: float  # Period return


@dataclass
class StrategyPerformanceSnapshot:
    """Performance snapshot for a strategy at a specific time."""
    strategy_id: str
    timestamp: datetime
    returns: float
    benchmark_returns: Dict[str, float]  # Symbol -> return
    total_capital: float
    net_exposure: float
    strategy_pnl: float
    alpha_component: Optional[float] = None
    beta_component: Optional[float] = None


@dataclass
class ResidualAnalysisResult:
    """Result of residual return analysis."""
    strategy_id: str
    period_start: datetime
    period_end: datetime
    alpha_pnl: float  # Skill-based PnL
    beta_pnl: float   # Market-driven PnL
    total_pnl: float
    alpha_percentage: float  # Percentage of total PnL from alpha
    beta_percentage: float   # Percentage of total PnL from beta
    r_squared: float  # R² of the regression
    alpha_significance: float  # Statistical significance of alpha
    market_exposure: Dict[str, float]  # Beta coefficients for each benchmark
    analysis_quality: str  # 'high', 'medium', 'low', 'unreliable'
    fake_alpha_detected: bool  # Whether alpha is likely fake


@dataclass
class FakeAlphaAlert:
    """Alert for strategies detected to have fake alpha."""
    strategy_id: str
    timestamp: datetime
    reason: str
    action_required: str  # 'monitor', 'reduce', 'retire'
    alpha_pnl_ratio: float  # Alpha PnL as ratio of total PnL


class BenchmarkDataProvider(Protocol):
    """
    Protocol for data providers that the analyzer can use to
    collect market benchmark data.
    """
    async def get_benchmark_returns(self, symbols: List[str], 
                                   start_time: datetime, 
                                   end_time: datetime) -> Dict[str, List[MarketBenchmark]]:
        """Get market benchmark returns for specified symbols."""
        ...

    async def get_strategy_performance(self, strategy_id: str,
                                      start_time: datetime,
                                      end_time: datetime) -> List[StrategyPerformanceSnapshot]:
        """Get strategy performance data."""
        ...

    async def get_active_strategies(self) -> List[str]:
        """Get all active strategy IDs to analyze."""
        ...


class ResidualReturnAnalyzer:
    """
    Analyzes residual returns to separate Beta PnL (market drift) from Alpha PnL (skill).
    
    This component regresses daily strategy returns against market benchmarks (BTC, ETH)
    to identify skill-based vs. market-driven performance, enabling detection of fake alpha.
    """
    
    def __init__(self,
                 data_provider: Optional[BenchmarkDataProvider] = None,
                 benchmark_symbols: Optional[List[str]] = None,
                 minimum_data_points: int = 10,
                 alpha_significance_threshold: float = 0.1,
                 r_squared_significance_threshold: float = 0.1):
        """
        Initialize the residual return analyzer.
        
        Args:
            data_provider: Optional data provider for benchmark and strategy data
            benchmark_symbols: List of benchmark symbols (defaults to BTC, ETH)
            minimum_data_points: Minimum points needed for reliable analysis
            alpha_significance_threshold: Threshold for alpha statistical significance
            r_squared_significance_threshold: Minimum R² for reliable regression
        """
        self.data_provider = data_provider
        self.benchmark_symbols = benchmark_symbols or ["BTC", "ETH"]
        self.minimum_data_points = minimum_data_points
        self.alpha_significance_threshold = alpha_significance_threshold
        self.r_squared_significance_threshold = r_squared_significance_threshold
        self.logger = logging.getLogger(__name__)
        self._running = False
        
        # Store historical data for analysis
        self._benchmark_data: Dict[str, List[MarketBenchmark]] = {}  # symbol -> benchmarks
        self._strategy_performance: Dict[str, List[StrategyPerformanceSnapshot]] = {}  # strategy_id -> performance
        self._analysis_history: Dict[str, List[ResidualAnalysisResult]] = {}  # strategy_id -> analysis results
        
        # Fake alpha detection thresholds
        self._fake_alpha_alpha_threshold = 0.05  # If alpha component < 5% of total, likely fake

    async def start(self):
        """Start the residual return analyzer."""
        self._running = True
        self.logger.info("Residual Return Analyzer started")
        
        if self.data_provider:
            # Initial data loading would happen here in a real system
            pass
        
        self.logger.info("Residual return analysis monitoring started")

    async def stop(self):
        """Stop the residual return analyzer."""
        self._running = False
        self.logger.info("Residual Return Analyzer stopped")

    async def record_benchmark_data(self, benchmark: MarketBenchmark):
        """Record market benchmark data."""
        symbol = benchmark.symbol
        
        if symbol not in self._benchmark_data:
            self._benchmark_data[symbol] = []
            
        self._benchmark_data[symbol].append(benchmark)
        self.logger.debug(f"Recorded benchmark data for {symbol}: {benchmark.returns:.4f}")

    async def record_strategy_performance(self, performance: StrategyPerformanceSnapshot):
        """Record strategy performance data."""
        strategy_id = performance.strategy_id
        
        if strategy_id not in self._strategy_performance:
            self._strategy_performance[strategy_id] = []
            
        self._strategy_performance[strategy_id].append(performance)
        self.logger.debug(f"Recorded performance for {strategy_id}: {performance.returns:.4f}")

    async def perform_residual_analysis(self, 
                                      strategy_id: str,
                                      period_start: Optional[datetime] = None,
                                      period_end: Optional[datetime] = None) -> Optional[ResidualAnalysisResult]:
        """
        Perform residual return analysis for a strategy over a period.
        
        Args:
            strategy_id: The strategy to analyze
            period_start: Start of analysis period (defaults to available data start)
            period_end: End of analysis period (defaults to now)
            
        Returns:
            Analysis result or None if insufficient data
        """
        # Get strategy performance data
        all_performance = self._strategy_performance.get(strategy_id, [])
        
        if not all_performance:
            self.logger.warning(f"No performance data available for {strategy_id}")
            return None
            
        # Filter by time period if specified
        if period_start or period_end:
            filtered_performance = []
            for perf in all_performance:
                if period_start and perf.timestamp < period_start:
                    continue
                if period_end and perf.timestamp > period_end:
                    continue
                filtered_performance.append(perf)
        else:
            filtered_performance = all_performance
            
        if len(filtered_performance) < self.minimum_data_points:
            self.logger.warning(f"Insufficient data for {strategy_id}: {len(filtered_performance)} points (min {self.minimum_data_points})")
            return None
            
        # Prepare benchmark data for regression
        # Align strategy and benchmark timestamps
        strategy_returns = []
        benchmark_returns_matrix = []  # Each row is [BTC_return, ETH_return, ...]
        
        for perf in filtered_performance:
            # Find corresponding benchmark returns
            benchmark_row = []
            for symbol in self.benchmark_symbols:
                # Find the benchmark return closest to this timestamp
                closest_benchmark = self._find_closest_benchmark(symbol, perf.timestamp)
                if closest_benchmark:
                    benchmark_row.append(closest_benchmark.returns)
                else:
                    # If no benchmark data, use 0 as default
                    benchmark_row.append(0.0)
            
            if len(benchmark_row) == len(self.benchmark_symbols):
                strategy_returns.append(perf.returns)
                benchmark_returns_matrix.append(benchmark_row)
        
        if len(strategy_returns) < self.minimum_data_points:
            self.logger.warning(f"Insufficient aligned data for {strategy_id}: {len(strategy_returns)} points")
            return None
        
        # Perform regression: strategy_returns = alpha + beta * benchmark_returns + epsilon
        X = np.array(benchmark_returns_matrix)  # Market factors
        y = np.array(strategy_returns)          # Strategy returns
        
        try:
            # Fit linear regression model
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)
            
            # Predict and calculate R²
            y_pred = model.predict(X)
            r_squared = r2_score(y, y_pred)
            
            # Extract coefficients (betas for each benchmark)
            betas = model.coef_
            alpha_intercept = model.intercept_
            
            # Calculate alpha and beta components
            # For each period: total_return = alpha + sum(betas * benchmark_returns) + residuals
            predicted_returns = model.predict(X)
            alpha_component = alpha_intercept  # Average skill component
            beta_components = np.mean(X * betas, axis=0)  # Average market exposure
            
            # Estimate PnL components based on average returns
            avg_strategy_return = np.mean(y)
            avg_predicted_return = np.mean(predicted_returns)
            avg_residual = avg_strategy_return - avg_predicted_return  # Pure alpha component after regression
            
            # Calculate total PnL based on average returns (simplified)
            # In practice, you'd use actual PnL data
            total_pnl = avg_strategy_return * perf.total_capital if hasattr(perf, 'total_capital') and perf.total_capital else avg_strategy_return
            beta_pnl = avg_predicted_return * perf.total_capital if hasattr(perf, 'total_capital') and perf.total_capital else avg_predicted_return
            alpha_pnl = avg_residual * perf.total_capital if hasattr(perf, 'total_capital') and perf.total_capital else avg_residual
            
            # Calculate percentages
            if abs(total_pnl) > 1e-8:  # Avoid division by zero
                alpha_percentage = alpha_pnl / total_pnl
                beta_percentage = beta_pnl / total_pnl
            else:
                alpha_percentage = 0.0
                beta_percentage = 0.0
            
            # Determine analysis quality based on R² and data quality
            if r_squared >= 0.3:
                quality = 'high'
            elif r_squared >= 0.1:
                quality = 'medium'
            elif r_squared >= 0.01:
                quality = 'low'
            else:
                quality = 'unreliable'
            
            # Check for fake alpha (alpha close to zero while showing profit from market)
            fake_alpha_detected = (
                abs(alpha_pnl / total_pnl) < self._fake_alpha_alpha_threshold
                and total_pnl > 0  # Strategy is profitable
                and abs(beta_pnl / total_pnl) > 0.8  # Most of profit is from market
            ) if abs(total_pnl) > 1e-8 else False
            
            # Calculate alpha significance (simplified)
            alpha_significance = abs(avg_residual) / (np.std(y) + 1e-8) if len(y) > 1 else 0.0
            
            # Create market exposure dictionary
            market_exposure = dict(zip(self.benchmark_symbols, betas))
            
            result = ResidualAnalysisResult(
                strategy_id=strategy_id,
                period_start=period_start or min(p.timestamp for p in filtered_performance),
                period_end=period_end or max(p.timestamp for p in filtered_performance),
                alpha_pnl=alpha_pnl,
                beta_pnl=beta_pnl,
                total_pnl=total_pnl,
                alpha_percentage=alpha_percentage,
                beta_percentage=beta_percentage,
                r_squared=r_squared,
                alpha_significance=alpha_significance,
                market_exposure=market_exposure,
                analysis_quality=quality,
                fake_alpha_detected=fake_alpha_detected
            )
            
            # Store the result
            if strategy_id not in self._analysis_history:
                self._analysis_history[strategy_id] = []
            self._analysis_history[strategy_id].append(result)
            
            self.logger.info(f"Residual analysis for {strategy_id}: alpha={alpha_pnl:.4f}, beta={beta_pnl:.4f}, R²={r_squared:.3f}, fake_alpha={fake_alpha_detected}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in residual analysis for {strategy_id}: {e}")
            return None

    def _find_closest_benchmark(self, symbol: str, target_time: datetime) -> Optional[MarketBenchmark]:
        """Find the benchmark data point closest to the target time."""
        benchmarks = self._benchmark_data.get(symbol, [])
        if not benchmarks:
            return None
            
        # Find the benchmark with timestamp closest to target_time
        closest = None
        min_diff = timedelta.max
        
        for benchmark in benchmarks:
            time_diff = abs(benchmark.timestamp - target_time)
            if time_diff < min_diff:
                min_diff = time_diff
                closest = benchmark
                
        # Only return if the time difference is within a reasonable threshold (e.g., 1 hour)
        if min_diff < timedelta(hours=1):
            return closest
        else:
            return None

    async def get_strategy_alpha_beta_breakdown(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get the alpha/beta breakdown for a strategy.
        
        Args:
            strategy_id: The strategy to analyze
            
        Returns:
            Alpha/beta breakdown information
        """
        if strategy_id not in self._analysis_history:
            # Try to run analysis if not available
            result = await self.perform_residual_analysis(strategy_id)
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
            'total_pnl': latest_analysis.total_pnl,
            'alpha_pnl': latest_analysis.alpha_pnl,
            'beta_pnl': latest_analysis.beta_pnl,
            'alpha_percentage': latest_analysis.alpha_percentage,
            'beta_percentage': latest_analysis.beta_percentage,
            'r_squared': latest_analysis.r_squared,
            'analysis_quality': latest_analysis.analysis_quality,
            'market_exposure': latest_analysis.market_exposure,
            'fake_alpha_detected': latest_analysis.fake_alpha_detected,
            'recommendation': self._get_recommendation(latest_analysis)
        }

    def _get_recommendation(self, analysis: ResidualAnalysisResult) -> str:
        """
        Get a recommendation based on the analysis results.
        
        Args:
            analysis: The analysis result to evaluate
            
        Returns:
            Recommendation string
        """
        if analysis.fake_alpha_detected:
            return "FAKE ALPHA DETECTED: Strategy likely just leveraged market exposure, consider retirement"
        elif analysis.alpha_pnl > 0 and analysis.alpha_significance > self.alpha_significance_threshold:
            return "Strong alpha detected, continue monitoring"
        elif analysis.alpha_pnl > 0 and abs(analysis.alpha_pnl) > abs(analysis.beta_pnl):
            return "Primarily alpha-driven, good strategy to continue"
        elif analysis.beta_pnl > 0 and abs(analysis.beta_pnl) > abs(analysis.alpha_pnl):
            return "Primarily beta-driven, monitor for market dependency risk"
        else:
            return "Mixed or negative performance, evaluate strategy effectiveness"

    async def detect_fake_alpha_strategies(self) -> List[FakeAlphaAlert]:
        """
        Detect strategies that are likely generating fake alpha (leveraged market exposure).
        
        Returns:
            List of fake alpha alerts
        """
        alerts = []
        
        for strategy_id, analysis_list in self._analysis_history.items():
            if not analysis_list:
                continue
                
            latest_analysis = analysis_list[-1]
            
            if latest_analysis.fake_alpha_detected:
                alert = FakeAlphaAlert(
                    strategy_id=strategy_id,
                    timestamp=latest_analysis.period_end,
                    reason=f"Alpha component ({latest_analysis.alpha_pnl:.4f}) is {abs(latest_analysis.alpha_percentage*100):.1f}% of total PnL, dominated by market exposure ({latest_analysis.beta_pnl:.4f})",
                    action_required='retire' if latest_analysis.total_pnl > 0 else 'monitor',
                    alpha_pnl_ratio=latest_analysis.alpha_pnl / latest_analysis.total_pnl if latest_analysis.total_pnl != 0 else 0
                )
                alerts.append(alert)
                
        self.logger.info(f"Detected {len(alerts)} strategies with potential fake alpha")
        return alerts

    async def get_strategies_for_retirement_review(self) -> List[Dict[str, Any]]:
        """
        Get strategies that should be considered for retirement based on fake alpha detection.
        
        Returns:
            List of strategies with retirement considerations
        """
        retirement_candidates = []
        
        for strategy_id, analysis_list in self._analysis_history.items():
            if not analysis_list:
                continue
                
            latest_analysis = analysis_list[-1]
            
            # Consider for retirement if fake alpha detected OR alpha is consistently negative/very small
            should_review = (
                latest_analysis.fake_alpha_detected or
                (latest_analysis.alpha_pnl < 0 and abs(latest_analysis.alpha_pnl) > 0.5) or
                (latest_analysis.alpha_pnl > 0 and latest_analysis.alpha_pnl / latest_analysis.total_pnl < 0.1 and latest_analysis.r_squared > 0.5)
            )
            
            if should_review:
                retirement_candidates.append({
                    'strategy_id': strategy_id,
                    'alpha_pnl': latest_analysis.alpha_pnl,
                    'total_pnl': latest_analysis.total_pnl,
                    'alpha_percentage': latest_analysis.alpha_percentage,
                    'r_squared': latest_analysis.r_squared,
                    'is_fake_alpha': latest_analysis.fake_alpha_detected,
                    'market_exposure': latest_analysis.market_exposure,
                    'review_reason': 'fake_alpha' if latest_analysis.fake_alpha_detected else 'low_alpha_significance'
                })
        
        return retirement_candidates


class MockBenchmarkDataProvider:
    """
    Mock data provider for testing purposes.
    """
    
    def __init__(self):
        self._mock_benchmarks = {}
        self._mock_performance = {}
        
    async def get_benchmark_returns(self, symbols: List[str], 
                                   start_time: datetime, 
                                   end_time: datetime) -> Dict[str, List[MarketBenchmark]]:
        # Create mock benchmark data
        for symbol in symbols:
            if symbol not in self._mock_benchmarks:
                data = []
                current_time = start_time
                base_price = 40000.0 if symbol == "BTC" else 3000.0
                
                for i in range(30):  # 30 days of data
                    # Generate realistic price movements
                    daily_return = np.random.normal(0.0005, 0.02)  # Small drift, 2% volatility
                    new_price = base_price * (1 + daily_return)
                    data.append(MarketBenchmark(
                        symbol=symbol,
                        timestamp=current_time + timedelta(days=i),
                        price=new_price,
                        returns=daily_return
                    ))
                    base_price = new_price
                    
                self._mock_benchmarks[symbol] = data
        
        result = {}
        for symbol in symbols:
            result[symbol] = [b for b in self._mock_benchmarks[symbol] 
                             if start_time <= b.timestamp <= end_time]
        return result

    async def get_strategy_performance(self, strategy_id: str,
                                      start_time: datetime,
                                      end_time: datetime) -> List[StrategyPerformanceSnapshot]:
        # Create mock strategy performance that varies by type
        if strategy_id.startswith("alpha"):
            # Alpha strategy: mostly skill-based returns
            base_returns = [np.random.normal(0.001, 0.01) for _ in range(30)]
        elif strategy_id.startswith("beta"):
            # Beta strategy: market-correlated returns
            btc_benchmarks = await self.get_benchmark_returns(["BTC"], start_time, end_time)
            base_returns = [b.returns * 1.2 + np.random.normal(0, 0.005) for b in btc_benchmarks.get("BTC", [])[:30]]
        else:
            # Mixed strategy
            base_returns = [np.random.normal(0.0005, 0.015) for _ in range(30)]
            
        data = []
        current_time = start_time
        for i, ret in enumerate(base_returns):
            # Create benchmark returns mapping
            benchmark_returns = {
                "BTC": np.random.normal(0.0005, 0.015),
                "ETH": np.random.normal(0.0003, 0.012)
            }
            
            data.append(StrategyPerformanceSnapshot(
                strategy_id=strategy_id,
                timestamp=current_time + timedelta(days=i),
                returns=ret,
                benchmark_returns=benchmark_returns,
                total_capital=10000.0 + np.random.uniform(-1000, 1000),
                net_exposure=np.random.uniform(-2000, 2000),
                strategy_pnl=ret * 10000.0
            ))
        
        return data

    async def get_active_strategies(self) -> List[str]:
        return ["alpha_strategy_1", "beta_strategy_1", "mixed_strategy_1"]


__all__ = [
    "ResidualReturnAnalyzer",
    "MockBenchmarkDataProvider",
    "MarketBenchmark",
    "StrategyPerformanceSnapshot",
    "ResidualAnalysisResult",
    "FakeAlphaAlert",
    "BenchmarkDataProvider"
]