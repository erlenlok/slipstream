"""
Lifecycle Management System for Strategy Lifecycle Automation.

This module implements the automated lifecycle management system that manages strategies 
from incubation with minimal risk capital through evaluation, promotion based on 
statistical significance, and retirement when alpha decays or shortfall exceeds 
thresholds, as specified in the federated vision document.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd


class StrategyLifecycleStage(Enum):
    """Stages in the strategy lifecycle."""
    INCUBATION = "incubation"
    EVALUATION = "evaluation"
    GROWTH = "growth"
    MATURITY = "maturity"
    RETIREMENT = "retirement"


@dataclass
class StrategyMetrics:
    """Key metrics for a strategy at a specific time."""
    strategy_id: str
    timestamp: datetime
    returns: float
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
    implementation_shortfall: Optional[float] = None
    days_active: int = 1
    performance_score: Optional[float] = None


@dataclass
class StrategyLifecycleState:
    """Current state of a strategy in the lifecycle."""
    strategy_id: str
    current_stage: StrategyLifecycleStage
    start_date: datetime
    current_capital: float
    performance_history: List[StrategyMetrics]
    statistical_significance: float  # Measure of performance significance
    risk_metrics: Dict[str, float]  # Various risk metrics
    trigger_events: List[str]  # Events that triggered state changes
    status: str  # 'active', 'paused', 'retired'
    next_review_date: Optional[datetime] = None


@dataclass
class LifecycleTransition:
    """Record of a lifecycle stage transition."""
    strategy_id: str
    from_stage: StrategyLifecycleStage
    to_stage: StrategyLifecycleStage
    timestamp: datetime
    reason: str
    old_capital: float
    new_capital: float
    performance_snapshot: Optional[StrategyMetrics] = None


@dataclass
class LifecycleRecommendation:
    """Recommendation for lifecycle management."""
    strategy_id: str
    action: str  # 'promote', 'demote', 'retire', 'monitor', 'increase_capital', 'decrease_capital'
    confidence: float  # 0-1 confidence in recommendation
    reason: str
    impact: Dict[str, Any]  # Expected impact of the action


class MetricsDataProvider(Protocol):
    """
    Protocol for data providers that the lifecycle manager can use to
    collect strategy performance data.
    """
    async def get_strategy_metrics(self, strategy_id: str,
                                  start_time: datetime,
                                  end_time: datetime) -> List[StrategyMetrics]:
        """Get performance metrics for a strategy."""
        ...

    async def get_current_state(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get current metrics for a strategy."""
        ...

    async def get_historical_performance(self, strategy_id: str, 
                                       days: int = 30) -> List[StrategyMetrics]:
        """Get historical performance for statistical analysis."""
        ...


class LifecycleManager:
    """
    Manages the automated lifecycle of strategies from incubation to retirement.
    
    This system implements the complete lifecycle management as specified in the 
    federated vision: incubation with minimal capital, evaluation with performance
    tracking, promotion based on statistical significance, and retirement when
    alpha decays or shortfall exceeds thresholds.
    """
    
    def __init__(self,
                 data_provider: Optional[MetricsDataProvider] = None,
                 incubation_capital: float = 1000.0,  # Initial learning capital
                 evaluation_period_days: int = 30,   # Days in evaluation phase
                 promotion_significance_threshold: float = 2.0,  # Statistical significance required
                 retirement_alpha_threshold: float = 0.0,       # Alpha below this triggers retirement
                 retirement_shortfall_threshold: float = 0.005,  # 50 bps shortfall triggers review
                 min_promotion_capital: float = 5000.0,
                 max_promotion_capital: float = 50000.0):
        """
        Initialize the lifecycle manager.
        
        Args:
            data_provider: Optional data provider for metrics
            incubation_capital: Capital for incubation phase (minimal risk)
            evaluation_period_days: Days to evaluate before promotion decision
            promotion_significance_threshold: Statistical significance needed for promotion
            retirement_alpha_threshold: Alpha threshold below which to retire
            retirement_shortfall_threshold: Implementation shortfall threshold
            min_promotion_capital: Minimum capital after promotion
            max_promotion_capital: Maximum capital allocation
        """
        self.data_provider = data_provider
        self.incubation_capital = incubation_capital
        self.evaluation_period_days = evaluation_period_days
        self.promotion_significance_threshold = promotion_significance_threshold
        self.retirement_alpha_threshold = retirement_alpha_threshold
        self.retirement_shortfall_threshold = retirement_shortfall_threshold
        self.min_promotion_capital = min_promotion_capital
        self.max_promotion_capital = max_promotion_capital
        self.logger = logging.getLogger(__name__)
        self._running = False
        
        # Store lifecycle states
        self._lifecycle_states: Dict[str, StrategyLifecycleState] = {}
        self._transition_history: Dict[str, List[LifecycleTransition]] = {}
        self._recommendation_history: Dict[str, List[LifecycleRecommendation]] = {}
        
        # Track statistical significance of strategies
        self._performance_significance: Dict[str, float] = {}

    async def start(self):
        """Start the lifecycle manager."""
        self._running = True
        self.logger.info("Lifecycle Manager started")
        
        if self.data_provider:
            # Initial data loading would happen here in a real system
            pass
        
        self.logger.info("Lifecycle management started")

    async def stop(self):
        """Stop the lifecycle manager."""
        self._running = False
        self.logger.info("Lifecycle Manager stopped")

    async def register_new_strategy(self, strategy_id: str) -> StrategyLifecycleState:
        """
        Register a new strategy with the lifecycle management system.
        
        Args:
            strategy_id: The ID of the new strategy
            
        Returns:
            Initial lifecycle state in INCUBATION stage
        """
        state = StrategyLifecycleState(
            strategy_id=strategy_id,
            current_stage=StrategyLifecycleStage.INCUBATION,
            start_date=datetime.now(),
            current_capital=self.incubation_capital,
            performance_history=[],
            statistical_significance=0.0,
            risk_metrics={},
            trigger_events=["registration"],
            status="active"
        )
        
        self._lifecycle_states[strategy_id] = state
        self.logger.info(f"Registered new strategy in incubation: {strategy_id} with capital {self.incubation_capital}")
        
        # Create initial transition record
        transition = LifecycleTransition(
            strategy_id=strategy_id,
            from_stage=None,
            to_stage=StrategyLifecycleStage.INCUBATION,
            timestamp=datetime.now(),
            reason="initial_registration",
            old_capital=0.0,
            new_capital=self.incubation_capital
        )
        
        if strategy_id not in self._transition_history:
            self._transition_history[strategy_id] = []
        self._transition_history[strategy_id].append(transition)
        
        return state

    async def update_strategy_metrics(self, metrics: StrategyMetrics):
        """
        Update performance metrics for a strategy and trigger lifecycle evaluation.
        
        Args:
            metrics: Current performance metrics for the strategy
        """
        strategy_id = metrics.strategy_id
        
        if strategy_id not in self._lifecycle_states:
            # Auto-register unknown strategies in incubation
            await self.register_new_strategy(strategy_id)
        
        state = self._lifecycle_states[strategy_id]
        
        # Add metrics to history
        state.performance_history.append(metrics)
        
        # Update statistical significance
        if state.performance_history:
            await self._calculate_statistical_significance(state)
        
        # Update risk metrics
        await self._update_risk_metrics(state, metrics)
        
        # Check if lifecycle transition is needed
        await self._check_lifecycle_transition(state)
        
        self.logger.debug(f"Updated metrics for {strategy_id} in stage {state.current_stage}")

    async def _calculate_statistical_significance(self, state: StrategyLifecycleState):
        """Calculate statistical significance of a strategy's performance."""
        if not state.performance_history:
            state.statistical_significance = 0.0
            return
        
        # Calculate significance based on sharpe ratio and consistency over time
        recent_metrics = state.performance_history[-min(20, len(state.performance_history)):]  # Last 20 periods or all
        
        if not recent_metrics:
            state.statistical_significance = 0.0
            return
        
        # Calculate average Sharpe ratio and its consistency
        sharpes = [m.sharpe_ratio for m in recent_metrics if m.sharpe_ratio is not None]
        if not sharpes:
            state.statistical_significance = 0.0
            return
        
        avg_sharpe = np.mean(sharpes)
        sharpe_volatility = np.std(sharpes) if len(sharpes) > 1 else 0.0
        
        # Calculate significance score (0-1 scale)
        # Higher average Sharpe and lower volatility = higher significance
        normalized_sharpe = max(0, avg_sharpe)  # Only positive contribution
        stability_factor = 1.0 / (1.0 + sharpe_volatility) if sharpe_volatility > 0 else 1.0
        
        # Combine factors with weights
        significance = min(1.0, normalized_sharpe * 0.7 + stability_factor * 0.3)
        state.statistical_significance = significance

    async def _update_risk_metrics(self, state: StrategyLifecycleState, metrics: StrategyMetrics):
        """Update risk metrics for the strategy."""
        # Update or add risk metrics based on current performance
        state.risk_metrics.update({
            'current_volatility': metrics.volatility,
            'max_drawdown': metrics.max_drawdown,
            'current_sharpe': metrics.sharpe_ratio,
            'current_alpha': metrics.alpha or 0.0,
            'implementation_shortfall': metrics.implementation_shortfall or 0.0,
            'days_active': metrics.days_active
        })

    async def _check_lifecycle_transition(self, state: StrategyLifecycleState):
        """Check if a strategy should transition to a different lifecycle stage."""
        strategy_id = state.strategy_id
        
        # Check retirement criteria
        should_retire = await self._should_retire_strategy(state)
        if should_retire:
            await self._transition_to_retirement(state)
            return
        
        # Stage-specific transition logic
        if state.current_stage == StrategyLifecycleStage.INCUBATION:
            # Check if evaluation period is complete based on performance history duration
            if state.performance_history:
                # Use the time between first and last metrics as indicator of evaluation period
                first_time = state.performance_history[0].timestamp
                last_time = state.performance_history[-1].timestamp
                days_of_history = (last_time - first_time).days
                if days_of_history >= self.evaluation_period_days:
                    await self._evaluate_promotion(state)
            else:
                # If no history yet, use start date as fallback
                days_in_incubation = (datetime.now() - state.start_date).days
                if days_in_incubation >= self.evaluation_period_days:
                    await self._evaluate_promotion(state)

        elif state.current_stage == StrategyLifecycleStage.EVALUATION:
            # Evaluation is an ongoing assessment phase
            await self._evaluate_promotion(state)
        
        elif state.current_stage in [StrategyLifecycleStage.GROWTH, StrategyLifecycleStage.MATURITY]:
            # Check for demotion or promotion opportunities
            await self._evaluate_promotion(state)

    async def _should_retire_strategy(self, state: StrategyLifecycleState) -> bool:
        """Check if a strategy should be retired."""
        if not state.performance_history:
            return False
        
        latest_metrics = state.performance_history[-1]
        
        # Check alpha threshold (if alpha is too low for too long)
        if latest_metrics.alpha is not None and latest_metrics.alpha < self.retirement_alpha_threshold:
            # But only retire if it's consistently below threshold
            recent_metrics = state.performance_history[-min(10, len(state.performance_history)):]
            low_alpha_count = sum(1 for m in recent_metrics if m.alpha and m.alpha < self.retirement_alpha_threshold)
            if len(recent_metrics) > 0 and low_alpha_count / len(recent_metrics) > 0.6:  # 60% of recent periods
                state.trigger_events.append("low_alpha_retirement_criteria")
                return True
        
        # Check implementation shortfall threshold
        if latest_metrics.implementation_shortfall and latest_metrics.implementation_shortfall > self.retirement_shortfall_threshold:
            state.trigger_events.append("high_shortfall_retirement_criteria")
            return True
        
        # Check for severe drawdown
        if latest_metrics.max_drawdown and latest_metrics.max_drawdown < -0.2:  # -20% drawdown
            state.trigger_events.append("severe_drawdown_retirement_criteria")
            return True
        
        return False

    async def _transition_to_retirement(self, state: StrategyLifecycleState):
        """Transition a strategy to retirement."""
        old_stage = state.current_stage
        old_capital = state.current_capital
        
        state.current_stage = StrategyLifecycleStage.RETIREMENT
        state.status = "retired"
        state.current_capital = 0.0  # Remove all capital
        
        self.logger.info(f"Strategy {state.strategy_id} retired: {state.trigger_events[-1] if state.trigger_events else 'unknown_reason'}")
        
        # Record transition
        transition = LifecycleTransition(
            strategy_id=state.strategy_id,
            from_stage=old_stage,
            to_stage=StrategyLifecycleStage.RETIREMENT,
            timestamp=datetime.now(),
            reason=state.trigger_events[-1] if state.trigger_events else "retirement_criteria_met",
            old_capital=old_capital,
            new_capital=0.0,
            performance_snapshot=state.performance_history[-1] if state.performance_history else None
        )
        
        self._transition_history[state.strategy_id].append(transition)

    async def _evaluate_promotion(self, state: StrategyLifecycleState):
        """Evaluate if a strategy should be promoted/demoted."""
        if state.statistical_significance >= self.promotion_significance_threshold:
            # Promote to next stage or increase capital
            await self._attempt_promotion(state)
        else:
            # Check if performance is deteriorating
            await self._check_demotion(state)

    async def _attempt_promotion(self, state: StrategyLifecycleState):
        """Attempt to promote a strategy to the next stage."""
        old_stage = state.current_stage
        old_capital = state.current_capital
        
        if state.current_stage == StrategyLifecycleStage.INCUBATION:
            # Move to evaluation phase
            state.current_stage = StrategyLifecycleStage.EVALUATION
            # Increase capital for evaluation
            state.current_capital = min(self.min_promotion_capital, state.current_capital * 2)
        elif state.current_stage == StrategyLifecycleStage.EVALUATION:
            # Move to growth phase
            state.current_stage = StrategyLifecycleStage.GROWTH
            # Increase capital based on performance
            performance_multiplier = 1.0 + min(2.0, state.statistical_significance * 2)  # Max 3x increase
            state.current_capital = min(self.max_promotion_capital, self.min_promotion_capital * performance_multiplier)
        elif state.current_stage == StrategyLifecycleStage.GROWTH:
            # Potentially move to maturity or increase capital
            if state.statistical_significance > 3.0:  # Very high significance
                state.current_stage = StrategyLifecycleStage.MATURITY
                # Further capital increase
                state.current_capital = min(self.max_promotion_capital, state.current_capital * 1.5)
        elif state.current_stage == StrategyLifecycleStage.MATURITY:
            # Keep in maturity but adjust capital based on performance
            base_capital = self.max_promotion_capital * 0.8  # Use 80% as baseline
            performance_adjustment = min(1.5, 1.0 + (state.statistical_significance / 5.0))  # Cap at 1.5x
            state.current_capital = min(self.max_promotion_capital, base_capital * performance_adjustment)
        
        if old_stage != state.current_stage:
            self.logger.info(f"Strategy {state.strategy_id} promoted from {old_stage.value} to {state.current_stage.value}")
            state.trigger_events.append(f"promoted_from_{old_stage.value}_to_{state.current_stage.value}")
        elif old_capital != state.current_capital:
            self.logger.info(f"Strategy {state.strategy_id} capital adjusted from {old_capital:.2f} to {state.current_capital:.2f}")
            state.trigger_events.append(f"capital_increase_from_{old_capital:.2f}_to_{state.current_capital:.2f}")
        
        # Record transition if stage changed
        if old_stage != state.current_stage:
            transition = LifecycleTransition(
                strategy_id=state.strategy_id,
                from_stage=old_stage,
                to_stage=state.current_stage,
                timestamp=datetime.now(),
                reason=f"statistical_significance_{state.statistical_significance:.3f}",
                old_capital=old_capital,
                new_capital=state.current_capital,
                performance_snapshot=state.performance_history[-1] if state.performance_history else None
            )
            
            self._transition_history[state.strategy_id].append(transition)

    async def _check_demotion(self, state: StrategyLifecycleState):
        """Check if a strategy should be demoted."""
        # Check if statistical significance is deteriorating significantly
        if state.performance_history and len(state.performance_history) >= 5:
            # Compare recent performance to earlier performance
            recent_metrics = state.performance_history[-3:]  # Last 3 periods
            earlier_metrics = state.performance_history[-8:-5]  # 3 periods before that
            
            if recent_metrics and earlier_metrics:
                recent_avg_sharpe = np.mean([m.sharpe_ratio for m in recent_metrics if m.sharpe_ratio is not None])
                earlier_avg_sharpe = np.mean([m.sharpe_ratio for m in earlier_metrics if m.sharpe_ratio is not None])
                
                # If performance is significantly declining, consider demotion
                if earlier_avg_sharpe > 0 and recent_avg_sharpe < earlier_avg_sharpe * 0.5:  # 50% decline
                    # For now, just log the significant decline
                    self.logger.warning(f"Strategy {state.strategy_id} showing declining performance: "
                                      f"recent Sharpe {recent_avg_sharpe:.3f} vs earlier {earlier_avg_sharpe:.3f}")
                    state.trigger_events.append(f"performance_decline_detected_{recent_avg_sharpe:.3f}_vs_{earlier_avg_sharpe:.3f}")

    async def get_lifecycle_recommendations(self, strategy_id: str) -> List[LifecycleRecommendation]:
        """
        Get lifecycle recommendations for a strategy.
        
        Args:
            strategy_id: The strategy to get recommendations for
            
        Returns:
            List of lifecycle recommendations
        """
        if strategy_id not in self._lifecycle_states:
            return []
        
        state = self._lifecycle_states[strategy_id]
        recommendations = []
        
        # Check if strategy is ready for promotion
        if state.statistical_significance >= self.promotion_significance_threshold:
            if state.current_stage in [StrategyLifecycleStage.INCUBATION, StrategyLifecycleStage.EVALUATION]:
                recommendations.append(LifecycleRecommendation(
                    strategy_id=strategy_id,
                    action="promote",
                    confidence=min(1.0, state.statistical_significance / self.promotion_significance_threshold),
                    reason=f"Statistical significance {state.statistical_significance:.3f} exceeds threshold {self.promotion_significance_threshold:.3f}",
                    impact={"capital_increase": f"{state.current_capital:.2f} -> {state.current_capital * 2:.2f}"}
                ))
            elif state.current_stage == StrategyLifecycleStage.GROWTH:
                recommendations.append(LifecycleRecommendation(
                    strategy_id=strategy_id,
                    action="increase_capital",
                    confidence=min(0.8, state.statistical_significance / 3.0),
                    reason=f"High performing strategy in growth phase with significance {state.statistical_significance:.3f}",
                    impact={"capital_increase": f"{state.current_capital:.2f} -> {state.current_capital * 1.5:.2f}"}
                ))
        
        # Check if strategy should be retired
        if await self._should_retire_strategy(state):
            recommendations.append(LifecycleRecommendation(
                strategy_id=strategy_id,
                action="retire",
                confidence=1.0,
                reason="Retirement criteria met",
                impact={"capital_reduction": f"{state.current_capital:.2f} -> 0.0"}
            ))
        
        # Check for performance monitoring
        if state.statistical_significance < 0.1 and state.current_stage in [StrategyLifecycleStage.GROWTH, StrategyLifecycleStage.MATURITY]:
            recommendations.append(LifecycleRecommendation(
                strategy_id=strategy_id,
                action="monitor",
                confidence=0.9,
                reason=f"Low statistical significance {state.statistical_significance:.3f}, requires monitoring",
                impact={"status": "monitoring"}
            ))
        
        # Store recommendations
        if strategy_id not in self._recommendation_history:
            self._recommendation_history[strategy_id] = []
        self._recommendation_history[strategy_id].extend(recommendations)
        
        return recommendations

    async def get_strategy_lifecycle_status(self, strategy_id: str) -> Optional[StrategyLifecycleState]:
        """
        Get the current lifecycle status of a strategy.
        
        Args:
            strategy_id: The strategy to get status for
            
        Returns:
            Current lifecycle state or None if not found
        """
        return self._lifecycle_states.get(strategy_id)

    async def get_lifecycle_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all strategy lifecycles.
        
        Returns:
            Dictionary mapping strategy_id to lifecycle summary
        """
        summary = {}
        
        for strategy_id, state in self._lifecycle_states.items():
            if state.performance_history:
                latest_metrics = state.performance_history[-1]
                summary[strategy_id] = {
                    'stage': state.current_stage.value,
                    'status': state.status,
                    'current_capital': state.current_capital,
                    'statistical_significance': state.statistical_significance,
                    'latest_sharpe': latest_metrics.sharpe_ratio,
                    'latest_alpha': latest_metrics.alpha,
                    'days_active': latest_metrics.days_active,
                    'total_trades': latest_metrics.trades,
                    'latest_return': latest_metrics.returns,
                    'is_retirement_candidate': await self._should_retire_strategy(state)
                }
            else:
                summary[strategy_id] = {
                    'stage': state.current_stage.value,
                    'status': state.status,
                    'current_capital': state.current_capital,
                    'statistical_significance': 0.0,
                    'latest_sharpe': 0.0,
                    'latest_alpha': 0.0,
                    'days_active': 0,
                    'total_trades': 0,
                    'latest_return': 0.0,
                    'is_retirement_candidate': False
                }
        
        return summary

    async def get_lifecycle_transitions(self, strategy_id: str) -> List[LifecycleTransition]:
        """
        Get lifecycle transition history for a strategy.
        
        Args:
            strategy_id: The strategy to get transitions for
            
        Returns:
            List of lifecycle transitions
        """
        return self._transition_history.get(strategy_id, [])

    async def force_lifecycle_action(self, strategy_id: str, action: str) -> bool:
        """
        Force a specific lifecycle action (use with caution).
        
        Args:
            strategy_id: The strategy to act on
            action: The action to take ('retire', 'promote', 'demote', 'pause')
            
        Returns:
            True if action was successful, False otherwise
        """
        if strategy_id not in self._lifecycle_states:
            return False
        
        state = self._lifecycle_states[strategy_id]
        
        if action == 'retire':
            state.status = 'retired'
            state.current_capital = 0.0
            old_stage = state.current_stage
            state.current_stage = StrategyLifecycleStage.RETIREMENT
            state.trigger_events.append(f"forced_retirement_action")
            
            # Record transition
            transition = LifecycleTransition(
                strategy_id=state.strategy_id,
                from_stage=old_stage,
                to_stage=StrategyLifecycleStage.RETIREMENT,
                timestamp=datetime.now(),
                reason="forced_retirement_action",
                old_capital=state.current_capital,
                new_capital=0.0
            )
            self._transition_history[state.strategy_id].append(transition)
            
            self.logger.warning(f"Strategy {strategy_id} force-retired by admin action")
            return True
        
        # Add other force actions as needed...
        
        return False


class MockMetricsDataProvider:
    """
    Mock data provider for testing purposes.
    """
    
    def __init__(self):
        self._strategy_metrics = {}
        
    async def get_strategy_metrics(self, strategy_id: str,
                                  start_time: datetime,
                                  end_time: datetime) -> List[StrategyMetrics]:
        if strategy_id not in self._strategy_metrics:
            # Generate mock metrics with different performance characteristics
            metrics = []
            base_time = start_time
            
            for i in range(20):  # 20 periods of metrics
                if strategy_id.startswith("high_perf"):
                    # High performing strategy: consistent positive returns
                    avg_return = 0.002
                    volatility = 0.01
                    sharpe = 1.8
                    alpha = 0.0015
                elif strategy_id.startswith("low_perf"):
                    # Low performing strategy: negative returns
                    avg_return = -0.001
                    volatility = 0.015
                    sharpe = -0.8
                    alpha = -0.001
                else:
                    # Medium performing strategy
                    avg_return = 0.0005
                    volatility = 0.012
                    sharpe = 0.5
                    alpha = 0.0002
                
                # Add some variation
                actual_return = avg_return + np.random.normal(0, volatility)
                
                metrics.append(StrategyMetrics(
                    strategy_id=strategy_id,
                    timestamp=base_time + timedelta(days=i),
                    returns=actual_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe + np.random.normal(0, 0.2),
                    max_drawdown=-0.05 + (i * -0.001),
                    total_return=avg_return * (i + 1),
                    total_capital=10000.0,
                    net_exposure=2000.0 + (i * 100),
                    gross_exposure=3000.0 + (i * 150),
                    win_rate=0.6 + np.random.uniform(-0.1, 0.1),
                    avg_win=150.0 + np.random.uniform(-50, 50),
                    avg_loss=-100.0 + np.random.uniform(-30, 30),
                    trades=10 + i,
                    alpha=alpha + np.random.normal(0, 0.0005),
                    beta=0.01 + np.random.normal(0, 0.005),
                    implementation_shortfall=np.random.uniform(0.0001, 0.001) if i % 3 != 0 else 0.005,  # Varying shortfall
                    days_active=i + 1
                ))
            
            self._strategy_metrics[strategy_id] = metrics
        
        return [m for m in self._strategy_metrics[strategy_id] 
                if start_time <= m.timestamp <= end_time]

    async def get_current_state(self, strategy_id: str) -> Optional[StrategyMetrics]:
        metrics_list = await self.get_strategy_metrics(strategy_id, datetime.now() - timedelta(days=1), datetime.now())
        return metrics_list[-1] if metrics_list else None

    async def get_historical_performance(self, strategy_id: str, days: int = 30) -> List[StrategyMetrics]:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        return await self.get_strategy_metrics(strategy_id, start_time, end_time)


__all__ = [
    "LifecycleManager",
    "MockMetricsDataProvider",
    "StrategyMetrics",
    "StrategyLifecycleState",
    "LifecycleTransition",
    "LifecycleRecommendation",
    "StrategyLifecycleStage",
    "MetricsDataProvider"
]