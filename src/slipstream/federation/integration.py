"""
Federation Integration System: Complete Federated Trading Factory.

This module integrates all federation components into a complete federated trading system
that matches the vision outlined in the FEDERATED_VISION.md document. It orchestrates
the Strategy Pods (States), Meta-Optimizer (Central Bank), and Shared Infrastructure (Grid)
to create a factory for alphas that allocates risk and measures quality.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Any, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


@dataclass
class FederationMetrics:
    """Overall federation metrics combining all components."""
    timestamp: datetime
    total_strategies: int
    active_strategies: int
    total_capital_allocated: float
    portfolio_sharpe: float
    portfolio_volatility: float
    avg_strategy_correlation: float
    strategy_performance_entropy: float  # Measure of strategy diversity
    federation_stability_score: float  # Overall health score 0-100
    risk_metrics: Dict[str, float]  # Portfolio-level risk metrics


@dataclass
class FederationHealthStatus:
    """Comprehensive health status of the federated system."""
    timestamp: datetime
    overall_status: str  # 'healthy', 'degraded', 'at_risk', 'critical'
    component_health: Dict[str, str]  # Health of individual components
    strategies_health: Dict[str, str]  # Health of individual strategies
    allocation_efficiency: float  # How efficiently capital is allocated
    risk_exposure: float  # Overall risk exposure
    alerts: List[str]  # Current alerts
    recommendations: List[str]  # Recommendations for improvement


@dataclass
class FederationEvent:
    """Event in the federated system."""
    timestamp: datetime
    event_type: str  # 'strategy_registration', 'capital_allocation', 'performance_review', etc.
    source_component: str  # Which component generated event
    details: Dict[str, Any]
    severity: str  # 'info', 'warning', 'critical'


class FederationEventBus(Protocol):
    """Protocol for the federation event bus."""
    async def publish(self, event: FederationEvent):
        """Publish an event to the federation."""
        ...
    
    async def subscribe(self, event_type: str, callback):
        """Subscribe to events of a specific type."""
        ...


class FederationOrchestrator:
    """
    Orchestrates the complete federated trading system.
    
    This component ties together all federation components:
    - Strategy Pods (States): Individual strategies as autonomous units
    - Meta-Optimizer (Central Bank): Capital allocation and performance optimization  
    - Shared Infrastructure (Grid): Risk Auditor, Data Lake, Execution Gateway
    
    The orchestrator ensures the system operates as a true federation where
    individual strategies maintain autonomy while being coordinated through
    risk allocation rather than orchestration.
    """
    
    def __init__(self,
                 strategy_api_endpoints: Optional[Dict[str, str]] = None,  # strategy_id -> API URL
                 allocation_cycle_interval: timedelta = timedelta(hours=4),
                 performance_review_interval: timedelta = timedelta(hours=1),
                 health_check_interval: timedelta = timedelta(minutes=15)):
        """
        Initialize the federation orchestrator.
        
        Args:
            strategy_api_endpoints: Dictionary mapping strategy IDs to their API endpoints
            allocation_cycle_interval: How often to run capital allocation cycle
            performance_review_interval: How often to review strategy performance
            health_check_interval: How often to check federation health
        """
        self.strategy_api_endpoints = strategy_api_endpoints or {}
        self.allocation_cycle_interval = allocation_cycle_interval
        self.performance_review_interval = performance_review_interval
        self.health_check_interval = health_check_interval
        self.logger = logging.getLogger(__name__)
        self._running = False
        
        # Component references (in a real system, these would be actual instances)
        self._strategy_pods: Dict[str, Any] = {}  # Strategy pod handles
        self._meta_optimizer: Optional[Any] = None  # Meta-optimizer instance
        self._risk_auditor: Optional[Any] = None   # Risk auditor instance
        self._data_lake: Optional[Any] = None      # Data lake instance
        self._execution_gateway: Optional[Any] = None  # Execution gateway instance
        
        # Federation state
        self._federation_metrics: List[FederationMetrics] = []
        self._health_history: List[FederationHealthStatus] = []
        self._federation_events: List[FederationEvent] = []
        self._last_allocation_cycle: Optional[datetime] = None
        self._last_performance_review: Optional[datetime] = None
        
        # Event bus for federation-wide communication
        self._event_bus: Optional[FederationEventBus] = None

    async def start(self):
        """Start the federation orchestrator."""
        self._running = True
        self.logger.info("Federation Orchestrator started")
        
        # Initialize federation components
        await self._initialize_federation_components()
        
        # Start background processes
        asyncio.create_task(self._run_health_monitoring())
        asyncio.create_task(self._run_allocation_cycles())
        asyncio.create_task(self._run_performance_reviews())
        
        self.logger.info("Federation system initialized and monitoring started")

    async def stop(self):
        """Stop the federation orchestrator."""
        self._running = False
        self.logger.info("Federation Orchestrator stopped")

    async def _initialize_federation_components(self):
        """Initialize all federation components."""
        self.logger.info("Initializing federation components...")
        
        # This is where we would initialize actual instances of:
        # - Strategy API wrappers for each pod
        # - Meta-optimizer with data provider
        # - Risk auditor with exchange connectors
        # - Data lake connection
        # - Execution gateway
        
        # Register strategies
        for strategy_id, endpoint in self.strategy_api_endpoints.items():
            await self._register_strategy_pod(strategy_id, endpoint)
        
        # Initialize shared infrastructure components
        await self._initialize_shared_infrastructure()
        
        # Set up event bus for component communication
        await self._setup_event_bus()
        
        self.logger.info("Federation components initialized")

    async def _register_strategy_pod(self, strategy_id: str, api_endpoint: str):
        """Register a strategy pod with the federation."""
        # In a real system, this would create API client for the strategy
        self._strategy_pods[strategy_id] = {
            'api_endpoint': api_endpoint,
            'status': 'registered',
            'registration_time': datetime.now(),
            'health_status': 'unknown'
        }
        
        # Add to federation with minimal capital for incubation
        await self._initiate_incubation_process(strategy_id)
        
        self.logger.info(f"Registered strategy pod: {strategy_id}")

    async def _initiate_incubation_process(self, strategy_id: str):
        """Initiate the incubation process for a new strategy."""
        # This would trigger the lifecycle manager to start the strategy in incubation mode
        event = FederationEvent(
            timestamp=datetime.now(),
            event_type='strategy_incubation_started',
            source_component='orchestrator',
            details={'strategy_id': strategy_id, 'capital_allocated': 1000.0},  # Incubation capital
            severity='info'
        )
        await self._publish_event(event)
        
        self.logger.info(f"Initiated incubation for strategy: {strategy_id}")

    async def _initialize_shared_infrastructure(self):
        """Initialize shared infrastructure components."""
        # In a real system, this would instantiate actual components
        self.logger.info("Initializing shared infrastructure...")
        # - Data lake connection
        # - Risk auditor
        # - Execution gateway
        # - etc.

    async def _setup_event_bus(self):
        """Set up the federation event bus."""
        # In a real system, this would set up an actual event bus
        self.logger.info("Event bus setup completed")

    async def _publish_event(self, event: FederationEvent):
        """Publish an event to the federation."""
        self._federation_events.append(event)
        
        # In a real system, this would publish to the actual event bus
        if event.severity == 'critical':
            self.logger.error(f"FEDERATION EVENT: {event.event_type} - {event.details}")
        elif event.severity == 'warning':
            self.logger.warning(f"FEDERATION EVENT: {event.event_type} - {event.details}")
        else:
            self.logger.info(f"FEDERATION EVENT: {event.event_type}")

    async def _run_health_monitoring(self):
        """Background task to monitor federation health."""
        while self._running:
            try:
                health_status = await self.get_federation_health()
                self._health_history.append(health_status)
                
                if health_status.overall_status in ['at_risk', 'critical']:
                    self.logger.warning(f"Federation health degraded: {health_status.overall_status}")
                
                await asyncio.sleep(self.health_check_interval.total_seconds())
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(self.health_check_interval.total_seconds())

    async def _run_allocation_cycles(self):
        """Background task to run capital allocation cycles."""
        while self._running:
            try:
                # Run allocation optimization cycle
                await self._run_allocation_optimization()
                self._last_allocation_cycle = datetime.now()
                
                await asyncio.sleep(self.allocation_cycle_interval.total_seconds())
            except Exception as e:
                self.logger.error(f"Error in allocation cycle: {e}")
                await asyncio.sleep(self.allocation_cycle_interval.total_seconds())

    async def _run_performance_reviews(self):
        """Background task to run performance reviews."""
        while self._running:
            try:
                # Review strategy performance
                await self._review_strategy_performance()
                self._last_performance_review = datetime.now()
                
                await asyncio.sleep(self.performance_review_interval.total_seconds())
            except Exception as e:
                self.logger.error(f"Error in performance review: {e}")
                await asyncio.sleep(self.performance_review_interval.total_seconds())

    async def _run_allocation_optimization(self):
        """Run the allocation optimization cycle."""
        self.logger.info("Starting allocation optimization cycle...")
        
        # Collect performance data from all strategies
        performance_data = await self._collect_strategy_performance()
        
        # Calculate covariance and correlations
        covariance_matrix = await self._calculate_strategy_covariances(performance_data)
        
        # Optimize capital allocation based on performance and risk
        allocation_recommendations = await self._optimize_capital_allocation(
            performance_data, covariance_matrix
        )
        
        # Apply allocation changes
        await self._apply_allocation_changes(allocation_recommendations)
        
        # Record metrics
        metrics = await self._calculate_federation_metrics()
        self._federation_metrics.append(metrics)
        
        self.logger.info(f"Allocation optimization completed for {len(performance_data)} strategies")

    async def _review_strategy_performance(self):
        """Review strategy performance and make lifecycle recommendations."""
        self.logger.info("Starting performance review cycle...")
        
        # Get performance data
        performance_data = await self._collect_strategy_performance()
        
        # For each strategy, evaluate lifecycle status
        for strategy_id, perf in performance_data.items():
            # This would trigger lifecycle manager evaluation
            await self._evaluate_strategy_lifecycle(strategy_id, perf)
        
        self.logger.info(f"Performance review completed for {len(performance_data)} strategies")

    async def _collect_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Collect performance data from all registered strategies."""
        performance = {}
        
        for strategy_id in self._strategy_pods.keys():
            # In a real system, this would call the strategy's /status endpoint
            # and potentially other performance endpoints
            performance[strategy_id] = {
                'net_exposure': 0.0,  # Would come from strategy
                'pnl': 0.0,           # Would come from strategy
                'returns': 0.0,       # Would come from strategy
                'volatility': 0.0,    # Would come from strategy
                'sharpe': 0.0,        # Would come from strategy
                'active': True        # Would come from strategy status
            }
        
        return performance

    async def _calculate_strategy_covariances(self, performance_data: Dict[str, Dict[str, Any]]) -> Optional[np.ndarray]:
        """Calculate covariance matrix between strategies."""
        # This would use historical return data to calculate covariances
        # In this simplified version, we'll return a dummy matrix
        active_strategies = [sid for sid, perf in performance_data.items() if perf['active']]
        
        if len(active_strategies) < 2:
            return None
            
        # Create a dummy covariance matrix (in real system this would be calculated from data)
        n = len(active_strategies)
        cov_matrix = np.eye(n)  # Identity matrix as placeholder
        
        # Add some correlation for demonstration
        for i in range(n):
            for j in range(i+1, n):
                # Add some random correlation
                correlation = np.random.uniform(-0.1, 0.3)
                cov_matrix[i][j] = correlation
                cov_matrix[j][i] = correlation
        
        return cov_matrix

    async def _optimize_capital_allocation(self, 
                                         performance_data: Dict[str, Dict[str, Any]], 
                                         covariance_matrix: Optional[np.ndarray]) -> Dict[str, float]:
        """Optimize capital allocation based on performance and risk."""
        allocations = {}
        
        active_strategies = [sid for sid, perf in performance_data.items() if perf['active']]
        
        if not active_strategies:
            return allocations
        
        # Simple allocation algorithm: allocate more to higher Sharpe ratio strategies
        # but reduce allocation for highly correlated strategies
        total_performance_score = 0.0
        performance_scores = {}
        
        for strategy_id in active_strategies:
            perf = performance_data[strategy_id]
            # Calculate performance score based on Sharpe and other metrics
            performance_score = max(0.0, perf['sharpe'] if perf['sharpe'] is not None else 0.0)
            performance_scores[strategy_id] = performance_score
            total_performance_score += performance_score
        
        # Calculate total capital to allocate (simplified)
        total_capital = 100000.0  # Fixed amount for demo
        
        # Allocate proportionally to performance score
        for strategy_id in active_strategies:
            if total_performance_score > 0:
                base_allocation = (performance_scores[strategy_id] / total_performance_score) * total_capital
                # Apply diversification adjustment based on covariance
                # (simplified - in real system this would be more sophisticated)
                allocations[strategy_id] = base_allocation
            else:
                allocations[strategy_id] = total_capital / len(active_strategies)  # Equal allocation
        
        return allocations

    async def _apply_allocation_changes(self, allocation_recommendations: Dict[str, float]):
        """Apply allocation changes to strategies."""
        for strategy_id, new_allocation in allocation_recommendations.items():
            # In a real system, this would call the strategy's /configure endpoint
            # to update the MaxPosition limits
            current_allocation = self._strategy_pods[strategy_id].get('current_allocation', 1000.0)
            
            if abs(new_allocation - current_allocation) > 10:  # Only update if significant change
                self._strategy_pods[strategy_id]['current_allocation'] = new_allocation
                
                event = FederationEvent(
                    timestamp=datetime.now(),
                    event_type='capital_allocation_changed',
                    source_component='orchestrator',
                    details={
                        'strategy_id': strategy_id,
                        'old_allocation': current_allocation,
                        'new_allocation': new_allocation,
                        'change_amount': new_allocation - current_allocation
                    },
                    severity='info'
                )
                await self._publish_event(event)

    async def _evaluate_strategy_lifecycle(self, strategy_id: str, performance: Dict[str, Any]):
        """Evaluate strategy lifecycle status and trigger transitions if needed."""
        # In a real system, this would call the lifecycle manager
        # to evaluate if the strategy should be promoted, demoted, or retired
        
        # Check for retirement conditions
        if performance['sharpe'] is not None and performance['sharpe'] < -0.5:
            # Consider retirement for poor performance
            event = FederationEvent(
                timestamp=datetime.now(),
                event_type='retirement_consideration',
                source_component='orchestrator',
                details={
                    'strategy_id': strategy_id,
                    'reason': 'poor_sharpe_ratio',
                    'sharpe': performance['sharpe']
                },
                severity='warning'
            )
            await self._publish_event(event)

    async def _calculate_federation_metrics(self) -> FederationMetrics:
        """Calculate overall federation metrics."""
        active_strategies = [key for key, value in self._strategy_pods.items() 
                           if value.get('status') != 'retired']
        
        # Calculate metrics (simplified)
        total_capital = sum(pod.get('current_allocation', 1000.0) for pod in self._strategy_pods.values())
        
        metrics = FederationMetrics(
            timestamp=datetime.now(),
            total_strategies=len(self._strategy_pods),
            active_strategies=len(active_strategies),
            total_capital_allocated=total_capital,
            portfolio_sharpe=0.8,  # Placeholder
            portfolio_volatility=0.1,  # Placeholder 
            avg_strategy_correlation=0.2,  # Placeholder
            strategy_performance_entropy=0.5,  # Placeholder
            federation_stability_score=85.0,  # Placeholder
            risk_metrics={'var': 0.02, 'max_drawdown': -0.08}  # Placeholder
        )
        
        return metrics

    async def get_federation_health(self) -> FederationHealthStatus:
        """Get the current health status of the federation."""
        # Check component health
        component_health = {
            'orchestrator': 'healthy',
            'meta_optimizer': 'healthy',
            'risk_auditor': 'healthy',
            'data_lake': 'healthy'
        }
        
        # Check strategy health
        strategy_health = {}
        alerts = []
        
        for strategy_id, pod in self._strategy_pods.items():
            health = pod.get('health_status', 'unknown')
            strategy_health[strategy_id] = health
            
            # Add any alerts based on strategy status
            if pod.get('status') == 'error':
                alerts.append(f"Strategy {strategy_id} in error state")
        
        # Calculate overall status
        if any(status == 'critical' for status in component_health.values()):
            overall_status = 'critical'
        elif any(status == 'degraded' for status in component_health.values()) or alerts:
            overall_status = 'at_risk'
        else:
            overall_status = 'healthy'
        
        health_status = FederationHealthStatus(
            timestamp=datetime.now(),
            overall_status=overall_status,
            component_health=component_health,
            strategies_health=strategy_health,
            allocation_efficiency=0.85,  # Placeholder
            risk_exposure=0.12,  # Placeholder
            alerts=alerts,
            recommendations=[]
        )
        
        return health_status

    async def get_federation_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive federation dashboard data.
        
        Returns:
            Dictionary with all federation metrics and status information
        """
        health = await self.get_federation_health()
        metrics = await self._calculate_federation_metrics() if self._federation_metrics else await self._calculate_federation_metrics()
        
        # Get strategy-specific data
        strategy_details = {}
        for strategy_id, pod in self._strategy_pods.items():
            strategy_details[strategy_id] = {
                'status': pod.get('status', 'unknown'),
                'health': pod.get('health_status', 'unknown'),
                'capital_allocation': pod.get('current_allocation', 1000.0),
                'registration_time': pod.get('registration_time')
            }
        
        dashboard = {
            'timestamp': datetime.now(),
            'federation_health': health,
            'federation_metrics': metrics,
            'strategy_summary': {
                'total': len(self._strategy_pods),
                'active': len([s for s in self._strategy_pods.values() if s.get('status') != 'retired']),
                'retired': len([s for s in self._strategy_pods.values() if s.get('status') == 'retired'])
            },
            'strategy_details': strategy_details,
            'allocation_cycle_info': {
                'last_cycle': self._last_allocation_cycle,
                'interval': self.allocation_cycle_interval.total_seconds() / 3600,  # hours
            },
            'performance_review_info': {
                'last_review': self._last_performance_review,
                'interval': self.performance_review_interval.total_seconds() / 3600,  # hours
            }
        }
        
        return dashboard

    async def add_strategy_to_federation(self, strategy_id: str, api_endpoint: str, initial_capital: float = 1000.0):
        """
        Add a new strategy to the federation.
        
        Args:
            strategy_id: Unique identifier for the strategy
            api_endpoint: API endpoint URL for the strategy
            initial_capital: Initial capital allocation (typically low for incubation)
        """
        # Register the strategy API endpoint
        self.strategy_api_endpoints[strategy_id] = api_endpoint
        
        # Register the strategy pod
        await self._register_strategy_pod(strategy_id, api_endpoint)
        
        # Set initial allocation (incubation capital)
        self._strategy_pods[strategy_id]['current_allocation'] = initial_capital
        
        self.logger.info(f"Added strategy {strategy_id} to federation with {initial_capital} capital")

    async def remove_strategy_from_federation(self, strategy_id: str):
        """
        Remove a strategy from the federation (for retirement).
        
        Args:
            strategy_id: The strategy to remove
        """
        if strategy_id in self._strategy_pods:
            # Set to retired status
            self._strategy_pods[strategy_id]['status'] = 'retired'
            self._strategy_pods[strategy_id]['current_allocation'] = 0.0
            
            # Remove from active endpoints if needed
            if strategy_id in self.strategy_api_endpoints:
                del self.strategy_api_endpoints[strategy_id]
            
            event = FederationEvent(
                timestamp=datetime.now(),
                event_type='strategy_retired',
                source_component='orchestrator',
                details={'strategy_id': strategy_id},
                severity='info'
            )
            await self._publish_event(event)
            
            self.logger.info(f"Removed strategy {strategy_id} from federation")

    async def force_strategy_allocation(self, strategy_id: str, new_allocation: float):
        """
        Force a specific allocation for a strategy (admin function).
        
        Args:
            strategy_id: The strategy to adjust
            new_allocation: The new allocation amount
        """
        if strategy_id in self._strategy_pods:
            old_allocation = self._strategy_pods[strategy_id].get('current_allocation', 0.0)
            self._strategy_pods[strategy_id]['current_allocation'] = new_allocation
            
            event = FederationEvent(
                timestamp=datetime.now(),
                event_type='allocation_force_changed',
                source_component='admin',
                details={
                    'strategy_id': strategy_id,
                    'old_allocation': old_allocation,
                    'new_allocation': new_allocation
                },
                severity='info'
            )
            await self._publish_event(event)
            
            self.logger.info(f"Force changed allocation for {strategy_id}: {old_allocation} -> {new_allocation}")


class MockFederationEventBus:
    """Mock event bus for testing purposes."""
    
    def __init__(self):
        self._subscribers = {}
        self._events = []
    
    async def publish(self, event: FederationEvent):
        self._events.append(event)
        # Notify subscribers if any
        if event.event_type in self._subscribers:
            for callback in self._subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Error in event callback: {e}")
    
    async def subscribe(self, event_type: str, callback):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)


__all__ = [
    "FederationOrchestrator",
    "MockFederationEventBus",
    "FederationMetrics", 
    "FederationHealthStatus",
    "FederationEvent",
    "FederationEventBus"
]