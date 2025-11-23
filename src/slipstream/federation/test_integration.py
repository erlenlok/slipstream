"""
Test suite for the Federation Integration System component.

This module tests that all federation components are properly integrated into
a complete federated trading system that operates according to the vision.
"""

from datetime import datetime, timedelta
import asyncio
import numpy as np
from slipstream.federation.integration import FederationOrchestrator, FederationMetrics


async def test_federation_initialization():
    """
    Test that the federation system initializes correctly with all components.
    """
    print("Testing federation initialization...")
    
    orchestrator = FederationOrchestrator(
        strategy_api_endpoints={
            "test_strategy_1": "http://localhost:8001",
            "test_strategy_2": "http://localhost:8002"
        }
    )
    await orchestrator.start()
    
    print(f"  ✓ Federation orchestrator started")
    
    # Verify components are registered
    print(f"  ✓ Strategy pods registered: {len(orchestrator._strategy_pods)} strategies")
    print(f"  ✓ Strategy API endpoints: {len(orchestrator.strategy_api_endpoints)} endpoints")
    
    assert len(orchestrator._strategy_pods) > 0
    assert len(orchestrator.strategy_api_endpoints) > 0
    print("  ✓ Initialization completed correctly")
    
    await orchestrator.stop()
    print("  ✓ Federation initialization test passed")


async def test_strategy_registration_integration():
    """
    Test that strategies are properly registered and managed in the federation.
    """
    print("\nTesting strategy registration integration...")
    
    orchestrator = FederationOrchestrator()
    await orchestrator.start()
    
    # Add a new strategy to the federation
    await orchestrator.add_strategy_to_federation(
        strategy_id="integration_test_strategy",
        api_endpoint="http://localhost:8003",
        initial_capital=1500.0
    )
    
    print(f"  ✓ Strategy added to federation")
    print(f"  ✓ Strategy pods count: {len(orchestrator._strategy_pods)}")
    
    # Verify the strategy is in the system with correct capital
    assert "integration_test_strategy" in orchestrator._strategy_pods
    assert orchestrator._strategy_pods["integration_test_strategy"]["current_allocation"] == 1500.0
    assert orchestrator._strategy_pods["integration_test_strategy"]["status"] == "registered"
    print("  ✓ Strategy properly registered with correct parameters")
    
    # Test dashboard data includes the new strategy
    dashboard = await orchestrator.get_federation_dashboard()
    assert "integration_test_strategy" in dashboard["strategy_details"]
    print("  ✓ Strategy appears in federation dashboard")
    
    await orchestrator.stop()
    print("  ✓ Strategy registration integration test passed")


async def test_allocation_optimization_cycle():
    """
    Test that the allocation optimization cycle runs and adjusts capital correctly.
    """
    print("\nTesting allocation optimization cycle...")
    
    orchestrator = FederationOrchestrator(
        allocation_cycle_interval=timedelta(seconds=1),  # Fast interval for testing
        performance_review_interval=timedelta(seconds=2)
    )
    await orchestrator.start()
    
    # Add strategies to the federation
    await orchestrator.add_strategy_to_federation(
        strategy_id="alloc_test_1",
        api_endpoint="http://localhost:8004",
        initial_capital=2000.0
    )
    await orchestrator.add_strategy_to_federation(
        strategy_id="alloc_test_2", 
        api_endpoint="http://localhost:8005",
        initial_capital=2000.0
    )
    
    print(f"  ✓ Added 2 strategies for allocation testing")
    
    # Manually trigger allocation optimization
    await orchestrator._run_allocation_optimization()
    
    # Check metrics were calculated
    metrics = await orchestrator._calculate_federation_metrics()
    print(f"  ✓ Federation metrics calculated: total_strategies={metrics.total_strategies}, active_strategies={metrics.active_strategies}")
    
    # Verify metrics look reasonable
    assert metrics.total_strategies >= 2
    assert metrics.active_strategies >= 2
    assert metrics.total_capital_allocated >= 4000.0  # At least initial capital
    print("  ✓ Allocation metrics are reasonable")
    
    await orchestrator.stop()
    print("  ✓ Allocation optimization cycle test passed")


async def test_federation_health_monitoring():
    """
    Test that federation health monitoring works correctly.
    """
    print("\nTesting federation health monitoring...")
    
    orchestrator = FederationOrchestrator()
    await orchestrator.start()
    
    # Add some strategies
    await orchestrator.add_strategy_to_federation(
        strategy_id="health_test_1",
        api_endpoint="http://localhost:8006",
        initial_capital=1000.0
    )
    await orchestrator.add_strategy_to_federation(
        strategy_id="health_test_2",
        api_endpoint="http://localhost:8007", 
        initial_capital=1000.0
    )
    
    # Get health status
    health = await orchestrator.get_federation_health()
    
    print(f"  ✓ Health status: {health.overall_status}")
    print(f"  ✓ Component health: {len(health.component_health)} components")
    print(f"  ✓ Strategy health: {len(health.strategies_health)} strategies")
    print(f"  ✓ Alerts: {len(health.alerts)} alerts")
    
    # Verify health status components are correct
    assert health.overall_status in ['healthy', 'at_risk', 'critical']
    assert len(health.component_health) > 0
    assert len(health.strategies_health) == 2  # Both strategies
    assert 'orchestrator' in health.component_health
    print("  ✓ Health monitoring components are correct")
    
    # Get full dashboard to test comprehensive monitoring
    dashboard = await orchestrator.get_federation_dashboard()
    print(f"  ✓ Dashboard contains federation metrics: {type(dashboard['federation_metrics'])}")
    print(f"  ✓ Strategy summary: {dashboard['strategy_summary']}")
    
    assert 'federation_health' in dashboard
    assert 'federation_metrics' in dashboard
    assert 'strategy_summary' in dashboard
    print("  ✓ Federation dashboard provides comprehensive monitoring")
    
    await orchestrator.stop()
    print("  ✓ Federation health monitoring test passed")


async def test_lifecycle_integrations():
    """
    Test that lifecycle management integrates properly with federation.
    """
    print("\nTesting lifecycle management integrations...")
    
    orchestrator = FederationOrchestrator()
    await orchestrator.start()
    
    # Add a strategy that should be retired in testing
    await orchestrator.add_strategy_to_federation(
        strategy_id="retirement_test",
        api_endpoint="http://localhost:8008",
        initial_capital=1000.0
    )
    
    print(f"  ✓ Added strategy for lifecycle testing")
    
    # Simulate performance data that would trigger retirement
    # In the integration, the lifecycle evaluation happens in the performance review cycle
    await orchestrator._evaluate_strategy_lifecycle(
        "retirement_test",
        {"sharpe": -0.6, "active": True}  # Poor performance
    )
    
    # Test the retirement function
    await orchestrator.remove_strategy_from_federation("retirement_test")
    
    # Check that strategy is retired
    strategy_pod = orchestrator._strategy_pods.get("retirement_test")
    assert strategy_pod is not None
    assert strategy_pod["status"] == "retired"
    assert strategy_pod["current_allocation"] == 0.0
    print("  ✓ Strategy successfully retired")
    
    # Verify dashboard reflects the retirement
    dashboard = await orchestrator.get_federation_dashboard()
    retired_count = dashboard["strategy_summary"]["retired"]
    assert retired_count >= 1
    print(f"  ✓ Dashboard shows {retired_count} retired strategy(s)")
    
    await orchestrator.stop()
    print("  ✓ Lifecycle management integration test passed")


async def test_admin_functions():
    """
    Test that administrative functions work properly in the integrated system.
    """
    print("\nTesting administrative functions...")
    
    orchestrator = FederationOrchestrator()
    await orchestrator.start()
    
    # Add a strategy
    await orchestrator.add_strategy_to_federation(
        strategy_id="admin_test",
        api_endpoint="http://localhost:8009",
        initial_capital=1000.0
    )
    
    print(f"  ✓ Added strategy for admin testing")
    
    # Test force allocation change
    await orchestrator.force_strategy_allocation("admin_test", 5000.0)
    
    # Check that allocation was changed
    strategy_pod = orchestrator._strategy_pods["admin_test"]
    assert strategy_pod["current_allocation"] == 5000.0
    print(f"  ✓ Allocation force changed: 1000.0 -> 5000.0")
    
    # Test event generation
    events_after = len(orchestrator._federation_events)
    assert events_after > 0  # Should have generated events
    print(f"  ✓ Federation events generated: {events_after} events")
    
    # Look for the allocation change event
    allocation_events = [
        e for e in orchestrator._federation_events 
        if e.event_type == 'allocation_force_changed'
    ]
    assert len(allocation_events) > 0
    print("  ✓ Allocation change event recorded correctly")
    
    await orchestrator.stop()
    print("  ✓ Administrative functions test passed")


async def run_all_tests():
    """
    Run all federation integration tests.
    """
    print("Running Federation Integration System Tests...\n")
    
    await test_federation_initialization()
    await test_strategy_registration_integration()
    await test_allocation_optimization_cycle()
    await test_federation_health_monitoring()
    await test_lifecycle_integrations()
    await test_admin_functions()
    
    print("\n✅ All Federation Integration System tests passed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())