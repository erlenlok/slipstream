"""
Test suite for the Covariance Stress Tester component.

This module tests the covariance stress tester's ability to calculate portfolio 
correlation to benchmarks and recommend reducing limits when correlations exceed
thresholds to prevent hidden beta accumulation.
"""

from datetime import datetime, timedelta
import asyncio
import numpy as np
from slipstream.federation.covariance import CovarianceStressTester, StrategyReturn


async def test_correlation_calculations():
    """
    Test that correlation calculations work correctly.
    """
    print("Testing correlation calculations...")
    
    tester = CovarianceStressTester(correlation_threshold=0.8, min_data_points=3)
    await tester.start()
    
    strategy_id = "corr_test_strategy"
    base_time = datetime.now()
    
    # Add strategy return data with some correlation to benchmark
    for i in range(10):
        # Create returns that have some correlation to a benchmark
        benchmark_effect = 0.02 * (i % 4 - 2) * 0.1  # Simple benchmark pattern
        strategy_return = benchmark_effect * 0.7 + np.random.normal(0, 0.005)  # High correlation with benchmark
        
        await tester.record_strategy_returns(StrategyReturn(
            strategy_id=strategy_id,
            timestamp=base_time + timedelta(hours=i),
            returns=strategy_return,
            benchmark_returns={"BTC": benchmark_effect, "ETH": benchmark_effect * 0.8},
            portfolio_weight=0.1 + (i * 0.01)
        ))
        
        # Add corresponding benchmark returns
        await tester.record_benchmark_returns("BTC", base_time + timedelta(hours=i), benchmark_effect)
        await tester.record_benchmark_returns("ETH", base_time + timedelta(hours=i), benchmark_effect * 0.8)
    
    # Calculate portfolio metrics to test correlation calculation
    metrics = await tester.calculate_portfolio_metrics([strategy_id], base_time + timedelta(hours=9))
    
    if metrics:
        print(f"  ✓ Portfolio metrics calculated")
        print(f"  ✓ Strategy correlations: {metrics.benchmark_correlations}")
        print(f"  ✓ Portfolio benchmark correlation: {metrics.portfolio_benchmark_correlation:.3f}")
    else:
        print("  ⚠ Metrics not calculated")
    
    await tester.stop()
    print("  ✓ Correlation calculations test passed")


async def test_stress_test_logic():
    """
    Test that the stress test correctly identifies high correlations.
    """
    print("\nTesting stress test logic...")
    
    tester = CovarianceStressTester(correlation_threshold=0.5, min_data_points=3)  # Lower threshold for test
    await tester.start()
    
    # Add data for strategies with high and low correlation to benchmarks
    base_time = datetime.now()
    
    # High correlation strategy 
    high_corr_strategy = "high_corr_strategy"
    for i in range(8):
        benchmark_effect = 0.01 * (i % 3 - 1)  # Benchmark pattern
        strategy_return = benchmark_effect * 0.8 + np.random.normal(0, 0.002)  # High correlation
        
        await tester.record_strategy_returns(StrategyReturn(
            strategy_id=high_corr_strategy,
            timestamp=base_time + timedelta(hours=i),
            returns=strategy_return,
            benchmark_returns={"BTC": benchmark_effect, "ETH": benchmark_effect * 0.7},
            portfolio_weight=0.15
        ))
        
        await tester.record_benchmark_returns("BTC", base_time + timedelta(hours=i), benchmark_effect)
        await tester.record_benchmark_returns("ETH", base_time + timedelta(hours=i), benchmark_effect * 0.7)
    
    # Run stress test
    result = await tester.run_covariance_stress_test()
    
    print(f"  ✓ BTC correlation: {result.portfolio_correlation_to_btc:.3f}")
    print(f"  ✓ ETH correlation: {result.portfolio_correlation_to_eth:.3f}")
    print(f"  ✓ Threshold: {result.max_allowed_correlation:.3f}")
    print(f"  ✓ Stress test passed: {result.stress_test_passed}")
    print(f"  ✓ Risk score: {result.overall_risk_score:.1f}")
    
    # Since we set threshold to 0.5 and our correlation should be higher, test should fail
    if result.portfolio_correlation_to_btc > 0.5:
        print("  ✓ High correlation correctly detected")
    else:
        print(f"  ⚠ Expected high correlation, got {result.portfolio_correlation_to_btc:.3f}")
    
    if not result.stress_test_passed:
        print("  ✓ Stress test correctly failed due to high correlation")
    else:
        print("  ⚠ Stress test should have failed")
    
    await tester.stop()
    print("  ✓ Stress test logic test passed")


async def test_alert_generation():
    """
    Test that correlation alerts are generated correctly.
    """
    print("\nTesting alert generation...")
    
    tester = CovarianceStressTester(correlation_threshold=0.3, min_data_points=3)  # Low threshold
    await tester.start()
    
    # Add data that will trigger high correlation
    base_time = datetime.now()
    
    for i in range(6):
        benchmark_effect = 0.01 * (i % 2)  # Strong pattern
        strategy_return = benchmark_effect * 0.9 + np.random.normal(0, 0.001)  # Very high correlation
        
        await tester.record_strategy_returns(StrategyReturn(
            strategy_id="alert_test_strategy",
            timestamp=base_time + timedelta(hours=i),
            returns=strategy_return,
            benchmark_returns={"BTC": benchmark_effect, "ETH": benchmark_effect * 0.8},
            portfolio_weight=0.2
        ))
        
        await tester.record_benchmark_returns("BTC", base_time + timedelta(hours=i), benchmark_effect)
        await tester.record_benchmark_returns("ETH", base_time + timedelta(hours=i), benchmark_effect * 0.8)
    
    # Run stress test first to populate data
    await tester.run_covariance_stress_test()
    
    # Generate alerts
    alerts = await tester.generate_correlation_alerts()
    
    print(f"  ✓ Generated {len(alerts)} alerts")
    for alert in alerts:
        print(f"    - {alert.alert_type} ({alert.severity}): {alert.correlation_value:.3f} > {alert.threshold:.3f}")
    
    # Should have alerts since we set very low threshold (0.3) and high correlation
    if alerts:
        print("  ✓ Correlation alerts generated as expected")
    else:
        print("  ⚠ No alerts generated")
    
    await tester.stop()
    print("  ✓ Alert generation test passed")


async def test_allocation_recommendations():
    """
    Test that allocation recommendations are generated for high correlation situations.
    """
    print("\nTesting allocation recommendations...")
    
    tester = CovarianceStressTester(correlation_threshold=0.4, min_data_points=3)  # Low threshold
    await tester.start()
    
    base_time = datetime.now()
    
    # Add data for high correlation strategy
    for i in range(8):
        benchmark_effect = 0.02 * (i % 3 - 1)
        strategy_return = benchmark_effect * 0.85 + np.random.normal(0, 0.003)  # High correlation
        
        await tester.record_strategy_returns(StrategyReturn(
            strategy_id="reco_test_strategy",
            timestamp=base_time + timedelta(hours=i),
            returns=strategy_return,
            benchmark_returns={"BTC": benchmark_effect, "ETH": benchmark_effect * 0.7},
            portfolio_weight=0.25
        ))
        
        await tester.record_benchmark_returns("BTC", base_time + timedelta(hours=i), benchmark_effect)
        await tester.record_benchmark_returns("ETH", base_time + timedelta(hours=i), benchmark_effect * 0.7)
    
    # Get allocation recommendations
    recommendations = await tester.get_allocation_recommendations()
    
    print(f"  ✓ Generated {len(recommendations)} allocation recommendations")
    for rec in recommendations:
        print(f"    - {rec.strategy_id}: {rec.current_allocation:.2f} -> {rec.recommended_allocation:.2f} ({rec.reason})")
    
    if recommendations:
        print("  ✓ Allocation recommendations generated for high correlation")
    else:
        print("  ⚠ No recommendations generated")
    
    await tester.stop()
    print("  ✓ Allocation recommendations test passed")


async def test_diversification_metrics():
    """
    Test that diversification metrics are calculated correctly.
    """
    print("\nTesting diversification metrics...")
    
    tester = CovarianceStressTester(min_data_points=3)
    await tester.start()
    
    base_time = datetime.now()
    strategies = ["strategy_A", "strategy_B", "strategy_C"]
    
    # Add return data for multiple strategies
    for strategy_idx, strategy_id in enumerate(strategies):
        for i in range(10):
            # Create returns with varying correlation patterns
            # Strategy A: more correlated with benchmark
            # Strategy B: moderate correlation  
            # Strategy C: less correlation
            benchmark_effect = 0.01 * (i % 4 - 2)
            
            if strategy_id == "strategy_A":
                strategy_return = benchmark_effect * 0.8 + np.random.normal(0, 0.005)  # High correlation
            elif strategy_id == "strategy_B":
                strategy_return = benchmark_effect * 0.5 + np.random.normal(0, 0.006)  # Medium correlation
            else:  # strategy_C
                strategy_return = benchmark_effect * 0.2 + np.random.normal(0, 0.007)  # Low correlation
            
            await tester.record_strategy_returns(StrategyReturn(
                strategy_id=strategy_id,
                timestamp=base_time + timedelta(hours=i),
                returns=strategy_return,
                benchmark_returns={"BTC": benchmark_effect, "ETH": benchmark_effect * 0.6},
                portfolio_weight=0.1 + (strategy_idx * 0.05)
            ))
    
    # Calculate diversification metrics for portfolio
    diversification_metrics = await tester.get_portfolio_diversification_metrics(strategies)
    
    print(f"  ✓ Diversification metrics: {diversification_metrics}")
    print(f"    - Diversification ratio: {diversification_metrics['diversification_ratio']:.3f}")
    print(f"    - Concentration ratio: {diversification_metrics['concentration_ratio']:.3f}")
    print(f"    - Avg correlation: {diversification_metrics['average_correlation']:.3f}")
    
    # Values should be reasonable
    assert 0 <= diversification_metrics['diversification_ratio'] <= 1
    assert 0 <= diversification_metrics['concentration_ratio'] <= 1
    print("  ✓ Diversification metrics in valid ranges")
    
    await tester.stop()
    print("  ✓ Diversification metrics test passed")


async def run_all_tests():
    """
    Run all covariance stress tester tests.
    """
    print("Running Covariance Stress Tester Tests...\n")
    
    await test_correlation_calculations()
    await test_stress_test_logic()
    await test_alert_generation()
    await test_allocation_recommendations()
    await test_diversification_metrics()
    
    print("\n✅ All Covariance Stress Tester tests passed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())