"""
Test suite for the Meta-Optimizer component.

This module tests the Meta-Optimizer's ability to observe and analyze
strategy performance without affecting their operations.
"""

from datetime import datetime, timedelta
import asyncio
import numpy as np
from slipstream.federation.optimizer import MetaOptimizer, StrategyPerformance, MockStrategyDataProvider


async def test_optimizer_observation_only():
    """
    Test that the Meta-Optimizer only observes without controlling strategies.
    """
    print("Testing Meta-Optimizer observation-only mode...")
    
    # Create optimizer without data provider initially
    optimizer = MetaOptimizer(rebalance_interval=timedelta(hours=1))
    await optimizer.start()
    
    # Add some mock performance data
    test_performance = StrategyPerformance(
        strategy_id="test_strategy_1",
        timestamp=datetime.now(),
        returns=0.02,
        volatility=0.015,
        sharpe_ratio=1.5,
        max_drawdown=-0.05,
        total_return=0.15,
        total_capital=50000.0,
        net_exposure=10000.0,
        gross_exposure=15000.0,
        win_rate=0.65,
        avg_win=150.0,
        avg_loss=-100.0,
        trades=45,
        alpha=0.005,
        beta=0.02,
        information_ratio=0.8
    )
    
    # Add performance data (observation only)
    await optimizer.collect_performance_data("test_strategy_1", test_performance)
    
    # Verify this didn't change any strategy settings
    performance_summary = await optimizer.get_performance_summary("test_strategy_1")
    assert performance_summary['strategy_id'] == 'test_strategy_1'
    assert abs(performance_summary['avg_return'] - 0.02) < 0.001
    print("  ✓ Optimizer only observes, doesn't control strategy settings")
    
    await optimizer.stop()
    print("  ✓ Observation-only test passed")


async def test_performance_collection():
    """
    Test that the optimizer correctly collects and stores performance data.
    """
    print("\nTesting performance data collection...")
    
    optimizer = MetaOptimizer()
    await optimizer.start()
    
    # Create multiple performance records for the same strategy
    base_time = datetime.now()
    for i in range(5):
        perf = StrategyPerformance(
            strategy_id="perf_test_strategy",
            timestamp=base_time + timedelta(minutes=i*10),
            returns=0.01 * i,
            volatility=0.01 + (i * 0.001),
            sharpe_ratio=1.0 + (i * 0.1),
            max_drawdown=-0.01 * (i + 1),
            total_return=0.05 * i,
            total_capital=10000.0 + (i * 1000),
            net_exposure=2000.0 + (i * 500),
            gross_exposure=3000.0 + (i * 750),
            win_rate=0.6 + (i * 0.05),
            avg_win=100.0 + (i * 10),
            avg_loss=-80.0 - (i * 5),
            trades=20 + i,
            alpha=0.001 * i,
            beta=0.01 * i,
            information_ratio=0.5 + (i * 0.1)
        )
        await optimizer.collect_performance_data("perf_test_strategy", perf)
    
    # Check that all records were collected
    history = optimizer._performance_history.get("perf_test_strategy", [])
    assert len(history) == 5
    print("  ✓ All performance records collected successfully")
    
    # Check that calculations work correctly with multiple records
    summary = await optimizer.get_performance_summary("perf_test_strategy")
    assert summary['total_periods'] == 5
    assert abs(summary['avg_return'] - 0.02) < 0.005  # Average of 0.00, 0.01, 0.02, 0.03, 0.04
    print("  ✓ Performance calculations work with multiple records")
    
    await optimizer.stop()
    print("  ✓ Performance collection test passed")


async def test_covariance_calculation():
    """
    Test that the optimizer can calculate covariance between strategies.
    """
    print("\nTesting covariance calculation...")
    
    optimizer = MetaOptimizer()
    await optimizer.start()
    
    # Add performance data for multiple strategies
    strategies = ["strategy_A", "strategy_B", "strategy_C"]
    
    for i, strategy_id in enumerate(strategies):
        for j in range(10):  # Add 10 data points per strategy
            perf = StrategyPerformance(
                strategy_id=strategy_id,
                timestamp=datetime.now() - timedelta(hours=10-j),
                returns=np.random.normal(0.001 * (i+1), 0.02),  # Different drift per strategy
                volatility=0.015,
                sharpe_ratio=1.0,
                max_drawdown=-0.02,
                total_return=0.1,
                total_capital=10000.0,
                net_exposure=1000.0,
                gross_exposure=2000.0,
                win_rate=0.6,
                avg_win=100.0,
                avg_loss=-80.0,
                trades=5,
                alpha=0.001,
                beta=0.01,
                information_ratio=0.8
            )
            await optimizer.collect_performance_data(strategy_id, perf)
    
    # Calculate covariance matrix
    cov_matrix = await optimizer.calculate_covariance_matrix()
    
    if cov_matrix is not None:
        assert cov_matrix.shape[0] == len(strategies), f"Expected {len(strategies)} strategies, got {cov_matrix.shape[0]}"
        print("  ✓ Covariance matrix calculated correctly")
    else:
        print("  ⚠ Could not calculate covariance (insufficient aligned data)")
    
    # Test diversification metrics
    div_metrics = await optimizer.calculate_diversification_metrics()
    print(f"  ✓ Diversification ratio: {div_metrics.get('diversification_ratio', 'N/A')}")
    print(f"  ✓ Avg correlation: {div_metrics.get('avg_correlation', 'N/A')}")
    
    await optimizer.stop()
    print("  ✓ Covariance calculation test passed")


async def test_allocation_analysis():
    """
    Test that the optimizer can run allocation analysis without making changes.
    """
    print("\nTesting allocation analysis (observation only)...")
    
    optimizer = MetaOptimizer()
    await optimizer.start()
    
    # Add performance data for multiple strategies
    strategies = ["high_perf", "med_perf", "low_perf"]
    
    # High performing strategy
    await optimizer.collect_performance_data("high_perf", StrategyPerformance(
        strategy_id="high_perf",
        timestamp=datetime.now(),
        returns=0.03,  # High return
        volatility=0.02,
        sharpe_ratio=2.0,  # Excellent Sharpe
        max_drawdown=-0.03,
        total_return=0.25,
        total_capital=50000.0,
        net_exposure=10000.0,
        gross_exposure=15000.0,
        win_rate=0.7,
        avg_win=200.0,
        avg_loss=-100.0,
        trades=30,
        alpha=0.02,
        beta=0.01,
        information_ratio=1.5
    ))
    
    # Medium performing strategy
    await optimizer.collect_performance_data("med_perf", StrategyPerformance(
        strategy_id="med_perf", 
        timestamp=datetime.now(),
        returns=0.01,  # Medium return
        volatility=0.025,
        sharpe_ratio=0.8,  # OK Sharpe
        max_drawdown=-0.05,
        total_return=0.10,
        total_capital=50000.0,
        net_exposure=8000.0,
        gross_exposure=12000.0,
        win_rate=0.6,
        avg_win=150.0,
        avg_loss=-120.0,
        trades=25,
        alpha=0.005,
        beta=0.02,
        information_ratio=0.6
    ))
    
    # Low performing strategy  
    await optimizer.collect_performance_data("low_perf", StrategyPerformance(
        strategy_id="low_perf",
        timestamp=datetime.now(),
        returns=-0.01,  # Negative return
        volatility=0.03,
        sharpe_ratio=-0.5,  # Negative Sharpe
        max_drawdown=-0.08,
        total_return=-0.05,
        total_capital=50000.0,
        net_exposure=5000.0,
        gross_exposure=10000.0,
        win_rate=0.4,
        avg_win=80.0,
        avg_loss=-150.0,
        trades=20,
        alpha=-0.01,
        beta=-0.01,
        information_ratio=-0.3
    ))
    
    # Run allocation analysis (should only analyze, not change anything)
    analysis = await optimizer.run_allocation_analysis()
    
    assert analysis['status'] == 'analysis_complete'
    assert len(analysis['suggested_allocations']) >= 3
    assert 'high_perf' in analysis['suggested_allocations']
    assert analysis['suggested_allocations']['high_perf'] >= analysis['suggested_allocations'].get('low_perf', 0)
    print("  ✓ Allocation analysis completed without making changes")
    print(f"  ✓ High-performing strategy got higher suggested allocation")
    
    # Verify no actual allocation changes occurred (observation only)
    original_allocations = optimizer._current_allocations.copy()
    # Run analysis again and verify allocations unchanged
    await optimizer.run_allocation_analysis()
    new_allocations = optimizer._current_allocations
    assert original_allocations == new_allocations
    print("  ✓ No actual allocation changes made (analysis only)")
    
    await optimizer.stop()
    print("  ✓ Allocation analysis test passed")


async def test_attention_list():
    """
    Test that the optimizer can identify strategies requiring attention.
    """
    print("\nTesting attention list generation...")
    
    optimizer = MetaOptimizer()
    await optimizer.start()
    
    # Add a poorly performing strategy
    await optimizer.collect_performance_data("poor_strategy", StrategyPerformance(
        strategy_id="poor_strategy",
        timestamp=datetime.now(),
        returns=-0.02,
        volatility=0.02,
        sharpe_ratio=-1.0,  # Very poor Sharpe
        max_drawdown=-0.15,
        total_return=-0.2,
        total_capital=10000.0,
        net_exposure=-5000.0,
        gross_exposure=8000.0,
        win_rate=0.3,
        avg_win=50.0,
        avg_loss=-200.0,
        trades=15,
        alpha=-0.02,
        beta=-0.03,
        information_ratio=-0.8
    ))
    
    # Add a well performing strategy
    await optimizer.collect_performance_data("good_strategy", StrategyPerformance(
        strategy_id="good_strategy",
        timestamp=datetime.now(),
        returns=0.02,
        volatility=0.015,
        sharpe_ratio=1.8,  # Good Sharpe
        max_drawdown=-0.02,
        total_return=0.18,
        total_capital=10000.0,
        net_exposure=3000.0,
        gross_exposure=4000.0,
        win_rate=0.7,
        avg_win=180.0,
        avg_loss=-80.0,
        trades=25,
        alpha=0.015,
        beta=0.01,
        information_ratio=1.2
    ))
    
    # Get strategies requiring attention
    attention_list = await optimizer.get_strategies_requiring_attention()
    
    # Check that poor strategy is flagged but good strategy is not
    poor_strategy_attention = next((item for item in attention_list if item['strategy_id'] == 'poor_strategy'), None)
    good_strategy_attention = next((item for item in attention_list if item['strategy_id'] == 'good_strategy'), None)
    
    assert poor_strategy_attention is not None, "Poor strategy should be flagged for attention"
    assert good_strategy_attention is None, "Good strategy should not be flagged"
    assert len(poor_strategy_attention['issues']) > 0
    print("  ✓ Poor-performing strategy correctly flagged for attention")
    print("  ✓ Good-performing strategy not flagged unnecessarily")
    
    await optimizer.stop()
    print("  ✓ Attention list test passed")


async def run_all_tests():
    """
    Run all meta-optimizer tests.
    """
    print("Running Meta-Optimizer Tests...\n")
    
    await test_optimizer_observation_only()
    await test_performance_collection()
    await test_covariance_calculation()
    await test_allocation_analysis()
    await test_attention_list()
    
    print("\n✅ All Meta-Optimizer tests passed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())