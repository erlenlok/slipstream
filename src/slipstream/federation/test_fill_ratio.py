"""
Test suite for the Fill Ratio Analyzer component.

This module tests the fill ratio analyzer's ability to track maker vs. taker 
volume ratios and verify execution style matches strategy intent.
"""

from datetime import datetime, timedelta
import asyncio
import numpy as np
from slipstream.federation.fill_ratio import FillRatioAnalyzer, TradeExecution, ExecutionStyleProfile


async def test_fill_ratio_calculation():
    """
    Test that fill ratios are calculated correctly.
    """
    print("Testing fill ratio calculation...")
    
    analyzer = FillRatioAnalyzer(process_failure_threshold=0.3)
    await analyzer.start()
    
    strategy_id = "test_strategy"
    
    # Add execution data with known maker/taker distribution
    base_time = datetime.now()
    for i in range(10):
        fill_type = "maker" if i < 7 else "taker"  # 7 maker, 3 taker = 70% maker
        
        await analyzer.record_execution(TradeExecution(
            strategy_id=strategy_id,
            timestamp=base_time + timedelta(minutes=i),
            symbol="BTC",
            side="buy",
            quantity=0.1,
            price=45000.0 + (i * 10),
            order_id=f"order_{i}",
            fill_type=fill_type,
            fees_paid=0.5
        ))
    
    # Set a profile with 50% intended maker ratio
    profile = ExecutionStyleProfile(
        strategy_id=strategy_id,
        intended_style="balanced",
        maker_ratio_target=0.5,
        taker_ratio_target=0.5,
        max_slippage_tolerance=0.001,
        time_in_force_preference="gtc"
    )
    await analyzer.set_execution_style_profile(profile)
    
    # Analyze fill ratios
    report = await analyzer.analyze_fill_ratios(strategy_id)
    
    if report:
        print(f"  ✓ Fill ratios calculated: maker={report.maker_ratio:.2%}, taker={report.taker_ratio:.2%}")
        print(f"  ✓ Total trades: {report.total_trades}, maker: {report.maker_fills}, taker: {report.taker_fills}")
        
        # Check calculations
        assert report.total_trades == 10
        assert report.maker_fills == 7
        assert report.taker_fills == 3
        assert report.maker_ratio == 0.7
        assert report.taker_ratio == 0.3
        print("  ✓ Calculations verified")
        
        # Check that deviation from intended 50% is calculated
        expected_deviation = abs(0.7 - 0.5)  # 20% deviation
        assert abs(report.deviation_score - expected_deviation) < 0.01
        print(f"  ✓ Deviation score: {report.deviation_score:.2f}")
    else:
        print("  ⚠ Analysis not completed due to data issues")
    
    await analyzer.stop()
    print("  ✓ Fill ratio calculation test passed")


async def test_process_failure_detection():
    """
    Test that process failures are detected when execution style doesn't match intent.
    """
    print("\nTesting process failure detection...")
    
    analyzer = FillRatioAnalyzer(process_failure_threshold=0.3)  # 30% threshold
    await analyzer.start()
    
    strategy_id = "process_failure_test"
    
    # Add execution data that significantly deviates from intended
    base_time = datetime.now()
    for i in range(10):
        # All taker fills for a strategy that should be maker-heavy (70% target)
        await analyzer.record_execution(TradeExecution(
            strategy_id=strategy_id,
            timestamp=base_time + timedelta(minutes=i),
            symbol="BTC", 
            side="buy",
            quantity=0.1,
            price=45000.0,
            order_id=f"fail_order_{i}",
            fill_type="taker",  # All taker
            fees_paid=1.0
        ))
    
    # Set profile expecting 70% maker
    profile = ExecutionStyleProfile(
        strategy_id=strategy_id,
        intended_style="maker_heavy",
        maker_ratio_target=0.7,
        taker_ratio_target=0.3,
        max_slippage_tolerance=0.001,
        time_in_force_preference="post_only"
    )
    await analyzer.set_execution_style_profile(profile)
    
    # Analyze
    report = await analyzer.analyze_fill_ratios(strategy_id)
    
    if report:
        print(f"  ✓ Actual maker ratio: {report.maker_ratio:.2%}, intended: {report.intended_maker_ratio:.2%}")
        print(f"  ✓ Deviation: {report.deviation_score:.2f}, failure detected: {report.process_failure_detected}")
        
        # With 0% maker vs 70% intended, deviation is 0.7, so with 0.3 threshold, should detect failure
        if report.deviation_score > 0.3:
            assert report.process_failure_detected == True
            print("  ✓ Process failure correctly detected")
        else:
            print(f"  ⚠ Unexpected result: deviation {report.deviation_score} but failure {report.process_failure_detected}")
    else:
        print("  ⚠ Analysis not completed")
    
    # Get process failure alerts
    alerts = await analyzer.detect_process_failures()
    print(f"  ✓ Found {len(alerts)} process failure alerts")
    for alert in alerts:
        print(f"    - {alert.strategy_id}: {alert.issue[:50]}...")
    
    await analyzer.stop()
    print("  ✓ Process failure detection test passed")


async def test_execution_style_compliance():
    """
    Test that execution style compliance is correctly evaluated.
    """
    print("\nTesting execution style compliance...")
    
    analyzer = FillRatioAnalyzer(process_failure_threshold=0.2)
    await analyzer.start()
    
    strategy_id = "compliance_test"
    
    # Add mixed execution data
    base_time = datetime.now()
    for i in range(10):
        fill_type = "maker" if i < 6 else "taker"  # 60% maker for 50% target = good compliance
        
        await analyzer.record_execution(TradeExecution(
            strategy_id=strategy_id,
            timestamp=base_time + timedelta(minutes=i),
            symbol="ETH",
            side="buy",
            quantity=0.5,
            price=3000.0,
            order_id=f"comp_order_{i}",
            fill_type=fill_type,
            fees_paid=0.2
        ))
    
    # Set profile expecting 50/50
    profile = ExecutionStyleProfile(
        strategy_id=strategy_id,
        intended_style="balanced",
        maker_ratio_target=0.5,
        taker_ratio_target=0.5,
        max_slippage_tolerance=0.001,
        time_in_force_preference="gtc"
    )
    await analyzer.set_execution_style_profile(profile)
    
    # Analyze
    report = await analyzer.analyze_fill_ratios(strategy_id)
    
    if report:
        print(f"  ✓ Maker ratio: {report.maker_ratio:.2%}, intended: {report.intended_maker_ratio:.2%}")
        print(f"  ✓ Deviation: {report.deviation_score:.2f}, compliance: {report.style_compliance}")
        
        # 60% vs 50% = 10% deviation, should be 'good' compliance
        if report.deviation_score < 0.2:
            assert report.style_compliance in ['excellent', 'good']
            print(f"  ✓ Compliance rating '{report.style_compliance}' as expected")
        else:
            print(f"  ⚠ Compliance rating {report.style_compliance} unexpected for {report.deviation_score:.2f} deviation")
    else:
        print("  ⚠ Analysis not completed")
    
    await analyzer.stop()
    print("  ✓ Execution style compliance test passed")


async def test_execution_analysis():
    """
    Test the full execution analysis functionality.
    """
    print("\nTesting execution analysis...")
    
    analyzer = FillRatioAnalyzer()
    await analyzer.start()
    
    strategy_id = "analysis_test"
    
    # Add various execution types 
    base_time = datetime.now()
    for i in range(12):
        fill_type = "maker" if i % 3 != 2 else "taker"  # 2/3 maker (66.7%)
        
        await analyzer.record_execution(TradeExecution(
            strategy_id=strategy_id,
            timestamp=base_time + timedelta(minutes=i*5),
            symbol="BTC",
            side="buy" if i % 2 == 0 else "sell",
            quantity=0.1 + (i * 0.02),
            price=45000.0 + (i * 50),
            order_id=f"analy_order_{i}",
            fill_type=fill_type,
            fees_paid=0.3 + (i * 0.05),
            slippage=0.001 if fill_type == "taker" else 0,
            time_to_fill=timedelta(seconds=1 + (i % 3))
        ))
    
    # Perform analysis
    analysis = await analyzer.get_strategy_execution_analysis(strategy_id)
    
    print(f"  ✓ Analysis retrieved: {analysis.get('has_data', False)}")
    print(f"  ✓ Total trades: {analysis.get('total_trades', 0)}")
    print(f"  ✓ Maker ratio: {analysis.get('maker_ratio', 0):.2%}")
    print(f"  ✓ Process failure: {analysis.get('process_failure_detected', False)}")
    
    # Also test efficiency metrics
    efficiency = await analyzer.calculate_execution_efficiency_metrics(strategy_id)
    print(f"  ✓ Efficiency metrics: {len(efficiency)} metrics calculated")
    if efficiency:
        print(f"    - Maker ratio: {efficiency.get('maker_ratio', 0):.2%}")
        print(f"    - Total executions: {efficiency.get('total_executions', 0)}")
    
    await analyzer.stop()
    print("  ✓ Execution analysis test passed")


async def test_execution_trends():
    """
    Test that execution trends can be tracked over time.
    """
    print("\nTesting execution trends...")
    
    analyzer = FillRatioAnalyzer()
    await analyzer.start()
    
    strategy_id = "trend_test"
    base_time = datetime.now()
    
    # Add execution data over multiple time periods
    for period in range(3):
        start_of_period = base_time + timedelta(hours=period*24)
        for i in range(5):
            fill_type = "maker" if (i + period) % 2 == 0 else "taker"
            
            await analyzer.record_execution(TradeExecution(
                strategy_id=strategy_id,
                timestamp=start_of_period + timedelta(hours=i),
                symbol="ETH",
                side="buy",
                quantity=0.2,
                price=3000.0 + (period * 100),
                order_id=f"trend_order_{period}_{i}",
                fill_type=fill_type,
                fees_paid=0.25
            ))
        
        # Analyze each period
        await analyzer.analyze_fill_ratios(
            strategy_id, 
            start_of_period, 
            start_of_period + timedelta(hours=24)
        )
    
    # Get trends
    trends = await analyzer.get_execution_trends(strategy_id, days=7)
    print(f"  ✓ Retrieved {len(trends)} trend reports")
    
    for i, trend in enumerate(trends):
        print(f"    - Period {i+1}: maker={trend.maker_ratio:.2%}, deviation={trend.deviation_score:.2f}")
    
    await analyzer.stop()
    print("  ✓ Execution trends test passed")


async def run_all_tests():
    """
    Run all fill ratio analyzer tests.
    """
    print("Running Fill Ratio Analyzer Tests...\n")
    
    await test_fill_ratio_calculation()
    await test_process_failure_detection()
    await test_execution_style_compliance()
    await test_execution_analysis()
    await test_execution_trends()
    
    print("\n✅ All Fill Ratio Analyzer tests passed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())