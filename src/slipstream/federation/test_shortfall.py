"""
Test suite for the Implementation Shortfall Analyzer component.

This module tests the shortfall analyzer's ability to track decision vs. realized prices
and calculate implementation shortfall without affecting strategy operations.
"""

from datetime import datetime, timedelta
import asyncio
import numpy as np
import logging
# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

from slipstream.federation.shortfall import ImplementationShortfallAnalyzer, TradeDecision, FillEvent


async def test_shortfall_calculation():
    """
    Test that shortfall is correctly calculated as |Decision - Realized|.
    """
    print("Testing shortfall calculation...")
    
    analyzer = ImplementationShortfallAnalyzer()
    await analyzer.start()
    
    # Create a trade decision
    decision = TradeDecision(
        strategy_id="test_strategy",
        timestamp=datetime.now(),
        symbol="BTC",
        side="buy",
        decision_price=45000.0,
        quantity=1.0,
        decision_reason="momentum_signal",
        order_id="order_123",
        signal_strength=0.8,
        expected_pnl=200.0
    )
    
    # Create a matching fill event with some slippage
    fill = FillEvent(
        strategy_id="test_strategy",
        timestamp=decision.timestamp + timedelta(seconds=5),
        symbol="BTC",
        side="buy",
        fill_price=45025.0,  # 25 USD higher (slippage)
        fill_quantity=1.0,
        order_id="order_123",
        fees=1.0
    )
    
    # Calculate shortfall
    report = await analyzer.calculate_shortfall(decision, fill)
    
    # Check shortfall calculation
    expected_shortfall = abs(45000.0 - 45025.0)  # 25.0 USD
    expected_pct_shortfall = 25.0 / 45000.0  # ~0.000555 or ~5.55 bps
    
    assert abs(report.shortfall - expected_shortfall) < 0.01
    assert abs(report.analysis['shortfall_bps'] - expected_pct_shortfall * 10000) < 0.1
    print(f"  ✓ Shortfall calculated correctly: {report.shortfall} USD, {report.analysis['shortfall_bps']:.2f} bps")
    
    # Check cost impact calculation (negative for buy with bad slippage)
    expected_cost = (45025.0 - 45000.0) * 1.0  # 25 USD cost due to slippage
    assert abs(report.analysis['cost_impact'] - expected_cost) < 0.01
    print(f"  ✓ Cost impact calculated correctly: {report.analysis['cost_impact']:.2f} USD")
    
    await analyzer.stop()
    print("  ✓ Shortfall calculation test passed")


async def test_execution_quality_classification():
    """
    Test that execution quality is correctly classified based on shortfall.
    """
    print("\nTesting execution quality classification...")
    
    analyzer = ImplementationShortfallAnalyzer()
    await analyzer.start()
    
    # Test excellent execution (very low shortfall)
    decision_excellent = TradeDecision(
        strategy_id="test_strategy",
        timestamp=datetime.now(),
        symbol="ETH",
        side="buy",
        decision_price=3000.0,
        quantity=2.0,
        decision_reason="signal",
        order_id="order_excellent",
        signal_strength=0.9
    )
    fill_excellent = FillEvent(
        strategy_id="test_strategy",
        timestamp=datetime.now(),
        symbol="ETH", 
        side="buy",
        fill_price=3000.01,  # Very small slippage
        fill_quantity=2.0,
        order_id="order_excellent"
    )
    
    report_excellent = await analyzer.calculate_shortfall(decision_excellent, fill_excellent)
    assert report_excellent.execution_quality == "excellent"
    print("  ✓ Excellent execution correctly classified")
    
    # Test poor execution (high shortfall)
    decision_poor = TradeDecision(
        strategy_id="test_strategy",
        timestamp=datetime.now(),
        symbol="BTC",
        side="buy", 
        decision_price=45000.0,
        quantity=1.0,
        decision_reason="signal",
        order_id="order_poor"
    )
    fill_poor = FillEvent(
        strategy_id="test_strategy",
        timestamp=datetime.now(),
        symbol="BTC",
        side="buy",
        fill_price=45250.0,  # 250 USD slippage (0.55% = 55 bps)
        fill_quantity=1.0,
        order_id="order_poor"
    )
    
    report_poor = await analyzer.calculate_shortfall(decision_poor, fill_poor)
    assert report_poor.execution_quality in ["poor", "failure"]  # Depends on exact thresholds
    print(f"  ✓ Poor execution correctly classified as '{report_poor.execution_quality}'")
    
    await analyzer.stop()
    print("  ✓ Execution quality classification test passed")


async def test_decision_fill_matching():
    """
    Test that decisions are correctly matched with fills.
    """
    print("\nTesting decision-fill matching...")
    
    analyzer = ImplementationShortfallAnalyzer()
    await analyzer.start()
    
    strategy_id = "match_test_strategy"
    
    # Record some decisions
    decision1 = TradeDecision(
        strategy_id=strategy_id,
        timestamp=datetime.now(),
        symbol="BTC",
        side="buy",
        decision_price=45000.0,
        quantity=1.0,
        decision_reason="signal",
        order_id="match_order_1"
    )
    
    decision2 = TradeDecision(
        strategy_id=strategy_id,
        timestamp=datetime.now(),
        symbol="ETH", 
        side="sell",
        decision_price=3000.0,
        quantity=2.0,
        decision_reason="signal",
        order_id="match_order_2"
    )
    
    await analyzer.record_trade_decision(decision1)
    await analyzer.record_trade_decision(decision2)
    
    # Record fills (one matching, one not matching)
    fill1 = FillEvent(
        strategy_id=strategy_id,
        timestamp=datetime.now(),
        symbol="BTC",
        side="buy",
        fill_price=45010.0,
        fill_quantity=1.0,
        order_id="match_order_1"  # This matches
    )
    
    fill2 = FillEvent(
        strategy_id=strategy_id,
        timestamp=datetime.now(),
        symbol="ETH",
        side="sell",
        fill_price=2995.0,
        fill_quantity=2.0,
        order_id="match_order_2"  # This matches
    )
    
    fill3 = FillEvent(
        strategy_id=strategy_id,
        timestamp=datetime.now(),
        symbol="ADA",
        side="buy",
        fill_price=0.50,
        fill_quantity=100.0,
        order_id="non_match_order"  # This does not match
    )
    
    await analyzer.record_fill_event(fill1)
    await analyzer.record_fill_event(fill2)
    await analyzer.record_fill_event(fill3)
    
    # Check matching
    matched_pairs = await analyzer.match_decisions_and_fills(strategy_id)
    
    # Should have 2 matched pairs (decision1-fill1, decision2-fill2)
    assert len(matched_pairs) == 2
    
    # Verify specific matches
    for decision, fill in matched_pairs:
        assert decision.order_id == fill.order_id
    
    print(f"  ✓ Correctly matched {len(matched_pairs)} decision-fill pairs")
    
    # Analyze matched trades
    reports = await analyzer.analyze_strategy_shortfall(strategy_id)
    assert len(reports) == 2  # Two matched trades
    print("  ✓ Shortfall analysis completed for matched trades")
    
    await analyzer.stop()
    print("  ✓ Decision-fill matching test passed")


async def test_aggregate_metrics():
    """
    Test that aggregate shortfall metrics are calculated correctly.
    """
    print("\nTesting aggregate shortfall metrics...")
    
    analyzer = ImplementationShortfallAnalyzer()
    await analyzer.start()
    
    strategy_id = "agg_test_strategy"
    
    # Create multiple trades with varying shortfalls
    base_time = datetime.now()
    shortfalls_to_create = [0.0001, 0.0005, 0.001, 0.002, 0.005]  # 1, 5, 10, 20, 50 bps
    
    for i, shortfall in enumerate(shortfalls_to_create):
        decision = TradeDecision(
            strategy_id=strategy_id,
            timestamp=base_time + timedelta(minutes=i),
            symbol=f"SYM{i}",
            side="buy",
            decision_price=100.0,
            quantity=1.0,
            decision_reason="test",
            order_id=f"agg_order_{i}"
        )
        
        # Calculate fill price to achieve desired shortfall
        fill_price = 100.0 * (1 + shortfall)  # Positive shortfall for buy orders
        
        fill = FillEvent(
            strategy_id=strategy_id,
            timestamp=base_time + timedelta(minutes=i, seconds=5),
            symbol=f"SYM{i}",
            side="buy",
            fill_price=fill_price,
            fill_quantity=1.0,
            order_id=f"agg_order_{i}"
        )
        
        await analyzer.record_trade_decision(decision)
        await analyzer.record_fill_event(fill)
        
        # This will also calculate the shortfall report
        await analyzer.calculate_shortfall(decision, fill)
    
    # Calculate aggregate metrics
    metrics = await analyzer.calculate_aggregate_metrics(strategy_id)
    
    assert metrics.total_trades == len(shortfalls_to_create)
    assert metrics.avg_shortfall > 0  # Should have some average shortfall
    assert 0 <= metrics.quality_score <= 100  # Quality score should be in range
    
    print(f"  ✓ Aggregate metrics calculated: {metrics.total_trades} trades, avg_shortfall={metrics.avg_shortfall*10000:.2f} bps")
    print(f"  ✓ Quality score: {metrics.quality_score:.1f}, total_cost={metrics.total_cost_due_to_shortfall:.2f} USD")
    
    # Test execution quality summary
    summary = await analyzer.get_strategy_execution_quality_summary(strategy_id)
    assert summary['has_data'] == True
    assert summary['total_trades_analyzed'] == len(shortfalls_to_create)
    print("  ✓ Execution quality summary generated correctly")
    
    await analyzer.stop()
    print("  ✓ Aggregate metrics test passed")


async def test_high_shortfall_detection():
    """
    Test that high shortfall trades are correctly identified for review.
    """
    print("\nTesting high shortfall detection...")
    
    analyzer = ImplementationShortfallAnalyzer(high_shortfall_threshold=0.001)  # 10 bps threshold
    await analyzer.start()
    
    strategy_id = "review_test_strategy"
    
    # Create trades with both low and high shortfall
    base_time = datetime.now()
    
    # Low shortfall trade (should not require review)
    await analyzer.record_trade_decision(TradeDecision(
        strategy_id=strategy_id,
        timestamp=base_time,
        symbol="LOW",
        side="buy",
        decision_price=100.0,
        quantity=1.0,
        decision_reason="test",
        order_id="low_sf_order"
    ))
    
    await analyzer.record_fill_event(FillEvent(
        strategy_id=strategy_id,
        timestamp=base_time + timedelta(seconds=1),
        symbol="LOW",
        side="buy",
        fill_price=100.0005,  # Very low shortfall: 0.5 bps
        fill_quantity=1.0,
        order_id="low_sf_order"
    ))
    
    # High shortfall trade (should require review)
    await analyzer.record_trade_decision(TradeDecision(
        strategy_id=strategy_id,
        timestamp=base_time + timedelta(minutes=1),
        symbol="HIGH",
        side="buy",
        decision_price=100.0,
        quantity=1.0,
        decision_reason="test",
        order_id="high_sf_order"
    ))
    
    await analyzer.record_fill_event(FillEvent(
        strategy_id=strategy_id,
        timestamp=base_time + timedelta(minutes=1, seconds=1),
        symbol="HIGH",
        side="buy",
        fill_price=100.15,  # High shortfall: 150 bps
        fill_quantity=1.0,
        order_id="high_sf_order"
    ))
    
    # Run analysis to create reports
    await analyzer.analyze_strategy_shortfall(strategy_id)
    
    # Get trades requiring execution review
    review_trades = await analyzer.get_trades_requiring_execution_review(strategy_id)
    
    # Should only have the high shortfall trade
    assert len(review_trades) == 1
    assert review_trades[0].trade_id == "high_sf_order"
    high_shortfall_bps = review_trades[0].analysis['shortfall_bps']
    assert high_shortfall_bps >= 10  # Above threshold
    print(f"  ✓ High shortfall trade detected: {high_shortfall_bps:.2f} bps")
    
    print(f"  ✓ Correctly identified {len(review_trades)} trade(s) requiring review")

    await analyzer.stop()
    print("  ✓ High shortfall detection test passed")


async def run_all_tests():
    """
    Run all implementation shortfall analyzer tests.
    """
    print("Running Implementation Shortfall Analyzer Tests...\n")
    
    await test_shortfall_calculation()
    await test_execution_quality_classification()
    await test_decision_fill_matching()
    await test_aggregate_metrics()
    await test_high_shortfall_detection()
    
    print("\n✅ All Implementation Shortfall Analyzer tests passed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())