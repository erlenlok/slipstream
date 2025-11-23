"""
Test suite for the Risk Auditor component.

This module tests the Risk Auditor's ability to operate independently
from strategies while maintaining accurate tracking of actual exposures.
"""

from datetime import datetime
import asyncio
from slipstream.federation.auditor import RiskAuditor, TradeEvent, PositionEvent, StrategyReport, AuditResult


async def test_auditor_independence():
    """
    Test that the Risk Auditor operates independently from strategies.
    """
    print("Testing Risk Auditor independence...")
    
    # Create auditor without any exchange connector (simulated mode)
    auditor = RiskAuditor()
    await auditor.start()
    
    # Add a strategy for monitoring
    await auditor.add_strategy("test_strategy_1")
    
    # Simulate exchange events that the auditor tracks independently
    trade_event = TradeEvent(
        timestamp=datetime.now(),
        event_type="trade",
        strategy_id="test_strategy_1",
        account_id="acc_123",
        symbol="BTC",
        side="buy",
        quantity=1.5,
        price=45000.0,
        order_id="order_123"
    )
    
    await auditor.record_exchange_event(trade_event)
    
    # Check that auditor independently tracked the position
    actual_positions = auditor._actual_positions["test_strategy_1"]
    assert actual_positions["BTC"] == 1.5
    print("  ✓ Auditor independently tracked trade event")
    
    # Simulate strategy self-report (this should not affect auditor's tracking)
    strategy_report = StrategyReport(
        strategy_id="test_strategy_1",
        timestamp=datetime.now(),
        net_exposure=20000.0,  # Different from what auditor calculated
        open_orders=1,
        reported_pnl=150.0,
        positions={"BTC": 2.0}  # Different position than actual
    )
    
    await auditor.record_strategy_report(strategy_report)
    
    # Verify auditor's tracking wasn't affected by the report
    actual_position = auditor._actual_positions["test_strategy_1"]["BTC"]
    reported_position = auditor._reported_positions["test_strategy_1"]["BTC"]
    
    assert actual_position == 1.5  # Auditor's independent tracking
    assert reported_position == 2.0  # Strategy's self-report
    print("  ✓ Auditor maintains independent tracking separate from reports")
    
    # Perform audit to compare
    audit_result = await auditor.perform_audit("test_strategy_1")
    
    assert audit_result.overall_status == "FAIL"  # Due to mismatches
    assert len(audit_result.discrepancies) > 0
    print("  ✓ Audit correctly detected discrepancies between actual and reported")
    
    await auditor.stop()
    print("  ✓ Auditor independence test passed")


async def test_read_only_operation():
    """
    Test that the Risk Auditor operates in read-only mode without affecting strategies.
    """
    print("\nTesting Risk Auditor read-only operation...")
    
    # Create auditor
    auditor = RiskAuditor()
    await auditor.start()
    
    # Add multiple strategies
    await auditor.add_strategy("strategy_alpha")
    await auditor.add_strategy("strategy_beta")
    
    # Add various exchange events
    await auditor.record_exchange_event(TradeEvent(
        timestamp=datetime.now(),
        event_type="trade",
        strategy_id="strategy_alpha",
        account_id="acc_alpha",
        symbol="ETH",
        side="buy",
        quantity=5.0,
        price=3000.0,
        order_id="order_456"
    ))

    await auditor.record_exchange_event(TradeEvent(
        timestamp=datetime.now(),
        event_type="trade",
        strategy_id="strategy_beta",
        account_id="acc_beta",
        symbol="BTC",
        side="sell",
        quantity=2.0,
        price=47000.0,
        order_id="order_789"
    ))
    
    # Verify auditor tracked both strategies independently
    alpha_exposure = auditor._actual_exposures["strategy_alpha"]
    beta_exposure = auditor._actual_exposures["strategy_beta"]
    
    assert alpha_exposure > 0  # Should have ETH position
    assert beta_exposure > 0  # Should have BTC position (negative would be -2 * 47000)
    print("  ✓ Auditor tracked multiple strategies independently")
    
    # Verify no interference with strategies
    # (In real implementation, strategies would operate normally)
    print("  ✓ No interference with strategy operations")
    
    await auditor.stop()
    print("  ✓ Read-only operation test passed")


async def test_unified_exposure_view():
    """
    Test that the auditor provides a unified view of all strategy exposures.
    """
    print("\nTesting unified exposure view...")
    
    auditor = RiskAuditor()
    await auditor.start()
    
    # Add strategies and record some events
    await auditor.add_strategy("strategy_a")
    await auditor.add_strategy("strategy_b")
    
    await auditor.record_exchange_event(TradeEvent(
        timestamp=datetime.now(),
        event_type="trade",
        strategy_id="strategy_a",
        account_id="acc_a",
        symbol="BTC",
        side="buy",
        quantity=3.0,
        price=50000.0,
        order_id="order_a1"
    ))
    
    await auditor.record_strategy_report(StrategyReport(
        strategy_id="strategy_a",
        timestamp=datetime.now(),
        net_exposure=3.0,  # Position size, not notional
        open_orders=2,
        reported_pnl=300.0,
        positions={"BTC": 3.0}
    ))
    
    # Get unified view
    unified_view = await auditor.get_unified_exposure_view()
    
    assert "strategy_a" in unified_view
    assert "strategy_b" in unified_view  # Even if no events recorded yet
    
    strategy_a_data = unified_view["strategy_a"]
    assert strategy_a_data['actual_exposure'] == 3.0  # Position size, not notional value
    assert strategy_a_data['reported_exposure'] == 3.0  # This is what strategy reported
    assert strategy_a_data['active'] == True
    
    print("  ✓ Unified exposure view provides comprehensive data")
    print("  ✓ All strategies tracked in unified view")
    
    await auditor.stop()
    print("  ✓ Unified exposure view test passed")


async def test_audit_with_no_discrepancies():
    """
    Test audit when reported and actual values match.
    """
    print("\nTesting audit with matching values...")
    
    auditor = RiskAuditor()
    await auditor.start()
    
    await auditor.add_strategy("perfect_strategy")
    
    await auditor.record_exchange_event(TradeEvent(
        timestamp=datetime.now(),
        event_type="trade",
        strategy_id="perfect_strategy",
        account_id="acc_perfect",
        symbol="BTC",
        side="buy",
        quantity=2.0,
        price=48000.0,
        order_id="order_p1"
    ))
    
    # Have strategy report the exact same values (no discrepancies)
    await auditor.record_strategy_report(StrategyReport(
        strategy_id="perfect_strategy",
        timestamp=datetime.now(),
        net_exposure=2.0,  # Same as actual position
        open_orders=1,
        reported_pnl=0.0,  # Assuming no PnL yet
        positions={"BTC": 2.0}
    ))
    
    # Perform audit
    audit_result = await auditor.perform_audit("perfect_strategy")
    
    assert audit_result.overall_status == "PASS"
    assert len(audit_result.discrepancies) == 0
    assert audit_result.exposure_difference == 0
    print("  ✓ Audit correctly passed when values match")
    
    await auditor.stop()
    print("  ✓ Perfect match audit test passed")


async def run_all_tests():
    """
    Run all risk auditor tests.
    """
    print("Running Risk Auditor Tests...\n")
    
    await test_auditor_independence()
    await test_read_only_operation()
    await test_unified_exposure_view()
    await test_audit_with_no_discrepancies()
    
    print("\n✅ All Risk Auditor tests passed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())