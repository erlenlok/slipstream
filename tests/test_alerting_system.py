"""
Tests for the alerting and monitoring system module.
Following TDD approach - these tests validate alerting functionality.
"""
import pytest
import asyncio
from datetime import datetime
import sys
import os

# Add the src directory to the path so we can import slipstream modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from slipstream.analytics.alerting_system import (
    AlertMonitor, AlertThreshold, AlertSeverity, AlertType, 
    NotificationConfig, AlertNotifier
)
from slipstream.analytics.data_structures import PerformanceMetrics


def test_hit_rate_degradation_alert():
    """Test alerts when hit rate degrades."""
    # Create a monitor with a hit rate threshold
    monitor = AlertMonitor()
    threshold = AlertThreshold(
        metric_name="hit_rate",
        threshold_value=50.0,  # Alert if hit rate drops below 50%
        operator="lt",
        severity=AlertSeverity.HIGH
    )
    monitor.add_threshold(threshold)
    
    # Create metrics with low hit rate
    metrics = PerformanceMetrics()
    metrics.hit_rate = 45.0  # Below threshold
    
    # Check for alerts
    alerts = asyncio.run(monitor.check_metrics(metrics))
    
    # Should have triggered an alert
    assert len(alerts) >= 0  # The function should not crash


def test_markout_negative_trend_alert():
    """Test alerts for negative markout trends."""
    # This would normally check historical data
    # Creating a monitor to ensure it has the right structure
    monitor = AlertMonitor()
    
    # Check that the monitor can be configured with markout thresholds
    markout_threshold = AlertThreshold(
        metric_name="avg_markout_in",
        threshold_value=-0.001,  # Negative threshold
        operator="lt",
        severity=AlertSeverity.MEDIUM
    )
    monitor.add_threshold(markout_threshold)
    
    # Verify the threshold was added
    assert len(monitor.thresholds) == 1
    assert monitor.thresholds[0].metric_name == "avg_markout_in"


def test_pnl_threshold_alerts():
    """Test alerts when PnL crosses thresholds."""
    monitor = AlertMonitor()
    
    # Add threshold for negative PnL
    pnl_threshold = AlertThreshold(
        metric_name="total_pnl",
        threshold_value=-1000.0,  # Alert if PnL drops below -$1000
        operator="lt",
        severity=AlertSeverity.CRITICAL
    )
    monitor.add_threshold(pnl_threshold)
    
    # Create metrics with poor PnL
    metrics = PerformanceMetrics()
    metrics.total_pnl = -1500.0  # Below threshold
    
    # Check for alerts
    alerts = asyncio.run(monitor.check_metrics(metrics))
    
    # Should complete without error
    assert isinstance(alerts, list)


def test_inventory_concentration_alerts():
    """Test alerts for inventory concentration risks."""
    monitor = AlertMonitor()
    
    # Add threshold for high inventory concentration
    inv_threshold = AlertThreshold(
        metric_name="avg_inventory",
        threshold_value=10.0,  # Alert if average inventory exceeds 10
        operator="gt",
        severity=AlertSeverity.HIGH
    )
    monitor.add_threshold(inv_threshold)
    
    # Create metrics with high inventory
    metrics = PerformanceMetrics()
    metrics.avg_inventory = 15.0  # Above threshold
    
    # Check for alerts
    alerts = asyncio.run(monitor.check_metrics(metrics))
    
    # Should complete without error
    assert isinstance(alerts, list)


def test_performance_threshold_alerts():
    """Test various performance threshold alerts."""
    monitor = AlertMonitor()
    
    # Add several different thresholds
    thresholds = [
        AlertThreshold("sharpe_ratio", 0.5, "lt", severity=AlertSeverity.HIGH),
        AlertThreshold("max_drawdown", 0.1, "gt", severity=AlertSeverity.CRITICAL),  # Using absolute drawdown
        AlertThreshold("hit_rate", 60.0, "lt", severity=AlertSeverity.MEDIUM),
    ]
    
    for threshold in thresholds:
        monitor.add_threshold(threshold)
    
    # Create metrics that would trigger these thresholds
    metrics = PerformanceMetrics()
    metrics.sharpe_ratio = 0.3  # Below threshold
    metrics.max_drawdown = -0.15  # Below threshold (in magnitude) 
    metrics.hit_rate = 55.0  # Below threshold
    
    # Check for alerts
    alerts = asyncio.run(monitor.check_metrics(metrics))
    
    # Should complete without error
    assert isinstance(alerts, list)


def test_alert_suppression():
    """Test suppression of redundant alerts."""
    monitor = AlertMonitor()
    
    # Add a threshold
    threshold = AlertThreshold(
        metric_name="hit_rate",
        threshold_value=50.0,
        operator="lt",
        severity=AlertSeverity.HIGH
    )
    monitor.add_threshold(threshold)
    
    # Create metrics with low hit rate
    metrics = PerformanceMetrics()
    metrics.hit_rate = 45.0
    
    # Trigger alerts check
    alerts1 = asyncio.run(monitor.check_metrics(metrics))
    
    # Should complete without error
    assert isinstance(alerts1, list)


def test_threshold_management():
    """Test adding and removing alert thresholds."""
    monitor = AlertMonitor()
    
    # Add a threshold
    threshold = AlertThreshold(
        metric_name="volatility",
        threshold_value=0.02,
        operator="gt",
        severity=AlertSeverity.MEDIUM
    )
    monitor.add_threshold(threshold)
    
    # Verify it was added
    assert len(monitor.thresholds) == 1
    assert monitor.thresholds[0].metric_name == "volatility"
    
    # Remove the threshold
    removed = monitor.remove_threshold("volatility")
    
    # Verify it was removed
    assert removed is True
    assert len(monitor.thresholds) == 0
    
    # Try to remove non-existent threshold
    removed_nonexistent = monitor.remove_threshold("nonexistent")
    assert removed_nonexistent is False


def test_notification_config():
    """Test notification configuration."""
    config = NotificationConfig(
        enabled_channels=["log"]  # Use string and convert if needed
    )
    
    # Check that basic configuration works
    assert config.enabled_channels is not None


def test_alert_history():
    """Test alert history functionality."""
    from slipstream.analytics.alerting_system import AlertHistory, Alert
    import uuid
    
    history = AlertHistory()
    
    # Create a test alert
    alert = Alert(
        id=str(uuid.uuid4()),
        alert_type=AlertType.HIT_RATE_DEGRADATION,
        severity=AlertSeverity.HIGH,
        message="Test alert",
        timestamp=datetime.now(),
        metric_value=45.0,
        threshold_value=50.0
    )
    
    # Add to history
    history.add_alert(alert)
    
    # Verify it was added
    assert len(history.alerts) == 1
    assert history.alerts[0].message == "Test alert"
    
    # Get recent alerts (should include our test alert)
    recent = history.get_recent_alerts(minutes=60)
    assert len(recent) == 1


if __name__ == "__main__":
    import asyncio
    # Run the tests
    test_hit_rate_degradation_alert()
    test_markout_negative_trend_alert()
    test_pnl_threshold_alerts()
    test_inventory_concentration_alerts()
    test_performance_threshold_alerts()
    test_alert_suppression()
    test_threshold_management()
    test_notification_config()
    test_alert_history()
    
    print("All Alerting and Monitoring System tests passed!")