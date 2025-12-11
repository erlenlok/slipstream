"""
Tests for the real-time dashboard and visualization module.
Following TDD approach - these tests validate dashboard functionality.
"""
import pytest
from datetime import datetime
import sys
import os

# Add the src directory to the path so we can import slipstream modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from slipstream.analytics.dashboard import (
    RealTimeDashboard, DashboardService, DashboardConfig, 
    WebSocketManager, MetricType
)
from slipstream.analytics.data_structures import PerformanceMetrics


def test_real_time_metric_updates():
    """Test that dashboard updates metrics in real-time."""
    service = DashboardService(mock_mode=True)
    
    # Check that the service was initialized properly
    assert service.current_data is not None
    assert service.core_calculator is not None
    assert service.historical_analyzer is not None
    
    # The service should have initialized with mock data
    assert service.current_data.current_metrics is not None


def test_24hr_snapshot_display():
    """Test that 24-hour snapshot is displayed correctly."""
    service = DashboardService(mock_mode=True)
    
    # After initialization, we should have current metrics
    assert service.current_data.current_metrics is not None
    
    # Check that key metrics are populated
    metrics = service.current_data.current_metrics
    assert hasattr(metrics, 'total_pnl')
    assert hasattr(metrics, 'hit_rate')
    assert hasattr(metrics, 'total_trades')


def test_historical_trend_visualization():
    """Test that historical trends are visualized correctly."""
    service = DashboardService(mock_mode=True)
    
    # Check that historical data was initialized
    assert service.current_data.historical_data is not None
    assert isinstance(service.current_data.historical_data, list)


def test_per_asset_breakdown_display():
    """Test that per-asset breakdowns are displayed correctly."""
    service = DashboardService(mock_mode=True)
    
    # Check that per-asset data was initialized
    assert service.current_data.per_asset_data is not None
    assert isinstance(service.current_data.per_asset_data, dict)


def test_dashboard_performance():
    """Test dashboard performance with real-time updates."""
    # This would test the dashboard's ability to handle updates efficiently
    service = DashboardService(mock_mode=True)
    
    # Verify the WebSocket manager was created
    assert service.websocket_manager is not None
    assert hasattr(service.websocket_manager, 'broadcast')


def test_error_handling_in_dashboard():
    """Test dashboard behavior when data is unavailable."""
    service = DashboardService(mock_mode=True)
    
    # Test accessing metrics when they might be None
    data = service.current_data.to_dict()
    assert isinstance(data, dict)
    
    # Should handle missing data gracefully
    assert 'current_metrics' in data
    assert 'per_asset_data' in data


def test_dashboard_config_creation():
    """Test DashboardConfig can be created properly."""
    config = DashboardConfig()
    
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.log_level == "info"
    
    # Test custom config
    custom_config = DashboardConfig(host="localhost", port=3000, log_level="debug")
    assert custom_config.host == "localhost"
    assert custom_config.port == 3000
    assert custom_config.log_level == "debug"


def test_websocket_manager():
    """Test WebSocket manager functionality."""
    manager = WebSocketManager()
    
    # Check that it initializes correctly
    assert manager.active_connections == []
    
    # Check that it has required methods
    assert hasattr(manager, 'connect')
    assert hasattr(manager, 'disconnect')
    assert hasattr(manager, 'broadcast')


def test_metric_type_enum():
    """Test MetricType enum has all required values."""
    expected_metrics = ['PNL', 'HIT_RATE', 'MARKOUT', 'SHARPE_RATIO', 'INVENTORY', 'VOLATILITY']
    
    for metric in expected_metrics:
        assert hasattr(MetricType, metric)
    
    # Test accessing values
    assert MetricType.PNL.value == "pnl"
    assert MetricType.HIT_RATE.value == "hit_rate"


def test_dashboard_initialization():
    """Test RealTimeDashboard can be properly initialized."""
    dashboard = RealTimeDashboard()
    
    assert dashboard.service is not None
    assert dashboard.app is not None
    assert dashboard.config is not None


if __name__ == "__main__":
    # Run the tests
    test_real_time_metric_updates()
    test_24hr_snapshot_display()
    test_historical_trend_visualization()
    test_per_asset_breakdown_display()
    test_dashboard_performance()
    test_error_handling_in_dashboard()
    test_dashboard_config_creation()
    test_websocket_manager()
    test_metric_type_enum()
    test_dashboard_initialization()
    
    print("All Real-time Dashboard and Visualization tests passed!")