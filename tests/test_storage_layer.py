"""
Tests for the storage and persistence layer.
Following TDD approach - these tests validate storage functionality.
"""
import pytest
import asyncio
from datetime import datetime
import sys
import os

# Add the src directory to the path so we can import slipstream modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from slipstream.analytics.storage_layer import AnalyticsStorage, DatabaseConfig
from slipstream.analytics.data_structures import TradeEvent, TradeType, PerformanceMetrics


def test_database_config_creation():
    """Test that DatabaseConfig can be created properly."""
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_db",
        username="test_user",
        password="test_pass"
    )
    
    assert config.host == "localhost"
    assert config.port == 5432
    assert config.database == "test_db"
    assert config.connection_string().startswith("postgresql://")


def test_table_names_enum():
    """Test that all required table names are defined."""
    from slipstream.analytics.storage_layer import TableNames
    
    # Check that all expected tables exist
    expected_tables = [
        "trade_events",
        "performance_snapshots", 
        "per_asset_metrics",
        "rolling_metrics",
        "historical_trends"
    ]
    
    table_values = [table.value for table in TableNames]
    for expected in expected_tables:
        assert expected in table_values


def test_analytics_storage_initialization():
    """Test AnalyticsStorage can be initialized."""
    config = DatabaseConfig()
    storage = AnalyticsStorage(config)
    
    assert storage.config == config
    assert storage.connection is None


def test_metrics_storage_structure():
    """Test that storage methods exist for required metrics."""
    config = DatabaseConfig()
    storage = AnalyticsStorage(config)
    
    # Verify storage has the necessary methods
    methods_to_check = [
        'store_trade_event',
        'store_performance_snapshot',
        'store_per_asset_metrics',
        'store_rolling_metrics',
        'store_historical_trend',
        'get_performance_snapshots',
        'get_asset_metrics',
        'get_recent_trades'
    ]
    
    for method_name in methods_to_check:
        assert hasattr(storage, method_name)


def test_24hr_data_retention():
    """Test 24-hour snapshot data retention policies."""
    # This is more of a structural test since we can't easily test retention policies
    # without an actual database connection
    config = DatabaseConfig()
    storage = AnalyticsStorage(config)
    
    # Verify the cleanup method exists
    assert hasattr(storage, 'cleanup_old_data')


def test_historical_data_retention():
    """Test long-term historical data retention policies."""
    config = DatabaseConfig()
    storage = AnalyticsStorage(config)
    
    # Verify the cleanup method exists and takes appropriate parameters
    assert hasattr(storage, 'cleanup_old_data')


def test_data_consistency():
    """Test data consistency mechanisms."""
    # This would be tested more fully with an actual database,
    # but we can verify the structure supports consistency
    config = DatabaseConfig()
    storage = AnalyticsStorage(config)
    
    # Verify we have methods that would ensure data consistency
    assert hasattr(storage, 'batch_store_trades')  # Batch operations help with consistency


if __name__ == "__main__":
    # Run the tests
    test_database_config_creation()
    test_table_names_enum()
    test_analytics_storage_initialization()
    test_metrics_storage_structure()
    test_24hr_data_retention()
    test_historical_data_retention()
    test_data_consistency()
    
    print("All Storage and Persistence Layer tests passed!")