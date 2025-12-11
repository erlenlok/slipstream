"""
Tests for the integration testing and validation module.
Following TDD approach - these tests validate the complete system integration.
"""
import pytest
import asyncio
from datetime import datetime
import sys
import os

# Add the src directory to the path so we can import slipstream modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from slipstream.analytics.integration_tests import (
    run_integration_tests,
    EndToEndPerformanceTrackingTest,
    MockBrawlerIntegrationTest,
    MultiInstrumentIntegrationTest,
    DataConsistencyIntegrationTest,
    ErrorRecoveryIntegrationTest,
    test_end_to_end_performance_tracking,
    test_mock_brawler_integration,
    test_multiple_instruments_integration,
    test_long_running_performance,
    test_data_consistency_across_components,
    test_error_recovery
)


def test_end_to_end_performance_tracking_integration():
    """Test complete performance tracking pipeline from trade to dashboard."""
    # This test should run without errors
    from slipstream.analytics.integration_tests import test_end_to_end_performance_tracking as func
    func()


def test_mock_brawler_integration_test():
    """Test integration with mocked Brawler event stream."""
    # This test should run without errors
    from slipstream.analytics.integration_tests import test_mock_brawler_integration as func
    func()


def test_multiple_instruments_integration_test():
    """Test performance tracking with multiple instruments."""
    # This test should run without errors
    from slipstream.analytics.integration_tests import test_multiple_instruments_integration as func
    func()


def test_long_running_performance_integration():
    """Test system performance over extended time periods."""
    # This test should run without errors
    from slipstream.analytics.integration_tests import test_long_running_performance as func
    func()


def test_data_consistency_across_components_integration():
    """Test data consistency across all system components."""
    # This test should run without errors
    from slipstream.analytics.integration_tests import test_data_consistency_across_components as func
    func()


def test_error_recovery_integration():
    """Test system recovery from various error conditions."""
    # This test should run without errors
    from slipstream.analytics.integration_tests import test_error_recovery as func
    func()


def test_integration_test_classes():
    """Test that integration test classes can be instantiated and run."""
    # Test individual integration test classes
    tests = [
        EndToEndPerformanceTrackingTest(),
        MockBrawlerIntegrationTest(), 
        MultiInstrumentIntegrationTest(),
        DataConsistencyIntegrationTest(),
        ErrorRecoveryIntegrationTest()
    ]
    
    for test in tests:
        assert hasattr(test, 'run_test')
        assert hasattr(test, 'validate_results')
        assert test.name is not None and test.name != ""


def test_run_integration_tests_function():
    """Test the main integration test runner function."""
    # Run integration tests and check results format
    results = asyncio.run(run_integration_tests())
    
    # Should have results for all test types
    expected_tests = [
        "End-to-End Performance Tracking",
        "Mock Brawler Integration", 
        "Multi-Instrument Integration",
        "Data Consistency Integration",
        "Error Recovery Integration"
    ]
    
    for test_name in expected_tests:
        assert test_name in results
        assert 'success' in results[test_name]
        assert 'details' in results[test_name]
        assert isinstance(results[test_name]['details'], dict)


if __name__ == "__main__":
    # Run the tests
    test_end_to_end_performance_tracking_integration()
    test_mock_brawler_integration_test()
    test_multiple_instruments_integration_test()
    test_long_running_performance_integration()
    test_data_consistency_across_components_integration()
    test_error_recovery_integration()
    test_integration_test_classes()
    test_run_integration_tests_function()

    print("All Integration Testing and Validation tests passed!")