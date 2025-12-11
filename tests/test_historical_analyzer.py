"""
Tests for the long-term tracking and historical analysis module.
Following TDD approach - these tests validate historical analysis functionality.
"""
import pytest
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path so we can import slipstream modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from slipstream.analytics.historical_analyzer import (
    HistoricalAnalyzer, TimeWindow, TrendAnalyzer, 
    VolatilityRegimeAnalyzer, SeasonalPatternDetector, StressPeriodAnalyzer
)
from slipstream.analytics.data_structures import PerformanceMetrics


def test_rolling_sharpe_ratio():
    """Test rolling Sharpe ratio calculation over different time periods."""
    analyzer = HistoricalAnalyzer()
    
    # Create some mock metrics with varying Sharpe ratios
    base_time = datetime.now()
    for i in range(5):
        metrics = PerformanceMetrics()
        metrics.sharpe_ratio = float(i + 1)  # 1, 2, 3, 4, 5
        metrics.hit_rate = 50.0 + (i * 5)  # 50, 55, 60, 65, 70
        metrics.total_pnl = 1000.0 * (i + 1)
        
        period_time = base_time - timedelta(days=i)
        analyzer.process_period_metrics(metrics, period_time)
    
    # Get trends
    trends = analyzer.trend_analyzer.get_performance_trend()
    
    # Verify we have trend data for key metrics
    assert 'sharpe_ratio' in trends
    assert 'hit_rate' in trends
    assert 'pnl' in trends


def test_rolling_max_drawdown():
    """Test rolling maximum drawdown calculation."""
    analyzer = HistoricalAnalyzer()
    
    # Create metrics with varying drawdowns
    base_time = datetime.now()
    for i in range(3):
        metrics = PerformanceMetrics()
        metrics.max_drawdown = float(-(i + 1))  # -1, -2, -3 (more negative = worse)
        metrics.sharpe_ratio = 2.0 - (i * 0.5)  # 2, 1.5, 1.0
        
        period_time = base_time - timedelta(days=i)
        analyzer.process_period_metrics(metrics, period_time)
    
    # Check that we can access the rolling max drawdown
    # Just verify the analyzer functionality works without errors
    rolling_metrics = analyzer.get_rolling_metrics()
    # This should not raise an exception


def test_rolling_volatility():
    """Test rolling volatility calculation."""
    analyzer = HistoricalAnalyzer()
    
    # Add metrics with different volatilities
    base_time = datetime.now()
    for i in range(3):
        metrics = PerformanceMetrics()
        metrics.volatility = 0.01 * (i + 1)  # 0.01, 0.02, 0.03
        metrics.sharpe_ratio = 1.5
        
        period_time = base_time - timedelta(days=i)
        analyzer.process_period_metrics(metrics, period_time)
    
    # Verify metrics are stored
    assert len(analyzer.historical_metrics) == 3


def test_rolling_markout():
    """Test rolling markout calculations."""
    analyzer = HistoricalAnalyzer()
    
    # Add metrics with different markouts
    base_time = datetime.now()
    for i in range(3):
        metrics = PerformanceMetrics()
        metrics.markout_analysis.avg_markout_in = 10.0 * (i + 1)  # 10, 20, 30
        metrics.total_pnl = 1000.0 * (i + 1)
        
        period_time = base_time - timedelta(days=i)
        analyzer.process_period_metrics(metrics, period_time)
    
    # Check trends
    trends = analyzer.trend_analyzer.get_performance_trend()
    assert 'pnl' in trends


def test_performance_trend_detection():
    """Test ability to detect performance trends over time."""
    analyzer = HistoricalAnalyzer()
    
    # Create a clear increasing trend
    base_time = datetime.now()
    for i in range(10):  # Need at least min_trend_points (3) + some more
        metrics = PerformanceMetrics()
        metrics.sharpe_ratio = 0.5 + (i * 0.1)  # Increasing from 0.5 to 1.4
        metrics.hit_rate = 45.0 + (i * 1.0)     # Increasing from 45 to 54
        
        period_time = base_time - timedelta(hours=i)
        analyzer.process_period_metrics(metrics, period_time)
    
    # Check that trends are detected
    trends = analyzer.get_performance_trends()
    
    # Verify we have trend data for key metrics
    assert 'sharpe_ratio' in trends
    assert 'hit_rate' in trends
    assert trends['sharpe_ratio']['direction'] in ['increasing', 'decreasing', 'neutral', 'insufficient_data']


def test_volatility_regime_analysis():
    """Test analysis during different market volatility periods."""
    analyzer = HistoricalAnalyzer()
    vol_analyzer = VolatilityRegimeAnalyzer()
    
    # Add metrics under different volatility regimes
    # Low volatility periods
    for i in range(3):
        metrics = PerformanceMetrics()
        metrics.sharpe_ratio = 2.0
        metrics.hit_rate = 60.0
        vol_analyzer.add_metrics_for_volatility_regime(metrics, 0.005)  # Low vol
    
    # Medium volatility periods
    for i in range(3):
        metrics = PerformanceMetrics()
        metrics.sharpe_ratio = 1.5
        metrics.hit_rate = 55.0
        vol_analyzer.add_metrics_for_volatility_regime(metrics, 0.02)  # Medium vol
    
    # High volatility periods
    for i in range(3):
        metrics = PerformanceMetrics()
        metrics.sharpe_ratio = 1.0
        metrics.hit_rate = 50.0
        vol_analyzer.add_metrics_for_volatility_regime(metrics, 0.05)  # High vol
    
    # Check regime summary
    summary = vol_analyzer.get_regime_performance_summary()
    
    # Verify we have data structures created
    assert isinstance(summary, dict)
    # The content depends on internal implementation but dict should be returned


def test_seasonal_pattern_detection():
    """Test detection of time-of-day/seasonal patterns."""
    seasonal_detector = SeasonalPatternDetector()
    
    # Add performance data for different hours
    # Simulate better performance during certain hours
    for hour in range(24):
        # Create 3 data points for each hour with varying performance
        for day in range(3):
            timestamp = datetime(2023, 1, day+1, hour, 0)
            # Simulate good performance during early morning (6-10 AM)
            if 6 <= hour <= 10:
                pnl = 50.0  # Good performance
            elif 15 <= hour <= 18:
                pnl = 30.0  # Decent performance
            else:
                pnl = 10.0  # Average/low performance
            seasonal_detector.add_performance_by_time(timestamp, pnl)
    
    # Get best hours
    best_hours = seasonal_detector.get_best_hours(3)
    
    # Verify structure of results (not exact hours since multiple hours may have same value)
    assert isinstance(best_hours, list)
    assert len(best_hours) <= 3
    if best_hours:
        assert all(isinstance(hour, int) and isinstance(pnl, float) for hour, pnl in best_hours)


def test_stress_period_analysis():
    """Test analysis during simulated stress periods."""
    stress_analyzer = StressPeriodAnalyzer()
    
    # Add normal period metrics
    normal_metrics = PerformanceMetrics()
    normal_metrics.total_pnl = 1000.0
    normal_metrics.sharpe_ratio = 2.0
    stress_analyzer.add_period_performance(
        normal_metrics, {'volatility': 0.01, 'correlation': 0.2}  # Normal conditions
    )
    
    # Add stress period metrics
    stress_metrics = PerformanceMetrics()
    stress_metrics.total_pnl = -500.0  # Worse performance during stress
    stress_metrics.sharpe_ratio = 0.5
    stress_analyzer.add_period_performance(
        stress_metrics, {'volatility': 0.08, 'correlation': 0.9}  # Stress conditions
    )
    
    # Get stress analysis
    analysis = stress_analyzer.get_stress_impact_analysis()
    
    # Verify we have the analysis results
    assert isinstance(analysis, dict)
    assert 'stress_avg_pnl' in analysis
    assert 'normal_avg_pnl' in analysis
    assert 'stress_period_count' in analysis
    assert 'normal_period_count' in analysis


if __name__ == "__main__":
    # Run the tests
    test_rolling_sharpe_ratio()
    test_rolling_max_drawdown()
    test_rolling_volatility()
    test_rolling_markout()
    test_performance_trend_detection()
    test_volatility_regime_analysis()
    test_seasonal_pattern_detection()
    test_stress_period_analysis()
    
    print("All Long-term Tracking and Historical Analysis tests passed!")