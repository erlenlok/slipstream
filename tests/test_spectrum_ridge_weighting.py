"""
Unit tests for Spectrum Dynamic Weighting - Module C

Tests for ridge_weighting.py following the requirements in spectrum_spec.md
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.slipstream.strategies.spectrum.ridge_weighting import (
    prepare_pooled_data,
    fit_ridge_regression,
    compute_factor_weights_rolling,
    apply_factor_weights
)


def test_prepare_pooled_data():
    """Test pooled data preparation for Ridge regression."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    
    # Create signal matrix
    signal_matrix = {
        'idio_carry': pd.DataFrame({
            'SOL': np.random.normal(0, 1, len(dates)),
            'ADA': np.random.normal(0, 1, len(dates)),
            'XRP': np.random.normal(0, 1, len(dates))
        }, index=dates),
        'idio_momentum': pd.DataFrame({
            'SOL': np.random.normal(0, 1, len(dates)),
            'ADA': np.random.normal(0, 1, len(dates)),
            'XRP': np.random.normal(0, 1, len(dates))
        }, index=dates),
        'idio_meanrev': pd.DataFrame({
            'SOL': np.random.normal(0, 1, len(dates)),
            'ADA': np.random.normal(0, 1, len(dates)),
            'XRP': np.random.normal(0, 1, len(dates))
        }, index=dates)
    }
    
    # Create target returns
    target_returns = pd.DataFrame({
        'SOL': np.random.normal(0.001, 0.02, len(dates)),
        'ADA': np.random.normal(0.0005, 0.015, len(dates)),
        'XRP': np.random.normal(-0.0002, 0.018, len(dates))
    }, index=dates)
    
    # Test with different lookback periods
    X_pooled, y_pooled, valid_mask = prepare_pooled_data(
        signal_matrix, target_returns, lookback_period=15
    )
    
    # Verify shapes: X should have 3 features (one per factor), y should be 1D
    assert X_pooled.shape[1] == 3  # 3 factors
    assert len(y_pooled) == len(X_pooled)  # Same number of samples
    assert valid_mask.shape == signal_matrix['idio_carry'].shape  # Same shape as input
    
    # Check that all values are finite
    assert np.isfinite(X_pooled).all()
    assert np.isfinite(y_pooled).all()


def test_fit_ridge_regression():
    """Test Ridge regression fitting."""
    # Create simple test data
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([0.5, -0.3, 0.8])
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)  # Add noise
    
    # Test Ridge fitting
    coefficients, best_alpha, cv_results = fit_ridge_regression(
        X, y, alphas=[0.01, 0.1, 1.0], cv_folds=3
    )
    
    # Verify output shapes and types
    assert len(coefficients) == n_features
    assert isinstance(best_alpha, (int, float))
    assert 'r2_score' in cv_results
    assert 'best_alpha' in cv_results
    
    # Coefficients should be reasonable (finite)
    assert np.isfinite(coefficients).all()
    assert np.isfinite(best_alpha)
    assert np.isfinite(cv_results['r2_score'])


def test_compute_factor_weights_rolling():
    """Test rolling factor weights computation."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    
    # Create signal matrix with some correlation to targets
    np.random.seed(42)
    base_signal = np.random.randn(len(dates), 3)  # 3 factors
    
    signal_matrix = {
        'idio_carry': pd.DataFrame({
            'SOL': base_signal[:, 0],
            'ADA': base_signal[:, 0] * 0.8,
            'XRP': base_signal[:, 0] * 1.2
        }, index=dates),
        'idio_momentum': pd.DataFrame({
            'SOL': base_signal[:, 1],
            'ADA': base_signal[:, 1] * 0.9,
            'XRP': base_signal[:, 1] * 1.1
        }, index=dates),
        'idio_meanrev': pd.DataFrame({
            'SOL': base_signal[:, 2],
            'ADA': base_signal[:, 2] * 0.7,
            'XRP': base_signal[:, 2] * 1.3
        }, index=dates)
    }
    
    # Create target returns that have some relationship to signals
    target_returns = pd.DataFrame({
        'SOL': 0.3 * base_signal[:, 0] + 0.2 * base_signal[:, 1] + 0.1 * base_signal[:, 2] + np.random.randn(len(dates)) * 0.01,
        'ADA': 0.2 * base_signal[:, 0] + 0.3 * base_signal[:, 1] + 0.1 * base_signal[:, 2] + np.random.randn(len(dates)) * 0.01,
        'XRP': 0.1 * base_signal[:, 0] + 0.1 * base_signal[:, 1] + 0.4 * base_signal[:, 2] + np.random.randn(len(dates)) * 0.01
    }, index=dates)
    
    # Compute factor weights
    factor_weights, training_results = compute_factor_weights_rolling(
        signal_matrix, target_returns, lookback_period=20, cv_folds=3
    )
    
    # Verify output structure
    assert isinstance(factor_weights, dict)
    assert set(factor_weights.keys()) == {'idio_carry', 'idio_momentum', 'idio_meanrev'}
    assert isinstance(training_results, dict)
    
    # Verify that weights are finite numbers
    for weight in factor_weights.values():
        assert np.isfinite(weight)
    
    # Verify that training results contain expected elements
    assert 'r2_score' in training_results
    assert 'n_samples' in training_results
    assert 'best_alpha' in training_results


def test_apply_factor_weights():
    """Test applying factor weights to signals."""
    # Create sample signal matrix
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    
    signal_matrix = {
        'idio_carry': pd.DataFrame({
            'SOL': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'ADA': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        }, index=dates),
        'idio_momentum': pd.DataFrame({
            'SOL': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'ADA': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        }, index=dates),
        'idio_meanrev': pd.DataFrame({
            'SOL': [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0],
            'ADA': [-0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4, -1.6, -1.8, -2.0]
        }, index=dates)
    }
    
    # Define factor weights
    factor_weights = {
        'idio_carry': 0.5,
        'idio_momentum': 0.3,
        'idio_meanrev': -0.2
    }
    
    # Apply weights
    composite_alpha = apply_factor_weights(signal_matrix, factor_weights)
    
    # Verify output shape
    assert composite_alpha.shape == signal_matrix['idio_carry'].shape
    
    # Verify that the calculation is correct
    # For SOL on first day: 1.0*0.5 + 0.1*0.3 + (-0.1)*(-0.2) = 0.5 + 0.03 + 0.02 = 0.55
    expected_first_sol = 1.0*0.5 + 0.1*0.3 + (-0.1)*(-0.2)
    assert abs(composite_alpha.iloc[0, 0] - expected_first_sol) < 1e-10
    
    # For ADA on first day: 0.5*0.5 + 0.2*0.3 + (-0.2)*(-0.2) = 0.25 + 0.06 + 0.04 = 0.35
    expected_first_ada = 0.5*0.5 + 0.2*0.3 + (-0.2)*(-0.2)
    assert abs(composite_alpha.iloc[0, 1] - expected_first_ada) < 1e-10


def test_ridge_weighting_edge_cases():
    """Test edge cases in ridge weighting."""
    # Test with empty signal matrix
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')

    target_returns = pd.DataFrame({
        'SOL': np.random.normal(0.001, 0.02, len(dates)),
        'ADA': np.random.normal(0.0005, 0.015, len(dates))
    }, index=dates)

    # Empty signal matrix should fail
    with pytest.raises(ValueError):
        compute_factor_weights_rolling({}, target_returns)

    # Test with no valid data (all NaN) - should raise exception from prepare_pooled_data
    signal_matrix_nan = {
        'idio_carry': pd.DataFrame(np.nan, index=dates, columns=['SOL', 'ADA']),
        'idio_momentum': pd.DataFrame(np.nan, index=dates, columns=['SOL', 'ADA']),
        'idio_meanrev': pd.DataFrame(np.nan, index=dates, columns=['SOL', 'ADA'])
    }

    # This should raise a ValueError because no valid data points are found for pooling
    with pytest.raises(ValueError, match="No valid data points found for pooling"):
        compute_factor_weights_rolling(signal_matrix_nan, target_returns)


def test_prepare_pooled_data_with_nans():
    """Test pooled data preparation with NaN values."""
    # Create sample data with some NaN values
    dates = pd.date_range(start='2023-01-01', periods=15, freq='D')
    
    signal_matrix = {
        'idio_carry': pd.DataFrame({
            'SOL': [1.0, 2.0, np.nan, 4.0, 5.0] + [i for i in np.random.normal(0, 1, 10)],
            'ADA': [0.5, np.nan, 1.5, 2.0, 2.5] + [i*0.8 for i in np.random.normal(0, 1, 10)]
        }, index=dates),
        'idio_momentum': pd.DataFrame({
            'SOL': [0.1, 0.2, 0.3, np.nan, 0.5] + [i*0.5 for i in np.random.normal(0, 1, 10)],
            'ADA': [0.2, 0.4, 0.6, 0.8, np.nan] + [i*0.6 for i in np.random.normal(0, 1, 10)]
        }, index=dates),
        'idio_meanrev': pd.DataFrame({
            'SOL': [np.nan, -0.2, -0.3, -0.4, -0.5] + [i*0.7 for i in np.random.normal(0, 1, 10)],
            'ADA': [-0.2, -0.4, np.nan, -0.8, -1.0] + [i*0.9 for i in np.random.normal(0, 1, 10)]
        }, index=dates)
    }
    
    target_returns = pd.DataFrame({
        'SOL': [0.01, 0.02, -0.01, 0.03, 0.02] + [i for i in np.random.normal(0.001, 0.02, 10)],
        'ADA': [0.005, -0.01, 0.015, 0.02, -0.02] + [i*0.8 for i in np.random.normal(0.001, 0.015, 10)]
    }, index=dates)
    
    # This should handle NaN values gracefully
    X_pooled, y_pooled, valid_mask = prepare_pooled_data(
        signal_matrix, target_returns, lookback_period=15
    )
    
    # No samples should be finite if there are NaNs, but the function should not crash
    assert X_pooled.shape[1] == 3  # Should still have 3 features
    assert len(y_pooled) == len(X_pooled)  # Same number of samples


if __name__ == "__main__":
    # Run tests
    test_prepare_pooled_data()
    test_fit_ridge_regression()
    test_compute_factor_weights_rolling()
    test_apply_factor_weights()
    test_ridge_weighting_edge_cases()
    test_prepare_pooled_data_with_nans()
    print("All ridge weighting tests passed!")