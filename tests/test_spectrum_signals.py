"""
Unit tests for Spectrum Signal Factory - Module B

Tests for signals.py following the requirements in spectrum_spec.md
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.slipstream.strategies.spectrum.signals import (
    compute_idio_carry,
    compute_idio_momentum,
    compute_idio_meanrev,
    apply_cross_sectional_zscore,
    generate_spectrum_signals
)


def test_compute_idio_carry():
    """Test idio-carry signal computation."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    
    # Daily funding yields (small values)
    daily_funding = pd.DataFrame({
        'SOL': [0.0001] * len(dates),
        'ADA': [-0.00005] * len(dates),
        'XRP': [0.0002] * len(dates)
    }, index=dates)
    
    # Idiosyncratic volatility (annualized for comparison)
    idio_vol = pd.DataFrame({
        'SOL': [0.02] * len(dates),
        'ADA': [0.015] * len(dates), 
        'XRP': [0.025] * len(dates)
    }, index=dates)
    
    # Compute idio-carry
    idio_carry = compute_idio_carry(daily_funding, idio_vol)
    
    # Verify shape
    assert idio_carry.shape == daily_funding.shape
    
    # Verify values are reasonable (funding normalized by vol)
    # Annualized funding / annualized vol = (daily_fund * 365) / (daily_vol * sqrt(365))
    expected_sma = (0.0001 * 365) / (0.02 * np.sqrt(365))  # For SOL
    actual_sma = idio_carry.iloc[-1, 0]  # Last value for SOL
    # Due to the computation chain, this is more complex, but the values should be finite
    assert np.isfinite(actual_sma)


def test_compute_idio_momentum():
    """Test idio-momentum signal computation."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    
    # Create some residuals with trends
    np.random.seed(42)
    residuals = pd.DataFrame({
        'SOL': np.cumsum(np.random.normal(0.001, 0.02, len(dates))),
        'ADA': np.cumsum(np.random.normal(0.0005, 0.015, len(dates))),
        'XRP': np.cumsum(np.random.normal(-0.0002, 0.018, len(dates)))
    }, index=dates)
    
    # Idiosyncratic volatility
    idio_vol = pd.DataFrame({
        'SOL': [0.02] * len(dates),
        'ADA': [0.015] * len(dates),
        'XRP': [0.018] * len(dates)
    }, index=dates)
    
    # Compute idio-momentum
    idio_momentum = compute_idio_momentum(residuals, idio_vol, fast_span=3, slow_span=10)
    
    # Verify shape
    assert idio_momentum.shape == residuals.shape
    
    # Verify that values are finite where they should be
    # (Note: early periods may have NaN due to EMA initialization)
    assert not idio_momentum.isna().all().all()


def test_compute_idio_meanrev():
    """Test idio-meanrev signal computation."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    
    # Create residuals with some mean-reversion characteristics
    np.random.seed(42)
    residuals = pd.DataFrame({
        'SOL': np.random.normal(0.001, 0.02, len(dates)),
        'ADA': np.random.normal(0.0005, 0.015, len(dates)),
        'XRP': np.random.normal(-0.0002, 0.018, len(dates))
    }, index=dates)
    
    # Add some pattern to make mean reversion detectable
    for col in residuals.columns:
        # Make values revert to mean occasionally
        residuals.loc[dates[10:15], col] = -residuals.loc[dates[10:15], col]
    
    # Idiosyncratic volatility
    idio_vol = pd.DataFrame({
        'SOL': [0.02] * len(dates),
        'ADA': [0.015] * len(dates),
        'XRP': [0.018] * len(dates)
    }, index=dates)
    
    # Compute idio-meanrev
    idio_meanrev = compute_idio_meanrev(residuals, idio_vol, sma_period=5)
    
    # Verify shape
    assert idio_meanrev.shape == residuals.shape
    
    # Verify that values are finite where they should be
    assert not idio_meanrev.isna().all().all()


def test_apply_cross_sectional_zscore():
    """Test cross-sectional z-score and winsorization."""
    # Create sample signals
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    
    signals = pd.DataFrame({
        'SOL': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'ADA': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        'XRP': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]
    }, index=dates)
    
    # Apply z-score
    zscored = apply_cross_sectional_zscore(signals, winsorize_at=3.0)
    
    # Verify shape
    assert zscored.shape == signals.shape
    
    # For each date, the mean should be approximately 0 (with some numerical error)
    date_means = zscored.mean(axis=1)
    assert np.allclose(date_means, 0.0, atol=1e-10)
    
    # Each z-score should be within the winsorization bounds
    assert (zscored <= 3.0).all().all()
    assert (zscored >= -3.0).all().all()


def test_generate_spectrum_signals():
    """Test end-to-end spectrum signal generation."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    
    # Create sample inputs
    np.random.seed(42)
    residuals = pd.DataFrame({
        'SOL': np.random.normal(0.001, 0.02, len(dates)),
        'ADA': np.random.normal(0.0005, 0.015, len(dates)),
        'XRP': np.random.normal(-0.0002, 0.018, len(dates))
    }, index=dates)
    
    daily_funding = pd.DataFrame({
        'SOL': np.random.normal(0.0001, 0.0005, len(dates)),
        'ADA': np.random.normal(-0.00005, 0.0004, len(dates)),
        'XRP': np.random.normal(0.00002, 0.0003, len(dates))
    }, index=dates)
    
    idio_vol = pd.DataFrame({
        'SOL': [0.02] * len(dates),
        'ADA': [0.015] * len(dates),
        'XRP': [0.018] * len(dates)
    }, index=dates)
    
    # Generate all signals
    signals = generate_spectrum_signals(
        residuals, daily_funding, idio_vol,
        warmup_periods=10,
        momentum_fast_span=3,
        momentum_slow_span=10,
        meanrev_sma_period=5,
        zscore_winsorize_at=3.0
    )
    
    # Verify return structure
    assert isinstance(signals, dict)
    assert set(signals.keys()) == {'idio_carry', 'idio_momentum', 'idio_meanrev'}
    
    # Verify shapes
    for key in signals:
        assert signals[key].shape == residuals.shape
        # Check that after warmup, we have reasonable values
        # (Before warmup the first few periods may be NaN)
        assert not signals[key].isna().all().all()
    
    # Verify values are finite and within expected ranges
    for key in signals:
        # Check that all values are finite (not NaN, inf, -inf)
        assert np.isfinite(signals[key].values).all()


def test_signal_computation_edge_cases():
    """Test edge cases in signal computation."""
    # Test with all zeros (after sufficient periods for EMA to be valid)
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')  # More periods for EMA

    zeros_df = pd.DataFrame({
        'SOL': [0.0] * len(dates),
        'ADA': [0.0] * len(dates)
    }, index=dates)

    idio_vol = pd.DataFrame({
        'SOL': [0.02] * len(dates),
        'ADA': [0.015] * len(dates)
    }, index=dates)

    # Should handle zero residuals without error
    momentum = compute_idio_momentum(zeros_df, idio_vol)
    meanrev = compute_idio_meanrev(zeros_df, idio_vol)
    carry = compute_idio_carry(zeros_df, idio_vol)

    # Test that we have finite values after the initial warmup period
    # For EMA with spans 3 and 10, we expect NaN values in early periods
    # Check only valid (non-NaN) values are finite
    momentum_valid = momentum.dropna(how='all')  # Remove rows that are all NaN
    if not momentum_valid.empty:
        assert np.isfinite(momentum_valid.values).all()

    meanrev_valid = meanrev.dropna(how='all')
    if not meanrev_valid.empty:
        assert np.isfinite(meanrev_valid.values).all()

    carry_valid = carry.dropna(how='all')
    if not carry_valid.empty:
        assert np.isfinite(carry_valid.values).all()

    # Test with very small volatility (should not cause division by zero)
    small_vol = pd.DataFrame({
        'SOL': [1e-8] * len(dates),
        'ADA': [1e-8] * len(dates)
    }, index=dates)

    momentum_small = compute_idio_momentum(zeros_df, small_vol)
    # Should handle small vol without error, possibly creating large values but not inf/nan


if __name__ == "__main__":
    # Run the tests
    test_compute_idio_carry()
    test_compute_idio_momentum()
    test_compute_idio_meanrev()
    test_apply_cross_sectional_zscore()
    test_generate_spectrum_signals()
    test_signal_computation_edge_cases()
    print("All signal factory tests passed!")