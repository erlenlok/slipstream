"""
Unit tests for Spectrum Robust Optimizer - Module D

Tests for optimizer.py following the requirements in spectrum_spec.md
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.slipstream.strategies.spectrum.optimizer import (
    compute_cost_vector,
    compute_ledoit_wolf_covariance,
    compute_robust_optimization,
    prepare_asset_universe,
    optimize_spectrum_portfolio
)


def test_compute_cost_vector():
    """Test cost vector computation."""
    # Create sample data
    volatilities = pd.Series([0.02, 0.03, 0.015], index=['SOL', 'ADA', 'XRP'])
    
    # Test with volatility-based costs
    cost_vector = compute_cost_vector(volatilities, base_cost=0.0002)
    
    # Verify output
    assert len(cost_vector) == 3
    assert list(cost_vector.index) == ['SOL', 'ADA', 'XRP']
    # Costs should be positive and higher for higher volatility assets
    assert (cost_vector > 0).all()
    assert cost_vector['ADA'] > cost_vector['XRP']  # ADA has higher volatility
    
    # Test with spread proxy
    spreads = pd.Series([0.0005, 0.0008, 0.0003], index=['SOL', 'ADA', 'XRP'])
    cost_vector_with_spread = compute_cost_vector(volatilities, spreads)
    
    assert len(cost_vector_with_spread) == 3
    # Should use spread values when provided
    assert cost_vector_with_spread['ADA'] > cost_vector_with_spread['XRP']


def test_compute_ledoit_wolf_covariance():
    """Test Ledoit-Wolf covariance computation."""
    # Create sample residuals data
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    np.random.seed(42)
    
    # Create correlated residuals
    residuals = pd.DataFrame({
        'SOL': np.random.normal(0.0, 0.02, len(dates)),
        'ADA': np.random.normal(0.0, 0.015, len(dates)),
        'XRP': np.random.normal(0.0, 0.018, len(dates))
    }, index=dates)
    
    # Add some correlation
    residuals['ADA'] += 0.3 * residuals['SOL']
    residuals['XRP'] += 0.2 * residuals['SOL']
    
    cov_matrix = compute_ledoit_wolf_covariance(residuals)
    
    # Verify output shape and properties
    assert cov_matrix.shape == (3, 3)
    assert np.allclose(cov_matrix, cov_matrix.T)  # Symmetric
    assert np.all(np.linalg.eigvals(cov_matrix) >= -1e-10)  # PSD (with tolerance)
    
    # Diagonal should be variances (positive)
    assert np.all(np.diag(cov_matrix) > 0)


def test_compute_robust_optimization():
    """Test robust optimization."""
    # Create sample inputs
    n_assets = 3
    alpha = np.array([0.01, -0.005, 0.02])  # Expected returns
    cov_matrix = np.array([[0.0004, 0.0001, 0.00005],
                           [0.0001, 0.000225, 0.00003],
                           [0.00005, 0.00003, 0.000324]])  # Covariance matrix
    betas = np.array([0.8, 1.2, 0.5])  # Market betas
    w_prev = np.array([0.1, -0.05, 0.15])  # Previous weights
    cost_vector = np.array([0.0002, 0.0003, 0.00025])  # Cost coefficients
    
    # Run optimization
    w_opt, info = compute_robust_optimization(
        alpha=alpha,
        cov_matrix=cov_matrix,
        betas=betas,
        w_prev=w_prev,
        cost_vector=cost_vector,
        target_leverage=1.0,
        max_single_pos=0.5,
        gamma=1.0
    )
    
    # Verify output
    assert len(w_opt) == n_assets
    assert np.isfinite(w_opt).all()
    assert np.isfinite(info['actual_leverage'])
    assert np.abs(info['beta_exposure']) < 1e-5  # Should be approximately beta neutral
    assert info['actual_leverage'] <= 1.0001  # Should respect leverage constraint (with small tolerance for numerical precision)


def test_prepare_asset_universe():
    """Test asset universe preparation."""
    current_assets = ['SOL', 'ADA', 'XRP', 'BTC']
    previous_weights = {'SOL': 0.2, 'ADA': -0.1, 'ETH': 0.3}  # ETH not in current
    
    asset_names, w_prev = prepare_asset_universe(current_assets, previous_weights)
    
    # Verify outputs
    assert len(asset_names) == 4
    assert len(w_prev) == 4
    assert list(asset_names) == current_assets
    # SOL should have previous weight of 0.2
    assert w_prev[0] == 0.2  # SOL
    # ADA should have previous weight of -0.1
    assert w_prev[1] == -0.1  # ADA
    # XRP and BTC should have 0 weights (new entrants)
    assert w_prev[2] == 0.0  # XRP
    assert w_prev[3] == 0.0  # BTC


def test_optimize_spectrum_portfolio():
    """Test end-to-end spectrum portfolio optimization."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    np.random.seed(42)
    
    # Composite alpha (predictions)
    composite_alpha = pd.DataFrame({
        'SOL': np.random.normal(0.001, 0.01, len(dates)),
        'ADA': np.random.normal(-0.0005, 0.008, len(dates)),
        'XRP': np.random.normal(0.002, 0.012, len(dates))
    }, index=dates)
    
    # Idiosyncratic residuals
    residuals = pd.DataFrame({
        'SOL': np.random.normal(0.0, 0.02, len(dates)),
        'ADA': np.random.normal(0.0, 0.015, len(dates)),
        'XRP': np.random.normal(0.0, 0.018, len(dates))
    }, index=dates)
    
    # Add some correlation structure
    residuals['ADA'] += 0.3 * 0.02 * np.random.normal(0, 0.02, len(dates))
    residuals['XRP'] += 0.2 * 0.02 * np.random.normal(0, 0.02, len(dates))
    
    # Beta coefficients
    betas = pd.DataFrame({
        'SOL': [0.8] * len(dates),
        'ADA': [1.2] * len(dates), 
        'XRP': [0.6] * len(dates)
    }, index=dates)
    
    # Previous weights
    previous_weights = {'SOL': 0.1, 'ADA': -0.05, 'XRP': 0.15}
    
    # Run portfolio optimization
    weights, info = optimize_spectrum_portfolio(
        composite_alpha=composite_alpha,
        residuals=residuals,
        betas=betas,
        previous_weights=previous_weights,
        target_leverage=1.0,
        max_single_pos=0.3
    )
    
    # Verify output
    assert isinstance(weights, pd.Series)
    assert set(weights.index) == set(['SOL', 'ADA', 'XRP'])
    assert np.isfinite(weights).all()
    # The portfolio should be BETA neutral, not market neutral (weights don't sum to 0)
    # Calculate beta neutrality: weights.T * betas should be ~0
    final_betas = betas.iloc[-1]  # Get the latest betas
    beta_neutrality = sum(weights[asset] * final_betas[asset] for asset in weights.index)
    assert abs(beta_neutrality) < 0.01  # Should be approximately beta neutral
    assert np.sum(np.abs(weights)) <= 1.1  # Should respect leverage constraint (with small tolerance)

    # Verify info contains expected fields
    assert 'actual_leverage' in info
    assert 'beta_exposure' in info
    assert 'n_positions' in info


def test_optimizer_edge_cases():
    """Test optimizer edge cases."""
    # Test with empty inputs
    weights, info = optimize_spectrum_portfolio(
        composite_alpha=pd.DataFrame(),
        residuals=pd.DataFrame(),
        betas=pd.DataFrame()
    )
    assert len(weights) == 0
    assert 'error' in info
    
    # Test with single asset
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    single_asset_alpha = pd.DataFrame({'SOL': [0.01] * len(dates)}, index=dates)
    single_asset_residuals = pd.DataFrame({'SOL': np.random.normal(0, 0.02, len(dates))}, index=dates)
    single_asset_betas = pd.DataFrame({'SOL': [0.8] * len(dates)}, index=dates)
    
    weights, info = optimize_spectrum_portfolio(
        composite_alpha=single_asset_alpha,
        residuals=single_asset_residuals,
        betas=single_asset_betas,
        target_leverage=0.5
    )
    
    assert len(weights) == 1
    assert 'SOL' in weights.index
    assert np.isfinite(weights['SOL'])
    
    # Test with NaN values in alpha
    nan_alpha = single_asset_alpha.copy()
    nan_alpha.iloc[0, 0] = np.nan  # Introduce NaN
    
    weights_nan, info_nan = optimize_spectrum_portfolio(
        composite_alpha=nan_alpha,
        residuals=single_asset_residuals,
        betas=single_asset_betas
    )
    
    # Should handle NaN gracefully
    assert isinstance(weights_nan, pd.Series)
    assert 'SOL' in weights_nan.index


def test_beta_neutrality_constraint():
    """Test that the optimization actually achieves beta neutrality."""
    # Create sample data with clear alpha and beta signals
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    np.random.seed(123)
    
    # Create strongly positive alpha for SOL and negative for ADA
    composite_alpha = pd.DataFrame({
        'SOL': [0.02] * len(dates),  # Strong positive signal
        'ADA': [-0.02] * len(dates),  # Strong negative signal
        'XRP': [0.0] * len(dates)     # Neutral signal
    }, index=dates)
    
    # Create residuals data
    residuals = pd.DataFrame({
        'SOL': np.random.normal(0, 0.02, len(dates)),
        'ADA': np.random.normal(0, 0.015, len(dates)),
        'XRP': np.random.normal(0, 0.018, len(dates))
    }, index=dates)
    
    # Different betas for each asset
    betas = pd.DataFrame({
        'SOL': [1.5] * len(dates),  # High beta
        'ADA': [0.5] * len(dates),  # Low beta
        'XRP': [1.0] * len(dates)   # Medium beta
    }, index=dates)
    
    # Run optimization
    weights, info = optimize_spectrum_portfolio(
        composite_alpha=composite_alpha,
        residuals=residuals,
        betas=betas,
        target_leverage=1.0,
        max_single_pos=0.8
    )
    
    # Calculate actual beta exposure
    final_betas = betas.iloc[-1]  # Use the latest beta values
    beta_exposure = sum(weights[asset] * final_betas[asset] for asset in weights.index)
    
    # Beta exposure should be very close to zero (beta neutrality)
    assert abs(beta_exposure) < 0.01, f"Beta neutrality failed: exposure = {beta_exposure}"


if __name__ == "__main__":
    # Run tests
    test_compute_cost_vector()
    test_compute_ledoit_wolf_covariance()
    test_compute_robust_optimization()
    test_prepare_asset_universe()
    test_optimize_spectrum_portfolio()
    test_optimizer_edge_cases()
    test_beta_neutrality_constraint()
    print("All optimizer tests passed!")