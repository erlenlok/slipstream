"""
Unit tests for risk management utilities (VAR and RIE covariance).
"""

import numpy as np
import pandas as pd
import pytest

from slipstream.common.risk import (
    compute_daily_returns,
    compute_portfolio_var,
    estimate_covariance_rie,
)


class TestComputeDailyReturns:
    """Test 4h to daily returns resampling."""

    def test_basic_resampling(self):
        """Test that 6 consecutive 4h returns sum to daily return."""
        # Create 12 periods (2 days) of 4h returns
        dates = pd.date_range("2024-01-01", periods=12, freq="4h")
        returns_4h = pd.DataFrame(
            {
                "BTC": [0.01] * 12,  # Constant 1% per 4h
                "ETH": [0.02] * 12,  # Constant 2% per 4h
            },
            index=dates,
        )

        daily = compute_daily_returns(returns_4h, window=6)

        # First 5 rows should be NaN (need 6 periods for first daily return)
        assert daily.iloc[:5].isna().all().all()

        # Row 6 should have sum of first 6 periods
        assert daily.iloc[5]["BTC"] == pytest.approx(0.06)  # 6 * 0.01
        assert daily.iloc[5]["ETH"] == pytest.approx(0.12)  # 6 * 0.02

    def test_handles_nans(self):
        """Test that NaNs are handled correctly in rolling sum."""
        dates = pd.date_range("2024-01-01", periods=10, freq="4h")
        returns_4h = pd.DataFrame(
            {
                "BTC": [0.01, np.nan, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            },
            index=dates,
        )

        daily = compute_daily_returns(returns_4h, window=6)

        # Row with NaN input should have NaN output
        assert daily.iloc[6].isna().all()


class TestEstimateCovarianceRIE:
    """Test RIE covariance estimation."""

    def test_diagonal_fallback_when_t_less_than_n(self):
        """When T < N, should fallback to diagonal covariance."""
        # Create 30 days of data for 50 assets (T=30 < N=50)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        n_assets = 50
        returns = pd.DataFrame(
            np.random.randn(30, n_assets) * 0.02,
            index=dates,
            columns=[f"ASSET_{i}" for i in range(n_assets)],
        )

        cov = estimate_covariance_rie(returns, lookback_days=30, fallback_diagonal=True)

        # Check it's diagonal (off-diagonals should be zero)
        off_diagonal_sum = np.abs(cov.values - np.diag(np.diag(cov.values))).sum()
        assert off_diagonal_sum == pytest.approx(0.0)

        # Check diagonal elements are positive
        assert (np.diag(cov.values) > 0).all()

    def test_rie_cleaning_shrinks_small_eigenvalues(self):
        """RIE should shrink small (noisy) eigenvalues toward zero."""
        # Create 60 days of returns for 50 assets with known structure
        np.random.seed(42)
        T, N = 60, 50

        # Generate returns with modest correlations
        factor = np.random.randn(T, 1) * 0.01  # Common factor
        idio = np.random.randn(T, N) * 0.015  # Idiosyncratic noise
        returns_data = 0.3 * factor + 0.7 * idio  # Returns = 30% factor + 70% noise

        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        returns = pd.DataFrame(
            returns_data,
            index=dates,
            columns=[f"ASSET_{i}" for i in range(N)],
        )

        # Compute both empirical and RIE covariance
        cov_rie = estimate_covariance_rie(returns, lookback_days=T)

        # Compute empirical for comparison
        cov_emp = returns.cov()

        # RIE should shrink small eigenvalues (noise reduction)
        eig_rie = np.linalg.eigvalsh(cov_rie.values)
        eig_emp = np.linalg.eigvalsh(cov_emp.values)

        # Number of near-zero eigenvalues should increase with RIE
        zero_threshold = 1e-6
        n_zero_rie = np.sum(eig_rie < zero_threshold)
        n_zero_emp = np.sum(eig_emp < zero_threshold)

        assert n_zero_rie > n_zero_emp  # RIE sets more noisy eigenvalues to ~zero

    def test_covariance_is_symmetric(self):
        """Cleaned covariance matrix should be symmetric."""
        np.random.seed(42)
        T, N = 60, 30

        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        returns = pd.DataFrame(
            np.random.randn(T, N) * 0.02,
            index=dates,
            columns=[f"ASSET_{i}" for i in range(N)],
        )

        cov = estimate_covariance_rie(returns, lookback_days=T)

        # Check symmetry
        assert np.allclose(cov.values, cov.values.T)

    def test_covariance_is_positive_semidefinite(self):
        """Cleaned covariance should be positive semidefinite."""
        np.random.seed(42)
        T, N = 60, 30

        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        returns = pd.DataFrame(
            np.random.randn(T, N) * 0.02,
            index=dates,
            columns=[f"ASSET_{i}" for i in range(N)],
        )

        cov = estimate_covariance_rie(returns, lookback_days=T)

        # Check all eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert (eigenvalues >= -1e-10).all()  # Allow small numerical errors


class TestComputePortfolioVAR:
    """Test parametric VAR calculation."""

    def test_var_scales_with_weights(self):
        """Doubling all weights should double VAR."""
        # Simple 2-asset covariance
        cov = pd.DataFrame(
            [[0.01, 0.005], [0.005, 0.01]],
            index=["BTC", "ETH"],
            columns=["BTC", "ETH"],
        )

        weights1 = np.array([0.5, 0.5])
        weights2 = np.array([1.0, 1.0])

        var1 = compute_portfolio_var(weights1, cov, confidence=0.95)
        var2 = compute_portfolio_var(weights2, cov, confidence=0.95)

        assert var2 == pytest.approx(2 * var1)

    def test_var_with_dict_weights(self):
        """VAR should work with dict weights."""
        cov = pd.DataFrame(
            [[0.01, 0.005], [0.005, 0.01]],
            index=["BTC", "ETH"],
            columns=["BTC", "ETH"],
        )

        weights_array = np.array([0.6, 0.4])
        weights_dict = {"BTC": 0.6, "ETH": 0.4}

        var_array = compute_portfolio_var(weights_array, cov)
        var_dict = compute_portfolio_var(weights_dict, cov)

        assert var_array == pytest.approx(var_dict)

    def test_var_increases_with_confidence(self):
        """Higher confidence should give higher VAR."""
        cov = pd.DataFrame(
            [[0.01, 0.005], [0.005, 0.01]],
            index=["BTC", "ETH"],
            columns=["BTC", "ETH"],
        )

        weights = np.array([0.5, 0.5])

        var_95 = compute_portfolio_var(weights, cov, confidence=0.95)
        var_99 = compute_portfolio_var(weights, cov, confidence=0.99)

        assert var_99 > var_95

    def test_var_known_case(self):
        """Test VAR against known analytical result."""
        # Single asset, variance = 0.0004 (2% daily std)
        cov = pd.DataFrame([[0.0004]], index=["BTC"], columns=["BTC"])
        weights = np.array([1.0])

        var_95 = compute_portfolio_var(weights, cov, confidence=0.95)

        # VAR = 1.645 * 0.02 = 0.0329
        expected = 1.645 * 0.02
        assert var_95 == pytest.approx(expected, rel=1e-3)

    def test_diversification_reduces_var(self):
        """Uncorrelated assets should reduce VAR via diversification."""
        # Two uncorrelated assets with same variance
        cov = pd.DataFrame(
            [[0.0004, 0.0], [0.0, 0.0004]],  # Independent
            index=["BTC", "ETH"],
            columns=["BTC", "ETH"],
        )

        # Single asset
        weights_single = np.array([1.0, 0.0])
        var_single = compute_portfolio_var(weights_single, cov)

        # Equally weighted portfolio
        weights_diversified = np.array([0.5, 0.5])
        var_diversified = compute_portfolio_var(weights_diversified, cov)

        # Diversified VAR should be lower
        # For independent assets: var_portfolio = sqrt(0.5^2 * var_1 + 0.5^2 * var_2)
        #                                        = sqrt(2 * 0.5^2 * 0.0004) = 0.0141
        # vs single asset std = 0.02
        assert var_diversified < var_single
        assert var_diversified == pytest.approx(1.645 * np.sqrt(2 * 0.25 * 0.0004), rel=1e-3)
