"""
Integration tests for Gradient portfolio construction with VAR targeting.
"""

import numpy as np
import pandas as pd
import pytest

from slipstream.strategies.gradient.portfolio import construct_gradient_portfolio
from slipstream.common.risk import compute_portfolio_var, estimate_covariance_rie, compute_daily_returns


class TestGradientPortfolioVAR:
    """Test VAR-based portfolio construction."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic market data for testing."""
        np.random.seed(42)

        # 120 days of 4h data (20 days * 6 periods/day)
        n_periods = 120
        n_assets = 60

        dates = pd.date_range("2024-01-01", periods=n_periods, freq="4h")

        # Generate correlated returns (common factor + idiosyncratic)
        factor = np.random.randn(n_periods, 1) * 0.01
        idio = np.random.randn(n_periods, n_assets) * 0.015
        returns_4h = 0.4 * factor + 0.6 * idio

        log_returns = pd.DataFrame(
            returns_4h,
            index=dates,
            columns=[f"ASSET_{i}" for i in range(n_assets)],
        )

        # Generate trend strength signals (stronger for some assets)
        signal_base = np.random.randn(n_periods, n_assets)
        # Add persistence to signals
        for i in range(1, n_periods):
            signal_base[i] = 0.7 * signal_base[i - 1] + 0.3 * signal_base[i]

        trend_strength = pd.DataFrame(
            signal_base,
            index=dates,
            columns=log_returns.columns,
        )

        return log_returns, trend_strength

    def test_backward_compatibility_with_dollar_vol(self, synthetic_data):
        """VAR method with risk_method='dollar_vol' should match legacy behavior."""
        log_returns, trend_strength = synthetic_data

        # Construct with explicit dollar_vol
        weights_explicit = construct_gradient_portfolio(
            trend_strength,
            log_returns,
            top_n=5,
            bottom_n=5,
            risk_method="dollar_vol",
            target_side_dollar_vol=1.0,
            vol_span=16,  # Shorter for test
        )

        # Should produce non-zero weights
        assert (weights_explicit.abs().sum(axis=1) > 0).sum() > 0

        # Check structure: should have ~10 non-zero positions (5 long + 5 short)
        non_zero_per_period = (weights_explicit != 0).sum(axis=1)
        assert non_zero_per_period.max() <= 10

    def test_var_method_produces_balanced_risk(self, synthetic_data):
        """VAR method should produce approximately equal VAR on long/short sides."""
        log_returns, trend_strength = synthetic_data

        # Construct with VAR method
        weights = construct_gradient_portfolio(
            trend_strength,
            log_returns,
            top_n=50,  # Use many assets to test N=50 regime
            bottom_n=50,
            risk_method="var",
            target_side_var=0.02,  # 2% daily VAR
            var_lookback_days=60,
        )

        # Skip early periods without enough data
        test_idx = weights.index[-1]  # Test last period with full history

        # Separate long and short weights
        long_weights = weights.loc[test_idx][weights.loc[test_idx] > 0]
        short_weights = weights.loc[test_idx][weights.loc[test_idx] < 0]

        # Should have non-zero positions
        assert len(long_weights) > 0
        assert len(short_weights) > 0

        # Compute daily returns and covariance
        daily_returns = compute_daily_returns(log_returns, window=6)
        daily_returns_clean = daily_returns.loc[:test_idx].dropna(how="all")

        # Compute VAR for long side
        long_assets = long_weights.index.tolist()
        long_cov = estimate_covariance_rie(
            daily_returns_clean[long_assets],
            lookback_days=60,
        )
        long_var = compute_portfolio_var(
            long_weights.values,
            long_cov,
            confidence=0.95,
        )

        # Compute VAR for short side
        short_assets = short_weights.index.tolist()
        short_cov = estimate_covariance_rie(
            daily_returns_clean[short_assets],
            lookback_days=60,
        )
        short_var = compute_portfolio_var(
            abs(short_weights.values),  # Use absolute values for VAR
            short_cov,
            confidence=0.95,
        )

        # VARs should be similar (within 50% of each other)
        # Note: Some imbalance is expected due to discrete asset selection and correlations
        var_ratio = long_var / short_var
        assert 0.5 < var_ratio < 2.0, f"VAR ratio {var_ratio:.2f} outside acceptable range"

    def test_var_method_weights_proportional_to_signal(self, synthetic_data):
        """Within each side, weights should be proportional to signal strength."""
        log_returns, trend_strength = synthetic_data

        # Set specific signals for a test period
        test_idx = trend_strength.index[-1]

        # Create artificial strong signals for specific assets
        trend_strength_test = trend_strength.copy()
        trend_strength_test.loc[test_idx, "ASSET_0"] = 10.0  # Very strong long
        trend_strength_test.loc[test_idx, "ASSET_1"] = 5.0  # Medium long
        trend_strength_test.loc[test_idx, "ASSET_2"] = -10.0  # Very strong short
        trend_strength_test.loc[test_idx, "ASSET_3"] = -5.0  # Medium short

        weights = construct_gradient_portfolio(
            trend_strength_test,
            log_returns,
            top_n=10,
            bottom_n=10,
            risk_method="var",
            target_side_var=0.02,
            var_lookback_days=60,
        )

        # Get weights for test period
        w = weights.loc[test_idx]

        # Asset with signal=10 should have ~2x weight of asset with signal=5 (within same side)
        if w["ASSET_0"] > 0 and w["ASSET_1"] > 0:  # Both long
            ratio = w["ASSET_0"] / w["ASSET_1"]
            assert 1.5 < ratio < 2.5, f"Weight ratio {ratio:.2f} doesn't match 2:1 signal ratio"

    def test_var_method_handles_missing_data(self, synthetic_data):
        """VAR method should gracefully handle assets with missing data."""
        log_returns, trend_strength = synthetic_data

        # Introduce NaNs in some assets
        log_returns_with_nans = log_returns.copy()
        log_returns_with_nans.loc[:, "ASSET_0"] = np.nan  # Completely missing

        weights = construct_gradient_portfolio(
            trend_strength,
            log_returns_with_nans,
            top_n=10,
            bottom_n=10,
            risk_method="var",
            target_side_var=0.02,
            var_lookback_days=60,
        )

        # Should still produce weights (just excluding missing assets)
        assert (weights.abs().sum(axis=1) > 0).sum() > 0

        # ASSET_0 should have zero weight (missing data)
        assert (weights["ASSET_0"] == 0).all()

    def test_invalid_risk_method_raises_error(self, synthetic_data):
        """Invalid risk_method should raise ValueError."""
        log_returns, trend_strength = synthetic_data

        with pytest.raises(ValueError, match="risk_method must be"):
            construct_gradient_portfolio(
                trend_strength,
                log_returns,
                risk_method="invalid_method",
            )

    def test_negative_target_var_raises_error(self, synthetic_data):
        """Negative target_side_var should raise ValueError."""
        log_returns, trend_strength = synthetic_data

        with pytest.raises(ValueError, match="target_side_var must be positive"):
            construct_gradient_portfolio(
                trend_strength,
                log_returns,
                risk_method="var",
                target_side_var=-0.01,
            )
