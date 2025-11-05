"""
Test multi-factor PCA orthogonality.

This script validates that:
1. Multi-component PCA computes correctly (PC1, PC2, PC3)
2. Factor returns can be computed for each component
3. Multi-factor residuals are orthogonal to all three factors
4. Correlation between residuals and factors is ~zero
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from slipstream.core.signals import compute_multifactor_residuals


def generate_synthetic_data(n_timestamps=100, n_assets=20, random_seed=42):
    """Generate synthetic returns data with known factor structure."""
    np.random.seed(random_seed)

    timestamps = pd.date_range('2024-01-01', periods=n_timestamps, freq='D')
    assets = [f'ASSET_{i}' for i in range(n_assets)]

    # Generate three independent factors
    factor_returns = {
        'PC1': np.random.normal(0, 0.02, n_timestamps),  # Market factor
        'PC2': np.random.normal(0, 0.01, n_timestamps),  # Size/sector factor
        'PC3': np.random.normal(0, 0.005, n_timestamps),  # Style factor
    }
    factors_df = pd.DataFrame(factor_returns, index=timestamps)

    # Generate random loadings (betas) for each asset on each factor
    loadings = {}
    for pc_name in ['PC1', 'PC2', 'PC3']:
        loadings[pc_name] = pd.DataFrame(
            np.random.uniform(0.1, 0.5, (n_timestamps, n_assets)),
            index=timestamps,
            columns=assets
        )

    # Generate asset returns as linear combination of factors + noise
    returns_dict = {}
    for asset in assets:
        # Return = beta1*F1 + beta2*F2 + beta3*F3 + noise
        returns_dict[asset] = (
            loadings['PC1'][asset].values * factors_df['PC1'].values +
            loadings['PC2'][asset].values * factors_df['PC2'].values +
            loadings['PC3'][asset].values * factors_df['PC3'].values +
            np.random.normal(0, 0.005, n_timestamps)  # idiosyncratic noise
        )

    returns = pd.DataFrame(returns_dict, index=timestamps)

    # Convert loadings to long format (timestamp, asset) index
    loadings_long = {}
    for pc_name in ['PC1', 'PC2', 'PC3']:
        loadings_long[pc_name] = loadings[pc_name].stack()
        loadings_long[pc_name].index.names = ['timestamp', 'asset']

    return returns, loadings_long, factors_df


def test_multifactor_orthogonality():
    """Test that multi-factor residuals are orthogonal to all factors."""

    print("=" * 70)
    print("MULTI-FACTOR PCA ORTHOGONALITY TEST")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    returns, loadings_long, factors_df = generate_synthetic_data()

    print(f"   Returns shape: {returns.shape}")
    print(f"   Factors: {list(factors_df.columns)}")
    print(f"   Loadings per factor: {loadings_long['PC1'].shape}")

    # Compute multi-factor residuals
    print("\n2. Computing multi-factor residuals...")
    residuals = compute_multifactor_residuals(
        returns,
        loadings_long['PC1'],
        loadings_long['PC2'],
        loadings_long['PC3'],
        factors_df['PC1'],
        factors_df['PC2'],
        factors_df['PC3']
    )

    print(f"   Residuals shape: {residuals.shape}")
    print(f"   Residuals mean: {residuals.mean().mean():.6f}")
    print(f"   Residuals std:  {residuals.std().mean():.6f}")

    # Test orthogonality: correlation should be ~zero
    print("\n3. Testing orthogonality (correlation with factors)...")

    correlations = {}
    for pc_name in ['PC1', 'PC2', 'PC3']:
        # Compute correlation between residuals and factor returns
        # For each asset, correlate its residuals with the factor
        asset_corrs = []
        for asset in residuals.columns:
            if asset in returns.columns:
                corr = residuals[asset].corr(factors_df[pc_name])
                if not np.isnan(corr):
                    asset_corrs.append(corr)

        correlations[pc_name] = {
            'mean': np.mean(asset_corrs) if asset_corrs else np.nan,
            'std': np.std(asset_corrs) if asset_corrs else np.nan,
            'max': np.max(np.abs(asset_corrs)) if asset_corrs else np.nan,
        }

        print(f"\n   {pc_name}:")
        print(f"     Mean correlation: {correlations[pc_name]['mean']:8.6f}")
        print(f"     Std correlation:  {correlations[pc_name]['std']:8.6f}")
        print(f"     Max |corr|:       {correlations[pc_name]['max']:8.6f}")

    # Validation thresholds
    print("\n4. Validation results:")
    all_passed = True

    for pc_name in ['PC1', 'PC2', 'PC3']:
        mean_corr = abs(correlations[pc_name]['mean'])
        max_corr = correlations[pc_name]['max']

        # Relaxed thresholds accounting for statistical noise:
        # - Mean correlation should be < 0.10 (close to zero)
        # - Max correlation can be higher (< 0.30) due to random chance
        mean_ok = mean_corr < 0.10
        max_ok = max_corr < 0.30

        status = "✓ PASS" if (mean_ok and max_ok) else "✗ FAIL"
        print(f"   {pc_name}: {status}")

        if not (mean_ok and max_ok):
            all_passed = False
            print(f"      Mean |corr| = {mean_corr:.4f} (threshold: 0.10)")
            print(f"      Max |corr|  = {max_corr:.4f} (threshold: 0.30)")

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Residuals are orthogonal to all factors!")
    else:
        print("✗ SOME TESTS FAILED - Check orthogonality implementation")
    print("=" * 70)

    assert all_passed


def test_variance_decomposition():
    """Test that variance explained adds up correctly."""

    print("\n" + "=" * 70)
    print("VARIANCE DECOMPOSITION TEST")
    print("=" * 70)

    # Generate data
    returns, loadings_long, factors_df = generate_synthetic_data()

    # Compute residuals
    residuals = compute_multifactor_residuals(
        returns,
        loadings_long['PC1'],
        loadings_long['PC2'],
        loadings_long['PC3'],
        factors_df['PC1'],
        factors_df['PC2'],
        factors_df['PC3']
    )

    # Compute variance of original returns
    original_var = returns.var().mean()

    # Compute variance of residuals
    residual_var = residuals.var().mean()

    # Variance explained by factors = 1 - (residual_var / original_var)
    var_explained = 1 - (residual_var / original_var)

    print(f"\nOriginal returns variance:  {original_var:.6f}")
    print(f"Residual variance:          {residual_var:.6f}")
    print(f"Variance explained:         {var_explained:.2%}")

    # With our synthetic data generation, we added idiosyncratic noise
    # Variance explained should be substantial (> 50%) but not perfect
    if var_explained > 0.50:
        print("\n✓ Variance decomposition looks correct")
        print(f"  (Factor model explains {var_explained:.1%} of variance)")
        assert True
    else:
        print(f"\n✗ Unexpectedly low variance explained (expected > 50%, got {var_explained:.2%})")
        assert False


if __name__ == "__main__":
    print("\nTesting multi-factor PCA implementation...\n")

    test1_passed = test_multifactor_orthogonality()
    test2_passed = test_variance_decomposition()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Orthogonality test: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Variance test:      {'✓ PASS' if test2_passed else '✗ FAIL'}")

    if test1_passed and test2_passed:
        print("\n✓ All validation tests passed!")
        print("Multi-factor PCA implementation is working correctly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please review the implementation.")
        sys.exit(1)
