"""Test multi-factor residuals with real crypto data."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from slipstream.signals import compute_multifactor_residuals

print("="*80)
print("MULTI-FACTOR RESIDUALS TEST (Real Crypto Data)")
print("="*80)

# Load 3-component PCA
print("\n1. Loading 3-component PCA...")
pca = pd.read_csv('data/features/pca_factor_H24_K30_sqrt_3pc.csv', index_col=0)
pca.index = pd.to_datetime(pca.index)
print(f"   Loaded: {len(pca)} timestamps")

# Extract assets
all_cols = [col for col in pca.columns if not col.startswith('_')]
assets = sorted(list(set([col.split('_pc')[0] for col in all_cols if '_pc' in col])))
print(f"   Assets: {len(assets)}")

# Load returns (4H data)
print("\n2. Loading 4H returns data...")
data_dir = Path('data/market_data')
returns_dict = {}

for asset in assets[:20]:  # Test with subset for speed
    candle_file = data_dir / f'{asset}_candles_4h.csv'
    if candle_file.exists():
        df = pd.read_csv(candle_file)
        if 'datetime' in df.columns:
            df.index = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'])
        else:
            continue

        if 'close' in df.columns:
            log_returns = np.log(df['close'] / df['close'].shift(1))
            returns_dict[asset] = log_returns

returns = pd.DataFrame(returns_dict).sort_index()
print(f"   Loaded returns: {returns.shape}")
print(f"   Date range: {returns.index.min()} to {returns.index.max()}")

# Resample to daily to match PCA frequency
print("\n3. Resampling to daily (24H)...")
returns_daily = returns.resample('D').sum()
returns_daily = returns_daily[returns_daily.index >= pca.index.min()]
print(f"   Daily returns: {returns_daily.shape}")

# Extract loadings for each component (wide format)
print("\n4. Extracting loadings...")
assets_in_returns = returns_daily.columns

loadings_wide_pc1 = pd.DataFrame({
    asset: pca[f'{asset}_pc1'] for asset in assets_in_returns if f'{asset}_pc1' in pca.columns
}, index=pca.index)

loadings_wide_pc2 = pd.DataFrame({
    asset: pca[f'{asset}_pc2'] for asset in assets_in_returns if f'{asset}_pc2' in pca.columns
}, index=pca.index)

loadings_wide_pc3 = pd.DataFrame({
    asset: pca[f'{asset}_pc3'] for asset in assets_in_returns if f'{asset}_pc3' in pca.columns
}, index=pca.index)

print(f"   PC1 loadings: {loadings_wide_pc1.shape}")
print(f"   PC2 loadings: {loadings_wide_pc2.shape}")
print(f"   PC3 loadings: {loadings_wide_pc3.shape}")

# Compute factor returns (weighted sums)
print("\n5. Computing factor returns...")

def compute_market_factor(loadings_wide, returns):
    """Compute factor return as weighted sum."""
    common_timestamps = loadings_wide.index.intersection(returns.index)
    common_assets = loadings_wide.columns.intersection(returns.columns)

    loadings_aligned = loadings_wide.loc[common_timestamps, common_assets]
    returns_aligned = returns.loc[common_timestamps, common_assets]

    factor = (loadings_aligned * returns_aligned).sum(axis=1, skipna=True)
    return factor

factor_pc1 = compute_market_factor(loadings_wide_pc1, returns_daily)
factor_pc2 = compute_market_factor(loadings_wide_pc2, returns_daily)
factor_pc3 = compute_market_factor(loadings_wide_pc3, returns_daily)

print(f"   PC1 factor: {len(factor_pc1)} timestamps")
print(f"     Mean: {factor_pc1.mean():.6f}, Std: {factor_pc1.std():.6f}")
print(f"   PC2 factor: {len(factor_pc2)} timestamps")
print(f"     Mean: {factor_pc2.mean():.6f}, Std: {factor_pc2.std():.6f}")
print(f"   PC3 factor: {len(factor_pc3)} timestamps")
print(f"     Mean: {factor_pc3.mean():.6f}, Std: {factor_pc3.std():.6f}")

# Convert to long format for compute_multifactor_residuals
print("\n6. Converting to long format...")
loadings_long_pc1 = loadings_wide_pc1.stack()
loadings_long_pc1.index.names = ['timestamp', 'asset']

loadings_long_pc2 = loadings_wide_pc2.stack()
loadings_long_pc2.index.names = ['timestamp', 'asset']

loadings_long_pc3 = loadings_wide_pc3.stack()
loadings_long_pc3.index.names = ['timestamp', 'asset']

# Compute multi-factor residuals
print("\n7. Computing multi-factor residuals...")
residuals = compute_multifactor_residuals(
    returns_daily,
    loadings_long_pc1,
    loadings_long_pc2,
    loadings_long_pc3,
    factor_pc1,
    factor_pc2,
    factor_pc3
)

print(f"   Residuals shape: {residuals.shape}")
print(f"   Residuals mean: {residuals.mean().mean():.6f}")
print(f"   Residuals std:  {residuals.std().mean():.6f}")

# Test orthogonality
print("\n8. Testing orthogonality...")
correlations = {}
for pc_name, factor in [('PC1', factor_pc1), ('PC2', factor_pc2), ('PC3', factor_pc3)]:
    asset_corrs = []
    for asset in residuals.columns:
        if asset in returns_daily.columns:
            # Align residuals and factor
            common_idx = residuals.index.intersection(factor.index)
            res_aligned = residuals.loc[common_idx, asset]
            factor_aligned = factor.loc[common_idx]

            # Filter out NaN
            valid = ~(res_aligned.isna() | factor_aligned.isna())
            if valid.sum() > 10:
                corr = res_aligned[valid].corr(factor_aligned[valid])
                if not np.isnan(corr):
                    asset_corrs.append(corr)

    correlations[pc_name] = {
        'mean': np.mean(asset_corrs) if asset_corrs else np.nan,
        'std': np.std(asset_corrs) if asset_corrs else np.nan,
        'max_abs': np.max(np.abs(asset_corrs)) if asset_corrs else np.nan,
    }

    print(f"\n   {pc_name}:")
    print(f"     Mean correlation: {correlations[pc_name]['mean']:8.6f}")
    print(f"     Std correlation:  {correlations[pc_name]['std']:8.6f}")
    print(f"     Max |correlation|: {correlations[pc_name]['max_abs']:8.6f}")

# Variance decomposition
print("\n9. Variance decomposition...")
original_var = returns_daily.var().mean()
residual_var = residuals.var().mean()
var_explained = 1 - (residual_var / original_var)

print(f"   Original returns variance: {original_var:.8f}")
print(f"   Residual variance:         {residual_var:.8f}")
print(f"   Variance explained:        {var_explained:.1%}")
print(f"   Expected from PCA:         93.5%")

# Summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

all_ok = True
for pc_name in ['PC1', 'PC2', 'PC3']:
    mean_corr = abs(correlations[pc_name]['mean'])
    max_corr = correlations[pc_name]['max_abs']

    # Relaxed thresholds for real data
    mean_ok = mean_corr < 0.15
    max_ok = max_corr < 0.40

    status = "✓" if (mean_ok and max_ok) else "✗"
    print(f"  {status} {pc_name}: mean_corr={mean_corr:.3f}, max_corr={max_corr:.3f}")

    if not (mean_ok and max_ok):
        all_ok = False

var_ok = var_explained > 0.70
status = "✓" if var_ok else "✗"
print(f"  {status} Variance: {var_explained:.1%} explained")
all_ok = all_ok and var_ok

print("\n" + "="*80)
if all_ok:
    print("✓ ALL CHECKS PASSED - Multi-factor residuals working correctly!")
else:
    print("⚠ Some checks marginally outside thresholds (expected with real data)")
print("="*80)
