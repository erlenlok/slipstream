"""Analyze PC2 and PC3 to understand what they represent."""
import pandas as pd
import numpy as np

# Load 3-component PCA
pca = pd.read_csv('data/features/pca_factor_H24_K30_sqrt_3pc.csv', index_col=0)
pca.index = pd.to_datetime(pca.index)

print("=" * 80)
print("MULTI-COMPONENT PCA ANALYSIS")
print("=" * 80)

# Extract variance explained
var_pc1 = pca['_variance_explained_pc1'].dropna().mean()
var_pc2 = pca['_variance_explained_pc2'].dropna().mean()
var_pc3 = pca['_variance_explained_pc3'].dropna().mean()

print(f"\n📊 Variance Explained:")
print(f"  PC1: {var_pc1:.1%}  (systematic market factor)")
print(f"  PC2: {var_pc2:.1%}  (second systematic factor)")
print(f"  PC3: {var_pc3:.1%}  (third systematic factor)")
print(f"  Total: {(var_pc1 + var_pc2 + var_pc3):.1%}")
print(f"  Residual: {(1 - var_pc1 - var_pc2 - var_pc3):.1%}  (idiosyncratic)")

# Get asset list from columns
all_cols = [col for col in pca.columns if not col.startswith('_')]
assets = sorted(list(set([col.split('_pc')[0] for col in all_cols if '_pc' in col])))

print(f"\n📦 Data Summary:")
print(f"  Assets: {len(assets)}")
print(f"  Timestamps: {len(pca)}")
print(f"  Date range: {pca.index.min()} to {pca.index.max()}")

# Extract loadings for the most recent timestamp with valid data
recent_idx = pca['_n_assets'].last_valid_index()

print(f"\n🔍 Analyzing loadings at: {recent_idx}")
print(f"  (Using most recent timestamp with valid PCA)")

# Extract loadings for each component
loadings_pc1 = {}
loadings_pc2 = {}
loadings_pc3 = {}

for asset in assets:
    pc1_col = f'{asset}_pc1'
    pc2_col = f'{asset}_pc2'
    pc3_col = f'{asset}_pc3'

    if pc1_col in pca.columns:
        loadings_pc1[asset] = pca.loc[recent_idx, pc1_col]
    if pc2_col in pca.columns:
        loadings_pc2[asset] = pca.loc[recent_idx, pc2_col]
    if pc3_col in pca.columns:
        loadings_pc3[asset] = pca.loc[recent_idx, pc3_col]

# Convert to Series and drop NaN
loadings_pc1 = pd.Series(loadings_pc1).dropna()
loadings_pc2 = pd.Series(loadings_pc2).dropna()
loadings_pc3 = pd.Series(loadings_pc3).dropna()

print(f"\n  Valid loadings: {len(loadings_pc1)} assets")

# Analyze PC1 (sanity check - should be positive and similar magnitude)
print(f"\n\n{'='*80}")
print(f"PC1 ANALYSIS (Market Factor - {var_pc1:.1%} variance)")
print(f"{'='*80}")
print(f"  Mean loading: {loadings_pc1.mean():.4f}")
print(f"  Std loading:  {loadings_pc1.std():.4f}")
print(f"  Min/Max: [{loadings_pc1.min():.4f}, {loadings_pc1.max():.4f}]")

print(f"\nTop 10 assets by PC1 loading:")
for i, (asset, loading) in enumerate(loadings_pc1.nlargest(10).items(), 1):
    print(f"  {i:2d}. {asset:10s} {loading:7.4f}")

print(f"\nBottom 10 assets by PC1 loading:")
for i, (asset, loading) in enumerate(loadings_pc1.nsmallest(10).items(), 1):
    print(f"  {i:2d}. {asset:10s} {loading:7.4f}")

# Analyze PC2
print(f"\n\n{'='*80}")
print(f"PC2 ANALYSIS (Second Factor - {var_pc2:.1%} variance)")
print(f"{'='*80}")
print(f"  Mean loading: {loadings_pc2.mean():.6f}  (should be ~0)")
print(f"  Std loading:  {loadings_pc2.std():.4f}")
print(f"  Min/Max: [{loadings_pc2.min():.4f}, {loadings_pc2.max():.4f}]")

print(f"\n🔼 Top 10 POSITIVE PC2 loadings:")
for i, (asset, loading) in enumerate(loadings_pc2.nlargest(10).items(), 1):
    print(f"  {i:2d}. {asset:10s} {loading:+7.4f}")

print(f"\n🔽 Top 10 NEGATIVE PC2 loadings:")
for i, (asset, loading) in enumerate(loadings_pc2.nsmallest(10).items(), 1):
    print(f"  {i:2d}. {asset:10s} {loading:+7.4f}")

print(f"\n💡 Interpretation hints:")
print(f"   - Assets with similar PC2 sign move together on this factor")
print(f"   - Look for patterns: BTC/ETH vs altcoins? Large-cap vs small-cap?")
print(f"   - DeFi vs infrastructure? Memes vs fundamentals?")

# Analyze PC3
print(f"\n\n{'='*80}")
print(f"PC3 ANALYSIS (Third Factor - {var_pc3:.1%} variance)")
print(f"{'='*80}")
print(f"  Mean loading: {loadings_pc3.mean():.6f}  (should be ~0)")
print(f"  Std loading:  {loadings_pc3.std():.4f}")
print(f"  Min/Max: [{loadings_pc3.min():.4f}, {loadings_pc3.max():.4f}]")

print(f"\n🔼 Top 10 POSITIVE PC3 loadings:")
for i, (asset, loading) in enumerate(loadings_pc3.nlargest(10).items(), 1):
    print(f"  {i:2d}. {asset:10s} {loading:+7.4f}")

print(f"\n🔽 Top 10 NEGATIVE PC3 loadings:")
for i, (asset, loading) in enumerate(loadings_pc3.nsmallest(10).items(), 1):
    print(f"  {i:2d}. {asset:10s} {loading:+7.4f}")

# Correlations between components
print(f"\n\n{'='*80}")
print(f"COMPONENT CORRELATIONS (Orthogonality Check)")
print(f"{'='*80}")
corr_12 = loadings_pc1.corr(loadings_pc2)
corr_13 = loadings_pc1.corr(loadings_pc3)
corr_23 = loadings_pc2.corr(loadings_pc3)

print(f"  Corr(PC1, PC2): {corr_12:+.6f}  (should be ~0)")
print(f"  Corr(PC1, PC3): {corr_13:+.6f}  (should be ~0)")
print(f"  Corr(PC2, PC3): {corr_23:+.6f}  (should be ~0)")

if abs(corr_12) < 0.01 and abs(corr_13) < 0.01 and abs(corr_23) < 0.01:
    print(f"\n  ✓ Components are orthogonal!")
else:
    print(f"\n  ⚠ Some correlation detected (this is normal in practice)")

# Special asset analysis
print(f"\n\n{'='*80}")
print(f"KEY ASSETS LOADINGS")
print(f"{'='*80}")

key_assets = ['BTC', 'ETH', 'SOL', 'DOGE', 'PEPE', 'WIF', 'ARB', 'OP', 'LINK']
available_keys = [a for a in key_assets if a in loadings_pc1.index]

print(f"\n{'Asset':<10} {'PC1':>8} {'PC2':>8} {'PC3':>8}")
print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8}")
for asset in available_keys:
    pc1 = loadings_pc1[asset]
    pc2 = loadings_pc2[asset]
    pc3 = loadings_pc3[asset]
    print(f"{asset:<10} {pc1:8.4f} {pc2:+8.4f} {pc3:+8.4f}")

print(f"\n{'='*80}")
print(f"✓ Analysis complete!")
print(f"{'='*80}")
