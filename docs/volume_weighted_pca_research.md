# Volume-Weighted PCA Research

## Executive Summary

This document explores methodologies for incorporating volume data into PCA-based market factor construction for the Slipstream trading framework. Volume weighting addresses the economic intuition that high-volume assets carry more market information and should have greater influence on the market factor.

## Motivation

**Why Volume-Weight PCA?**

1. **Economic Significance**: Higher volume assets represent more active trading and price discovery
2. **Liquidity Proxy**: Volume indicates tradability and market depth
3. **Noise Reduction**: Low-volume assets may have noisier price movements that shouldn't dominate the market factor
4. **Market Impact Alignment**: Portfolio construction will face less slippage on high-volume assets

**Current Limitation**: Equal-weighted PCA treats a low-volume altcoin the same as BTC, potentially giving too much weight to illiquid, noisy assets.

## Methodological Approaches

### Approach 1: Pre-Weighted Returns (Return Scaling)

**Method**: Scale each asset's returns by its volume before applying PCA.

**Implementation**:
```python
# For each asset i at time t:
weighted_return[i,t] = return[i,t] * weight[i,t]

# Where weight[i,t] is derived from volume[i,t]
```

**Weighting Schemes**:
- **Linear**: `weight = volume / mean(volume)`
- **Square Root**: `weight = sqrt(volume / mean(volume))` - dampens extreme volume spikes
- **Rank-Based**: `weight = rank(volume) / n_assets` - robust to outliers
- **Dollar Volume**: `weight = (price * volume) / mean(price * volume)` - captures economic value

**Advantages**:
- Conceptually simple
- Can use standard sklearn PCA
- Directly interpretable: "returns weighted by trading activity"

**Disadvantages**:
- Changes the meaning of the return variable
- May over-emphasize volume spikes (mitigated by sqrt or rank transforms)
- Loadings are harder to interpret

**Normalization Considerations**:
- Standardize within each rolling window
- Use trailing average volume to avoid forward-looking bias
- Handle zero/missing volume gracefully

### Approach 2: Weighted Covariance Matrix

**Method**: Compute a volume-weighted covariance matrix, then apply eigen-decomposition.

**Implementation**:
```python
# Construct diagonal weight matrix W from volumes
W = diag(w1, w2, ..., wn)

# Weighted covariance:
Σ_weighted = (X^T W X) / sum(weights)

# Eigen-decomposition:
eigenvalues, eigenvectors = eig(Σ_weighted)
```

**Advantages**:
- Theoretically clean - weights observations by economic importance
- Loadings remain in return space (more interpretable)
- Can incorporate cross-sectional volume differences

**Disadvantages**:
- More complex implementation (can't use sklearn PCA directly)
- Requires custom eigen-decomposition
- Need to handle weight normalization carefully

### Approach 3: Iteratively Reweighted PCA (Robust PCA)

**Method**: Use volume to down-weight outliers in an iterative refinement process.

**Implementation**:
1. Compute initial PCA
2. Calculate reconstruction errors
3. Down-weight low-volume + high-error observations
4. Re-fit PCA
5. Iterate until convergence

**Advantages**:
- Robust to outliers
- Combines volume with statistical properties
- Well-studied in robust statistics literature

**Disadvantages**:
- Computationally expensive (iteration required)
- May be overkill for our use case
- Harder to tune (multiple hyperparameters)

### Approach 4: Two-Stage: Filter by Volume, Then Equal-Weight PCA

**Method**: Pre-filter asset universe by minimum volume threshold, then apply standard PCA.

**Implementation**:
```python
# Within each rolling window:
volume_threshold = percentile(volume, 25)  # Keep top 75% by volume
filtered_assets = assets[volume > volume_threshold]
pca.fit(returns[filtered_assets])
```

**Advantages**:
- Simplest to implement
- Removes clearly illiquid assets
- Standard PCA interpretation preserved

**Disadvantages**:
- Binary (in/out) rather than continuous weighting
- Loses information from excluded assets
- Threshold selection is arbitrary

## Recommended Implementation Strategy

### Phase 1: Implement Approach 1 (Pre-Weighted Returns) with Multiple Weighting Schemes

**Rationale**:
- Straightforward to implement
- Easy to compare variants
- Compatible with existing rolling PCA infrastructure

**Weighting Functions to Test**:
1. **None** (baseline): Equal-weighted PCA
2. **Sqrt Volume**: `w[i,t] = sqrt(volume[i,t])`
3. **Log Volume**: `w[i,t] = log(1 + volume[i,t])`
4. **Dollar Volume**: `w[i,t] = price[i,t] * volume[i,t]`
5. **Sqrt Dollar Volume**: `w[i,t] = sqrt(price[i,t] * volume[i,t])`

**Volume Aggregation**:
- Use **trailing window average** (e.g., same 60-day window as PCA)
- Normalize weights to mean=1 within each window
- Handle missing volume with asset-specific median

### Phase 2: Evaluation Framework

**Metrics for Comparison**:

1. **Factor Quality**:
   - Variance explained by PC1
   - Stability of loadings over time (turnover)
   - Correlation with broad market index (if available)

2. **Economic Relevance**:
   - Do high-volume assets get higher loadings? (expected)
   - Is factor return smooth or erratic?
   - Does it capture market-wide moves vs idiosyncratic noise?

3. **Strategy Impact** (downstream):
   - Beta neutrality quality (how well does factor hedge?)
   - Sharpe ratio of beta-neutral portfolios
   - Turnover and transaction costs

**Output Format**:
Each PCA variant should produce:
- `pca_factor_loadings_{method}.csv` - Daily loadings per asset
- Summary statistics: variance explained, asset counts, loading stability
- Visualization: loadings over time, volume vs loading scatter

### Phase 3: Production Selection

After backtesting with the beta-neutral strategy:
- Select the volume-weighting method that maximizes net-of-cost Sharpe ratio
- Document the choice and rationale
- Implement as default in production pipeline

## Implementation Plan for build_pca_factor.py

### New Functions to Add

```python
def load_volume_data(data_dir: Path, pattern: str = "*_candles_1h.csv") -> pd.DataFrame:
    """Load volume data from candle files and construct wide matrix."""
    # Load all candle files
    # Extract volume column
    # Align timestamps with returns
    # Return: (hours × assets) DataFrame of volumes

def compute_volume_weights(
    volumes: pd.DataFrame,
    method: str = "sqrt",
    window_days: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling volume weights for PCA.

    Args:
        volumes: Raw volume matrix (hours × assets)
        method: Weighting scheme - "none", "sqrt", "log", "dollar", "sqrt_dollar"
        window_days: Rolling window for computing average volume

    Returns:
        Weight matrix (hours × assets), normalized to mean=1 within each window
    """

def compute_rolling_pca_loadings_weighted(
    returns_daily: pd.DataFrame,
    volumes_daily: pd.DataFrame,
    weight_method: str = "sqrt",
    window_days: int = 60,
    min_assets: int = 10,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling PCA with volume weighting.

    Extends compute_rolling_pca_loadings() to incorporate volume weights.
    """
```

### CLI Updates

Add arguments to control weighting:
```python
parser.add_argument(
    "--weight-method",
    type=str,
    default="none",
    choices=["none", "sqrt", "log", "dollar", "sqrt_dollar"],
    help="Volume weighting method (default: none for equal-weight)",
)

parser.add_argument(
    "--compare-all",
    action="store_true",
    help="Compute PCA for all weighting methods and save separately",
)
```

## Technical Considerations

### Volume Data Alignment
- Candle files have OHLCV at hourly frequency
- Returns are computed from candles
- Volume must be aggregated to daily (sum over 24 hours)
- Handle missing volume: use forward-fill or asset median

### Numerical Stability
- Normalize weights to prevent numerical issues
- Use log-transform for very large volume values
- Check for degenerate cases (all zero volume in window)

### Bias Prevention
- Use **trailing** volume windows only (no forward-looking)
- Volume weights computed on same window as returns for PCA
- Don't let a single volume spike dominate the entire factor

### Edge Cases
- New assets with limited volume history: use unweighted or exclude
- Delisted assets: handle gracefully in rolling window
- Volume = 0: assign minimum weight (e.g., 1% of median)

## Research Questions to Answer

1. **Does volume weighting improve PC1 variance explained?**
2. **Do volume-weighted loadings have lower turnover (more stable)?**
3. **Which weighting function (linear, sqrt, log) performs best?**
4. **Is dollar volume (price × volume) better than raw volume?**
5. **Does the optimal weighting vary with the PCA window length?**
6. **Can we predict which method works best based on market regime?**

## Next Steps

1. ✅ Ensure volume data is available in candle files
2. ⏳ Implement `load_volume_data()` function
3. ⏳ Implement `compute_volume_weights()` with multiple schemes
4. ⏳ Extend `compute_rolling_pca_loadings()` to accept weights
5. ⏳ Generate comparison outputs for all methods
6. ⏳ Analyze results and document findings
7. ⏳ Integrate best method into strategy backtesting pipeline

---

**Document Version**: 1.0
**Author**: Claude Code
**Date**: 2025-10-22
