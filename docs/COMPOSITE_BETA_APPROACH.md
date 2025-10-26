# Composite Beta Approach: One Constraint for Multiple Factors

## The Problem

Multi-factor model requires multiple constraints:
```
R_i = α_i + β_i,1*F₁ + β_i,2*F₂ + β_i,3*F₃ + ε_i

Optimizer constraints:
- β₁ᵀw = 0
- β₂ᵀw = 0
- β₃ᵀw = 0
```

This is complex and reduces degrees of freedom significantly.

## The Insight: Pre-Orthogonalization

**Key idea:** Remove all systematic components BEFORE building signals, not AFTER optimizing.

### Step 1: Compute Multi-Factor Residuals

```python
# For each asset i at time t:
R_systematic_i(t) = (
    β_i,1 * R_m,1(t) +  # PC1 exposure
    β_i,2 * R_m,2(t) +  # PC2 exposure
    β_i,3 * R_m,3(t)    # PC3 exposure
)

# Idiosyncratic returns
ε_i(t) = R_i(t) - R_systematic_i(t)
```

### Step 2: Build Signals from Pure Idiosyncratic Returns

```python
# Your momentum signals are now based on ε_i
# These have ZERO correlation with PC1, PC2, PC3 by construction
momentum_i = EWMA(ε_i)
```

### Step 3: The Subtle Part - Portfolio Constraints

**Question:** If signals are already orthogonal to factors, why do we need constraints?

**Answer:** Because you're forming positions in actual assets, which still have factor loadings!

Example:
- Your signal says: "BTC idiosyncratic returns trending up"
- You buy BTC at weight w_BTC = 0.05
- But BTC has loadings: [β_BTC,1=0.45, β_BTC,2=0.12, β_BTC,3=0.08]
- Your portfolio just took on systematic exposure!

## The Elegant Solution: Effective Beta

Define a **composite beta** that captures total factor exposure:

### Option A: L2 Norm (Euclidean Distance)

```python
# Total factor exposure magnitude
β̃_i = sqrt(β_i,1² + β_i,2² + β_i,3²)
```

**Interpretation:** How "far" is asset i from being purely idiosyncratic?

**Constraint:**
```
β̃ᵀw = 0
```

**Problem:** This constrains the MAGNITUDE of exposure, not the DIRECTION. You could still be long PC1 and short PC2.

### Option B: Variance-Weighted Composite (RECOMMENDED)

Weight each factor by its variance contribution:

```python
# Variance explained by each PC
var₁ = 0.81  # PC1 explains 81% (from our data)
var₂ = 0.12  # PC2 explains ~12% (estimated)
var₃ = 0.05  # PC3 explains ~5% (estimated)

# Composite beta = variance-weighted sum
β̃_i = var₁*β_i,1 + var₂*β_i,2 + var₃*β_i,3
```

**Interpretation:** Assets with high PC1 loading dominate (since PC1 explains most variance)

**Constraint:**
```
β̃ᵀw = 0
```

**Benefit:** Single constraint, but you're implicitly more neutral to PC1 (the biggest risk) than PC3 (small risk)

### Option C: Just Constrain PC1 (Pragmatic)

Since PC1 explains 81% of variance:

```python
# Build signals from multi-factor residuals
ε_i = R_i - (β_i,1*R_m,1 + β_i,2*R_m,2 + β_i,3*R_m,3)

# But only constrain PC1 in optimizer
Constraint: β₁ᵀw = 0
```

**Rationale:**
- Signals already orthogonal to PC1, PC2, PC3 (by construction)
- Portfolio formed from these signals will naturally be ~neutral to all factors
- β₁ᵀw = 0 is a "safety check" for the biggest risk (81% variance)
- PC2/PC3 exposures will be small in practice

## Mathematical Justification for Option C

If your alpha signals α̃ satisfy:
```
Cov(α̃, R_m,1) = 0
Cov(α̃, R_m,2) = 0
Cov(α̃, R_m,3) = 0
```

Then the optimal unconstrained portfolio w* will have:
```
E[wᵀβ₂] ≈ 0
E[wᵀβ₃] ≈ 0
```

naturally, because you're not trying to time PC2 or PC3!

You only need β₁ᵀw = 0 as a hard constraint because:
1. PC1 is the biggest risk (81% variance)
2. Numerical/estimation errors could create drift
3. Transaction costs might cause you to "ride" a PC1 move temporarily

## Implementation

### Current (Single-Factor)
```python
# Compute idiosyncratic returns (PC1 only)
idio = compute_idiosyncratic_returns(
    returns,
    pca_loadings_pc1,
    market_factor_pc1
)

# Build signals
momentum = EWMA(idio)

# Optimize with constraint
constraint: beta1.T @ w == 0
```

### Proposed (Multi-Factor Residuals, Single Constraint)
```python
# Compute multi-factor idiosyncratic returns
idio = compute_multifactor_residuals(
    returns,
    loadings_pc1, loadings_pc2, loadings_pc3,
    factor_pc1, factor_pc2, factor_pc3
)

# Build signals (already orthogonal to all 3 PCs!)
momentum = EWMA(idio)

# Optimize with single constraint (safety check on biggest risk)
constraint: beta1.T @ w == 0
```

**Result:**
- You get the benefits of multi-factor hedging (cleaner signals)
- Without the complexity of multiple constraints
- One constraint preserves optimization degrees of freedom

## Code Changes Required

### 1. Modify Signal Computation
**File:** `src/slipstream/signals/idiosyncratic_momentum.py`

```python
def compute_multifactor_residuals(
    returns: pd.DataFrame,
    loadings_pc1: pd.Series,
    loadings_pc2: pd.Series,
    loadings_pc3: pd.Series,
    factor_pc1: pd.Series,
    factor_pc2: pd.Series,
    factor_pc3: pd.Series,
) -> pd.DataFrame:
    """Remove PC1, PC2, PC3 exposures to get pure idiosyncratic returns.

    This pre-orthogonalizes returns so that signals built from them are
    automatically neutral to all three principal components.
    """
    # Convert to long format
    returns_long = returns.stack()

    # Align all components
    # ... (similar to current compute_idiosyncratic_returns)

    # Remove all three systematic components
    idio = (
        returns_aligned -
        (loadings_pc1_aligned * factor_pc1_aligned) -
        (loadings_pc2_aligned * factor_pc2_aligned) -
        (loadings_pc3_aligned * factor_pc3_aligned)
    )

    return idio.unstack()
```

### 2. Update PCA Factor Generation
**File:** `scripts/build_pca_factor.py`

```python
# Change from n_components=1 to n_components=3
pca = PCA(n_components=3)

# Save all three components in output CSV
# Columns: BTC_pc1, ETH_pc1, ..., BTC_pc2, ETH_pc2, ..., BTC_pc3, ETH_pc3, ...
```

## Validation Test

To verify this works, check:

```python
# After computing multi-factor residuals
residuals = compute_multifactor_residuals(...)

# These should all be ~0
print(f"Corr(residuals, R_m,1): {residuals.corrwith(factor_pc1).mean():.4f}")
print(f"Corr(residuals, R_m,2): {residuals.corrwith(factor_pc2).mean():.4f}")
print(f"Corr(residuals, R_m,3): {residuals.corrwith(factor_pc3).mean():.4f}")

# Expected output:
# Corr(residuals, R_m,1): 0.0001  (basically zero)
# Corr(residuals, R_m,2): 0.0002
# Corr(residuals, R_m,3): 0.0001
```

## Summary

**The key insight:** You can get multi-factor neutrality with a single constraint by:

1. **Pre-orthogonalizing:** Remove PC1+PC2+PC3 from returns before building signals
2. **Natural neutrality:** Signals built from orthogonalized returns automatically avoid factor timing
3. **Safety constraint:** Keep β₁ᵀw = 0 to prevent drift on the biggest risk factor
4. **Simpler optimization:** One constraint instead of three

This is mathematically cleaner and computationally simpler than explicit multi-factor constraints.
