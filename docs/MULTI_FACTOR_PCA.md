# Multi-Factor PCA: Using PC1, PC2, PC3

## Motivation

Currently, Slipstream uses a **single-factor model** with PC1 as the market factor. This leaves ~18% of systematic variance unexplained, which could introduce:
- Unintended factor exposures (sector tilts, size effects)
- Higher portfolio volatility than necessary
- Correlation with non-market risk factors

Using **multi-factor PCA** (PC1 + PC2 + PC3) would create a "flatter" portfolio by hedging additional systematic risks.

---

## Theoretical Framework

### Single-Factor Model (Current)
```
R_i = α_i + β_i,1 * R_m,1 + ε_i

Portfolio constraint: β₁ᵀw = 0
```

**Variance decomposition:**
- PC1: ~81% (systematic market risk)
- Residual: ~19% (idiosyncratic + PC2/PC3/...)

### Multi-Factor Model (Proposed)
```
R_i = α_i + β_i,1*R_m,1 + β_i,2*R_m,2 + β_i,3*R_m,3 + ε_i

Portfolio constraints:
- β₁ᵀw = 0  (neutral to PC1)
- β₂ᵀw = 0  (neutral to PC2)
- β₃ᵀw = 0  (neutral to PC3)
```

**Variance decomposition:**
- PC1: ~81%
- PC2: ~10-12% (e.g., sector/size factor)
- PC3: ~4-6% (e.g., style/momentum factor)
- Residual: ~2-4% (truly idiosyncratic)

---

## Tradeoffs

### Pros (Why You Want This)
1. **Lower Portfolio Volatility**
   - Hedge out PC2/PC3 → reduce systematic risk
   - If alpha is uncorrelated with PC2/PC3 → higher Sharpe ratio

2. **Cleaner Alpha Signal**
   - Residuals are more truly idiosyncratic
   - Less contamination from sector/style tilts

3. **More Robust Hedging**
   - Single-factor model assumes PC1 captures all systematic risk (not true!)
   - Multi-factor captures 95%+ of systematic variance

### Cons (Why You Might Not)
1. **Fewer Degrees of Freedom**
   - 3 constraints vs 1 → less flexibility in optimization
   - Harder to find large positions while staying neutral

2. **Potential Alpha Loss**
   - If your momentum signal is correlated with PC2/PC3, you'll hedge it out
   - Example: If PC2 represents "altcoin vs BTC/ETH", and your alpha picks altcoins, you lose alpha

3. **Estimation Risk**
   - PC2/PC3 loadings are noisier (less variance explained)
   - Rolling window estimates may be unstable

---

## Implementation Steps

### Step 1: Modify PCA Script to Compute Multiple Components

**File:** `scripts/build_pca_factor.py`

```python
# Current (line 279):
pca = PCA(n_components=1)

# Modified:
pca = PCA(n_components=3)  # Compute PC1, PC2, PC3

# Extract all 3 components
pc1_loadings = pca.components_[0]
pc2_loadings = pca.components_[1]
pc3_loadings = pca.components_[2]

# Store in separate columns (wide format)
# Asset columns: BTC, ETH, ... (PC1 loadings)
# _BTC_pc2, _ETH_pc2, ... (PC2 loadings)
# _BTC_pc3, _ETH_pc3, ... (PC3 loadings)
```

**Output format:** `pca_factor_H24_K30_sqrt_3pc.csv`
- Columns for each asset: 3 values (PC1, PC2, PC3 loadings)
- Metadata: `_variance_explained_pc1`, `_variance_explained_pc2`, `_variance_explained_pc3`

### Step 2: Update Signal Computation

**File:** `src/slipstream/signals/idiosyncratic_momentum.py`

```python
def compute_idiosyncratic_returns_multifactor(
    returns: pd.DataFrame,
    pca_loadings_pc1: pd.Series,
    pca_loadings_pc2: pd.Series,
    pca_loadings_pc3: pd.Series,
    market_factor_pc1: pd.Series,
    market_factor_pc2: pd.Series,
    market_factor_pc3: pd.Series,
) -> pd.DataFrame:
    """Remove PC1, PC2, PC3 exposures to get truly idiosyncratic returns."""

    # Multi-factor model residuals
    idio = returns - (
        beta1 * R_m1 +
        beta2 * R_m2 +
        beta3 * R_m3
    )

    return idio
```

### Step 3: Update Portfolio Optimization

**File:** `src/slipstream/portfolio/optimizer.py` (future)

```python
# Cost-aware optimization with 3 beta constraints
objective = w.T @ alpha - 0.5 * w.T @ S @ w - C(Δw)

constraints = [
    beta1.T @ w == 0,  # PC1 neutral
    beta2.T @ w == 0,  # PC2 neutral
    beta3.T @ w == 0,  # PC3 neutral
]
```

---

## Research Validation Steps

Before fully committing, run these experiments:

### Experiment 1: Variance Decomposition
Compute PC1, PC2, PC3 and track:
- Variance explained by each
- Cumulative variance explained
- Stability of loadings over time

**Expected:** PC1=81%, PC2=10-12%, PC3=4-6%, Total=95%+

### Experiment 2: Alpha Correlation
Correlate your momentum signal with PC2/PC3 returns:
```python
corr(idio_momentum, R_m2)
corr(idio_momentum, R_m3)
```

**Decision rule:**
- If |corr| < 0.2: Safe to hedge PC2/PC3 (won't lose alpha)
- If |corr| > 0.4: Warning - you might be hedging your alpha source!

### Experiment 3: Backtest Comparison
Run parallel backtests:
1. **Single-factor** (PC1 only) - Current implementation
2. **Multi-factor** (PC1 + PC2 + PC3) - New implementation

Compare:
- Sharpe ratio
- Portfolio volatility
- Turnover (might increase with more constraints)
- Max drawdown

### Experiment 4: Factor Interpretation
Inspect PC2 and PC3 loadings to understand what they represent:
```python
# Which assets load heavily on PC2?
pc2_top = loadings_pc2.nlargest(10)
pc2_bottom = loadings_pc2.nsmallest(10)

# Hypothesis: PC2 might separate BTC/ETH vs altcoins
# Hypothesis: PC3 might separate DeFi vs infrastructure tokens
```

---

## Recommended Next Steps

1. **Quick Win:** Modify `build_pca_factor.py` to save variance_explained for PC1, PC2, PC3
2. **Diagnostic:** Plot cumulative variance explained over time
3. **Research:** Compute PC2/PC3 and inspect their interpretation
4. **Test:** Run alpha correlation analysis (Experiment 2)
5. **Decision:** If alpha is uncorrelated with PC2/PC3 → implement multi-factor
6. **Backtest:** Validate that multi-factor improves risk-adjusted returns

---

## CLI Interface (Proposed)

```bash
# Generate 3-component PCA factors
python scripts/build_pca_factor.py --freq 24H --window 720 --n-components 3

# Generate comparison (1 vs 3 factors)
python scripts/compare_factor_models.py --H 24 --K 30 --components 1 3

# Output:
# - pca_factor_H24_K30_sqrt_1pc.csv (current)
# - pca_factor_H24_K30_sqrt_3pc.csv (multi-factor)
```

---

## References

- Fama-French 3-factor model (equities): Market + Size + Value
- PCA factor models typically use 3-5 components for 95% variance coverage
- Crypto-specific: PC2 often represents "beta vs altcoins" or sector rotation
