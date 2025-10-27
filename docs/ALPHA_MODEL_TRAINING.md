# Alpha Model Training: Bootstrap Methodology

## Overview

This document specifies the alpha model training framework for Slipstream, adapting the bootstrap methodology from Schmidhuber (2021) "Trends, reversion, and critical phenomena in financial markets" to our multi-span momentum signals.

**Core Philosophy**: Train trend strengths separately many times on random subsets of the data to get a distribution of fitted parameter values. Select the mean as our best estimate.

**Reference Paper**: `docs/Trends, reversion, and critical phenomena in financial markets.pdf`

---

## 1. Model Specification

### 1.1 The Alpha Model

For asset `i` at time `t`, we predict the H-period forward **vol-normalized** idiosyncratic return:

```
alpha_norm_i(t → t+H) = Σ β_s · momentum_i,s(t) + Σ γ_s · funding_i,s(t) + ε_i(t)
                         s ∈ {2, 4, 8, 16, 32, 64}       s ∈ {2, 4, 8, 16, 32, 64}
```

**Components**:
- `alpha_norm_i`: Vol-normalized H-period forward return (target variable)
- `momentum_i,s(t)`: EWMA-based idiosyncratic momentum at span s (price feature)
- `funding_i,s(t)`: EWMA-based funding rate at span s (funding feature)
- `β_s`: Coefficients for momentum spans (to be estimated via Ridge regression + bootstrap)
- `γ_s`: Coefficients for funding spans (to be estimated via Ridge regression + bootstrap)
- `ε_i(t)`: Random noise / unexplained variance
- `H`: Holding period in hours (e.g., 24 for daily rebalancing)

**Total Parameters**: 12 (6 momentum + 6 funding)

**Key Differences from Schmidhuber**:
- **Funding rates included**: Critical for crypto perps where funding can be 10-50% annualized
- **L2 regularization**: Ridge regression on every bootstrap iteration (not just parameter counting)
- **More features allowed**: We have higher-resolution data (hourly vs daily), more assets, and more independent series after PC filtering
- **Linear only** (for now): Can add polynomial terms later if residual analysis suggests non-linearity

---

## 2. Target Variable: Vol-Normalized Returns ✓

### 2.1 The Choice

We predict **vol-normalized** returns and scale back at prediction time.

**Formula**:
```python
# Training target:
y_i(t) = forward_return_i(t→t+H) / vol_i(t)

# At prediction time:
alpha_i(t) = model.predict(X_i(t)) * vol_i(t)
```

### 2.2 Why Vol-Normalized? (Arguments FOR)

1. **Stationarity**: Vol-normalized returns are much more stationary than raw returns
   - Crypto volatility is highly time-varying
   - Heteroskedasticity is severe without normalization

2. **Consistency**: Features are vol-normalized idiosyncratic momentum → target should match same scale
   - All signals in `idiosyncratic_momentum()` are already vol-normalized
   - Feature-target scale alignment improves regression stability

3. **Schmidhuber precedent**: Paper uses normalized returns (Eq. 1) as both features AND targets throughout
   - `R_i(t) = r_i(t) / σ_i` (volatility-normalized)
   - Entire analysis on normalized space

4. **Kelly framework alignment**: Maximizing log-wealth cares about Sharpe (mean/vol), not absolute returns
   - Vol-normalized prediction directly estimates Sharpe contribution
   - Portfolio optimization uses Sharpe-like objectives

5. **Cross-sectional pooling**: With vol-normalization, all assets contribute equally to parameter estimation
   - Without it, high-vol assets (e.g., DOGE, PEPE) dominate the regression
   - BTC/ETH would have minimal impact despite being more liquid

6. **Adaptive predictions**: Time-varying volatility handled naturally
   ```python
   # Model learns stable relationship in normalized space
   alpha_norm = model.predict(momentum_features)

   # Scale by current volatility at prediction time
   alpha_dollar = alpha_norm * current_vol_t
   ```
   - Predictions automatically adjust to current market regime
   - More robust to volatility regime changes

### 2.3 Why NOT Raw Returns? (Arguments AGAINST raw targets)

1. **Non-stationarity**: Raw returns have time-varying variance → violates OLS assumptions
2. **Asset imbalance**: High-vol coins get 100x more weight in loss function
3. **Overfitting to volatile periods**: Model would optimize for rare high-vol events
4. **Poor generalization**: Coefficients estimated during low-vol periods won't work in high-vol periods

### 2.4 Implementation Details

**Volatility Estimator**:
```python
# For target normalization (at time t, looking backward):
vol_i(t) = EWMA_volatility(idio_returns_i, span=128)

# Same span as used in signal generation (max(momentum_spans) * 2)
```

**Forward Return Calculation**:
```python
# H-period forward return (overlapping for more data):
forward_return_i(t) = sum(idio_returns_i[t+1 : t+H+1])

# Vol-normalized target:
y_i(t) = forward_return_i(t) / vol_i(t)
```

**At Prediction Time**:
```python
# Current volatility estimate:
vol_current = EWMA_volatility(recent_returns, span=128)

# Scale prediction:
alpha_total = alpha_normalized * vol_current
```

---

## 3. Funding Rate Features

### 3.1 Why Include Funding Rates?

**Crypto-Specific Feature**: Perpetual futures on crypto exchanges have funding rates that can be highly predictive of future returns.

**Economic Intuition**:
1. **Mean reversion**: Extreme funding rates (high positive or negative) tend to revert
2. **Carry signal**: Negative funding = get paid to hold long → bullish signal
3. **Crowding indicator**: High positive funding = longs crowded → potential reversal
4. **Predictive power**: Funding can be 10-50% annualized → material impact on returns

**Strategy Specification** (Section 3.1):
- Funding rate prediction is a key component of α^total = α^price - F̂
- EWMA of recent funding rates is a simple but effective predictor of future funding

### 3.2 Funding Rate Normalization

**Problem**: How to make funding rates comparable across assets and time?

**Funding Rate Characteristics**:
- Already in percentage units (e.g., 0.01% = 1 basis point per 8 hours)
- Can be positive or negative
- Magnitude varies by asset (BTC funding ≈ ±0.05%, meme coins ≈ ±0.5%)
- Time-varying (regime changes)

**Normalization Options**:

**Option A: Standardize by Historical Volatility** (RECOMMENDED)
```python
# For each asset i at time t:
funding_norm_i,s(t) = EWMA_s(funding_i) / std(funding_i)

# Where std(funding_i) is measured over a long lookback (e.g., 90 days)
```
**Pros**:
- Comparable across assets (BTC and DOGE on same scale)
- Accounts for regime changes
- Consistent with our vol-normalized returns

**Cons**:
- Need enough history to estimate std(funding)
- Outliers can distort std

**Option B: Raw Funding Rates** (ALTERNATIVE)
```python
# Use raw EWMA of funding rates
funding_i,s(t) = EWMA_s(funding_i)
```
**Pros**:
- Simpler
- Interpretable (1% funding = 1% carry)
- No additional estimation error

**Cons**:
- High-funding assets (meme coins) dominate regression
- Not comparable across assets
- Magnitude drift over time

**Option C: Cross-Sectional Z-Score**
```python
# At each time t, z-score across all assets:
funding_z_i(t) = (funding_i(t) - mean_t(funding)) / std_t(funding)
```
**Pros**:
- Relative funding (vs. market) is more predictive than absolute
- Automatically normalized

**Cons**:
- Loses information about absolute funding level
- Requires sufficient cross-section at each time

### 3.3 Recommended Approach

**Hybrid**: Vol-normalize funding rates by asset-specific historical std, similar to returns

```python
def compute_funding_features(
    funding_rates: pd.DataFrame,  # Wide format, hourly funding rates
    spans: list = [2, 4, 8, 16, 32, 64],
    std_lookback: int = 90 * 24  # 90 days
) -> pd.DataFrame:
    """
    Compute EWMA funding features, normalized by historical volatility.

    Returns:
        DataFrame with MultiIndex (timestamp, asset) and columns=funding_2, funding_4, ...
    """
    # 1. Compute funding volatility for each asset
    funding_std = funding_rates.rolling(std_lookback, min_periods=30*24).std()

    # 2. Vol-normalize funding rates
    funding_norm = funding_rates / funding_std

    # 3. Compute EWMA at multiple spans
    funding_features = {}
    for span in spans:
        ewma = funding_norm.ewm(span=span, min_periods=span//2).mean()
        funding_features[f'funding_{span}'] = ewma.stack()

    # 4. Combine into panel
    return pd.DataFrame(funding_features)
```

**Why This Works**:
- Consistent with vol-normalized returns (target and features on same scale)
- High-funding assets (meme coins) don't dominate low-funding assets (BTC/ETH)
- Accounts for time-varying funding volatility
- L2 regularization will handle any remaining scale issues

### 3.4 Implementation Details

**Data Requirements**:
- Hourly funding rate data (already available in `data/market_data/`)
- At least 90 days of history for std estimation (warm-up period)

**Feature Engineering**:
```python
# Parallel to momentum features:
X: pd.DataFrame
    Index: MultiIndex(timestamp, asset)
    Columns: [
        'mom_2', 'mom_4', 'mom_8', 'mom_16', 'mom_32', 'mom_64',      # Momentum
        'funding_2', 'funding_4', 'funding_8', 'funding_16', 'funding_32', 'funding_64'  # Funding
    ]
```

**Expected Signs**:
- Momentum coefficients β_s: Likely **positive** (trend persistence)
- Funding coefficients γ_s: Likely **negative** (mean reversion)
  - High positive funding (longs pay shorts) → future returns negative (reversion)
  - High negative funding (shorts pay longs) → future returns positive (carry)

---

## 4. L2 Regularization (Ridge Regression)

### 4.1 Why Regularization?

**Problem**: With 12 features (6 momentum + 6 funding), OLS can overfit, especially with correlated features

**Schmidhuber's Approach**: Strict parameter budget (max 6 params for 28 years)

**Our Approach**: L2 regularization + bootstrap
- **More flexible**: Can include all relevant features
- **Better statistical properties**: Shrinks coefficients toward zero (reduces variance)
- **Handles multicollinearity**: Momentum spans are correlated, funding spans are correlated
- **Cross-validated λ**: Optimal regularization strength chosen via CV

### 4.2 Ridge Regression Formulation

**Objective Function**:
```
min_β  ||y - Xβ||² + λ ||β||²
        \_________/   \______/
        OLS loss      L2 penalty
```

Where:
- `λ > 0`: Regularization strength (hyperparameter)
- `||β||²`: Sum of squared coefficients (penalizes large coefficients)

**Effect**:
- `λ = 0`: Regular OLS (no shrinkage)
- `λ → ∞`: All coefficients → 0 (full shrinkage)
- `λ optimal`: Bias-variance trade-off optimized via cross-validation

### 4.3 Choosing λ via Cross-Validation

**Process**:
1. Try a grid of λ values: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
2. For each λ, run K-fold CV and compute out-of-sample R²
3. Select λ* that maximizes R²_oos
4. Retrain on full data with λ*

**Important**: λ selection happens BEFORE bootstrap
- Find optimal λ via CV once
- Use that λ for all bootstrap iterations
- This avoids overfitting λ to any particular bootstrap sample

### 4.4 Integration with Bootstrap

**Updated Workflow**:
```python
1. Choose optimal λ via cross-validation (once)
2. For each bootstrap iteration:
   a. Sample timestamps with replacement
   b. Fit Ridge(alpha=λ) on bootstrap sample
   c. Store coefficients
3. Compute mean/std of coefficients across bootstrap samples
```

**Code Update**:
```python
from sklearn.linear_model import Ridge, RidgeCV

# Step 1: Find optimal λ via cross-validation
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=10)
ridge_cv.fit(X, y)
lambda_optimal = ridge_cv.alpha_

print(f"Optimal λ: {lambda_optimal}")

# Step 2: Bootstrap with optimal λ
def bootstrap_train_alpha_model(X, y, lambda_reg, n_bootstrap=1000):
    beta_samples = []

    for i in range(n_bootstrap):
        # Sample timestamps
        sampled_times = sample_with_replacement(timestamps)
        X_boot, y_boot = X.loc[sampled_times], y.loc[sampled_times]

        # Fit Ridge regression with fixed λ
        model = Ridge(alpha=lambda_reg)
        model.fit(X_boot, y_boot)

        beta_samples.append(model.coef_)

    return np.array(beta_samples)
```

---

## 5. Bootstrap Methodology (Adapted for Crypto)

### 5.1 Key Innovation: Random Subset Training

**Problem**: Financial returns are non-IID, correlated, non-normal → standard errors unreliable

**Solution**: Bootstrap resampling to empirically measure coefficient distributions

**Process**:
1. From N hours of data, create B bootstrap samples (sampling timestamps with replacement)
2. Fit Ridge regression on each bootstrap sample → B coefficient estimates
3. Take mean as final estimate
4. Use 16th/84th percentiles for standard errors (= ±1σ for normal distribution)

### 5.2 How Many Bootstrap Samples?

**Schmidhuber Used**: 5000 samples for 28 years of daily data

**Our Data**:
- ~1.5 years of hourly data ≈ 13,000 hours
- ~100 assets (vs. Schmidhuber's 24)
- After PC filtering: ~10-20 independent factors

**Rule of Thumb**: Bootstrap until coefficient distribution stabilizes

**Test for Stability**:
```python
# Compute coefficient std as function of bootstrap samples
stds_vs_n = []
for n in [100, 200, 500, 1000, 2000, 5000]:
    beta_samples_n = beta_samples[:n]
    std_n = beta_samples_n.std(axis=0).mean()
    stds_vs_n.append(std_n)

# If std plateaus by n=1000 → 1000 samples sufficient
# If still decreasing at n=5000 → need more
```

**Recommendation**: Start with 1000-2000 bootstrap samples
- Faster iteration during development
- Increase to 5000+ for final model if std not stable
- Computational cost: ~1-2 seconds per bootstrap (Ridge fit) → 1000 samples ≈ 20-30 minutes

### 5.3 Cross-Validation for Out-of-Sample R²

**Problem**: In-sample R² overestimates predictive power (overfitting bias)

**Solution**: K-fold time-series cross-validation (expanding window)

**Our Implementation**:
- 10-15 folds (vs. Schmidhuber's 15 for 28 years)
- Expanding window: train on all data BEFORE validation window
- For 1.5 years of data: each fold ≈ 1-2 months

**Key Difference from Schmidhuber**:
- We use Ridge regression in CV (not OLS)
- λ is chosen to maximize R²_oos across all folds
- This prevents overfitting even with 12 features

**Expected Correction**:
```
Schmidhuber: R² - R²_oos ≈ 0.93 bp (for 2 params on 28 years)

Crypto (our case): R² - R²_oos ≈ 1-3 bp (for 12 params on 1.5 years)
- Higher correction due to less data
- But Ridge regularization reduces it
- Bootstrap standard errors account for this
```

---

## 6. Implementation Plan

### 6.1 Data Pipeline

**Input**:
- Idiosyncratic returns (wide DataFrame, already computed)
- Funding rates (wide DataFrame, hourly)
- PCA loadings and market factor (if using single-factor residuals)
- OR: Multi-factor residuals (if using 3-PC approach)

**Output**:
```python
# Features: momentum + funding panel
X: pd.DataFrame
    Index: MultiIndex(timestamp, asset)
    Columns: [
        'mom_2', 'mom_4', 'mom_8', 'mom_16', 'mom_32', 'mom_64',
        'funding_2', 'funding_4', 'funding_8', 'funding_16', 'funding_32', 'funding_64'
    ]

# Target: vol-normalized H-period forward returns
y: pd.Series
    Index: MultiIndex(timestamp, asset)
    Values: forward_return / volatility

# Metadata: volatility estimates (for scaling predictions)
vol: pd.Series
    Index: MultiIndex(timestamp, asset)
    Values: EWMA volatility

# Regularization: optimal λ from cross-validation
lambda_opt: float
```

### 6.2 Bootstrap Training with Ridge Regression

**Pseudocode**:
```python
from sklearn.linear_model import Ridge

def bootstrap_train_alpha_model(
    X: pd.DataFrame,  # Features (12 columns: 6 momentum + 6 funding)
    y: pd.Series,     # Target (vol-normalized forward returns)
    lambda_reg: float,  # Regularization strength (from CV)
    n_bootstrap: int = 1000,
    random_seed: int = 42
) -> dict:
    """
    Train alpha model using bootstrap sampling with Ridge regression.

    Key changes from Schmidhuber:
    - Uses Ridge(alpha=lambda_reg) instead of OLS
    - 1000-2000 bootstrap samples (vs. 5000)
    - 12 features (momentum + funding) vs. 2

    Returns:
        {
            'beta_mean': Mean coefficients (12,),
            'beta_std': Std dev of coefficients,
            'beta_distribution': Full distribution (n_bootstrap x 12),
            'r2_insample': Mean in-sample R² across bootstraps,
            't_statistics': t-stats for each coefficient,
            'lambda': Regularization strength used
        }
    """
    np.random.seed(random_seed)

    # Get unique timestamps
    timestamps = X.index.get_level_values('timestamp').unique()
    n_times = len(timestamps)

    # Storage for bootstrap results
    beta_samples = np.zeros((n_bootstrap, X.shape[1]))
    r2_samples = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Sample timestamps with replacement
        sampled_times = np.random.choice(timestamps, size=n_times, replace=True)

        # Get all assets at sampled timestamps
        mask = X.index.get_level_values('timestamp').isin(sampled_times)
        X_boot = X[mask]
        y_boot = y[mask]

        # Fit Ridge regression with fixed λ
        model = Ridge(alpha=lambda_reg)
        model.fit(X_boot, y_boot)

        # Store coefficients and R²
        beta_samples[i] = model.coef_
        r2_samples[i] = model.score(X_boot, y_boot)

    # Compute statistics
    beta_mean = beta_samples.mean(axis=0)
    beta_std = beta_samples.std(axis=0)

    # t-statistics (16th-84th percentile for robustness)
    beta_p16 = np.percentile(beta_samples, 16, axis=0)
    beta_p84 = np.percentile(beta_samples, 84, axis=0)
    beta_se = (beta_p84 - beta_p16) / 2  # Robust std error
    t_stats = beta_mean / beta_se

    return {
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_se': beta_se,
        'beta_distribution': beta_samples,
        'r2_insample': r2_samples.mean(),
        't_statistics': t_stats,
        'lambda': lambda_reg
    }
```

### 6.3 Walk-Forward Cross-Validation with Ridge

**Pseudocode**:
```python
from sklearn.linear_model import Ridge, RidgeCV

def find_optimal_lambda_and_validate(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    n_splits: int = 10,
    min_train_size: int = 180 * 24  # 180 days in hours
) -> dict:
    """
    Find optimal λ via cross-validation, then compute out-of-sample R².

    Two-stage process:
    1. RidgeCV to find optimal λ (10-fold CV on full data)
    2. Walk-forward CV to measure true out-of-sample performance

    Returns:
        {
            'lambda_opt': Optimal regularization strength,
            'r2_oos': Out-of-sample R² from walk-forward CV,
            'predictions_oos': All OOS predictions,
            'actuals_oos': All OOS actuals,
            'fold_r2': R² for each fold
        }
    """
    # Stage 1: Find optimal λ
    print("Stage 1: Finding optimal λ via RidgeCV...")
    ridge_cv = RidgeCV(alphas=alphas, cv=10)
    ridge_cv.fit(X, y)
    lambda_opt = ridge_cv.alpha_
    print(f"Optimal λ = {lambda_opt}")

    # Stage 2: Walk-forward validation with optimal λ
    print("Stage 2: Walk-forward validation...")
    timestamps = X.index.get_level_values('timestamp').unique().sort_values()
    n_times = len(timestamps)
    fold_size = n_times // n_splits

    predictions_oos = []
    actuals_oos = []
    fold_r2 = []

    for i in range(n_splits):
        # Validation window
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_splits - 1 else n_times
        val_times = timestamps[val_start:val_end]

        # Training window (all data before validation)
        if val_start < min_train_size:
            continue  # Skip if not enough training data
        train_times = timestamps[:val_start]

        # Split data
        train_mask = X.index.get_level_values('timestamp').isin(train_times)
        val_mask = X.index.get_level_values('timestamp').isin(val_times)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        # Fit Ridge with optimal λ
        model = Ridge(alpha=lambda_opt)
        model.fit(X_train, y_train)

        # Predict on validation data
        y_pred = model.predict(X_val)

        # Store results
        predictions_oos.append(y_pred)
        actuals_oos.append(y_val.values)

        # Calculate fold R²
        ss_res = np.sum((y_val.values - y_pred) ** 2)
        ss_tot = np.sum((y_val.values - y_val.mean()) ** 2)
        r2_fold = 1 - (ss_res / ss_tot)
        fold_r2.append(r2_fold)
        print(f"  Fold {i+1}: R² = {r2_fold:.4f}")

    # Concatenate all OOS predictions
    all_predictions = np.concatenate(predictions_oos)
    all_actuals = np.concatenate(actuals_oos)

    # Overall OOS R²
    ss_res = np.sum((all_actuals - all_predictions) ** 2)
    ss_tot = np.sum((all_actuals - all_actuals.mean()) ** 2)
    r2_oos = 1 - (ss_res / ss_tot)

    print(f"\nOut-of-sample R² = {r2_oos:.4f}")

    return {
        'lambda_opt': lambda_opt,
        'r2_oos': r2_oos,
        'predictions_oos': all_predictions,
        'actuals_oos': all_actuals,
        'fold_r2': np.array(fold_r2),
        'mean_fold_r2': np.mean(fold_r2)
    }
```

### 6.4 Complete Training Pipeline

**Full workflow** (λ selection → bootstrap → validation):

```python
def train_alpha_model_complete(
    X: pd.DataFrame,  # 12 features (mom + funding)
    y: pd.Series,     # Vol-normalized forward returns
    n_bootstrap: int = 1000
) -> dict:
    """
    Complete alpha model training pipeline.

    Steps:
    1. Find optimal λ via cross-validation
    2. Bootstrap training with optimal λ
    3. Walk-forward validation for out-of-sample R²
    4. Return coefficient distributions + diagnostics

    Returns:
        Complete model specification + diagnostics
    """
    # Step 1: Find optimal λ and validate
    cv_results = find_optimal_lambda_and_validate(X, y)
    lambda_opt = cv_results['lambda_opt']
    r2_oos = cv_results['r2_oos']

    # Step 2: Bootstrap training with optimal λ
    print(f"\nBootstrap training with λ={lambda_opt}, n={n_bootstrap}...")
    bootstrap_results = bootstrap_train_alpha_model(
        X, y, lambda_reg=lambda_opt, n_bootstrap=n_bootstrap
    )

    # Step 3: Compile results
    r2_in = bootstrap_results['r2_insample']
    correction = r2_in - r2_oos

    print(f"\n{'='*70}")
    print(f"FINAL MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"Regularization (λ):    {lambda_opt:.4f}")
    print(f"R² (in-sample):        {r2_in:.4f}")
    print(f"R² (out-of-sample):    {r2_oos:.4f}")
    print(f"Correction:            {correction:.4f} ({100*correction/r2_in:.1f}% of R²)")
    print(f"Bootstrap samples:     {n_bootstrap}")
    print(f"\nCoefficients:")
    print(f"{'Feature':<15} {'Beta':>10} {'Std Err':>10} {'t-stat':>8}")
    print(f"{'-'*50}")

    for i, col in enumerate(X.columns):
        beta = bootstrap_results['beta_mean'][i]
        se = bootstrap_results['beta_se'][i]
        t = bootstrap_results['t_statistics'][i]
        sig = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.64 else ''
        print(f"{col:<15} {beta:>10.4f} {se:>10.4f} {t:>8.2f} {sig}")

    print(f"\n*** p<0.01, ** p<0.05, * p<0.10")
    print(f"{'='*70}")

    return {
        'coefficients': bootstrap_results['beta_mean'],
        'std_errors': bootstrap_results['beta_se'],
        't_statistics': bootstrap_results['t_statistics'],
        'distribution': bootstrap_results['beta_distribution'],
        'lambda': lambda_opt,
        'r2_insample': r2_in,
        'r2_oos': r2_oos,
        'correction': correction,
        'feature_names': X.columns.tolist(),
        'cv_results': cv_results
    }
```

---

## 7. Key Decisions & Recommendations

### 7.1 Holding Period (H) ✓

**Recommendation**: Start with H=24 (daily rebalancing)

**Rationale**:
- Matches PCA timescale if using `pca_factor_H24_K30_sqrt.csv`
- Daily rebalancing is practical for crypto (24/7 markets)
- Can extend to H* optimization later (test H ∈ {6, 12, 24, 48})

### 7.2 Bootstrap Sample Size ✓

**Recommendation**: Start with 1000-2000 samples, test for stability

**Rationale**:
- We have MORE data than Schmidhuber (hourly vs daily, more assets)
- Computational cost: 1000 samples ≈ 20-30 minutes
- Test stability: if coefficient std plateaus at n=1000 → sufficient
- Can increase to 5000 if needed for final model

**Stability Test**: Run with n ∈ {100, 500, 1000, 2000, 5000}, plot coefficient std vs. n

### 7.3 Model Form ✓

**Recommendation**: Linear only (12 params: 6 momentum + 6 funding)

**Rationale**:
- Start simple, L2 regularization handles overfitting
- Our signals are pre-processed (vol-normalized, idiosyncratic)
- Schmidhuber's cubic terms were for RAW returns (trend reversion)
- Can add polynomial terms later if residual analysis suggests non-linearity

### 7.4 Volatility Estimation ✓

**Recommendation**: vol_span = 128 hours (max(spans) * 2)

**Rationale**:
- Consistent with Schmidhuber's approach
- Same as used in signal generation
- Long enough to be stable, short enough to adapt

### 7.5 PCA Components ✓

**Recommendation**: Start with 1-PC residuals, test 3-PC later

**Rationale**:
- 1-PC: Simpler, more data, existing implementation
- Can A/B test: train two models (1-PC vs 3-PC), compare R²_oos
- 3-PC may improve if PC2/PC3 have predictive power for returns

### 7.6 Overlapping Returns ✓

**Recommendation**: Use overlapping returns (every hour)

**Rationale**:
- More data (13,000 hours vs ~500 non-overlapping samples)
- Bootstrap handles autocorrelation properly
- Ridge regularization reduces overfitting
- Schmidhuber used overlapping successfully

### 7.7 Funding Rate Normalization ✓

**Recommendation**: Vol-normalize by asset-specific historical std

**Rationale**:
- Consistent with vol-normalized returns (target scale)
- Prevents meme coins from dominating regression
- Accounts for time-varying funding volatility
- L2 regularization handles any remaining scale issues

### 7.8 Tail Diagnostics & Guardrails ✓

**What changed**:

- **Warm-up enforcement**: every (timestamp, asset) observation must accumulate at least one full volatility span (default 128 h) before entering the training set. This removes the extreme spikes that occurred immediately after new listings when EWMA volatility was unreliable.
- **Clipping**:
  - Momentum panels are capped at ±2.5 (Schmidhuber convention).
  - Funding EWMAs are clipped at ±5 to suppress outlier prints in thin markets.
  - Vol-normalised forward returns are winsorised to ±10 to keep a single point from dominating loss/metrics.
- **Robust PCA metadata**: rolling PCA now stores `_pc1_eigenvalue` and `_condition_number`. Windows with eigenvalues below 1e-8 or near-zero loading norms are skipped, ensuring the beta vector stays well-conditioned.
- **Quantile analysis**: `train_alpha_model_complete` now prints (and serialises) a 10-decile table with counts, mean predictions, mean realised returns, and t-stats. This is the primary “go/no-go” check before trading a new horizon \(H\).

**Current H = 4 h snapshot**:

- Deciles 1–7 (most negative predictions) realise statistically significant negative returns (t-stats between −5.7 and −2.0).
- Decile 9 (most positive predictions) realises +2.5 bp per 4 h with t ≈ 4.8, even though aggregate \(R^2_{\text{oos}}\) is slightly negative (~−1 bp). This confirms momentum lives in the tails.
- Deciles 0 and 8 are neutral → production sizing should concentrate on the extreme buckets.

---

## 8. Expected Outcomes

### 8.1 Statistical Targets

**With L2 regularization and 12 features** (vs. Schmidhuber's 2-parameter OLS):

**Optimistic**:
- R² (in-sample): 10-20 bp (0.10%-0.20%)
  - Higher than Schmidhuber due to funding rate features
- R²_oos: 6-12 bp
  - Ridge regularization reduces overfitting
- t-statistics: |t| > 2.0 for at least 6-8 features
- Correction: 30-50% of R²

**Realistic**:
- R² (in-sample): 5-10 bp
- R²_oos: 2-5 bp
- t-statistics: |t| > 2.0 for 4-6 features
- Correction: 50-70% of R²

**Red flags**:
- R²_oos < 0: Model has no predictive power
- Correction > 90% of R²: Severe overfitting (even with Ridge)
- No significant coefficients (all |t| < 2): Signals don't predict returns
- λ → 0: Ridge not helping (features not correlated)
- λ → ∞: All coefficients shrunk to zero (no signal)

### 8.2 Economic Interpretation

**If successful**, coefficient estimates might look like:

```
Feature        Beta     Std Err  t-stat  Interpretation
─────────────────────────────────────────────────────────
mom_2        +0.15%    0.08%    1.9     Short-term momentum (weak)
mom_4        +0.45%    0.12%    3.8 *** Medium-fast momentum (strong)
mom_8        +0.60%    0.15%    4.0 *** Medium momentum (strongest)
mom_16       +0.40%    0.12%    3.3 *** Medium-slow momentum (strong)
mom_32       +0.15%    0.09%    1.7     Long-term momentum (weak)
mom_64       -0.05%    0.07%    0.7     Very long-term (reversion, n.s.)

funding_2    -0.10%    0.06%    1.7     Short-term funding reversion
funding_4    -0.25%    0.09%    2.8 **  Medium-fast funding reversion
funding_8    -0.35%    0.11%    3.2 *** Medium funding reversion (strongest)
funding_16   -0.30%    0.10%    3.0 **  Medium-slow funding reversion
funding_32   -0.15%    0.08%    1.9     Long-term funding reversion
funding_64   -0.05%    0.06%    0.8     Very long-term funding (n.s.)
```

**Interpretation**:
- **Momentum**: Positive, peak at 8-16 hours (trend persistence)
- **Funding**: Negative, peak at 8-16 hours (mean reversion)
- **Combined effect**: Momentum drives direction, funding provides entry/exit timing
- **λ optimal**: Likely 0.1-10 (modest shrinkage)

### 8.3 Why Higher R² Than Schmidhuber?

**Our advantages**:
1. **Funding rates**: Crypto-specific feature with strong predictive power
2. **Higher frequency**: Hourly data captures more dynamics than daily
3. **More independent series**: PC filtering removes systematic risk
4. **Better regularization**: Ridge vs. hard parameter limit

**Our disadvantages**:
1. **Less history**: 1.5 years vs. 28 years
2. **More noise**: Crypto is noisier than traditional futures
3. **Regime changes**: Crypto markets evolve faster

**Net effect**: R²_oos of 5-10 bp is realistic (vs. Schmidhuber's 4 bp)

---

## 9. Next Steps

### 9.1 Immediate (This Week)
1. ✓ Document alpha model specification (this doc)
2. ✓ Finalize key decisions (Section 7)
3. Implement data preparation pipeline:
   - Load idiosyncratic returns from existing signals
   - Load funding rates from `data/market_data/`
   - Compute vol-normalized funding features
   - Create feature matrix X (12 columns)
   - Create target vector y (H-period forward returns)
4. Implement Ridge + bootstrap training (Section 6.2)
5. Implement λ selection + walk-forward CV (Section 6.3)
6. Test on synthetic data first (known coefficients)

### 9.2 After Initial Implementation
1. Run on real crypto data (1.5 years)
2. Analyze coefficient distributions:
   - Plot bootstrap distributions (histograms)
   - Check for bimodality (sign instability)
   - Verify t-statistics > 2 for key features
3. Validate out-of-sample R²:
   - Should be positive and > 2bp
   - Correction < 70% of R²
   - Stable across CV folds
4. Residual analysis:
   - Plot residuals vs. fitted values
   - Check for non-linearity (polynomial terms needed?)
   - Check for heteroskedasticity
5. Compare 1-PC vs. 3-PC residuals:
   - Train both models
   - Compare R²_oos
   - Check if funding rate coefficients change

### 9.3 Extensions
1. **H* optimization**: Train separate models for H ∈ {6, 12, 24, 48}
   - Backtest each with transaction costs
   - Find H* that maximizes net Sharpe ratio
2. **Polynomial terms**: Add cubic terms if residuals show non-linearity
   - `mom_s³` and `funding_s³`
   - Increases to 24 parameters (still manageable with Ridge)
3. **Interaction terms**: Test `mom_s * funding_s` cross-terms
   - Captures non-linear relationship between momentum and funding
4. **Adaptive λ**: Retrain λ on rolling windows
   - Check if optimal λ changes over time
   - Could indicate regime changes

### 9.4 Production Integration
1. Integrate with portfolio optimizer:
   - α^total = α_model * current_vol
   - Feed into mean-variance optimization
2. Implement retraining pipeline:
   - Retrain monthly or quarterly
   - Monitor coefficient stability
   - Detect regime changes
3. Live monitoring:
   - Track realized Sharpe vs. predicted
   - Coefficient drift detection
   - R²_oos degradation alerts
4. Funding rate prediction (future):
   - Separate model for F̂(t → t+H)
   - Currently using EWMA as placeholder

---

## References

- Schmidhuber, C. (2021). "Trends, reversion, and critical phenomena in financial markets." Physica A: Statistical Mechanics and its Applications, 566, 125642.
- `docs/strategy_spec.md` - Slipstream strategy specification
- `src/slipstream/signals/idiosyncratic_momentum.py` - Current signal implementation
