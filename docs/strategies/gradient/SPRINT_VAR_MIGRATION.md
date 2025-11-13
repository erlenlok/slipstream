# Sprint Plan: VAR-Based Risk Balancing for Gradient

## Executive Summary

**Objective**: Migrate Gradient from dollar-volatility balanced sizing to Value-at-Risk (VAR) balanced sizing on long and short sides.

**Current State**: The strategy allocates equal dollar-volatility to long and short sides, but this does not balance actual **risk** (tail losses). Assets with similar volatility can have very different tail risk profiles, and correlation structures mean the portfolio risk ≠ sum of individual risks.

**Target State**: Each side (long/short) maintains equal one-day Value-at-Risk at a specified confidence level (95%), ensuring symmetric risk exposure in both directions.

**Approach**: Parametric VAR using empirical covariance estimation.

**Timeline**: 2 days

---

## Current Implementation Analysis

### Existing Portfolio Construction Logic

File: `src/slipstream/strategies/gradient/portfolio.py:104-129`

```python
def _allocate_side(assets, vol_row, target_dollar_vol, sign):
    """Current: Allocates equal dollar-vol per asset on each side."""

    n_assets = len(valid_assets)
    per_asset_risk = target_dollar_vol / n_assets  # Equal dollar-vol split

    for asset, vol_value in valid_assets:
        weight = sign * per_asset_risk / max(vol_value, EPSILON)  # w = risk / vol
```

**Problems**:
1. **Ignores correlations**: Treats each asset independently, missing diversification benefits
2. **No signal weighting**: Equal risk allocation regardless of signal strength
3. **Dollar-vol ≠ risk**: Volatility is not the same as tail risk (VAR)

### Example Scenario

Assume we have:
- **Long side**: 5 assets with σ = 2%, normal returns distribution
- **Short side**: 5 assets with σ = 2%, fat-tailed distribution (high kurtosis)

Current approach allocates equal dollar-vol to both sides, but the short side has much higher VAR due to fat tails. In a crisis, losses will be asymmetric.

---

## Technical Design

### 1. Parametric VAR Methodology

We will use **parametric VAR** assuming normal returns (simple, fast, industry-standard):

#### Portfolio VAR Formula
```
VAR_portfolio(α) = z_α * sqrt(w^T Σ w)
```

Where:
- `w` = vector of portfolio weights
- `Σ` = covariance matrix of daily returns
- `z_α` = critical value from standard normal (e.g., 1.645 for 95% confidence)

For a **long** position: VAR = potential loss if returns are negative
For a **short** position: VAR = potential loss if returns are positive (flip sign)

#### Why Parametric?
- **Simple**: Closed-form calculation, no bootstrap needed
- **Fast**: Scales to hundreds of assets
- **Sufficient**: Captures correlations and diversification
- **Better than current**: Current method ignores correlations entirely

#### Covariance Estimation Strategy

**Use debiased RIE (Rotationally-Invariant Estimator)** from Random Matrix Theory:

Given:
- N ≈ 50 assets per side (top/bottom 50 by signal strength)
- T = 60 days of daily returns
- **q = N/T ≈ 0.83** (challenging regime!)

At this q ratio, empirical covariance eigenvalues are heavily corrupted by Marchenko-Pastur noise. RIE is essential.

**Algorithm** (Bun, Bouchaud & Potters 2016):
1. Compute empirical covariance C_emp from daily returns
2. Eigendecompose: C_emp = U Λ U^T
3. Clean eigenvalues: λ_i^clean = debias(λ_i, q) using RMT
4. Reconstruct: Σ = U Λ^clean U^T

**Implementation**: Use `pyRMT` library
```python
from pyRMT import optimalShrinkage
Σ = optimalShrinkage(daily_returns, return_covariance=True, method='rie')
```

**Fallback**: If insufficient data (T < N), use diagonal covariance (independence assumption).

### 2. Side-Level VAR Targeting with Signal-Proportional Weights

#### Signal-Weighted Allocation Algorithm

```python
def _allocate_side_var_targeted(
    assets: list[str],
    signal_strengths: dict[str, float],  # Trend strength for each asset
    daily_returns: pd.DataFrame,  # Wide daily returns
    target_var: float,  # Target one-day VAR for this side
    sign: int,  # +1 for long, -1 for short
) -> dict[str, float]:
    """
    Allocate weights to achieve target VAR, proportional to signal strength.

    Algorithm:
    1. Normalize signals to sum to 1: w_i = |signal_i| / sum(|signals|)
    2. Compute portfolio VAR using covariance matrix
    3. Scale all weights by: target_var / computed_var
    """
    # Step 1: Signal-proportional weights (normalized to sum to 1)
    abs_signals = {a: abs(signal_strengths[a]) for a in assets}
    total_signal = sum(abs_signals.values())
    base_weights = np.array([abs_signals[a] / total_signal for a in assets])

    # Step 2: Compute covariance matrix and portfolio volatility
    cov_matrix = daily_returns[assets].cov()
    portfolio_std = np.sqrt(base_weights.T @ cov_matrix @ base_weights)

    # Step 3: Compute portfolio VAR (parametric)
    z_alpha = 1.645  # 95% confidence (one-tailed)
    computed_var = z_alpha * portfolio_std

    # Step 4: Scale to achieve target VAR
    scale = target_var / computed_var
    final_weights = {a: sign * base_weights[i] * scale for i, a in enumerate(assets)}

    return final_weights
```

**Key properties**:
- Within each side, weight is proportional to `|signal_i|` (stronger signals get more allocation)
- Portfolio VAR accounts for correlations via covariance matrix Σ
- Both sides scaled to achieve `VAR_long = VAR_short = target_var`

**Example**:
```
Long side (top 50 assets):
- Asset 1: signal = 5.2 → base_weight = 5.2 / total_signal
- Asset 2: signal = 4.8 → base_weight = 4.8 / total_signal
- ...
- Asset 50: signal = 1.1 → base_weight = 1.1 / total_signal

Compute Σ_RIE from 60 days of daily returns (debiased via RMT)
Portfolio volatility: sqrt(w^T Σ_RIE w) = 0.0121
VAR_computed = 1.645 * 0.0121 = 0.0199

If target_var = 0.02:
  scale = 0.02 / 0.0199 = 1.005
Final weights: w_i * 1.005 for each asset

Short side: (same process with bottom 50 assets)
```

### 3. Daily Returns Computation

Since Gradient runs on 4h candles, we need to convert to daily returns for VAR calculation:

```python
def compute_daily_returns(returns_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 4h log returns to daily.

    Daily return = sum of 6 consecutive 4h log returns (6 * 4h = 24h)
    """
    return returns_4h.rolling(window=6).sum()
```

**Lookback window**: Use 60 days of daily returns (360 4h bars) for covariance estimation.

### 4. Modified Portfolio Interface

#### New Parameters
```python
def construct_gradient_portfolio(
    trend_strength: pd.DataFrame,  # Used for both selection AND weighting
    log_returns: pd.DataFrame,
    *,
    top_n: int = 50,     # Top 50 assets by signal strength
    bottom_n: int = 50,  # Bottom 50 assets by signal strength

    # NEW VAR PARAMETERS
    risk_method: str = "dollar_vol",  # "dollar_vol" (default) or "var" (new)
    target_side_var: float = 0.02,    # Target 2% one-day VAR per side (95% confidence)
    var_lookback_days: int = 60,      # Days for covariance estimation

    # Legacy parameters (for backward compatibility)
    vol_span: int = 64,
    target_side_dollar_vol: float = 1.0,
) -> pd.DataFrame:
    """
    Build long/short weights with VAR-based risk balancing.

    Changes:
    - trend_strength now used for BOTH asset selection AND weighting within each side
    - When risk_method="var":
      * Allocate proportional to |signal_strength| within each side
      * Use RIE-cleaned covariance matrix (critical for N≈50, T=60 regime)
      * Scale each side to achieve VAR_long = VAR_short = target_side_var
    """
```

**Key changes**:
1. **Selection**: Pick top/bottom N assets by signal strength (existing)
2. **Weighting**: Allocate within each side proportional to |signal_strength| (new)
3. **RIE covariance**: Essential for N=50, T=60 regime where q≈0.83

---

## Implementation Plan

### Phase 1: VAR Calculation Infrastructure (Day 1)

**Goal**: Add parametric VAR calculation with RIE covariance cleaning

#### Tasks
1. **Install pyRMT dependency**
   ```bash
   uv add pyRMT
   ```

2. **Create `src/slipstream/common/risk.py`**
   - `compute_daily_returns()` - Resample 4h → daily (rolling sum of 6 periods)
   - `estimate_covariance_rie()` - RIE-cleaned covariance using pyRMT
   - `compute_portfolio_var()` - Parametric VAR: z_α * sqrt(w^T Σ w)

   **Key function**:
   ```python
   def estimate_covariance_rie(
       daily_returns: pd.DataFrame,
       lookback_days: int = 60,
       fallback_diagonal: bool = True,
   ) -> pd.DataFrame:
       """
       Estimate covariance using debiased RIE.

       Handles T < N case by falling back to diagonal.
       """
   ```

3. **Unit tests**: `tests/test_risk.py`
   - Test VAR on synthetic data (known covariance)
   - Test RIE vs empirical covariance (RIE should have lower out-of-sample error)
   - Test T < N fallback to diagonal
   - Test missing data handling

**Deliverables**:
- Updated `pyproject.toml` with pyRMT dependency
- `src/slipstream/common/risk.py` (~120 lines)
- `tests/test_risk.py` (5-6 tests)

**Success Criteria**:
- All tests pass
- RIE covariance is well-conditioned (positive definite)
- VAR calculation works for N=50, T=60 regime
- Fast (<0.2s for 50x50 covariance)

---

### Phase 2: VAR-Targeted Portfolio Construction (Day 2)

**Goal**: Integrate VAR calculation into portfolio construction

#### Tasks
1. **Extend `src/slipstream/strategies/gradient/portfolio.py`**
   - Add `_allocate_side_var_targeted()` function (equal-weight + scale)
   - Modify `construct_gradient_portfolio()` to support `risk_method` parameter
   - Preserve backward compatibility with `dollar_vol` mode

2. **Integration testing**
   - `tests/test_gradient_portfolio_var.py`:
     - Test VAR targeting with synthetic data
     - Verify long/short sides have equal VAR
     - Test fallback to dollar_vol mode

3. **Quick backtest validation**
   - Run backtest on 3-6 months of data
   - Print VAR stats for long/short sides
   - Verify no obvious bugs

**Deliverables**:
- Updated `portfolio.py` (+80 lines)
- `tests/test_gradient_portfolio_var.py` (3-4 tests)

**Success Criteria**:
- Backtest runs without errors
- Long/short VAR within 20% of each other (not perfect, just balanced)
- Performance metrics (Sharpe) similar to dollar_vol method

---

### Optional: Documentation & CLI Updates

**If time permits**, update user-facing components:

1. **Update `docs/strategies/gradient/README.md`**
   - Add section explaining VAR balancing
   - Document new `risk_method` parameter

2. **Add CLI flag** (if using CLI directly)
   - `--risk-method var` or `--risk-method dollar_vol`
   - `--target-side-var 0.02`

**Not critical for MVP** - can run programmatically via backtest functions first.

---

## Risk Assessment & Mitigation

### Risk 1: Insufficient Historical Data
**Mitigation**: Fallback to diagonal covariance (independence assumption) if <60 days of overlap

### Risk 2: Correlations Break Down in Crisis
**Mitigation**: Use 60-day rolling window (adapts to recent correlation regime)

### Risk 3: VAR Method Underperforms
**Mitigation**: Keep `risk_method` configurable, maintain backward compatibility with `dollar_vol`

---

## Testing Strategy

### Unit Tests
- `tests/test_risk.py`: VAR calculation with synthetic data
- `tests/test_gradient_portfolio_var.py`: Portfolio construction + VAR targeting

### Integration Tests
- Run backtest with `risk_method="var"` on historical data
- Verify long/short sides have balanced VAR

### Regression Test
- Ensure `risk_method="dollar_vol"` produces identical results to current code

---

## Configuration Schema

### New Parameters for `construct_gradient_portfolio()`

```python
risk_method: str = "dollar_vol"  # "dollar_vol" (current) or "var" (new)
target_side_var: float = 0.02    # Target 2% daily VAR per side (95% confidence)
var_lookback_days: int = 60      # Days of history for covariance
```

**Backward compatible**: Defaults to `dollar_vol` mode

---

## Success Metrics

1. **VAR Balance**: Long/short sides have VAR within 20% of each other
2. **Performance**: Sharpe ratio similar to or better than dollar-vol method
3. **Backward Compatibility**: `risk_method="dollar_vol"` works unchanged
4. **Tests Pass**: All unit and integration tests pass

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Phase 1: VAR Infrastructure | Day 1 | `risk.py` module with parametric VAR |
| Phase 2: Portfolio Construction | Day 2 | VAR-targeted allocation + tests |
| **Total** | **2 days** | Working VAR-based portfolio construction |

---

## Mathematical Details

### Parametric VAR Formula
```
VAR_portfolio(α) = z_α * sqrt(w^T Σ w)
```

Where:
- `w` = portfolio weights (signal-proportional within each side)
- `Σ` = covariance matrix (60-day rolling)
- `z_α` = 1.645 for 95% confidence (one-tailed)

### Covariance Estimation with RIE
```python
from pyRMT import optimalShrinkage

# RIE-cleaned covariance (debiases eigenvalues via Random Matrix Theory)
recent_returns = daily_returns.iloc[-60:]  # Last 60 days

if len(recent_returns) >= len(assets):  # T >= N
    Σ = optimalShrinkage(recent_returns, return_covariance=True, method='rie')
else:  # Fallback to diagonal if T < N
    Σ = np.diag(recent_returns.var())
```

**Why RIE is critical**:
- With N≈50 assets and T=60 days, q = N/T ≈ 0.83
- Empirical eigenvalues are heavily distorted by Marchenko-Pastur noise
- RIE debiases eigenvalues and shrinks spurious correlations
- Resulting covariance is more stable and better conditioned

### Allocation Algorithm
1. Start with signal-proportional weights: `w_i = |signal_i| / sum(|signals|)`
2. Compute portfolio VAR: `VAR = 1.645 * sqrt(w^T Σ w)`
3. Scale to target: `w_final = w * (target_var / VAR)`

This ensures:
- **Within each side**: allocation proportional to signal strength
- **Between sides**: VAR_long = VAR_short = target_var

---

## References

**RIE Covariance Estimation**:
- Paper: "Cleaning large correlation matrices: tools from Random Matrix Theory" (2016)
- Authors: Joël Bun, Jean-Philippe Bouchaud, Marc Potters
- arXiv: 1610.08104
- Implementation: `pyRMT` library (Gregory Giecold)

**Key insight**: At q = N/T ≈ 0.83, empirical covariance eigenvalues are heavily corrupted by noise. RIE uses Random Matrix Theory to debias eigenvalues and produce optimal shrinkage estimators.

---

## Next Steps

Ready to implement with RIE-cleaned covariance from the start. Begin with:

1. **Install pyRMT**: `uv add pyRMT`
2. **Create `src/slipstream/common/risk.py`** with RIE covariance estimation
3. **Unit tests** to validate VAR calculation with N=50, T=60
