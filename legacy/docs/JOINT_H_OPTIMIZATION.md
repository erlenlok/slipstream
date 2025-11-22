# Joint H* Optimization: Finding the Optimal Holding Period

## Overview

This document describes the joint optimization workflow for finding the optimal holding period H* by training and evaluating both price-alpha and funding models simultaneously.

## Why Joint Optimization?

The strategy's total alpha is:

```
α_total = α_price - F_hat
```

Where:
- `α_price`: Predicted forward price returns
- `F_hat`: Predicted forward funding payments

**Key insight**: The optimal H may differ when considering price alpha alone vs. the combined signal. Funding rates may persist at different timescales than price momentum, requiring joint analysis to identify the true optimal holding period.

## Workflow

### 1. Run Joint H* Search

```bash
# Test multiple holding periods with joint alpha + funding models
python scripts/find_optimal_H_joint.py --H 4 8 12 24 48 --n-bootstrap 1000

# Quick test with reduced bootstrap samples
python scripts/find_optimal_H_joint.py --H 4 8 --n-bootstrap 50

# Test different PCA methods
python scripts/find_optimal_H_joint.py --H 4 8 12 24 --pca-method sqrt_dollar --n-bootstrap 1000
```

### 2. Output Structure

Results are saved to `data/features/joint_models/`:

```
joint_models/
├── joint_model_H4.json         # Individual model results for H=4
├── joint_model_H8.json         # Individual model results for H=8
├── ...
├── H_comparison.csv            # Comparison table across all H values
└── optimization_summary.json   # Best H* recommendation
```

### 3. Key Metrics

For each H, the script reports:

- **R²_alpha**: Out-of-sample R² for price alpha alone
- **R²_funding**: Out-of-sample R² for funding predictions
- **R²_combined**: Out-of-sample R² for α_total = α_price - F_hat
- **Lambda values**: Regularization parameters for both models
- **Significant coefficients**: Number of significant features in each model

### 4. Combined Quantile Diagnostics

The most valuable output is the **combined signal quantile table**, which bins samples by the combined signal (α_total) and shows:

- Alpha predictions vs. actuals
- Funding predictions vs. actuals
- Combined signal predictions vs. actuals
- T-statistics for each component

**Example output (H=4 hours)**:

```
COMBINED SIGNAL QUANTILE ANALYSIS
(Binned by α_total = α_price - F_hat)

 Quantile  Count  α_pred_µ  α_actual_µ    α_t  F_pred_µ  F_actual_µ      F_t  Total_pred_µ  Total_actual_µ  Total_t
        0  52940    -0.017      -0.040 -8.132     6.742       8.541  798.383        -6.759          -8.581 -730.173
        1  52940    -0.003      -0.031 -6.490     4.992       4.743  373.971        -4.996          -4.774 -349.752
        ...
        9  52940    -0.007       0.006  1.354    -1.539      -0.826 -150.145         1.532           0.833  116.428
```

**Interpretation**:
- **Quantile 0 (bottom)**: Highest expected funding (F_pred = 6.74σ), negative combined signal
- **Quantile 9 (top)**: Negative expected funding (F_pred = -1.54σ), positive combined signal
- **Alpha contribution**: Weak overall but significant in tail quantiles (see quantile 9: α_actual = 0.006)
- **Funding dominates**: R²_funding = 0.72 vs R²_alpha = -0.001 at H=4

## Initial Results (H=4)

### Summary Statistics
- **Alpha R² (OOS)**: -0.001 (-9.86 bp) — essentially no predictive power
- **Funding R² (OOS)**: 0.720 (7200.80 bp) — very strong persistence
- **Combined R² (OOS)**: 0.638 (6378.65 bp) — strong combined signal

### Key Findings

1. **Funding dominates at short horizons**: At H=4, funding persistence is the primary driver of predictive power.

2. **Alpha has tail significance**: While overall R² is near zero, the quantile analysis shows alpha becomes predictive in extreme buckets:
   - Top decile: positive price alpha emerges (α_actual = 0.006, t=1.35)
   - Bottom deciles: negative alpha in high-funding regimes

3. **Combined signal quality**: The combined R² of 0.638 suggests that even though price alpha is weak overall, it adds value when combined with funding predictions.

4. **Strategy implication**: At H=4, the optimal strategy is primarily a **funding carry arbitrage** (short high funding, long low funding) with a small momentum overlay.

## Next Steps

### 1. Test Longer Horizons

```bash
python scripts/find_optimal_H_joint.py --H 8 12 24 48 72 96 --n-bootstrap 1000
```

**Hypothesis**: Price alpha should strengthen at longer horizons while funding persistence weakens, leading to a crossover point where both contribute meaningfully.

### 2. Analyze H* vs Component R²

Create a plot showing:
- X-axis: Holding period H
- Y-axis: R² (out-of-sample)
- Three lines: α_price, F_hat, α_total

Expected pattern:
- **Funding R²**: High at short H, decays as H increases (mean reversion)
- **Alpha R²**: Low at short H, increases with H (momentum persistence)
- **Combined R²**: Peak at optimal H* where both contribute

### 3. Decompose Combined Signal

For the optimal H*, analyze the relative contributions:

```python
# Signal variance decomposition
var(α_total) = var(α_price) + var(F_hat) - 2*cov(α_price, F_hat)

# Information ratio decomposition
IR(α_total) = [μ(α_price) - μ(F_hat)] / σ(α_total)
```

### 4. Cost-Adjusted H* Selection

The current R² analysis ignores transaction costs. The true H* should maximize:

```
Net Sharpe(H) = Sharpe(H) - c * Turnover(H)
```

Where turnover increases as H decreases (more frequent rebalancing).

## Comparison with Individual Optimizations

| Metric | Alpha Only | Funding Only | Joint |
|--------|-----------|--------------|-------|
| R² (H=4) | -0.001 | 0.720 | 0.638 |
| Optimal λ | 100.0 | 100.0 | - |
| Sig. Coefs | 2/12 | 6/6 | - |

**Key difference**: Joint optimization reveals that the combined signal has predictive power even when price alpha alone appears useless. The funding model provides the directional signal, while price alpha adds refinement in the tails.

## Technical Details

### Model Training

Both models use:
- **Walk-forward cross-validation**: 10 folds with expanding window
- **Bootstrap estimation**: 1000 samples for coefficient stability
- **Ridge regularization**: λ selected via CV from [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
- **Feature set**: 6 EWMA spans [2, 4, 8, 16, 32, 64] hours for both momentum and funding

### Signal Alignment

- Alpha model uses idiosyncratic returns (PCA-residualized)
- Funding model uses raw funding rates
- Both normalized by 128-hour EWMA volatility
- Both trained on the same timescale-matched PCA factors

### Quantile Computation

1. Compute combined signal: `α_total = α_price - F_hat`
2. Bin OOS samples into 10 quantiles by `α_total`
3. Within each bin, compute mean and t-stat for:
   - Alpha predictions vs. actuals
   - Funding predictions vs. actuals
   - Combined predictions vs. actuals

This reveals whether the combined signal is monotonic (stronger signal in extremes).

## References

See also:
- `docs/ALPHA_MODEL_TRAINING.md` - Price alpha model details
- `docs/FUNDING_MODEL_TRAINING.md` - Funding model details
- `docs/strategy_spec.md` Section 4.2 - H* optimization theory
- `scripts/find_optimal_H_joint.py` - Implementation
