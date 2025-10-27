# Quick Start: Timescale-Matched PCA for H* Optimization

## Overview

This guide explains how to generate PCA market factors optimized for different holding periods (H) using the **timescale matching principle**.

### The Circular Dependency Problem

The Slipstream strategy requires finding the optimal holding period H*, but:
1. To find H*, you need to backtest at different H values
2. To backtest at period H, you need PCA beta estimates appropriate for that timescale
3. To choose PCA parameters, you need to know what H you're targeting

### The Solution: Timescale Matching

**Principle**: Match PCA estimation frequency to the candidate holding period.

If you're rebalancing every H hours, you care about beta risk over H-hour periods. Therefore:
- **PCA estimation frequency = H** (use H-hour returns, not daily or hourly)
- **PCA lookback window = K × H** (where K ≈ 20-60 is the number of independent samples)

This eliminates the circular dependency by making PCA parameters a deterministic function of H.

---

## Prerequisites

Ensure data download is complete:

```bash
ls data/market_data/*.csv | wc -l
# Expected: ~417 files (139 assets × 3 file types each)

# Verify volume data exists
head -3 data/market_data/BTC_candles_4h.csv
# Should show: datetime,open,high,low,close,volume
```

---

## Quick Start: Generate PCA Grid

### Option 1: Coarse Grid Search (Recommended)

Test a range of holding periods with matched PCA parameters:

```bash
python scripts/find_optimal_horizon.py --H 6 12 24 48 --K 30
```

**What this does**:
- Tests H ∈ {6, 12, 24, 48} hours
- For each H, uses:
  - PCA frequency = H (e.g., 24H for daily)
  - PCA window = 30 × H (e.g., 720 hours = 30 days for H=24)
  - Default weight method: sqrt volume
- Generates 4 PCA factor files in `data/features/`

**Runtime**: ~20-30 minutes

**Outputs**:
```
data/features/pca_factor_H6_K30_sqrt.csv
data/features/pca_factor_H12_K30_sqrt.csv
data/features/pca_factor_H24_K30_sqrt.csv
data/features/pca_factor_H48_K30_sqrt.csv
```

### Option 2: Test Multiple Weighting Methods

```bash
python scripts/find_optimal_horizon.py \
    --H 24 \
    --K 30 \
    --weight-method sqrt log sqrt_dollar
```

**Outputs** (for H=24):
```
data/features/pca_factor_H24_K30_sqrt.csv
data/features/pca_factor_H24_K30_log.csv
data/features/pca_factor_H24_K30_sqrt_dollar.csv
```

### Option 3: Fine-Tune Lookback Window (K)

Once you've identified a promising H, optimize the lookback multiplier:

```bash
python scripts/find_optimal_horizon.py \
    --H 24 \
    --K 20 30 40 60 \
    --weight-method sqrt
```

Tests different window lengths around H=24:
- K=20 → 480h (20 day) lookback
- K=30 → 720h (30 day) lookback
- K=40 → 960h (40 day) lookback
- K=60 → 1440h (60 day) lookback

---

## Manual Single-Factor Generation

You can also build individual PCA factors directly:

```bash
# 6-hourly PCA with 180-hour (7.5 day) lookback
python scripts/build_pca_factor.py \
    --freq 6H \
    --window 180 \
    --weight-method sqrt \
    --output data/features/pca_factor_H6_K30_sqrt.csv

# Daily PCA with 1440-hour (60 day) lookback
python scripts/build_pca_factor.py \
    --freq D \
    --window 1440 \
    --weight-method sqrt \
    --output data/features/pca_factor_H24_K60_sqrt.csv

# 4-hourly PCA with 480-hour (20 day) lookback
python scripts/build_pca_factor.py \
    --freq 4H \
    --window 480 \
    --weight-method sqrt_dollar \
    --output data/features/pca_factor_H4_K120_sqrt_dollar.csv
```

**Key Parameters**:
- `--freq`: Resampling frequency (`H`, `4H`, `6H`, `12H`, `D`, `W`)
- `--window`: Lookback window in **hours** (not days!)
- `--weight-method`: `none`, `sqrt`, `log`, `dollar`, `sqrt_dollar`

---

## Understanding the Output

Each PCA factor file contains:

```bash
head -5 data/features/pca_factor_H24_K30_sqrt.csv
```

**Format**:
- **Rows**: Timestamps (at frequency H)
- **Columns**: Asset loadings (BTC, ETH, ...) + metadata
  - `BTC`, `ETH`, etc.: PC1 loadings (market factor weights)
  - `_variance_explained`: % of variance captured by PC1
  - `_n_assets`: Number of assets included in PCA that day

**Interpretation**:
- Higher |loading| = asset more correlated with market factor
- Higher variance explained = stronger market regime (assets moving together)
- Loadings sum to ~0 (PCA is zero-mean)

---

## Quick Validation

Check that the PCA factors look reasonable:

```bash
python -c "
import pandas as pd

# Load a generated factor
df = pd.read_csv('data/features/pca_factor_H24_K30_sqrt.csv', index_col=0, parse_dates=True)

print('Shape:', df.shape)
print('Date range:', df.index[0], 'to', df.index[-1])
print()
print('Variance explained:')
print('  Mean:', df['_variance_explained'].mean())
print('  Std:', df['_variance_explained'].std())
print()
print('Assets per period:')
print('  Mean:', df['_n_assets'].mean())
print('  Min:', df['_n_assets'].min())
print('  Max:', df['_n_assets'].max())
print()
print('BTC loading stats:')
print(df['BTC'].describe())
"
```

**Expected Results**:
- Variance explained: 0.3-0.7 (depends on market regime)
- Assets per period: 60-120 (depends on H and universe evolution)
- BTC loadings: Should be non-zero and relatively stable

---

## Choosing H and K Values

### Recommended H Candidates

| Strategy Type | Suggested H Range | Rationale |
|--------------|-------------------|-----------|
| High-frequency | 1-6 hours | Capture intraday alpha before decay |
| Intraday | 6-12 hours | Balance signal quality vs turnover |
| Daily rebalance | 18-30 hours | Standard institutional frequency |
| Swing trading | 48-72 hours | Lower turnover, longer holding |

### Recommended K Multipliers

| K Value | Lookback Periods | Use Case |
|---------|-----------------|----------|
| 20 | 20 independent samples | Fast adaptation to regime changes |
| 30 | 30 samples | **Default - good balance** |
| 40-60 | 40-60 samples | More stable, slower to adapt |

**Rule of thumb**:
- Crypto markets → K=20-30 (more volatile, regimes change faster)
- Equity markets → K=40-60 (more stable, longer regimes)

---

## Next Steps: Finding H*

Once you've generated PCA factors for multiple H values:

### 1. Implement Backtesting Framework
```python
# Pseudocode - to be implemented
for H in [6, 12, 24, 48]:
    pca_factor = load_pca_factor(f"pca_factor_H{H}_K30_sqrt.csv")
    portfolio_returns = backtest_strategy(H=H, beta=pca_factor)
    sharpe_ratios[H] = portfolio_returns.sharpe()
```

### 2. Plot Sharpe vs H
```python
import matplotlib.pyplot as plt

plt.plot(H_values, sharpe_ratios)
plt.xlabel('Holding Period (hours)')
plt.ylabel('Annualized Sharpe Ratio')
plt.title('Optimal Holding Period Search')
plt.show()

H_star = H_values[argmax(sharpe_ratios)]
print(f"Optimal H*: {H_star} hours")
```

### 3. Local Refinement (Optional)
```bash
# If H_star ≈ 24, test finer grid around it
python scripts/find_optimal_horizon.py --H 18 20 22 24 26 28 30 --K 30
```

---

## Troubleshooting

### "No files matching *_candles_4h.csv found"
- Data download incomplete
- Check: `ls data/market_data/*.csv | wc -l` should show ~417 files

### "No volume column in {file}"
- Old data files don't have volume
- Re-run: `uv run hl-load --all --days 730`

### "ValueError: Cannot parse frequency: XYZ"
- Supported frequencies: `H`, `4H`, `6H`, `12H`, `D`, `W`
- Custom frequencies like `3H` or `18H` work too (any integer + `H`)

### Very low variance explained (<0.2)
- Not necessarily a problem - crypto assets can be less correlated
- Compare across H values - relative performance matters more than absolute

### High memory usage
- Large lookback windows (K > 60) on hourly data can use significant RAM
- Consider using daily frequency for long-horizon testing (H ≥ 48)

---

## Advanced: 2D Grid Search

If you want to exhaustively test (H, K) combinations:

```bash
# Full 2D grid: 4 × 4 = 16 configurations
python scripts/find_optimal_horizon.py \
    --H 6 12 24 48 \
    --K 20 30 40 60 \
    --weight-method sqrt
```

This generates 16 PCA factor files. After backtesting all 16, you can create a heatmap:

```python
import pandas as pd
import seaborn as sns

# Collect backtest results
results = []
for H in [6, 12, 24, 48]:
    for K in [20, 30, 40, 60]:
        sharpe = backtest_results[(H, K)]  # Your backtest output
        results.append({'H': H, 'K': K, 'Sharpe': sharpe})

df = pd.DataFrame(results).pivot(index='K', columns='H', values='Sharpe')

sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn')
plt.title('Sharpe Ratio: H vs K')
plt.xlabel('Holding Period (hours)')
plt.ylabel('Lookback Multiplier (K)')
plt.show()
```

This reveals the optimal (H*, K*) pair empirically.

---

## Command Reference

### find_optimal_horizon.py

```bash
# Basic usage
python scripts/find_optimal_horizon.py --H 6 12 24 --K 30

# All options
python scripts/find_optimal_horizon.py \
    --H 4 6 12 24 48 72 \
    --K 20 30 40 60 \
    --data-dir data \
    --weight-method sqrt log sqrt_dollar

# Help
python scripts/find_optimal_horizon.py --help
```

### build_pca_factor.py

```bash
# Basic usage (single factor)
python scripts/build_pca_factor.py --freq 24H --window 720 --weight-method sqrt

# All options
python scripts/build_pca_factor.py \
    --data-dir data \
    --freq 6H \
    --window 180 \
    --min-assets 10 \
    --min-periods 30 \
    --weight-method sqrt \
    --output data/features/my_factor.csv

# Compare all weight methods (at single H)
python scripts/build_pca_factor.py --freq D --window 1440 --compare-all

# Help
python scripts/build_pca_factor.py --help
```

---

## Theory: Why Timescale Matching Works

From `docs/strategy_spec.md` Section 4.3:

> The stability and relevance of the market factor R_m depend on the lookback window used for the PCA calculation. This window should be related to the rebalancing frequency H.

**Intuition**:
1. If you rebalance every 24 hours, fluctuations within those 24 hours average out in your P&L
2. Therefore, you should estimate beta on 24-hour returns, not 1-hour returns
3. Your PCA should answer: "What is the common factor driving 24-hour moves?"
4. Estimating it from 1-hour data would overfit to intraday noise that doesn't affect your strategy

**Mathematical justification**:
- Portfolio variance over H hours scales with covariance of H-hour returns
- Beta estimated from 1-hour returns measures exposure to 1-hour market moves
- Beta estimated from H-hour returns measures exposure to H-hour market moves
- The latter is what you actually care about when holding for H hours

**Result**: Matching PCA frequency to H eliminates a free parameter and makes the optimization search 1-dimensional (just find H*) instead of 3-dimensional (H, frequency, window).
