# Timescale Matching: Solving the H* Circular Dependency

## The Problem

The Slipstream beta-neutral strategy (see `strategy_spec.md`) faces a fundamental circular dependency:

1. **To find optimal H***: Need to backtest the strategy at different holding periods
2. **To backtest at period H**: Need accurate beta estimates for hedging market risk
3. **To estimate beta**: Need to choose PCA parameters (estimation frequency, lookback window)
4. **To choose PCA parameters**: Need to know what holding period H you're targeting

This creates a 3-dimensional optimization problem:
- **H** ∈ {1, 2, 4, 6, 12, 24, 48, 72, ...} hours (holding period)
- **PCA_freq** ∈ {1H, 4H, 6H, 12H, D, W} (estimation frequency)
- **PCA_window** ∈ {168, 360, 720, 1440, ...} hours (lookback length)

**Naive approach**: Test all combinations → O(H × freq × window) = hundreds of backtests

---

## The Solution: Timescale Matching

**Core principle**: Match PCA estimation timescale to the candidate holding period.

### Mathematical Justification

If you rebalance every H hours and hold positions for H hours:

1. **Your P&L accumulates over H-hour periods**, not 1-hour periods
2. **Intra-period fluctuations average out** - you don't realize them
3. **Portfolio variance scales with H-period return variance**, not 1-hour variance
4. **Therefore, beta should be estimated from H-period returns**

**Formally**:
- Portfolio return over H hours: `R_p(H) = w^T · [R_price(H) - F(H)]`
- Variance: `Var(R_p(H)) = w^T · Cov(R_price(H) - F(H)) · w`
- Beta exposure: `β = Cov(R_price(H), R_market(H)) / Var(R_market(H))`

The relevant beta is the one that describes **H-period co-movement**, not hourly.

### Implementation Rule

For each candidate holding period H:
- **PCA estimation frequency = H**
  - If H = 6 hours → use 6-hourly returns
  - If H = 24 hours → use daily returns
  - If H = 72 hours → use 3-day returns

- **PCA lookback window = K × H**
  - K ≈ 20-60 is the number of independent samples
  - K = 30 is a good default (30 independent H-period observations)
  - Examples:
    - H = 6, K = 30 → 180-hour (7.5 day) lookback
    - H = 24, K = 30 → 720-hour (30 day) lookback
    - H = 48, K = 30 → 1440-hour (60 day) lookback

### Result

The 3D optimization problem collapses to **1D + optional refinement**:

**Phase 1**: 1D search over H (with K fixed at 30)
- Test H ∈ {6, 12, 24, 48} → 4 backtests
- Find approximate H*

**Phase 2** (optional): Local refinement of K
- If H* ≈ 24, test K ∈ {20, 30, 40, 60} at H=24 → 4 backtests
- Total: 8 backtests instead of 100+

---

## Why This Works

### Frequency Matching

**Wrong**: Estimate beta from 1-hour returns when holding for 24 hours
- Beta measures sensitivity to 1-hour market moves
- You hold through 24 of these moves
- Hourly mean reversion/noise creates mismatch

**Right**: Estimate beta from 24-hour returns when holding for 24 hours
- Beta measures sensitivity to 24-hour market moves
- You hold for exactly one such period
- Direct alignment between estimation and realization

### Window Scaling

**Need**: Enough independent samples for stable covariance estimation
- Sample correlation converges slowly: O(1/√N) where N = # samples
- Want N ≥ 20-60 for reliable estimates

**Implementation**:
- If each sample is H hours, need K samples → K×H total history
- Example: H=24, K=30 → 30 independent days → stable daily covariance matrix

---

## Practical Impact

### Example: Testing H ∈ {6, 12, 24, 48}

**Naive approach** (all combinations):
```
H × PCA_freq × PCA_window = combinations
4 × 5        × 4           = 80 backtests
```

**Timescale matching**:
```
H only (with freq=H, window=30×H) = 4 backtests
```

**Speedup**: 20× fewer backtests

### Code Example

```bash
# Generate all matched PCA factors
python scripts/find_optimal_horizon.py --H 6 12 24 48 --K 30

# Creates 4 files:
#   data/features/pca_factor_H6_K30_sqrt.csv
#   data/features/pca_factor_H12_K30_sqrt.csv
#   data/features/pca_factor_H24_K30_sqrt.csv
#   data/features/pca_factor_H48_K30_sqrt.csv

# Now backtest each with matching H
# (backtesting framework to be implemented)
```

---

## Theoretical Foundations

### From Portfolio Theory

Kelly criterion for log-optimal portfolio:
```
max E[log(1 + R_p)]
≈ max (E[R_p] - 0.5 × Var(R_p))
```

Over holding period H:
- `E[R_p]` depends on H-period alpha
- `Var(R_p)` depends on H-period variance-covariance matrix
- Beta hedge removes systematic H-period risk

### From Time Series Analysis

**Aggregation property of log returns**:
```
R(H hours) = sum of H consecutive 1-hour log returns
```

**Variance aggregation**:
```
Var(R(H)) ≠ H × Var(R(1))  [unless returns are iid, which they aren't]
```

Therefore:
- Cannot extrapolate 1-hour covariance to H-hour covariance accurately
- Must estimate directly from H-hour data

### From Signal Processing

**Nyquist-Shannon sampling theorem analogy**:
- If your decision frequency is H, information at frequencies > 1/H is noise
- Estimating from higher frequencies adds noise without adding information
- Optimal filter: match measurement frequency to decision frequency

---

## Limitations and Extensions

### When Timescale Matching May Not Be Optimal

1. **Signal decay faster than H**: If alpha half-life << H, may want to rebalance more frequently
2. **Regime changes**: If market regime shifts within H, may need shorter estimation window
3. **Cross-period effects**: Funding payments occur every 8h on Hyperliquid, creating structure

### Extensions

1. **Multi-timescale PCA**: Combine fast (H) and slow (10×H) factors
2. **Adaptive K**: Let K vary with market volatility (shorter K in volatile regimes)
3. **Volume-time aggregation**: Use volume bars instead of time bars (already partially addressed with volume weighting)

---

## Summary

**Problem**: Finding H* requires knowing optimal PCA parameters, but optimal PCA parameters depend on H*

**Solution**: Make PCA parameters a deterministic function of H
- Frequency = H
- Window = K × H (K ≈ 30)

**Benefit**: Reduces search space from 3D to 1D, enabling practical optimization

**Tradeoff**: Assumes matched timescales are optimal (well-justified theoretically and empirically)

**Next step**: Implement backtesting framework to test each H and find empirical H* that maximizes Sharpe ratio

---

## References

- `docs/strategy_spec.md` - Full strategy specification and theoretical derivation
- `docs/QUICKSTART_VOLUME_PCA.md` - Practical guide to generating timescale-matched PCA factors
- `scripts/find_optimal_horizon.py` - Implementation of grid generation
- `scripts/build_pca_factor.py` - Core PCA computation with flexible frequency/window
