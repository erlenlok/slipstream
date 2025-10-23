# Volume-Weighted PCA Implementation Summary

## Overview

The volume-weighted PCA feature has been successfully implemented in `scripts/build_pca_factor.py`. This allows the Slipstream framework to compute market factors that give more weight to high-volume (more liquid) assets.

## What Was Built

### 1. Data Loading Enhancement (`data_load.py`)

**Changes**:
- Updated `fetch_candles_1h()` to capture full OHLCV data: `open`, `high`, `low`, `close`, `volume`
- Previously only captured `close` price
- Doubled API friendliness: `REQUEST_PAUSE_SECONDS` = 0.5s (was 0.25s)
- Initial retry delay = 1.0s (was 0.5s)
- Added retry logging for visibility

**Impact**: All candle files now contain volume data needed for weighting.

### 2. Volume Data Infrastructure (`build_pca_factor.py`)

**New Functions**:

#### `load_volume_data(data_dir, pattern="*_candles_1h.csv")`
- Loads volume from all candle files
- Returns wide DataFrame (hours × assets)
- Handles missing data gracefully

#### `compute_volume_weights(volumes_daily, prices_daily, method, window_days)`
- Computes rolling volume weights using trailing windows
- Supports 5 weighting methods:
  1. **none**: Equal weighting (baseline)
  2. **sqrt**: Square root of volume - dampens extreme spikes
  3. **log**: Log(1 + volume) - robust to outliers
  4. **dollar**: Price × Volume - captures economic value traded
  5. **sqrt_dollar**: Sqrt(Price × Volume) - balanced dollar volume
- Normalizes weights to mean=1 per day
- Handles zero/missing volume via median imputation

#### `compute_rolling_pca_loadings_weighted(returns_daily, weights_daily, ...)`
- Volume-weighted version of the original PCA function
- Scales returns by weights before PCA fitting
- Maintains same interface and output format as original
- Properly handles missing data alignment

### 3. CLI Interface

**New Arguments**:

```bash
--weight-method {none,sqrt,log,dollar,sqrt_dollar}
    Choose a single volume weighting method (default: none)

--compare-all
    Compute PCA for all methods and save with suffixes:
    - pca_factor_loadings_none.csv
    - pca_factor_loadings_sqrt.csv
    - pca_factor_loadings_log.csv
    - pca_factor_loadings_sqrt_dollar.csv
```

### 4. Orchestration Logic

The `main()` function now:
1. Detects which method(s) to compute
2. For volume-weighted methods:
   - Loads volume data from candle files
   - Loads price data if needed (dollar volume methods)
   - Resamples hourly data to daily (volume=sum, price=last)
   - Computes rolling weights with same window as PCA
   - Aligns weights with returns
   - Runs weighted PCA
3. Saves outputs with appropriate naming

## Usage Examples

### Single Method (Equal-Weight Baseline)
```bash
python scripts/build_pca_factor.py
# Output: data/features/pca_factor_loadings.csv
```

### Single Method (Volume-Weighted)
```bash
python scripts/build_pca_factor.py --weight-method sqrt
# Output: data/features/pca_factor_loadings.csv
```

### Compare All Methods
```bash
python scripts/build_pca_factor.py --compare-all --window 60
# Outputs:
#   data/features/pca_factor_loadings_none.csv
#   data/features/pca_factor_loadings_sqrt.csv
#   data/features/pca_factor_loadings_log.csv
#   data/features/pca_factor_loadings_sqrt_dollar.csv
```

### Custom Window and Dollar Volume
```bash
python scripts/build_pca_factor.py \
    --weight-method sqrt_dollar \
    --window 30 \
    --min-assets 20 \
    --output data/features/pca_30d_sqrt_dollar.csv
```

## Technical Design Decisions

### 1. Pre-Weighting vs Weighted Covariance

**Chosen**: Pre-weight returns (scale returns by weights before PCA)

**Rationale**:
- Simple to implement using standard sklearn PCA
- Compatible with existing infrastructure
- Straightforward interpretation
- Computationally efficient

**Alternative Considered**: Custom weighted covariance matrix
- More complex, requires manual eigen-decomposition
- Marginal theoretical benefit
- Can be added later if needed

### 2. Volume Aggregation

**Daily Aggregation**:
- Volume: **Sum** over 24 hours (total traded)
- Price: **Last** close price of the day
- Returns: **Sum** of log returns (equivalent to log(P_end/P_start))

**Rationale**: Daily frequency matches PCA window granularity and reduces noise.

### 3. Weight Normalization

**Method**: Normalize to mean=1 within each day (row-wise)

**Rationale**:
- Prevents absolute volume scales from affecting PCA
- Preserves relative weighting across assets
- Ensures numerical stability

### 4. Missing Data Handling

**Strategy**:
- Use trailing window average for volume
- Replace missing/zero volume with asset median
- Align returns and weights, drop NaN pairs
- Maintain existing min_periods and min_assets thresholds

**Rationale**: Conservative - only uses data when both returns and volume are available.

## Output Format

All outputs maintain the same format:

```csv
datetime,ASSET1,ASSET2,...,ASSETN,_variance_explained,_n_assets
2024-01-01,0.123,-0.045,...,0.089,0.67,120
2024-01-02,0.119,-0.052,...,0.091,0.68,122
...
```

- **Columns 1-N**: PC1 loadings per asset (NaN if excluded that day)
- **_variance_explained**: Fraction of variance captured by PC1
- **_n_assets**: Number of assets included in that day's PCA

## Next Steps for Research

### Phase 1: Generate Comparison Data
```bash
# Once data download completes:
python scripts/build_pca_factor.py --compare-all
```

### Phase 2: Analysis (in Jupyter notebook)

Compare methods on:
1. **Variance explained** - Does volume weighting capture more market variance?
2. **Loading stability** - Turnover of loadings over time
3. **Volume-loading correlation** - Do high-volume assets get higher loadings?
4. **Factor return smoothness** - Construct factor returns and check volatility

### Phase 3: Strategy Backtesting

Integrate each factor variant into the beta-neutral strategy:
- Run full backtest for each weighting method
- Compare net-of-cost Sharpe ratios
- Analyze transaction costs and turnover
- Select optimal method based on performance

### Phase 4: Production Deployment

- Set `--weight-method` to best performer as default
- Document choice in strategy specification
- Add automated testing for PCA stability

## Files Modified

1. **`scripts/data_load.py`**:
   - Lines 36, 41: Doubled rate limiting
   - Lines 171-189: Expanded OHLCV capture

2. **`scripts/build_pca_factor.py`**:
   - Lines 68-113: Added `load_volume_data()`
   - Lines 116-180: Added `compute_volume_weights()`
   - Lines 297-405: Added `compute_rolling_pca_loadings_weighted()`
   - Lines 455-466: Added CLI arguments
   - Lines 470-564: Rewrote `main()` with method loop

3. **`docs/volume_weighted_pca_research.md`**: Research design document

## Testing Checklist

- [ ] Verify data download completed with volume
- [ ] Run equal-weight PCA (baseline) - should match previous results
- [ ] Run sqrt-weighted PCA - should complete without errors
- [ ] Run compare-all mode - should produce 4 output files
- [ ] Inspect outputs - check for reasonable variance explained (>0.5)
- [ ] Visual check - plot loadings for BTC/ETH over time
- [ ] Volume correlation - verify high-volume assets have higher magnitude loadings

---

**Implementation Date**: 2025-10-22
**Status**: ✅ Complete and ready for testing
**Estimated Time to Test**: 10-15 minutes (after data download)
