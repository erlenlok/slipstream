# Gradient Strategy

The Gradient strategy is a lightweight companion to the original Slipstream framework. It keeps a balanced long/short book by allocating a fixed amount of dollar volatility to the assets with the strongest multi-horizon trend signals.

## Concept

- **Signals**: Trend strength is computed by summing the momentum of volatility-normalized 4-hour log returns across lookbacks {2, 4, 8, …, 1024}.
- **Portfolio**: At each rebalance the strategy:
  1. Selects the top *N* assets with the largest positive trend scores (long side).
  2. Selects the bottom *N* assets with the most negative scores (short side).
  3. Allocates equal dollar-volatility exposure to each side by sizing positions inverse to their recent volatility.
- **Backtest**: A thin wrapper reuses the shared utilities and reports per-period portfolio returns together with helper metrics (e.g., annualized Sharpe).

The end result is a constantly hedged book that tilts toward persistent trends without needing the full Slipstream alpha–funding pipeline.

## Code Layout

```
src/slipstream/strategies/gradient/
├── __init__.py            # Public API (signals, portfolio, backtest)
├── cli.py                 # Command-line entry points
├── signals.py             # Trend strength construction
├── universe.py            # Long/short candidate selection
├── portfolio.py           # Dollar-volatility balanced sizing
└── backtest.py            # Lightweight backtester + result container
```

Shared helpers live under `src/slipstream/common/` so both Slipstream and Gradient can reuse volatility and return utilities.

## Workflow

1. Prepare a wide CSV of 4-hour log returns (index = timestamp, columns = asset symbols).
2. Compute signals:

   ```bash
   uv run gradient-signals \
       --returns-csv data/market_data/log_returns_4h.csv \
       --output data/gradient/signals/trend_strength.csv
   ```

3. Run a backtest (recomputes signals if none are supplied):

   ```bash
   uv run gradient-backtest \
       --returns-csv data/market_data/log_returns_4h.csv \
       --returns-output data/gradient/backtests/portfolio_returns.csv
   ```

4. Inspect `portfolio_returns.csv` or load it into a notebook to analyse performance.

### Useful Flags

- `--lookbacks`: Override the default power-of-two lookback set.
- `--top-n` / `--bottom-n`: Control the number of assets per side (defaults to 5/5).
- `--vol-span`: Adjust the EWMA span used for volatility targeting.
- `--target-side-dollar-vol`: Change the dollar-volatility budget per side.

## Extending

- Plug the `compute_trend_strength` output straight into custom risk models.
- Swap the volatility estimator in `common/` for alternative techniques (e.g., GARCH).
- Replace the equal risk per asset sizing in `portfolio.py` with more advanced schemes.

For development notes and the rationale behind this structure see `docs/GRADIENT_PLAN.md`.

---

## Portfolio Concentration Sensitivity Analysis

### Objective

Determine the optimal portfolio concentration (n%) by measuring expected returns across different concentration levels from 1% to 50%. Instead of selecting a fixed number of assets (e.g., top/bottom 5), we continuously long the top n% most bullish perps and short the bottom n% most bearish perps.

### Methodology

#### 1. Universe Selection

Filter out illiquid perps where a $10,000 USD trade would exceed 2.5% of average daily volume:
```
ADV_USD = avg_volume_24h * avg_price
Include if: 10,000 < 0.025 * ADV_USD
```

#### 2. Signal Construction

Momentum score per asset = sum of EWMA(vol-normalized returns) across multiple lookback windows:
- **Lookback windows**: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] hours
- **Volatility normalization**: returns / ewma_vol (using 24-hour span)
- **Signal**: Σ_lookback EWMA(vol_norm_returns, span=lookback)

At each timestamp, rank all perps by this momentum score.

#### 3. Portfolio Construction

For a given concentration n%:
- **Long bucket**: Top n% of assets by momentum score
- **Short bucket**: Bottom n% of assets by momentum score
- **Weighting schemes** (to be compared):
  1. **Equal-weighted**: 1/N weight per asset in each bucket
  2. **Inverse-vol weighted**: w_i ∝ 1/σ_i (normalized to sum to 1 per side)

#### 4. Sampling Strategy

For each configuration (n%, rebalance_frequency, weight_scheme):
- Sample K=100 random 10-day periods from the dataset
- Ensure periods don't overlap
- For each sample:
  1. Rank assets at start of period
  2. Construct portfolio (long top n%, short bottom n%)
  3. Apply weighting scheme
  4. Rebalance at specified frequency (4h, 8h, 12h, ..., 48h)
  5. Calculate cumulative return over 10 days
  6. Annualize the return

#### 5. Rebalance Frequencies to Test

Since candles are 4h resolution, test all multiples of 4h from 4h to 48h:
- [4h, 8h, 12h, 16h, 20h, 24h, 28h, 32h, 36h, 40h, 44h, 48h]

#### 6. Output Metrics

For each (n%, rebalance_freq, weight_scheme):
- **Mean annualized return** across K samples
- **Standard deviation** of annualized returns (variability)
- **Sharpe ratio** = mean / std
- **Min/Max returns** (range)

### Expected Deliverables

#### 1. High-Quality Panel Data

CSV with columns:
```
timestamp, asset, momentum_score, vol_24h, adv_usd, include_in_universe
```

This panel can be used for cross-sectional ranking at any timestamp.

#### 2. Sensitivity Analysis Results

CSV with columns:
```
n_pct, rebalance_freq_h, weight_scheme, mean_ann_return, std_ann_return, sharpe, min_return, max_return, n_samples
```

#### 3. Visualization

**Primary plot**: n% (x-axis) vs Expected Annualized Return (y-axis)
- Multiple lines for different rebalance frequencies
- Separate subplots or colors for equal-weighted vs inverse-vol
- Error bands showing ±1 std dev from K samples

**Additional plots**:
- Heatmap: (n%, rebalance_freq) → Sharpe ratio
- Distribution plots: return distributions for selected n% values

### Implementation Plan

#### Phase 1: Data Preparation
1. Load 4h candles for all perps from `data/market_data/`
2. Compute average daily volume (ADV) and filter universe
3. Compute vol-normalized returns (using 24h EWMA vol)
4. Compute multi-span EWMA momentum scores
5. Output panel data: `data/gradient/sensitivity/panel_data.csv`

#### Phase 2: Backtest Framework
1. Create `concentration_backtest()` function:
   - Takes panel data, n%, rebalance_freq, weight_scheme as inputs
   - Returns annualized return for a single 10-day period
2. Create sampling framework:
   - Generate K=100 random 10-day windows
   - Ensure no overlaps
   - Run backtest for each sample
3. Parallelize across configurations if needed

#### Phase 3: Analysis & Visualization
1. Aggregate results across all configurations
2. Generate sensitivity plots
3. Identify optimal n% and rebalance frequency
4. Output summary table with top configurations

#### Phase 4: Documentation
1. Update this doc with findings
2. Add notebook with reproducible analysis

### Files to Create

```
data/gradient/sensitivity/
├── panel_data.csv                 # High-quality momentum panel
├── sample_periods.csv             # K random 10-day periods
├── results_equal_weighted.csv     # Sensitivity results
├── results_inverse_vol.csv
└── optimal_config.json            # Best n%, rebalance_freq, weight_scheme

notebooks/gradient_concentration_analysis.ipynb
src/slipstream/strategies/gradient/sensitivity.py  # Concentration backtest logic
scripts/strategies/gradient/concentration_study.py  # Sensitivity sweep CLI
scripts/strategies/gradient/capture_baseline.py      # Baseline metrics helper
```

### Next Steps

1. Implement panel data construction with universe filtering
2. Implement concentration backtest function
3. Run full sensitivity sweep (n% × rebalance_freq × weight_scheme)
4. Analyze results and visualize
5. Select optimal configuration for production
