# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Initialization

**IMPORTANT**: On every new conversation initialization, read all documentation files in `docs/` to gain full context:
- `docs/strategy_spec.md` - Complete strategy specification and trading logic
- `docs/JOINT_H_OPTIMIZATION.md` - Joint alpha + funding H* search results and workflow
- `docs/BACKTESTING_GUIDE.md` - Portfolio optimization and backtesting framework
- `docs/ALPHA_MODEL_TRAINING.md` - Price alpha model training details
- `docs/FUNDING_MODEL_TRAINING.md` - Funding rate prediction model
- `docs/TIMESCALE_MATCHING.md` - Theoretical framework for PCA timescale matching
- `docs/volume_weighted_pca_research.md` - Research on volume weighting methodologies
- `docs/volume_weighted_pca_implementation.md` - Implementation details for volume-weighted PCA
- `docs/QUICKSTART_VOLUME_PCA.md` - Practical guide for H* optimization workflow
- `docs/strategies/gradient/README.md` - Gradient companion strategy overview and workflow ✨ NEW

This context is essential for understanding the research goals, implementation decisions, and current state of the framework.

## ⚠️ CRITICAL: Live Trading Safety

**NEVER run live trading scripts without explicit user permission.**

The following scripts execute REAL trades with REAL money:
- `scripts/strategies/gradient/live/rebalance.sh` - Live rebalance script (called by cron)
- `python -m slipstream.strategies.gradient.live.rebalance` - Direct rebalance execution
- Any command that executes the gradient live trading module

**Rules:**
1. NEVER run these scripts for testing or verification purposes
2. NEVER run these scripts to "check if they work"
3. ALWAYS ask for explicit permission before executing any live trading code
4. If you need to test functionality, ask the user to run it manually
5. When debugging live trading issues, analyze logs and code only - do not execute

**Violation of these rules can result in:**
- Unintended trades
- Financial losses
- Position mismanagement
- API rate limiting or bans

## Repository Status

**IMPORTANT**: This section should be periodically revisited and refreshed during the session as work progresses. Update this section when:
- Completing major implementations or features
- Discovering new limitations or TODOs
- Changing implementation priorities
- After significant debugging or refactoring

Consider reviewing and updating this status every 20-30 messages or when switching to a new major task.

---

### Current Implementation Status (Updated: Oct 2025)

**✅ Complete:**
- Signal generation framework (EWMA idiosyncratic momentum)
- S3 historical data downloader (resumable, full coverage Oct 2023+)
- Timescale-matched PCA factor generation (1, 2, or 3 components)
- Market factor (PC1 returns) computation from loadings and returns
- Multi-factor residuals (PC1+PC2+PC3) with pre-orthogonalization
- Multi-span momentum panel computation
- Transaction cost model with liquidity sensitivity
- Volume-weighted PCA (sqrt, log, sqrt_dollar methods)
- Data pipeline (API + S3) with proper pagination and retry logic
- **Alpha model training pipeline** - Ridge regression with bootstrap + walk-forward CV
- **Funding model training pipeline** - EWMA-based funding prediction with quantile diagnostics
- **Joint H* optimization** - Simultaneous alpha + funding model training across horizons
- **Beta-neutral portfolio optimizer** - Closed-form + cost-aware optimization
- **Transaction cost modeling** - Power-law impact with liquidity-adjusted parameters
- **Walk-forward backtesting framework** - Full simulation with realistic costs
- **Discrete lot rounding** - Beta repair algorithm for production trading
- **Risk analytics** - Covariance estimation and portfolio decomposition

**🎯 Key Research Finding:**
- **H* = 8 hours** (optimal holding period from joint optimization)
- Strategy is **70% funding carry arbitrage**, 15% tail momentum, 15% diversification
- Funding R² = 0.78 (extremely strong), Price alpha R² ≈ 0 (weak overall, strong in tails)
- Combined R² = 0.67 (6,674 bp predictive power)

**⚠️ In Progress:**
- Full end-to-end backtest on historical data (framework complete, needs prediction pipeline)
- Cost parameter calibration from L2 orderbook data
- Production prediction pipeline (apply trained models to generate forecasts)

**📋 Planned:**
- Live trading integration
- Automated retraining pipeline for models
- Risk monitoring dashboard
- Performance attribution analysis
- Slippage modeling from bid-ask spreads

**Known Limitations:**
- Price alpha model has negative R² overall (but tail quantiles are significant)
- Cost model parameters currently use defaults (need empirical calibration)
- Covariance estimation simplified (assumes near-independence of price/funding)
- No automated model retraining yet

---

## Project Overview

Slipstream is a **beta-neutral statistical arbitrage framework** for Hyperliquid perpetual futures. The strategy combines:
1. **Funding rate prediction** (primary driver, R² = 0.78)
2. **Price momentum** (secondary, emerges in tail quantiles)
3. **Beta-neutral portfolio construction** (hedges systematic risk)

The project uses a `src/` layout with `uv` for dependency management, separating data acquisition tooling from core trading logic.

## Architecture

### Directory Structure

- **`src/slipstream/`**: Trading logic and strategy implementations (importable package)
  - `common/` - Shared utilities (return normalization, volatility helpers) ✨ **NEW**
  - `signals/` - Signal generation functions (single source of truth for alpha models)
    - `base.py` - Base interfaces and validation functions
    - `pca_momentum.py` - PCA-based idiosyncratic momentum signals
    - `utils.py` - Signal processing utilities (normalization, autocorrelation analysis)
  - `alpha/` - Price alpha model training
    - `training.py` - Ridge regression with bootstrap + walk-forward CV
    - `data_prep.py` - Feature engineering for alpha model
  - `funding/` - Funding rate prediction
    - `data_prep.py` - Funding feature computation and target preparation
  - `portfolio/` - Portfolio optimization and backtesting ✨ **NEW**
    - `optimizer.py` - Beta-neutral optimization (closed-form + cost-aware)
    - `costs.py` - Transaction cost modeling (power-law impact)
    - `backtest.py` - Walk-forward simulation framework
    - `risk.py` - Covariance estimation and risk analytics
  - `gradient/` - Gradient trend-following companion strategy ✨ **NEW**
    - `signals.py` - Multi-horizon trend strength construction
    - `portfolio.py` - Dollar-volatility balanced sizing
    - `backtest.py` - Lightweight simulation + result helpers
    - `cli.py` - Command-line entry points for signals/backtests
- **`scripts/`**:
  - `data_load.py` - Data acquisition utility for downloading hourly OHLCV, funding, and return data
  - `fetch_s3_historical.py` - S3 historical data downloader (resumable, Oct 2023+)
  - `build_pca_factor.py` - Rolling PCA market factor computation from returns data
  - `find_optimal_horizon.py` - H* optimization via timescale-matched PCA grid generation
  - `find_optimal_H_alpha.py` - Alpha model H* search (trains models across horizons)
  - `find_optimal_H_funding.py` - Funding model H* search
  - `find_optimal_H_joint.py` - **Joint optimization** (trains both models simultaneously) ✨ **NEW**
  - `gradient_compute_signals.py` / `gradient_run_backtest.py` - Gradient CLI wrappers ✨ **NEW**
- **`notebooks/`**: Research and backtesting analysis
- **`data/`**: (git-ignored)
  - `market_data/` - Raw market data CSVs from API (candles, funding, merged returns) - **PRIMARY DATA SOURCE**
  - `s3_historical/` - S3 historical archive (Oct 2023+) - **ONLY FOR COST MODEL CALIBRATION, NOT BACKTESTING**
  - `features/` - Computed features
    - `alpha_models/` - Trained alpha models
    - `funding_models/` - Trained funding models
    - `joint_models/` - Joint optimization results ✨ **NEW**

### Data Pipeline Components

**`scripts/data_load.py`** - Market data acquisition:
- `fetch_all_perp_markets()` - Enumerates all perp dex namespaces and unions their live universes
- `fetch_candles()` - Paginates 4h candles in 120-day chunks to stay under API limits (~5k candles)
- `fetch_funding_hourly()` - Fetches hourly funding data with pagination
- `compute_log_returns()` - Vectorized computation of 4h log returns from close prices
- `build_datasets()` - Orchestrates fetching candles+funding, computing returns, aligning, and writing CSVs to `data/market_data/`
- `build_for_universe()` - Fetches and writes datasets for all live markets (with optional dex filtering)

**`scripts/build_pca_factor.py`** - Factor construction:
- `load_all_returns()` - Loads all merged return files from `data/market_data/` into wide matrix
- `resample_returns()` - Aggregates hourly log returns to specified frequency (hourly, 4H, 6H, daily, etc.)
- `compute_rolling_pca_loadings()` - Computes rolling PCA with PC1 loadings as market factor weights, handling variable asset counts
- `compute_rolling_pca_loadings_weighted()` - Volume-weighted PCA variant
- Supports timescale-matched PCA: frequency and window adapt to holding period H
- Outputs to `data/features/pca_factor_*.csv` with period loadings per asset + metadata

**`scripts/find_optimal_H_joint.py`** - Joint H* optimization ✨ **NEW**:
- Trains both alpha and funding models for each H in [4, 8, 12, 24, 48, ...]
- Computes combined signal: α_total = α_price - F_hat
- Generates combined quantile diagnostics showing alpha, funding, and total signal
- Identifies optimal H* that maximizes combined R² (out-of-sample)
- Output: `data/features/joint_models/joint_model_H{H}.json` for each H
- Result: **H* = 8 hours** with combined R² = 0.667

**`src/slipstream/portfolio/`** - Portfolio optimization ✨ **NEW**:
- `optimizer.py`:
  - `optimize_portfolio()` - Closed-form beta-neutral optimization (cost-free)
  - `optimize_portfolio_with_costs()` - Numerical optimization with transaction costs
  - `round_to_lots()` - Discrete lot rounding with beta repair algorithm
- `costs.py`:
  - `TransactionCostModel` - Power-law cost model: C(Δw) = Σ|Δw|*fee + Σλ|Δw|^1.5
  - `compute_transaction_costs()` - Calculate cost breakdown for rebalance
  - `estimate_liquidity_adjusted_costs()` - Calibrate costs from volume/spread data
- `backtest.py`:
  - `run_backtest()` - Walk-forward simulation with realistic costs
  - `BacktestResult` - Results container with Sharpe, drawdown, equity curve
- `risk.py`:
  - `compute_total_covariance()` - S_total = S_price + S_funding - 2*C_cross
  - `decompose_risk()` - Portfolio risk attribution by asset

### HTTP Layer

The project implements retry logic with exponential backoff for retryable HTTP status codes (429, 500, 502, 503, 504). All API calls include a `REQUEST_PAUSE_SECONDS` (0.25s) throttle to respect rate limits. The base API endpoint is `https://api.hyperliquid.xyz/info`.

### Data Pipeline

1. Fetch candles and funding data in parallel using `asyncio.gather()`
2. Compute log returns: `log_returns = np.log1p(candles["close"].pct_change())`
3. Create aligned 4-hour index and join all datasets
4. Write three CSV outputs: `{prefix}_{coin}_candles_4h.csv`, `{prefix}_{coin}_funding_4h.csv`, `{prefix}_{coin}_merged_4h.csv`

## Joint H* Optimization Results

**Optimal Holding Period: H* = 8 hours**

| H (hours) | R²_alpha | R²_funding | R²_combined | Rank |
|-----------|----------|------------|-------------|------|
| **8** | -0.002 | 0.776 | **0.667** | **1st** |
| 4 | -0.001 | 0.720 | 0.638 | 2nd |
| 12 | -0.003 | 0.689 | 0.582 | 3rd |
| 24 | -0.006 | 0.449 | 0.356 | 4th |
| 48 | -0.009 | 0.282 | 0.203 | 5th |

**Strategy Characterization:**
- **70% Funding Carry**: Primary driver (R² = 0.78)
- **15% Tail Momentum**: Price alpha emerges in low-funding regimes (quantile 9)
- **15% Diversification**: Risk reduction via beta-neutral construction

**Key Quantile Insight (H=8, Top vs Bottom Decile):**
- **Bottom (Q0)**: High funding (+9.75σ) → SHORT to avoid paying funding
- **Top (Q9)**: Negative funding (-1.52σ) + positive momentum (+0.037) → LONG to collect funding

See `docs/JOINT_H_OPTIMIZATION.md` for complete analysis.

## Common Commands

### Setup
```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

### Data Loading

#### API Data (Recent ~7 months)
```bash
# Print live perp markets
uv run python -m slipstream

# Fetch single coin (90 days) -> saves to data/market_data/
uv run hl-load --coin BTC --days 90

# Fetch all markets
uv run hl-load --all --days 30

# Filter by dex
uv run hl-load --all --dex "(default)" --days 30

# Custom date range
uv run hl-load --coin ETH --start 2024-01-01 --end 2024-03-01
```

#### S3 Historical Data (Full history back to Oct 2023)
```bash
# Setup (one-time)
sudo apt install lz4
aws configure  # Enter AWS credentials

# Test setup
./.aws_setup_test.sh

# Fetch historical data from S3 (resumable)
python scripts/fetch_s3_historical.py --start 2023-10-01 --end 2025-03-27

# Fetch specific coins only
python scripts/fetch_s3_historical.py --start 2023-10-01 --end 2025-03-27 --coins BTC ETH SOL

# Validate against API data
python scripts/fetch_s3_historical.py --validate

# See docs/S3_HISTORICAL_DATA.md for detailed guide
```

### Model Training

#### Timescale-Matched PCA (Required First Step)
```bash
# Generate PCA factors for multiple holding periods
python scripts/find_optimal_horizon.py --H 4 8 12 24 48 --K 30 --weight-method sqrt

# Output: data/features/pca_factor_H{H}_K30_sqrt.csv for each H
```

#### Joint H* Optimization (Recommended)
```bash
# Train both alpha and funding models simultaneously
python scripts/find_optimal_H_joint.py --H 4 8 12 24 48 --n-bootstrap 1000

# Output: data/features/joint_models/joint_model_H{H}.json
# Also: optimization_summary.json with optimal H* recommendation

# Quick test (fewer bootstrap samples)
python scripts/find_optimal_H_joint.py --H 8 --n-bootstrap 50
```

#### Individual Model Training (Alternative)
```bash
# Alpha model only
python scripts/find_optimal_H_alpha.py --H 8 --n-bootstrap 1000

# Funding model only
python scripts/find_optimal_H_funding.py --H 8 --n-bootstrap 1000
```

### Portfolio Optimization & Backtesting

```python
from slipstream.portfolio import optimize_portfolio, run_backtest, BacktestConfig

# Beta-neutral optimization (closed-form)
w = optimize_portfolio(
    alpha=alpha_total,  # α_price - F_hat
    beta=beta_exposures,
    S=covariance_matrix,
    leverage=1.0,
)

# Cost-aware optimization
from slipstream.portfolio import optimize_portfolio_with_costs
w, info = optimize_portfolio_with_costs(
    alpha=alpha_total,
    beta=beta_exposures,
    S=covariance_matrix,
    w_old=current_weights,
    cost_linear=cost_model.fee_rate,
    cost_impact=cost_model.impact_coef,
    leverage=1.0,
)

# Run backtest
config = BacktestConfig(H=8, start_date='2024-01-01', end_date='2024-12-31')
result = run_backtest(config, alpha_price, alpha_funding, beta, S, returns, funding)
print(f"Sharpe: {result.sharpe_ratio():.2f}")
```

See `docs/BACKTESTING_GUIDE.md` for complete workflow.

### Testing
```bash
# Run all tests
uv run pytest

# Test optimizer
python tests/test_portfolio_optimizer.py

# Lint
uv run ruff check
```

## Signal Generation Architecture

The `src/slipstream/signals/` module implements the alpha model (strategy_spec.md Section 3.1) as pure DataFrame transformations. This provides a single source of truth for signal logic that can be imported into notebooks for research and used in production.

**Design principles:**
- Signals are pure functions: DataFrame in → DataFrame out
- No hidden state or side effects
- Composable and testable
- All signals use consistent long format: MultiIndex (timestamp, asset) with 'signal' column

**Key functions:**
- `compute_idiosyncratic_returns()`: Extract residuals after removing market factor (PC1)
- `idiosyncratic_momentum()`: Primary alpha signal - EWMA-based idiosyncratic momentum panel
- `normalize_signal_cross_sectional()`: Cross-sectional z-score/rank normalization
- `compute_signal_autocorrelation()`: For signal half-life analysis (Section 4.4)

**Example usage in notebooks:**
```python
from slipstream.signals import idiosyncratic_momentum

# Load data
returns = load_all_returns()  # Wide format
pca_data = pd.read_csv('data/features/pca_factor_H8_K30_sqrt.csv')
pca_data['timestamp'] = pd.to_datetime(pca_data['timestamp'])

# Extract PCA components
loadings = pca_data.set_index(['timestamp', 'asset'])['loading']
market_factor = pca_data.groupby('timestamp')['market_return'].first()

# Compute panel of momentum indicators (idio_mom_2, idio_mom_4, idio_mom_8, etc.)
momentum_panel = idiosyncratic_momentum(
    returns=returns,
    pca_loadings=loadings,
    market_factor=market_factor,
    spans=[2, 4, 8, 16, 32],  # Multi-timescale EWMA
    normalization='volatility'
)

# Access specific momentum: momentum_panel.xs(8, level='span') for idio_mom_8
```

## Development Notes

- Python version is pinned to 3.11 via `.python-version`
- **Code organization**:
  - `src/slipstream/signals/` - Signal generation (importable, single source of truth)
  - `src/slipstream/alpha/` - Price alpha model training
  - `src/slipstream/funding/` - Funding rate prediction
  - `src/slipstream/portfolio/` - Portfolio optimization and backtesting ✨ **NEW**
  - `scripts/` - Data acquisition and feature engineering scripts
  - `notebooks/` - Research and backtesting
  - `data/market_data/` - API-based recent market data CSVs (~7 months, git-ignored)
  - `data/s3_historical/` - S3-based historical candles (Oct 2023+, git-ignored)
  - `data/features/` - Computed features (git-ignored)
- The `hl-load` CLI entry point wraps `scripts/data_load.py` via `python -m slipstream`
- Data files use naming convention: `{COIN}_{type}_1h.csv` where type is `candles`, `funding`, or `merged`
- PCA factor files use naming: `pca_factor_H{hours}_K{multiplier}_{method}.csv` for timescale-matched factors
- Joint model files use naming: `joint_model_H{hours}.json` with combined diagnostics
- When adding data loading functionality, extend `scripts/data_load.py` following the async pattern
- When adding new features, create scripts in `scripts/` and output to `data/features/`
- **When adding signal logic, create/extend modules in `src/slipstream/signals/` (importable from notebooks)**
- **When adding portfolio logic, extend modules in `src/slipstream/portfolio/`**
- All timestamps are handled in UTC and converted to milliseconds for API calls via `ms(dt)`
- Candle timestamps are floored to the hour via `to_hour(dt_ms)`

## Strategy Implementation Notes

The Slipstream strategy (see `docs/strategy_spec.md`) has found its optimal configuration:

**Optimal Holding Period: H* = 8 hours**

Key implementation insights:
- **Joint optimization is critical**: Training alpha and funding models separately misses interactions
- **Funding dominates**: R² = 0.78 for funding vs R² ≈ 0 for alpha (overall)
- **Alpha emerges in tails**: Top quantile shows positive momentum (α_actual = +0.037, t=5.6)
- **Strategy is carry-focused**: Short high-funding assets, long negative-funding assets
- **Beta neutrality essential**: Hedges systematic risk, allows leverage

**Workflow:**
1. Generate timescale-matched PCA factors for target H
2. Train joint models (alpha + funding) via `find_optimal_H_joint.py`
3. Generate predictions using trained models
4. Optimize portfolio with `optimize_portfolio_with_costs()`
5. Run backtest with `run_backtest()`
6. Analyze performance and refine

See `docs/JOINT_H_OPTIMIZATION.md` and `docs/BACKTESTING_GUIDE.md` for detailed workflows.

## Next Steps for Full Production

1. **Build prediction pipeline**: Apply trained models to generate forecasts
2. **Calibrate cost parameters**: Use L2 orderbook data to estimate λ_i (market impact)
3. **Run full historical backtest**: Test H*=8 on 2023-2025 data
4. **Analyze attribution**: Decompose returns into alpha, funding, costs
5. **Live trading integration**: Connect to Hyperliquid API for execution
