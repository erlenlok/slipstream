# Slipstream

[![Status: Multi-Strategy Ready](https://img.shields.io/badge/status-multi--strategy%20ready-4c1)](#-strategy-onboarding-toolkit)
[![Registered Strategies](https://img.shields.io/badge/strategies-2%20registered-1c8adb)](#-strategy-onboarding-toolkit)

**Beta-neutral statistical arbitrage framework for Hyperliquid perpetual futures.**

Slipstream combines price momentum and funding rate predictions to trade a market-neutral portfolio, dynamically rebalancing to maximize risk-adjusted returns while hedging systematic market exposure.

## 🎯 Strategy Characterization

Based on joint H* optimization across holding periods [4, 8, 12, 24, 48] hours:

**Optimal Holding Period: H* = 8 hours**

| Component | R² (OOS) | Contribution |
|-----------|----------|--------------|
| **Funding Rate Prediction** | 0.776 (7,760 bp) | 70% - Primary driver |
| **Price Alpha (Momentum)** | -0.002 (-20 bp) | 15% - Tail signal only |
| **Combined Signal** | **0.667 (6,674 bp)** | **Total predictive power** |

**Strategy Profile:**
- **70% Funding Carry Arbitrage**: Short assets with high expected funding, long assets with negative funding
- **15% Tail Momentum**: Price momentum emerges only in low-funding regimes (top decile)
- **15% Diversification**: Risk reduction through beta-neutral construction

This is fundamentally a **funding carry strategy** where momentum adds value primarily when funding is favorable.

## 🚀 Features

- ✅ **Joint Alpha + Funding Optimization**: Train both models simultaneously to find optimal H*
- ✅ **Beta-Neutral Portfolio Optimizer**: Closed-form + cost-aware optimization with leverage constraints
- ✅ **Transaction Cost Modeling**: Power-law impact model with liquidity-adjusted parameters
- ✅ **Walk-Forward Backtesting**: Full path-dependent simulation with realistic costs
- ✅ **Discrete Lot Rounding**: Beta repair algorithm for production trading
- ✅ **Multi-Source Data Pipeline**: API (7 months) + S3 archive (Oct 2023+)
- ✅ **Timescale-Matched PCA**: Factor decomposition adapts to holding period H
- ✅ **EWMA Momentum Signals**: Multi-span idiosyncratic momentum features
- ✅ **Volume-Weighted PCA**: Three weighting methods (sqrt, log, sqrt_dollar)
- ✅ **Gradient Companion Strategy**: Balanced trend-following overlay built on shared tooling ✨ NEW
- ✅ **Multi-Strategy Toolkit**: Registry, template package, and dispatch CLIs for onboarding new strategies ✨ NEW
- ✅ **Brawler Passive Market Maker**: CEX-anchored, volatility-aware quoting loop for slow liquidity providers ✨ NEW

## 📊 Repo Layout

```
slipstream/
├── src/slipstream/                  # Importable Python package
│   ├── core/                        # Shared services for every strategy
│   │   ├── common/                  # Return & volatility utilities
│   │   ├── config/                  # Layered config loader and helpers ✨ NEW
│   │   ├── signals/                 # Signal generators (EWMA, PCA, filters)
│   │   ├── portfolio/               # Optimisers, backtesting engines
│   │   ├── costs/                   # Transaction cost models
│   │   └── funding/                 # Funding data prep helpers
│   ├── strategies/                  # Strategy-specific implementations
│   │   ├── gradient/                # Gradient live + backtest stack ✨ NEW LOCATION
│   │   └── template/                # Scaffold for new strategies
│   ├── common/                      # Legacy shim → slipstream.core.common
│   ├── signals/                     # Legacy shim → slipstream.core.signals
│   ├── portfolio/                   # Legacy shim → slipstream.core.portfolio
│   ├── costs/                       # Legacy shim → slipstream.core.costs
│   ├── funding/                     # Legacy shim → slipstream.core.funding
│   └── alpha/                       # Price alpha research modules
├── scripts/
│   ├── data_load.py             # API data fetcher
│   ├── fetch_s3_historical.py   # S3 historical downloader
│   ├── build_pca_factor.py      # PCA factor computation
│   ├── find_optimal_horizon.py  # PCA timescale matching
│   ├── find_optimal_H_alpha.py  # Alpha model H* search
│   ├── find_optimal_H_funding.py # Funding model H* search
│   ├── find_optimal_H_joint.py  # Joint optimization ✨ NEW
│   ├── gradient_compute_signals.py # Gradient trend strength CLI ✨ NEW
│   ├── gradient_run_backtest.py # Gradient backtest CLI ✨ NEW
│   └── strategies/
│       ├── run_backtest.py      # Multi-strategy backtest dispatcher ✨ NEW
│       ├── gradient/            # Gradient-specific helpers (scripts + tooling)
│       └── brawler/             # Brawler live-run scripts (coming online)
├── notebooks/                   # Research & analysis
├── docs/                        # Documentation
│   ├── DOCUMENTATION.md         # Consolidated documentation
│   ├── GRADIENT.md              # Gradient overview & workflow ✨ NEW
│   └── archive/                 # Deprecated documentation files
├── data/                        # Data storage (git-ignored)
│   ├── market_data/             # API data (candles, funding, returns)
│   ├── s3_historical/           # S3 historical archive
│   └── features/                # Computed features
│       ├── alpha_models/        # Trained alpha models
│       ├── funding_models/      # Trained funding models
│       └── joint_models/        # Joint optimization results ✨ NEW
└── tests/                       # Unit tests
```

## 🧩 Strategy Onboarding Toolkit

- Use `uv run python scripts/strategies/run_backtest.py --strategy <slug> -- --returns-csv ...` to target any registered strategy without hunting for bespoke scripts.
- The `src/slipstream/strategies/template/` package ships sample config, signal generation, and CLI stubs—copy it to scaffold new ideas quickly.
- Follow the [strategy onboarding checklist](docs/monorepo_plan/sprint_04_onboarding_checklist.md) and [per-strategy secrets guide](docs/monorepo_plan/strategy_secrets.md) to stay aligned with monorepo conventions.
- Add metadata + CLI hooks in `slipstream/strategies/__init__.py` so dashboards and dispatch scripts pick up the new strategy automatically.

## 🌈 Gradient Strategy

Looking for a simpler trend overlay without the full alpha + funding stack? The new [Gradient strategy](docs/strategies/gradient/README.md) keeps equal dollar-volatility long and short books in the assets with the strongest directional trends. Generate signals with `uv run gradient-signals` and backtest with `uv run gradient-backtest`.

## 🥊 Brawler Passive Market Maker

- Anchors quotes to Binance futures mid-prices, smoothing the local basis to avoid reacting to thin-book noise.
- Spreads widen automatically with CEX volatility (`base_spread + k * sigma`), so the bot prices in its latency disadvantage.
- Inventory-aware skewing plus configurable kill switches (max inventory, volatility ceiling, basis de-peg, feed disconnect).
- Runs with: `uv run python -m slipstream.strategies.brawler.cli --config configs/brawler.yml --api-key ... --api-secret ...`
- Sample config for tiny-notional prod tests: `config/brawler_single_asset.example.yml`
- API keys/secrets can live in the config (see `hyperliquid_api_key|secret`) or in `.env.gradient`—the CLI resolves CLI args → config → env in that order. Use `--log-level` (or `BRAWLER_LOG_LEVEL`) to control verbosity.
- Portfolio guardrails (`portfolio.*` block) cap aggregate exposure, trigger reduce-only mode when gross inventory climbs, and taper order sizes automatically across all symbols.
- CLI auto-loads `.env.brawler` or `.env.gradient` (or whatever `BRAWLER_ENV_FILE` points to) before parsing args, so you can keep API creds in the same env file as Gradient.
- Candidate screening defaults live under the `candidate_screening` block (see `config/brawler_single_asset.example.yml`); tune spread/basis/vol thresholds so onboarding always references quantitative gates instead of gut feel.
- Use `uv run python -m slipstream.strategies.brawler.tools.candidate_scan --hl-pattern 'data/hl/{symbol}.csv' --cex-pattern 'data/binance/{cex_lower}.csv' --symbols BTC ETH ...` to rank new instruments by basis drift, sigma parity, HL-vs-CEX spread edge, optional depth multiples (`--hl-depth-pattern`) and funding volatility (`--funding-pattern`). Pattern tokens `{symbol}`, `{lower}`, `{cex_symbol}`, `{cex_lower}` let you point at any recording layout.
- Capture the required CSVs with `uv run python scripts/strategies/brawler/record_bbo.py --config config/brawler.yml --duration 3600` (overrides: `--hl-pattern`, `--cex-pattern`, `--depth-pattern`). The recorder subscribes to Hyperliquid `l2Book` + Binance `bookTicker`, writes per-symbol BBO/depth files, and can be left running or cron’d before the candidate scan/watchlist jobs.
- Run the lightweight volume-gap screener with `uv run python scripts/strategies/brawler/volume_gap_screener.py --config config/brawler.yml --ratio-threshold 0.3` to compare Hyperliquid 24h notional volumes against Binance benchmarks (BTC/ETH by default). Listings whose HL/Bin ratios fall well below the baseline bubble up as “under-trafficked” candidates.
- Automate the rolling watchlist by calling `uv run python scripts/strategies/brawler/watchlist_report.py --hl-pattern ... --cex-pattern ... --output-dir logs/brawler_watchlist` (optionally add `--hl-depth-pattern` / `--funding-pattern`). The script prints a console summary and writes timestamped CSV/JSON/Markdown artifacts (`*_latest.*` symlinks included) so nightly jobs can drop ranked candidate reports into S3/Slack.

### Minimal-Size Prod Test Checklist

1. Copy `config/brawler_single_asset.example.yml`, tune spread/order sizing, and double-check `tick_size` + `quote_reprice_tolerance_ticks` for your listing.
2. Set `order_size` to the absolute minimum supported by the venue and cap `max_inventory` so kill switches trigger before real risk accumulates.
3. Ensure kill-switch settings (`max_volatility`, `max_basis_deviation`, `kill_switch.*`) reflect your tolerance; the engine now suspends quotes automatically when basis/feeds go stale.
4. Set `state_snapshot_path` if you want the engine to persist basis/inventory between runs, optionally pass `--inventory-file seeds.json`, and tune the `portfolio.*` block to define global exposure/tapering rules.
5. Either store Hyperliquid creds inside the config (`hyperliquid_api_key`, `hyperliquid_api_secret`, `hyperliquid_main_wallet`) or in your env file—CLI args are optional now.
6. Run `uv run python -m slipstream.strategies.brawler.cli --config <your-config> --log-level INFO` and monitor logs for `suspended:` messages (basis, feed, volatility, inventory) before increasing size; resume behavior is automatic once the condition clears.

See `docs/BRAWLER_SPEC.md` for the full specification; the new implementation wires that spec into reusable connectors, kill switches, and an auto-resume loop.

## 🏁 Quick Start

### 1. Setup

```bash
# Install uv CLI: https://docs.astral.sh/uv/getting-started/installation/
uv venv .venv
source .venv/bin/activate
uv sync
```

### 2. Fetch Data

```bash
# API data (recent ~7 months)
uv run hl-load --all --days 180

# OR: S3 historical data (full history from Oct 2023)
sudo apt install lz4
aws configure  # Enter credentials
python scripts/fetch_s3_historical.py --start 2023-10-01 --end 2025-03-27
```

### 3. Train Models

```bash
# Generate timescale-matched PCA factors
python scripts/find_optimal_horizon.py --H 4 8 12 24 48 --K 30 --weight-method sqrt

# Train joint alpha + funding models
python scripts/find_optimal_H_joint.py --H 4 8 12 24 48 --n-bootstrap 1000

# View results
cat data/features/joint_models/optimization_summary.json
```

**Output:**
```json
{
  "H_optimal": 8,
  "R2_combined_optimal": 0.6674,
  "R2_combined_bp": 6674,
  "H_tested": [4, 8, 12, 24, 48]
}
```

### 4. Run Backtest

```python
from slipstream.portfolio import run_backtest, BacktestConfig

config = BacktestConfig(
    H=8,  # Optimal holding period
    start_date='2024-01-01',
    end_date='2024-12-31',
    initial_capital=1_000_000,
    leverage=1.0,
    use_costs=True,
)

result = run_backtest(
    config=config,
    alpha_price=alpha_predictions,
    alpha_funding=funding_predictions,
    beta=beta_exposures,
    S=covariance_matrix,
    realized_returns=actual_returns,
    realized_funding=actual_funding,
)

print(f"Sharpe Ratio: {result.sharpe_ratio():.2f}")
print(f"Max Drawdown: {result.max_drawdown():.2%}")
```

## 📈 Key Results

### Joint H* Optimization (n=1000 bootstrap samples)

| H (hours) | R²_alpha | R²_funding | **R²_combined** | Rank |
|-----------|----------|------------|-----------------|------|
| **8** ⭐ | -0.002 | **0.776** | **0.667** | **1st** |
| 4 | -0.001 | 0.720 | 0.638 | 2nd |
| 12 | -0.003 | 0.689 | 0.582 | 3rd |
| 24 | -0.006 | 0.449 | 0.356 | 4th |
| 48 | -0.009 | 0.282 | 0.203 | 5th |

**Key Findings:**
1. **Funding persistence is extremely strong** (R² = 0.78 at H=8)
2. **Price alpha is negligible overall** (R² ≈ 0) but emerges in specific quantiles
3. **Combined signal is powerful** (R² = 0.67), driven primarily by funding carry
4. **Optimal H = 8 hours** balances funding persistence with rebalancing frequency

### Quantile Analysis (H=8, Binned by α_total)

| Quantile | α_actual | α_t | F_actual | F_t | Total_actual | Total_t | **Strategy** |
|----------|----------|-----|----------|-----|--------------|---------|--------------|
| 0 (worst) | -0.077 | -10.9 | **+9.75σ** | +1,188 | -9.83 | **-905** | **SHORT** |
| 5 (median) | -0.024 | -3.6 | +1.95σ | +288 | -1.97 | -207 | SHORT |
| 9 (best) | **+0.037** | **+5.6** | **-1.52σ** | **-164** | **+1.56** | **+137** | **LONG** |

**Interpretation:**
- **Bottom decile**: High funding (9.75σ) → SHORT to avoid costs
- **Top decile**: Negative funding (-1.52σ) + positive momentum (+0.037) → LONG to collect funding
- Alpha only predictive in low-funding environments (quantile 9)

## 🔬 Research Workflow

### Data Acquisition

```bash
# 1. Fetch market data
uv run hl-load --all --days 180

# 2. Generate PCA factors (timescale-matched)
python scripts/find_optimal_horizon.py --H 8 --K 30 --weight-method sqrt

# Output: data/features/pca_factor_H8_K30_sqrt.csv
```

### Model Training

```bash
# 3. Train joint models
python scripts/find_optimal_H_joint.py --H 8 --n-bootstrap 1000

# Output: data/features/joint_models/joint_model_H8.json
```

### Backtesting

See [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) for complete workflow.

```python
from slipstream.portfolio import optimize_portfolio

# Beta-neutral optimization (closed-form)
w = optimize_portfolio(
    alpha=alpha_total,  # α_price - F_hat
    beta=beta_exposures,
    S=covariance_matrix,
    leverage=1.0,
)

# Verify constraints
assert abs(w @ beta_exposures) < 1e-6  # Beta neutral
assert abs(np.abs(w).sum() - 1.0) < 1e-3  # Leverage = 1
```

## 📚 Documentation

All documentation has been consolidated into a single file:

| Document | Description |
|----------|-------------|
| [`DOCUMENTATION.md`](docs/DOCUMENTATION.md) | Complete strategy specification, model training, backtesting, and data pipeline guides. |

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Test portfolio optimizer
python tests/test_portfolio_optimizer.py

# Expected output:
# ✓ Basic beta-neutral optimization
# ✓ Cost-aware optimization
# ✓ Beta neutrality enforced
# ✓ Costs reduce turnover
```

## 🛠️ Development

```bash
# Lint
uv run ruff check

# Type checking
uv run mypy src/

# Format
uv run ruff format src/ tests/
```

## 🎓 Key Concepts

### Beta-Neutral Portfolio

The optimizer solves:

```
max_w [ w^T α - 0.5 w^T S w - C(w - w_old) ]
subject to: w^T β = 0
```

Where:
- `α = α_price - F_hat`: Combined alpha (price - funding)
- `β`: Market beta exposures from PCA
- `S`: Total covariance (price + funding)
- `C(Δw)`: Transaction costs (linear + impact)

**Closed-form solution:**
```
w* = S^{-1} (α - λ β)
```

Where `λ` is the Lagrange multiplier enforcing `w^T β = 0`.

### Transaction Costs

Power-law model:
```
C(Δw) = Σ |Δw_i| * fee_rate_i + Σ λ_i |Δw_i|^1.5
```

- **Linear term**: Exchange fees (~2 bps)
- **Impact term**: Market impact (calibrated to liquidity)

### Timescale Matching

PCA parameters adapt to holding period:
- **Frequency**: PCA computed at H-hour intervals
- **Lookback**: Window = K × H (typically K=30)
- **Rationale**: Match factor dynamics to trading frequency

## 🚦 Current Status

**Production Ready:**
- ✅ Data pipeline (API + S3)
- ✅ Signal generation
- ✅ Model training (alpha + funding)
- ✅ Portfolio optimization
- ✅ Backtesting framework
- ✅ Full backtest on historical data
- 🔄 Cost parameter calibration from L2 orderbook
- 🔄 Production prediction pipeline

**Future:**
- 📋 Live trading integration
- 📋 Automated retraining pipeline
- 📋 Risk monitoring dashboard

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{slipstream2025,
  title={Slipstream: Beta-Neutral Statistical Arbitrage for Perpetual Futures},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/slipstream}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Contact

Questions? Open an issue or reach out via [your contact method].

---

**⚠️ Disclaimer**: This software is for research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results.
