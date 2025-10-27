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
src/slipstream/gradient/
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
