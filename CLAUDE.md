# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slipstream is a Python trading and research framework for the Hyperliquid perpetual futures exchange. The project uses a `src/` layout with `uv` for dependency management, separating data acquisition tooling from core trading logic.

## Architecture

### Directory Structure

- **`src/slipstream/`**: Trading logic, strategy implementations, position management, order execution (currently empty - add your trading utilities here)
- **`scripts/`**:
  - `data_load.py` - Data acquisition utility for downloading hourly OHLCV, funding, and return data
  - `build_pca_factor.py` - Rolling PCA market factor computation from returns data
- **`notebooks/`**: Research and backtesting analysis
- **`data/`**: (git-ignored)
  - `market_data/` - Raw market data CSVs (candles, funding, merged returns)
  - `features/` - Computed features (PCA factors, etc.)

### Data Pipeline Components

**`scripts/data_load.py`** - Market data acquisition:
- `fetch_all_perp_markets()` - Enumerates all perp dex namespaces and unions their live universes
- `fetch_candles_1h()` - Paginates 1h candles in 120-day chunks to stay under API limits (~5k candles)
- `fetch_funding_hourly()` - Fetches hourly funding data with pagination
- `compute_hourly_log_returns()` - Vectorized computation of hourly log returns from 1h close prices
- `build_datasets()` - Orchestrates fetching candles+funding, computing returns, aligning, and writing CSVs to `data/market_data/`
- `build_for_universe()` - Fetches and writes datasets for all live markets (with optional dex filtering)

**`scripts/build_pca_factor.py`** - Factor construction:
- `load_all_returns()` - Loads all merged return files from `data/market_data/` into wide matrix
- `resample_to_daily()` - Aggregates hourly log returns to daily
- `compute_rolling_pca_loadings()` - Computes rolling PCA with PC1 loadings as market factor weights, handling variable asset counts
- Outputs to `data/features/pca_factor_loadings.csv` with daily loadings per asset + metadata

### HTTP Layer

The project implements retry logic with exponential backoff for retryable HTTP status codes (429, 500, 502, 503, 504). All API calls include a `REQUEST_PAUSE_SECONDS` (0.25s) throttle to respect rate limits. The base API endpoint is `https://api.hyperliquid.xyz/info`.

### Data Pipeline

1. Fetch candles and funding data in parallel using `asyncio.gather()`
2. Compute log returns: `log_returns = np.log1p(candles["close"].pct_change())`
3. Create aligned hourly index and join all datasets
4. Write three CSV outputs: `{prefix}_{coin}_candles_1h.csv`, `{prefix}_{coin}_funding_1h.csv`, `{prefix}_{coin}_merged_1h.csv`

## Common Commands

### Setup
```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

### Data Loading
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

### Factor Construction
```bash
# Build PCA market factor (default: 60-day rolling window)
python scripts/build_pca_factor.py

# Custom parameters
python scripts/build_pca_factor.py --window 30 --min-assets 5 --min-periods 20

# Outputs to data/features/pca_factor_loadings.csv
```

### Development
```bash
# Lint
uv run ruff check

# Run tests
uv run pytest
```

## Development Notes

- Python version is pinned to 3.11 via `.python-version`
- **Code organization**:
  - `src/slipstream/` - Trading and strategy code (importable package)
  - `scripts/` - Data acquisition and feature engineering scripts
  - `notebooks/` - Research and backtesting
  - `data/market_data/` - Raw market data CSVs (git-ignored)
  - `data/features/` - Computed features (git-ignored)
- The `hl-load` CLI entry point wraps `scripts/data_load.py` via `python -m slipstream`
- Data files use naming convention: `{COIN}_{type}_1h.csv` where type is `candles`, `funding`, or `merged`
- When adding data loading functionality, extend `scripts/data_load.py` following the async pattern
- When adding new features, create scripts in `scripts/` and output to `data/features/`
- When adding trading logic, create modules in `src/slipstream/` (importable from notebooks)
- All timestamps are handled in UTC and converted to milliseconds for API calls via `ms(dt)`
- Candle timestamps are floored to the hour via `to_hour(dt_ms)`
