# Slipstream

Trading and research framework for Hyperliquid perpetual futures.

**Features:**
- 📊 Multi-source data acquisition (API + S3 historical archive)
- 📈 EWMA-based idiosyncratic momentum signal generation
- 🔬 PCA-based market factor decomposition with timescale matching
- 📓 Research notebooks with single source of truth for signals
- ⚡ Resumable S3 downloader for full historical coverage (Oct 2023+)

## Repo layout

- **`src/slipstream/`** – Importable Python package
  - `signals/` – Signal generation (EWMA momentum, PCA decomposition)
  - `portfolio/` – (future) Portfolio construction and optimization
- **`scripts/`** – Data acquisition and feature engineering
  - `data_load.py` – API-based data fetcher
  - `fetch_s3_historical.py` – S3 historical data downloader
  - `build_pca_factor.py` – PCA factor computation
  - `find_optimal_horizon.py` – H* optimization via timescale matching
- **`notebooks/`** – Research and backtesting
- **`data/`** – Data storage (git-ignored)
  - `market_data/` – API data (~7 months)
  - `s3_historical/` – S3 historical data (Oct 2023+)
  - `features/` – Computed features (PCA factors, signals)
- **`docs/`** – Documentation
- **`tests/`** – Unit tests

## Getting started

1. Install the [uv CLI](https://docs.astral.sh/uv/getting-started/installation/).
2. Create a project-local virtual environment and sync deps:

   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv sync
   ```

   The `.python-version` file pins Python 3.11 so `uv` (and tools like `pyenv`) select an interpreter consistently.

3. Print the live perp universe:

   ```bash
   uv run python -m slipstream
   ```

4. Fetch recent data via API (~7 months):

   ```bash
   # Single coin
   uv run hl-load --coin BTC --days 90

   # All markets
   uv run hl-load --all --days 30
   ```

5. (Optional) Download full historical data from S3:

   ```bash
   # Setup (one-time)
   sudo apt install lz4
   aws configure  # Enter AWS credentials

   # Download historical candles (Oct 2023 - present)
   python scripts/fetch_s3_historical.py --start 2023-10-01 --end 2025-03-27

   # See S3_SETUP_README.md for detailed guide
   ```

6. Generate signals and analyze in notebooks:

   ```bash
   # Install notebook dependencies
   uv sync --extra notebook

   # Launch Jupyter
   uv run jupyter notebook notebooks/momentum_panel_test.ipynb
   ```

## Development tooling

- `uv run ruff check` – lint the package.
- `uv run pytest` – run tests (add them to `tests/`).
- `uv run python -m slipstream --help` – view CLI options.

The package exposes `slipstream.__main__` so `python -m slipstream` works out of the box, enabling easy integration with notebooks and other tooling.
