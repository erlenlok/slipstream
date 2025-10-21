# Slipstream

Trading and research framework for Hyperliquid perpetual futures. Includes utilities for downloading hourly OHLCV, funding, and return data from the Hyperliquid API. Built around a `uv` managed environment with a `src/` layout so data analysis notebooks and scripts share the same package.

## Repo layout

- `src/slipstream/` – reusable data loading code (Python package).
- `scripts/` – shell helpers and ad-hoc commands (empty scaffold).
- `data/` – raw and processed CSV outputs (ignored by git).
- `notebooks/` – exploratory analysis notebooks.
- `tests/` – unit tests (empty scaffold).

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

4. Fetch hourly candles, funding, and merged returns for BTC:

   ```bash
   uv run hl-load --coin BTC --days 90 --out-prefix data/hl
   ```

## Development tooling

- `uv run ruff check` – lint the package.
- `uv run pytest` – run tests (add them to `tests/`).
- `uv run python -m slipstream --help` – view CLI options.

The package exposes `slipstream.__main__` so `python -m slipstream` works out of the box, enabling easy integration with notebooks and other tooling.
