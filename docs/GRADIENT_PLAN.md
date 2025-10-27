# Gradient Strategy – Implementation Blueprint

This document captures the organization plan for the new **Gradient** strategy so we can continue implementation even if the session resets.

## 1. High-Level Concept

Gradient maintains a constant dollar-volatility exposure split evenly between:

- Long positions in the **top five** assets with the strongest positive trend strength.
- Short positions in the **bottom five** assets with the strongest negative trend strength.

Trend strength is the summed momentum of **volatility-normalized log returns** evaluated across lookbacks {2, 4, 8, …, 1024}.

## 2. Package Layout

Add a sibling package to the existing Slipstream modules:

```
src/slipstream/
├── gradient/
│   ├── __init__.py
│   ├── universe.py        # Universe selection + volatility scaling utilities
│   ├── signals.py         # Trend strength computation
│   ├── portfolio.py       # Position sizing + rebalancing logic
│   └── backtest.py        # Thin wrapper reusing shared backtest plumbing
└── common/ (new)
    ├── __init__.py
    ├── returns.py         # Vol-normalized returns, log return helpers
    └── volatility.py      # Annualization, dollar-vol targeting helpers
```

`common/` houses any helpers shared between Slipstream and Gradient to prevent duplication.

## 3. Data Flow & Artifacts

- Reuse existing market data (`data/market_data/`).
- Store Gradient-specific outputs in `data/gradient/`:
  - `signals/` – serialized trend-strength panels.
  - `positions/` – daily/hourly allocations.
  - `backtests/` – simulation results.

## 4. Scripts & CLI Integration

Introduce dedicated scripts mirroring the existing tooling:

- `scripts/gradient_compute_signals.py`
- `scripts/gradient_run_backtest.py`

Expose CLI entry points via `pyproject.toml`, e.g.:

```
[project.scripts]
gradient-signals = "slipstream.gradient.scripts:compute_signals_cli"
gradient-backtest = "slipstream.gradient.scripts:run_backtest_cli"
```

(Exact module paths will follow final code structure.)

## 5. Documentation Updates

- Update `README.md` with a short Gradient overview and link.
- Add `docs/GRADIENT.md` covering concept, workflow, commands, and configuration.
- Mention shared-helper moves (common utilities) in consolidated docs if needed.

## 6. Testing Strategy

- Create `tests/gradient/` with coverage for:
  - Signal aggregation across lookbacks.
  - Universe selection and volatility weighting.
  - Portfolio construction producing balanced long/short exposures.
  - Integration smoke test using synthetic data.

Reuse fixtures or helper utilities from the existing test suite wherever possible.

## 7. Notebooks (Optional)

- Keep exploratory research under `notebooks/gradient/` to avoid mixing with Slipstream notebooks.

## 8. Implementation Order

1. Create shared `common/` utilities and wire existing Slipstream imports to the new location if required.
2. Scaffold Gradient package modules with clear interfaces.
3. Implement CLI scripts and update `pyproject.toml`.
4. Add documentation and tests, ensuring CI still passes.

With this blueprint recorded, we can safely resume development even after a fresh session.

