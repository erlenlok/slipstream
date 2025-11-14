# Sprint 04 – Strategy Onboarding Checklist

Use this checklist when spinning up a new Slipstream strategy. It mirrors the structure of `src/slipstream/strategies/template/`.

## 1. Prep
- [ ] Decide on a strategy slug (e.g., `carry`, `meanrev`) and add it to `slipstream/strategies/__init__.py`.
- [ ] Copy the template package into `src/slipstream/strategies/<slug>/` and rename modules/CLI entry points.
- [ ] Duplicate the template tests/notebooks if needed for research sign-off.

## 2. Configuration & Secrets
- [ ] Create `config/<slug>.yml` with strategy defaults.
- [ ] Add required env vars following `docs/monorepo_plan/strategy_secrets.md`.
- [ ] Document any new secrets in `.env.example` + deployment runbooks.

## 3. Signals & Backtest
- [ ] Implement signal generation helpers (start from `template/signals.py`).
- [ ] Wire a backtest loop (see `template/backtest.py`) and expose a `run_backtest_cli`.
- [ ] Register the CLI in `STRATEGY_REGISTRY` so `scripts/strategies/run_backtest.py` can dispatch to it.

## 4. Tooling + Tests
- [ ] Add unit tests for critical components (signals, portfolio transforms).
- [ ] Extend integration tests/backtests to cover the new strategy.
- [ ] Ensure `uv run python scripts/strategies/run_backtest.py --strategy <slug> -- ...` succeeds.

## 5. Documentation & Handover
- [ ] Update `README.md` (Strategy section) with a short description + pointers.
- [ ] Add a section to `docs/strategies/<slug>/README.md` covering workflow, configs, and alerting.
- [ ] Capture onboarding notes / known issues in `docs/monorepo_plan/sprint_04_retro.md` for future teams.
