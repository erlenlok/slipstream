# Sprint 04 – Retro & QA Notes

## What Went Well
- Strategy registry now exposes metadata + CLI hooks, making it obvious how to register future strategies.
- Template package matured (config + signals + backtest + CLI) so teams have a real skeleton to copy.
- Unified `scripts/strategies/run_backtest.py` simplified QA: one command fans out to any registered strategy.

## What To Improve
- Need automated docs generation for strategy metadata so README badges stay in sync.
- Template tests are still lightweight; future work should add example pytest modules.
- Consider adding smoke tests for `scripts/strategies/run_backtest.py` in CI to avoid regressions.

## Action Items
1. Add pytest coverage for the template package once research finalizes requirements.
2. Hook registry metadata into CI to auto-populate documentation tables/badges.
3. Pair with DevOps to template per-strategy dashboards alongside the secrets conventions.
