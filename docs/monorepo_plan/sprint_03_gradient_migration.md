# Sprint 03 — Gradient Migration & Certification

**Duration:** ~1 day  
**Objective:** Port the Gradient live and backtest stacks onto the new monorepo architecture, ensuring seamless operations and automated verification.

## Outcomes
- Gradient strategy lives under `slipstream.strategies.gradient` using shared core services.
- Backtesting and live execution commands work end-to-end via the new entry points.
- Regression guardrails (tests + monitoring) confirm parity with pre-migration behaviour.

## Scope
1. **Strategy Package Migration**
   - Move remaining Gradient-specific logic (momentum config, universe selection, cron scripts) into `strategies/gradient`.
   - Update CLIs (`uv run python -m slipstream.strategies.gradient.<cmd>`) with compatibility wrappers. ✅
   - Relocate docs/scripts to `docs/strategies/gradient` and `scripts/strategies/gradient`.
2. **Backtest & Live Parity**
   - Refactor backtest harness to consume `core` interfaces (data loader, portfolio, execution sim).
   - Validate emergency stop, rebalance, and alerting flows with dry-run/live smoke tests.
   - Ensure logging/metrics segregate by strategy namespace.
3. **Testing & Monitoring**
   - Expand integration suite: 
     - deterministic backtest run vs. baseline snapshot,
     - simulated rebalance verifying orders/deltas,
     - emergency stop dry-run.
   - Wire CI to run Gradient tests in both dry-run and mock-live modes.

## Backlog Highlights
- [x] Update `config/gradient_live.*` to new layered config structure.
- [x] Provide backwards-compatible CLI wrappers (deprecated warnings) for existing ops scripts.
- [ ] Capture baseline metrics (universe size, turnover, PnL) and diff after migration. *(Tooling landed via `scripts/strategies/gradient/capture_baseline.py`)*
- [ ] Update documentation (deployment, monitoring) to reflect new paths/commands.
- [ ] Pair with ops for one supervised live rebalance after cutover.

## Dependencies & Coordination
- Requires core services from Sprint 02 to be stable.
- Coordinate with DevOps for updated cron jobs and logging destinations.
- Align with quant research to validate signal integrity post-move.

## Definition of Done
- `uv run python -m slipstream.strategies.gradient.rebalance` succeeds in staging with no regressions.
- Backtest CLI produces identical summary stats to pre-migration runs (within tolerance).
- All Gradient docs/reference scripts point at the new locations; old commands emit deprecation warnings.

## Risks & Mitigations
- **Risk:** Latent coupling in Gradient modules causing runtime errors post-move.  
  **Mitigation:** incremental migration with feature flags; keep old modules until parity confirmed.
- **Risk:** Live trading disruption.  
  **Mitigation:** schedule migration during maintenance window; maintain rollback path (old branch + configs).

## Metrics
- Integration test pipeline green across backtest/live/dry-run tasks.
- Live rebalance smoke (dry-run) completes in <5 min with zero open orders.
- Ops checklist signed off; Telegram alerts verified end-to-end.

## Status (2025-11-05)
- ✅ Legacy import paths now emit deprecation warnings while forwarding to `slipstream.strategies.gradient.*` modules.
- ✅ Layered config loader integrated with Gradient live configuration; regression tests updated.
- ⚙️ Backtest/execution parity verification and telemetry capture to be completed in collaboration with ops.
