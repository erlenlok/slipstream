# Sprint 02 — Shared Core Services Extraction

**Duration:** ~1 day  
**Objective:** Carve out strategy-agnostic services from the current Gradient implementation and solidify reusable APIs for data, analytics, execution, and risk management.

## Outcomes
- `slipstream.core` package provides clean interfaces with unit coverage.
- Gradient code depends on the new abstractions without changing behaviour.
- Common config schema separates global settings, strategy overrides, and secrets.

## Scope
1. **Core Modules**
   - Data ingestion (`core.data`) wrapping Hyperliquid clients, caching, candle handling.
   - Signal pipelines (`core.signals`) with reusable transforms (momentum, volatility, filters).
   - Execution services (`core.execution`) exposing order routing and monitoring.
   - Risk/portfolio utilities (`core.risk`, `core.portfolio`).
2. **Configuration & Secrets**
   - Implement layered config loader (`global.yml`, `<strategy>.yml`, env overrides).
   - Document secret handling (vault/ENV) and integrate with existing `load_config`.
3. **Testing & Validation**
   - Extract deterministic fixtures from Gradient tests for shared modules.
   - Add contract tests ensuring old Gradient modules produce same outputs via core APIs.

## Backlog Highlights
- [x] Move reusable pieces from `slipstream.strategies.gradient` into `slipstream.core`.
- [ ] Create interface definitions (`Protocol` / ABC) for strategies to plug into.
- [ ] Build sample data adapters (live, historical) with caching hooks.
- [ ] Expand unit tests covering new core modules and backward compatibility layer.
- [ ] Update docs: core module overview + dependency graph.

## Dependencies & Coordination
- Requires Sprint 01 structure + CI updates merged.
- Coordinate with research to confirm shared signal definitions and data requirements.

## Definition of Done
- Gradient strategy imports from `slipstream.core` for data/signals/execution without regressions.
- New unit tests cover ≥80% of moved code; integration smoke passes.
- Config loader supports both legacy JSON and new YAML, with migration guide drafted.

## Risks & Mitigations
- **Risk:** Hidden strategy-specific edge cases baked into shared modules.  
  **Mitigation:** add golden-metric comparisons (e.g., signal counts, portfolio stats) before/after extraction.
- **Risk:** Config churn confusing ops.  
  **Mitigation:** maintain dual loaders + docs until full cutover.

## Metrics
- Matching outputs (within tolerance) for: signal universe size, target positions, executed deltas vs. pre-refactor baselines.
- Unit test coverage delta for `slipstream.core` > +30% vs. Sprint 01.

## Status (2025-11-05)
- ✅ Layered config loader (`slipstream.core.config`) online with unit coverage and integrated into Gradient live config.
- ✅ Legacy docs/scripts updated to refer to `slipstream.strategies.gradient.*` namespaces.
- ⏳ Interface protocols and data adapter abstractions planned for a follow-up slice.
