# Brawler Multi-Instrument Sprint Plan

## Current Location
- Spec is complete (`docs/BRAWLER_SPEC.md`), but there is no sequenced delivery roadmap or multi-asset rollout plan.
- Core plumbing (`src/slipstream/strategies/brawler`) exists: configs/states, Binance + Hyperliquid streams, quote engine, and CLI wiring.
- Multi-asset hooks are nominal (config accepts many assets, streams subscribe to symbol lists), yet coordination controls are missing.
- Kill-switch config fields (`risk`, `kill_switch`, `max_basis_deviation`) are unused in the engine; feed staleness timestamps are recorded but never enforced.
- Order management cancels all per-symbol orders on every update, risking quote thrash as more instruments are added.
- There is no reference multi-asset config, ops playbook, or live-observability story; validation is expected to happen with tiny size directly in prod.
- Asset selection is ad hoc: we lack tooling to screen Hyperliquid perps for basis/volatility parity and HL-vs-CEX spread edge before turning them on.

## Key Gaps Before Multi-Instrument Readiness
- Missing safety gates: basis-depeg, feed staleness, disconnect, and reduce-only recovery logic are unimplemented despite being part of the spec.
- No cross-asset risk coordination: inventory, capital, and sizing are managed per symbol without awareness of aggregate exposure or concurrency.
- Execution layer is over-simplified (cancel-all before every replace, no throttling/jitter) and may overload Hyperliquid APIs when symbols scale.
- Observability + documentation gaps: no per-asset metrics/log taxonomy, no sample configs, and no runbook describing prod bring-up with multiple instruments.
- Testing philosophy relies on “test in prod with minimal notional”; we still need deterministic checks (unit/simulation) for safety-critical math.
- Candidate discovery is missing: we need quantitative screens that prove a listing’s HL mid tracks Binance, realized vol is comparable, and HL spreads are materially wider before whitelisting.

## Sprint Plan

### Sprint 1 — Spec Parity & Single-Asset Hardening
**Objective:** Bring the engine up to parity with the written spec for one instrument, so later multi-asset work builds on a safe core.
- Implement kill switches for volatility, basis deviation, feed lag, and disconnect using `AssetState` timestamps plus `BrawlerKillSwitchConfig`.
- Surface suspension reasons and auto-resume behavior through structured logs/metrics; add config-driven tick-size tolerances for cancels/replaces.
- Publish a single-asset reference YAML config demonstrating every field plus minimal prod-testing notes (tiny size, manual monitoring).
- Ship the candidate-screening CLI + scoring heuristic (basis drift, sigma parity, HL-vs-CEX spread edge, depth) so new assets must clear quantitative gates before entering the config. Capture the results in the ops checklist.
- Build a lightweight recorder that mirrors HL + Binance BBO streams into timestamped CSV/Parquet files so the scanner/watchlist tooling has real data to chew on (document rotation/retention alongside the CLI).

### Sprint 2 — Multi-Asset Data & State Plane
**Objective:** Ensure every instrument has clean data plumbing and restartable state.
- Normalize symbol routing: authoritative mapping between CEX symbols and Hyperliquid coins, validation errors on mismatch/duplicates.
- Bound queues per symbol and tag quotes with latency metadata to monitor feed health.
- Add pluggable inventory sources (REST snapshot + fill deltas) and persist last-known basis/inventory so restarts across multiple instruments are deterministic.
- Extend the candidate scanner into a rolling “watchlist” job that logs spread/vol/basis deltas for every enabled coin, producing a ranked report operators can use to decide the next onboarding target.
- Automate the recorder + watchlist pair (cron/CI) so fresh HL/CEX captures feed into the nightly report without manual steps; define who reviews the artifacts and where they live.

### Sprint 3 — Cross-Asset Risk & Execution Controls
**Objective:** Coordinate quoting/risk budgets when several instruments run simultaneously.
- Introduce a portfolio controller that enforces global inventory cap, per-asset order-size scaling, and reduce-only transitions when aggregate exposure breaches limits.
- Refine order management: replace only the touched side, add price-distance + time guards, and stagger requests so multiple instruments do not hammer `cancel_all`.
- Wire kill-switch fan-out (e.g., a feed outage can pause a single asset or the entire engine based on policy) and document operator levers for minimal-size prod tests.

### Sprint 4 — Deterministic Verification & Dry-Run Harness
**Objective:** Provide non-backtest validation that still de-risks prod rollouts.
- Build a data replay harness that feeds recorded Binance/Hyperliquid streams into the engine to verify spread math, suspension logic, and concurrency without needing historical PnL sims.
- Author scenario tests (vol spike, feed stall, inventory breach, disconnect/reconnect) to prove kill switches and resume logic behave identically across instruments.
- Automate smoke runs in CI so every change replays short multi-asset traces before tagging a build for prod testing.

### Sprint 5 — Productionization & Multi-Instrument Launch
**Objective:** Deliver the docs, observability, and operational guardrails needed for live rollout with more than one instrument.
- Ship per-asset metrics (basis, spread, sigma, inventory, suspension states) and structured logs/alerts so operators can monitor tiny-size prod sessions.
- Package the CLI/runbook: config discovery, env-var overrides, deployment checklist, and “minimal-size prod test” procedures (order sizing, health checks, rollback).
- Execute staged go-live: shadow quotes on a second instrument, then graduate to simultaneous quoting with strict notional caps; capture learnings for future asset onboarding.

> **Testing philosophy:** Backtests are intentionally out of scope. Each sprint hardens deterministic checks and data replay so the final sign-off happens via controlled, minimal-size production sessions—the only environment that truly reflects MM behavior.
