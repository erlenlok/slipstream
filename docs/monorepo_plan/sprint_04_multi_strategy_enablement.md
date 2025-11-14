# Sprint 04 — Multi-Strategy Enablement & Final QA

**Duration:** ~1 day  
**Objective:** Finalise monorepo capabilities for hosting multiple strategies, deliver onboarding assets, and harden the platform through testing and documentation.

## Outcomes
- Strategy template and onboarding guide published for new systematic strategies.
- Cross-strategy tooling (backtest matrix, risk dashboard, alerting) operational.
- Gradient strategy certified post-migration with final regression sign-off.

## Scope
1. **Strategy Onboarding Toolkit**
   - Create `strategies/template` package with sample signals, config, and CLI stubs.
   - Provide cookiecutter or script to scaffold new strategies with wiring to core services.
   - Document coding standards, testing expectations, and deployment checklist for new strategies.
2. **Unified Tooling & QA**
   - Extend backtest runner to accept strategy IDs and schedule regression suites.
   - Build consolidated monitoring dashboards/log structure per strategy.
   - Finalise CI matrix: unit tests, integration tests, lint, type checks per strategy.
3. **Final Certification**
   - Run Gradient live dry-run and backtest suites; capture artefacts in `docs/monorepo_plan/certification`.
   - Host post-mortem/retro and update docs with lessons learned.

## Backlog Highlights
- [x] Implement strategy registry (`strategies/__init__.py`) enumerating available strategies + metadata.
- [x] Build `scripts/strategies/run_backtest.py` supporting `--strategy` flag.
- [x] Add documentation for secrets management per strategy (env var naming convention).
- [x] Publish onboarding checklist and retro report.
- [x] Update README badges/status to reflect multi-strategy readiness.

## Dependencies & Coordination
- Requires Gradient fully migrated (Sprint 03 DoD met).
- Coordinate with analytics/devops for dashboard work.
- Engage stakeholders for final sign-off (quant, trading ops, compliance).

## Definition of Done
- New strategy template can run lint + unit tests out of the box.
- CI pipelines green across full matrix; coverage reports generated.
- Gradient regression artefacts stored, with sign-off from trading ops.

## Risks & Mitigations
- **Risk:** Template diverges from real-world needs.  
  **Mitigation:** workshop with research/engineering to validate before publishing.
- **Risk:** QA gaps for future strategies.  
  **Mitigation:** embed mandatory test harness and documentation in template skeleton.

## Metrics
- Time to scaffold new strategy < 10 minutes.
- 100% of CI jobs tagged per strategy passing.
- Gradient certification report completed and archived.
