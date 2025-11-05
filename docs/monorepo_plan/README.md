# Slipstream Monorepo Transformation Plan

This folder captures the multi-sprint roadmap for turning the existing Gradient-focused codebase into a cohesive Slipstream monorepo that can host multiple systematic trading strategies. The plan emphasises incremental refactors, heavy code reuse, and continuous validation so that the Gradient live stack and backtests keep working throughout the migration.

## Current State
- Single-strategy bias: repo structure and tooling assume Gradient is the only strategy.
- Mixed concerns: `src/slipstream` blends shared utilities with Gradient-specific logic.
- Operational scripts and docs are tightly coupled to Gradient deployment.
- CI/testing focuses on a narrow slice (`test_trade_and_read.py`, `test_full_workflow.py`), leaving gaps for modular reuse.

## Target End State
- Monorepo organised by **core services** (data, execution, risk, tooling) and **strategy packages**.
- Strategy-agnostic backtesting and live orchestration layers with strategy-specific plug-ins.
- Gradient strategy fully migrated to the new layout with green tests (unit, integration, live dry-run).
- Onboarding playbook and templates for future strategies (e.g., momentum variants, mean reversion, carry).

## Guiding Principles
1. **Staged refactor:** make structure changes in small, reversible steps; de-risk via feature flags/double writes.
2. **Code reuse first:** lift and share the Gradient building blocks (data fetchers, signal engines, execution) before rewriting.
3. **Testing as a rail:** expand automated tests before and after moves to catch regressions early.
4. **Documentation alongside code:** update runbooks, diagrams, and onboarding docs in each sprint.
5. **Operational continuity:** live Gradient trading and emergency tooling must stay functional in every sprint.

## Sprint Structure
Four focused sprints (approx. 1–1.5 days each) move us from preparation to a production-ready monorepo:

1. **Monorepo Foundation & Governance**
2. **Shared Core Services Extraction**
3. **Gradient Migration & Certification**
4. **Multi-Strategy Enablement & Final QA**

Each sprint document details goals, scope, backlog items, dependencies, and Definition of Done.

## Cross-Cutting Enablers
- **Version control strategy:** short-lived feature branches merged via squash after review.
- **CI upgrades:** add lint/type checks and strategy matrix to GitHub Actions/CI pipeline.
- **Telemetry:** extend logging/metrics to differentiate strategy namespaces.
- **Security & secrets:** centralise credential management (vault/ENV) with per-strategy overrides.

## Next Steps
Review the sprint files in this folder, align stakeholders (quant research, engineering, devops), and lock scheduling. Start with Sprint 1 once backlog grooming and resource assignment are complete.

