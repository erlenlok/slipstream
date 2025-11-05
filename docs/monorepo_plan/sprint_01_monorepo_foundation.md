# Sprint 01 — Monorepo Foundation & Governance

**Duration:** ~1 day  
**Objective:** Establish the structural and tooling baseline for a multi-strategy Slipstream monorepo while keeping the existing Gradient workflow untouched.

## Outcomes
- Top-level repository layout supports shared core libraries and per-strategy packages.
- Build tooling (uv/poetry, lint, tests) updated to reflect the new layout.
- Clear ownership map and contribution guidelines for future strategy teams.

## Scope
1. **Repo Layout**
   - Introduce `src/slipstream/core/` for reusable services (data, execution, risk).
   - Create `src/slipstream/strategies/` namespace with placeholder packages (`gradient`, `template`).
   - Move legacy shared modules (`common`, `portfolio`, `signals`, `costs`, `funding`) under `core/` with compatibility imports to avoid breakage.
2. **Tooling & Automation**
   - Update `pyproject.toml` / `uv` settings to include new packages.
   - Refresh lint/type configs (ruff, mypy) with monorepo-aware paths.
   - Adjust test discovery to run under both `core` and `strategies`.
3. **Governance & Docs**
   - Draft CONTRIBUTING guidelines covering package ownership, review expectations, and release process.
   - Document repo layout diagram and migration rules in `docs/monorepo_plan`.

## Backlog Highlights
- [x] Create namespace packages and `__init__.py` shims for backwards compatibility.
- [x] Introduce `scripts/monorepo/check_structure.py` to validate layout in CI.
- [ ] Update CI pipeline matrix (lint, unit tests) to reflect new paths.
- [x] Write `CONTRIBUTING.md` and update `README.md` high-level architecture section.
- [ ] Align logging directory structure (`/var/log/slipstream/<strategy>`).

## Dependencies & Coordination
- Align with DevOps on CI updates and secret storage for future strategies.
- Coordinate with research to pause disruptive merges during directory reshuffle.

## Definition of Done
- Repo builds/tests pass locally and in CI using new structure.
- Docs merged: architecture diagram + contributing guide.
- No regression in current Gradient live/backtest commands (verified via smoke run).

## Risks & Mitigations
- **Risk:** Breaking relative imports during moves → **Mitigation:** keep compatibility imports and run full test suite before merge.
- **Risk:** CI drift → **Mitigation:** pair with DevOps on pipeline update PR.

## Metrics
- 100% pass rate on existing tests post-structure change.
- Zero open TODO items marked “breaking” in Gradient modules.

## Status (2025-11-05)
- ✅ Repository reshaped with `core/` and `strategies/` namespaces, including Gradient migration and shims.
- ✅ Full test suite (`uv run pytest`) green after path updates.
- ✅ CONTRIBUTING guide, monorepo structure checker, and README layout refresh merged.
