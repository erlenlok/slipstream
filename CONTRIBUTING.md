# Contributing to Slipstream

Thanks for helping evolve Slipstream into a multi-strategy trading platform! This guide outlines the workflow and expectations for contributors.

## Repository Layout

```
src/slipstream/
├── core/          # Shared services (data, signals, execution, risk)
└── strategies/    # Strategy packages (gradient, template, ...)
```

Legacy import shims (e.g. `slipstream.common`) forward to the new `slipstream.core` modules. When adding new code, prefer the `core` and `strategies` namespaces.

## Getting Started

1. Install the toolchain with [uv](https://docs.astral.sh/uv/):
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv sync --all-extras
   ```
2. Run the quality gates before opening a PR:
   ```bash
   uv run ruff check .
   uv run pytest
   ```
3. For strategy work, execute the relevant integration tests (see `tests/strategies/`).

## Branching & Reviews

- Create feature branches from `main` (`feature/<name>`). Keep them short-lived (<1 day).
- Open draft PRs early so we can align on direction.
- Every PR requires at least one reviewer from the owning team:
  - `core/*` modules → Core Services team.
  - `strategies/gradient/*` → Gradient Strategy team.
- Squash merge after approvals. Use descriptive commit summaries (max 72 chars).

## Code Style & Standards

- Python 3.10+, Ruff for linting, Black-style formatting (enforced via Ruff rules).
- Type annotations for public interfaces; leverage `typing.Protocol` for strategy plug-ins.
- Keep functions small and testable. Break complex notebooks into scripts + tests.
- Document important modules with a module docstring and docstrings for public APIs.

## Testing Expectations

| Change Type | Minimum Tests |
|-------------|---------------|
| Core utilities | Unit tests + regression fixtures |
| Strategy logic | Unit tests + strategy integration test |
| CLI / scripting | Smoke test or documented manual steps |
| Config / infra  | Update docs + add config schema validation |

When moving code between namespaces, add compatibility tests that assert old imports still resolve.

## Documentation

- Update `README.md` and `docs/` when behaviour or entry points change.
- Each sprint maintains a log in `docs/monorepo_plan/`—append notes to the relevant sprint file.
- For new strategies, add an onboarding guide under `docs/strategies/<name>/`.

## Secrets & Config

- Do not hard-code secrets. Use environment variables or the shared vault.
- Config files live in `config/`. Global settings go in `config/global.yml`; strategy overrides in `config/<strategy>.yml` (coming in Sprint 02).

## Release Process

1. Tag release (`vX.Y.Z`) after CI passes and trading ops sign-off.
2. Update changelog (TBD) and notify #trading-ops.
3. Monitor the first live rebalance after rollout.

## Support

- Bugs: open an issue with logs, strategy name, and reproduction steps.
- Questions: use `#slipstream-dev` Slack channel or tag the owning team.

Happy trading! 🏎️💨
