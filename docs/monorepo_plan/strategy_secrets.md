# Strategy-Specific Secrets & Env Vars

Sprint 04 introduces a consistent way to scope credentials and private config per strategy. Use this document as the canonical reference when onboarding new teams.

## Naming Convention

- Pattern: `SLIPSTREAM_<STRATEGY>_<SECRET_NAME>`
- Always use uppercase letters and snake case.
- `<STRATEGY>` **must** match the registry key (e.g., `GRADIENT`, `TEMPLATE`, `ALPHA2`).

| Example | Usage |
|---------|-------|
| `SLIPSTREAM_GRADIENT_HL_API_KEY` | Hyperliquid API key for Gradient live trading |
| `SLIPSTREAM_GRADIENT_TELEGRAM_TOKEN` | Alerting bot token scoped to Gradient |
| `SLIPSTREAM_TEMPLATE_RESEARCH_S3` | Placeholder credential when cloning the template |

## Resolution Order

When `slipstream.core.config.load_layered_config()` is invoked, secrets resolve in the following order:

1. **Global defaults** (`config/global.yml`)
2. **Strategy override** (`config/<strategy>.yml`)
3. **Environment variables** following the naming convention above
4. **Runtime overrides** (CLI flags / kwargs)

Environment variables always win, which keeps credentials out of VCS while still allowing ad-hoc overrides during testing.

## Adding a New Secret

1. Pick a descriptive `<SECRET_NAME>` (e.g., `EXCHANGE_API_KEY`, `DB_PASSWORD`).
2. Add the key (without value) to the relevant config file with a comment explaining its use.
3. Export the variable locally:
   ```bash
   export SLIPSTREAM_MYSTRAT_EXCHANGE_API_KEY="..."
   ```
4. In production, store it in the secret manager that backs the deployment (e.g., AWS SSM, Doppler).

## Tips

- Use the strategy slug rather than human-readable title to avoid spaces or punctuation.
- Mirror the variable in `.env.example` with a dummy value so onboarding engineers know it exists.
- Keep non-secret overrides (e.g., feature flags) inside the YAML configs; reserve env vars for sensitive values only.
