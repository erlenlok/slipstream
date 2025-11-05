#!/bin/bash
# Compatibility wrapper for scripts/strategies/gradient/live/rebalance.sh
# TODO: remove after downstream jobs migrate.

set -euo pipefail

warn="Use scripts/strategies/gradient/live/rebalance.sh instead of scripts/strategies/gradient/live/rebalance.sh"
>&2 echo "[DEPRECATED] ${warn}"
exec "$(dirname "$0")/../strategies/gradient/live/rebalance.sh" "$@"
