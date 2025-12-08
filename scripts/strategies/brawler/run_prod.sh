#!/bin/bash
set -e

# Load brawler env
if [ -f .env.brawler ]; then
    export $(cat .env.brawler | grep -v '#' | xargs)
fi

# Determine log level (default INFO for prod)
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "Starting Brawler (PRODUCTION)..."

# Using uv to run the module
exec uv run python -m slipstream.strategies.brawler.cli \
    --config config/brawler_prod.yaml
