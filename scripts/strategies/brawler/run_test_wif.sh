#!/bin/bash
set -e

# Load brawler env
if [ -f .env.brawler ]; then
    export $(cat .env.brawler | grep -v '#' | xargs)
fi

# Set LOG_LEVEL to INFO to see status summaries
export LOG_LEVEL=INFO

echo "Starting Brawler (WIF Test Config)..."
echo "Press Ctrl+C to stop."

uv run python -m slipstream.strategies.brawler.cli \
    --config config/brawler_wif_test.yaml
