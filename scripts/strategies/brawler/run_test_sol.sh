#!/bin/bash
set -e

# Ensure we're in the project root
cd "$(dirname "$0")/../../.."

# Load environment if present (optional helper, CLI also loads it)
if [ -f .env.gradient ]; then
    export $(grep -v '^#' .env.gradient | xargs)
fi

echo "Starting Brawler (SOL Test Config)..."
echo "Press Ctrl+C to stop."

uv run python -m slipstream.strategies.brawler.cli --config config/brawler_sol_test.yaml
