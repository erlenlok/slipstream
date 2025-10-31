#!/bin/bash
#
# Gradient Strategy Rebalance Script
# Called by cron every 4 hours
#

set -euo pipefail

# Configuration
PROJECT_DIR="/root/slipstream"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="/var/log/gradient"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Change to project directory
cd "$PROJECT_DIR"

# Run rebalance
python -m slipstream.gradient.live.rebalance

# Exit with rebalance script's exit code
exit $?
