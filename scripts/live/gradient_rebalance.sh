#!/bin/bash
#
# Gradient Strategy Rebalance Script
# Called by cron every 4 hours
#
# CRITICAL TIMING REQUIREMENT:
# ===========================
# This script MUST run at :01 of each 4-hour boundary (NOT :00)
#
# - Hyperliquid 4h candles close at: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
# - Data may not be available immediately at candle close
# - Running at :01 gives 1-minute buffer for data finalization
#
# REQUIRED CRON SCHEDULE:
# ======================
# 1 */4 * * * /root/slipstream/scripts/live/gradient_rebalance.sh
#
# This runs at: 00:01, 04:01, 08:01, 12:01, 16:01, 20:01 UTC
#
# WHY THIS MATTERS:
# ================
# - Backtest used candles that closed at :00
# - Live trading MUST use same candle boundaries
# - Misalignment = different performance than backtest
# - The :01 timing ensures we get complete candle data while staying aligned
#

set -euo pipefail

# Configuration
PROJECT_DIR="/root/slipstream"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="/var/log/gradient"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Export required environment variables
export HYPERLIQUID_API_KEY="0x998c0B58193faca878B55aE29165b68167A1BD30"
export HYPERLIQUID_API_SECRET="0xc414e72b39459d4a2cefb75a3ed82c3c9fdc6313593372a108372fceef354cad"
export TELEGRAM_BOT_TOKEN="8326508237:AAFI2rI2MhN_3CnukAcUgBmiOy8-JPnvlTA"
export TELEGRAM_CHAT_ID="1672609276"
export REDIS_ENABLED="true"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"

# Timing verification
CURRENT_HOUR=$(date -u +%H)
CURRENT_MIN=$(date -u +%M)
HOUR_MOD=$((10#$CURRENT_HOUR % 4))

# Warn if running at unexpected time
if [ $HOUR_MOD -ne 0 ] || [ $((10#$CURRENT_MIN)) -lt 1 ] || [ $((10#$CURRENT_MIN)) -gt 5 ]; then
    echo "⚠️  WARNING: Script running at unexpected time!" >&2
    echo "   Current time: ${CURRENT_HOUR}:${CURRENT_MIN} UTC" >&2
    echo "   Expected: :01-:05 of hours 00, 04, 08, 12, 16, 20" >&2
    echo "   This may cause candle alignment issues!" >&2
    echo "   Check your cron schedule: 1 */4 * * *" >&2
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Change to project directory
cd "$PROJECT_DIR"

# Run rebalance
python -m slipstream.gradient.live.rebalance

# Exit with rebalance script's exit code
exit $?
