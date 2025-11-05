#!/bin/bash
# Cron wrapper for daily Gradient performance email
# Run once per day to send email summary

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate virtual environment
cd "$PROJECT_ROOT"
source .venv/bin/activate

# Set environment variables if needed
# export EMAIL_FROM="your-email@gmail.com"
# export EMAIL_TO="your-email@gmail.com"
# export EMAIL_PASSWORD="your-app-password"

# Run daily summary script
python scripts/strategies/gradient/live/daily_summary.py

# Deactivate
deactivate
