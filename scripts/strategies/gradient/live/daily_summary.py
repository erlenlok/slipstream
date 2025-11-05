#!/usr/bin/env python3
"""
Send daily performance summary email.

This script should be run once per day via cron to send an email summary
of the previous day's trading activity plus all-time aggregated statistics.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from slipstream.strategies.gradient.live.config import load_config
from slipstream.strategies.gradient.live.performance import PerformanceTracker
from slipstream.strategies.gradient.live.notifications import send_email_daily_summary


def main():
    """Send daily summary email."""
    try:
        # Load configuration
        config = load_config()

        # Initialize performance tracker
        tracker = PerformanceTracker(log_dir=config.log_dir)

        # Get yesterday's summary (or today if you want today's data)
        # Using yesterday to ensure full day's data
        yesterday = datetime.now() - timedelta(days=1)
        daily_summary = tracker.get_daily_summary(yesterday)

        # Get all-time summary
        all_time_summary = tracker.get_all_time_summary()

        print(f"Sending daily summary for {daily_summary.get('date', 'N/A')}")
        print(f"Rebalances: {daily_summary.get('n_rebalances', 0)}")
        print(f"Turnover: ${daily_summary.get('total_turnover', 0):,.2f}")

        # Send email
        success = send_email_daily_summary(daily_summary, all_time_summary, config)

        if success:
            print("✓ Daily summary email sent successfully")
            return 0
        else:
            print("✗ Failed to send daily summary email")
            return 1

    except Exception as e:
        print(f"Error sending daily summary: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
