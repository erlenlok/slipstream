#!/usr/bin/env python3
"""Quick test script to verify Telegram setup."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_env():
    """Check if environment variables are set."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    print("=" * 60)
    print("TELEGRAM SETUP CHECK")
    print("=" * 60)

    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not set")
        print("   Run: export TELEGRAM_BOT_TOKEN='your-token'")
        return False
    else:
        print(f"✓ TELEGRAM_BOT_TOKEN set: {token[:10]}...")

    if not chat_id:
        print("❌ TELEGRAM_CHAT_ID not set")
        print("   Run: export TELEGRAM_CHAT_ID='your-chat-id'")
        return False
    else:
        print(f"✓ TELEGRAM_CHAT_ID set: {chat_id}")

    print("\n" + "=" * 60)
    return True


def test_telegram():
    """Send a test message."""
    if not check_env():
        return False

    print("Sending test message to Telegram...")
    print("=" * 60)

    try:
        from slipstream.gradient.live.notifications import send_telegram_rebalance_alert_sync
        from slipstream.gradient.live.config import load_config

        config = load_config()

        test_data = {
            'timestamp': '2025-10-31 16:00:00',
            'n_long': 18,
            'n_short': 18,
            'n_positions': 36,
            'total_turnover': 2500.0,
            'stage1_filled': 30,
            'stage2_filled': 6,
            'errors': 0,
            'dry_run': True
        }

        success = send_telegram_rebalance_alert_sync(test_data, config)

        print("\n" + "=" * 60)
        if success:
            print("✅ SUCCESS!")
            print("   Check your Telegram - you should have a test message!")
            print("=" * 60)
            return True
        else:
            print("❌ FAILED to send message")
            print("   Check token and chat ID are correct")
            print("=" * 60)
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_telegram()
    sys.exit(0 if success else 1)
