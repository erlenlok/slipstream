#!/usr/bin/env python3
"""Diagnostic script for Telegram bot issues."""

import asyncio
import os
from telegram import Bot
from telegram.error import TelegramError

async def diagnose():
    """Run diagnostic checks on Telegram bot."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    print("=" * 60)
    print("TELEGRAM DIAGNOSTIC")
    print("=" * 60)

    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not set")
        return

    if not chat_id:
        print("❌ TELEGRAM_CHAT_ID not set")
        return

    print(f"✓ Token: {token[:10]}...")
    print(f"✓ Chat ID: {chat_id}")
    print()

    try:
        bot = Bot(token=token)

        # Test 1: Check bot info
        print("Test 1: Checking bot info...")
        me = await bot.get_me()
        print(f"✓ Bot found: @{me.username} ({me.first_name})")
        print()

        # Test 2: Try to get chat info
        print("Test 2: Checking if chat exists...")
        try:
            # Try as integer
            chat = await bot.get_chat(int(chat_id))
            print(f"✓ Chat found: {chat.type}")
            if chat.username:
                print(f"  Username: @{chat.username}")
            print()
        except TelegramError as e:
            print(f"❌ Chat not found: {e}")
            print()
            print("SOLUTION: You need to start a conversation with your bot!")
            print(f"1. Open Telegram and search for @{me.username}")
            print("2. Click 'START' or send /start")
            print("3. Then run this script again")
            return

        # Test 3: Send test message
        print("Test 3: Sending test message...")
        await bot.send_message(
            chat_id=int(chat_id),
            text="✅ Telegram bot is working correctly!"
        )
        print("✓ Message sent successfully!")
        print()
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Check your Telegram - you should see a test message")
        print("=" * 60)

    except TelegramError as e:
        print(f"❌ Telegram error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose())
