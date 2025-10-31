# Gradient Notification Setup Guide

Complete guide for setting up Telegram and email notifications for live trading alerts.

## Overview

The Gradient trading system supports two types of notifications:

1. **Telegram Alerts** (After each 4h rebalance)
   - Sent immediately when rebalance completes
   - Shows positions entered, turnover, fill rates
   - Quick mobile notifications

2. **Daily Email Summary** (Once per day)
   - Comprehensive daily performance report
   - Aggregated statistics across all rebalances
   - All-time performance metrics
   - HTML-formatted with tables and charts

---

## 1. Telegram Setup

### Step 1: Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Choose a name (e.g., "Gradient Trading Bot")
4. Choose a username (e.g., "gradient_trading_bot")
5. **Copy the API token** (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 2: Get Your Chat ID

1. Search for `@userinfobot` on Telegram
2. Send `/start`
3. **Copy your Chat ID** (looks like `123456789`)

Alternatively, to send to a group:
1. Add your bot to a Telegram group
2. Use `@get_id_bot` in the group
3. Copy the group Chat ID (negative number like `-123456789`)

### Step 3: Set Environment Variables

Add to your `~/.bashrc` or `/etc/environment`:

```bash
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="123456789"
```

Apply changes:
```bash
source ~/.bashrc
```

### Step 4: Test Telegram Notification

```bash
# Test with Python
uv run python -c "
from slipstream.gradient.live.notifications import send_telegram_rebalance_alert_sync
from slipstream.gradient.live.config import load_config

config = load_config()
test_data = {
    'timestamp': '2025-10-31 12:00:00',
    'n_long': 15,
    'n_short': 15,
    'n_positions': 30,
    'total_turnover': 3500.0,
    'stage1_filled': 25,
    'stage2_filled': 5,
    'errors': 0,
    'dry_run': True
}
send_telegram_rebalance_alert_sync(test_data, config)
print('Check your Telegram!')
"
```

You should receive a message like:

```
✅ Gradient Rebalance Complete [DRY-RUN]

📅 Time: 2025-10-31 12:00:00

📊 Positions:
  • Long: 15 positions
  • Short: 15 positions
  • Total: 30

💰 Execution:
  • Turnover: $3,500.00
  • Stage 1 (passive): 25 fills
  • Stage 2 (aggressive): 5 fills
  • Errors: 0

⚠️ DRY-RUN MODE - No real orders placed
```

---

## 2. Email Setup

### Step 1: Generate App Password (Gmail)

For Gmail:
1. Go to https://myaccount.google.com/security
2. Enable 2-Step Verification if not already enabled
3. Go to https://myaccount.google.com/apppasswords
4. Select "Mail" and "Other (Custom name)"
5. Name it "Gradient Trading"
6. **Copy the 16-character app password**

For other providers:
- **Outlook**: Use account password or app password from security settings
- **Custom SMTP**: Get credentials from your email provider

### Step 2: Set Environment Variables

Add to `~/.bashrc` or `/etc/environment`:

```bash
export EMAIL_FROM="your-email@gmail.com"
export EMAIL_TO="your-email@gmail.com"  # Can be same or different
export EMAIL_PASSWORD="abcd efgh ijkl mnop"  # App password (16 chars)

# Optional: Custom SMTP settings (defaults to Gmail)
export SMTP_SERVER="smtp.gmail.com"  # Default
export SMTP_PORT="587"               # Default
```

Apply changes:
```bash
source ~/.bashrc
```

### Step 3: Test Email Notification

```bash
# Run daily summary script
uv run python scripts/live/gradient_daily_summary.py
```

Check your email inbox for a message titled:
**"Gradient Daily Summary - 2025-10-30"**

The email will include:
- Today's performance (number of rebalances, turnover, fill rates)
- Execution details (passive/aggressive fills)
- Recent rebalances table
- All-time statistics

---

## 3. Enable Notifications in Config

Edit `config/gradient_live.json`:

```json
{
  "alerts": {
    "enabled": true,           // Enable notifications
    "telegram_token": "",      // Leave empty (uses env var)
    "telegram_chat_id": "",    // Leave empty (uses env var)
    "email_enabled": true
  }
}
```

**Important**: Keep tokens empty in config file for security. Use environment variables instead.

---

## 4. Set Up Cron Jobs

### Telegram Notifications

Telegram notifications are sent automatically after each rebalance. No additional cron needed.

The existing rebalance cron job will send Telegram alerts:
```bash
# Already configured in gradient_rebalance.sh
0 */4 * * * /root/slipstream/scripts/live/gradient_rebalance.sh
```

### Daily Email Summary

Add a cron job to send daily email at 8 AM UTC:

```bash
crontab -e
```

Add this line:
```bash
# Send daily Gradient performance email at 8 AM UTC
0 8 * * * /root/slipstream/scripts/live/gradient_daily_email.sh >> /var/log/gradient/daily_email.log 2>&1
```

This will:
- Run once per day at 8:00 AM UTC
- Send summary of previous day's activity
- Include all-time aggregated statistics
- Log output to `/var/log/gradient/daily_email.log`

---

## 5. Environment Variables Summary

Here's a complete list of all environment variables you should set:

```bash
# Hyperliquid API (required for live trading)
export HYPERLIQUID_API_KEY="0x..."
export HYPERLIQUID_SECRET="your_secret_key"

# Telegram (required for Telegram alerts)
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="123456789"

# Email (required for daily summaries)
export EMAIL_FROM="your-email@gmail.com"
export EMAIL_TO="your-email@gmail.com"
export EMAIL_PASSWORD="your-app-password"

# Optional email settings (defaults shown)
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
```

Add all of these to `~/.bashrc` for persistence:

```bash
cat >> ~/.bashrc << 'EOF'

# Gradient Trading Environment Variables
export HYPERLIQUID_API_KEY="0x..."
export HYPERLIQUID_SECRET="your_secret_key"
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="123456789"
export EMAIL_FROM="your-email@gmail.com"
export EMAIL_TO="your-email@gmail.com"
export EMAIL_PASSWORD="your-app-password"
EOF

source ~/.bashrc
```

---

## 6. Verification Checklist

Before going live, verify all notifications work:

- [ ] Telegram bot created and token obtained
- [ ] Telegram chat ID obtained
- [ ] Email app password generated (Gmail) or credentials ready
- [ ] All environment variables set in `~/.bashrc`
- [ ] `source ~/.bashrc` applied
- [ ] Telegram test message sent successfully
- [ ] Email test sent successfully
- [ ] `alerts.enabled` set to `true` in config
- [ ] Cron job for daily email added
- [ ] Logs directory exists: `/var/log/gradient/`

---

## 7. Notification Examples

### Telegram Message (After Each Rebalance)

```
✅ Gradient Rebalance Complete [LIVE]

📅 Time: 2025-10-31 16:00:00

📊 Positions:
  • Long: 18 positions
  • Short: 18 positions
  • Total: 36

💰 Execution:
  • Turnover: $4,250.50
  • Stage 1 (passive): 28 fills
  • Stage 2 (aggressive): 8 fills
  • Errors: 0

✅ Live positions entered
```

### Email Subject Line

```
Gradient Daily Summary - 2025-10-31
```

### Email Content (HTML)

```
📊 Gradient Daily Summary - 2025-10-31

Today's Performance
-------------------
Rebalances:           6
Total Turnover:       $25,203
Avg Positions:        34.5
Passive Fill Rate:    72.3%
Errors:               0

Execution Details
-----------------
Average Turnover per Rebalance:  $4,200.50
Average Gross Exposure:          $9,800.25
Average Net Exposure:            $45.30
Passive Fills (Stage 1):         156 (72.3%)
Aggressive Fills (Stage 2):      60 (27.7%)

Recent Rebalances
-----------------
[Table showing last 5 rebalances with timestamps, positions, turnover, etc.]

All-Time Statistics
-------------------
Days Running:         14
Total Rebalances:     84
Lifetime Turnover:    $352,470
Lifetime Passive Rate: 68.9%
Lifetime Errors:      2
```

---

## 8. Troubleshooting

### Telegram Not Sending

**Check 1**: Verify bot token
```bash
echo $TELEGRAM_BOT_TOKEN
```

**Check 2**: Test bot directly
```bash
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"
```

**Check 3**: Verify chat ID
```bash
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage?chat_id=$TELEGRAM_CHAT_ID&text=Test"
```

**Common issues**:
- Bot token not set or incorrect
- Chat ID not set or incorrect
- Bot not started (send `/start` to your bot first)
- Bot not added to group (if using group chat ID)

### Email Not Sending

**Check 1**: Verify credentials
```bash
echo $EMAIL_FROM
echo $EMAIL_TO
echo $EMAIL_PASSWORD | head -c 5  # Show first 5 chars only
```

**Check 2**: Test SMTP connection
```bash
python3 -c "
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('$EMAIL_FROM', '$EMAIL_PASSWORD')
print('✓ SMTP login successful')
server.quit()
"
```

**Common issues**:
- App password not generated (using account password won't work)
- 2-Step Verification not enabled on Gmail
- Less secure app access blocked (use app password)
- Wrong SMTP server or port
- Firewall blocking port 587

### Logs Not Created

**Check**: Directory permissions
```bash
sudo mkdir -p /var/log/gradient
sudo chown $USER:$USER /var/log/gradient
chmod 755 /var/log/gradient
```

### Cron Job Not Running

**Check 1**: Cron service running
```bash
sudo systemctl status cron
```

**Check 2**: View cron logs
```bash
grep CRON /var/log/syslog | tail -20
```

**Check 3**: Test script manually
```bash
/root/slipstream/scripts/live/gradient_daily_email.sh
```

---

## 9. Security Best Practices

1. **Never commit tokens to git**
   - Keep config file empty of credentials
   - Use environment variables only

2. **Restrict file permissions**
   ```bash
   chmod 600 ~/.bashrc  # Only you can read
   ```

3. **Use app passwords, not account passwords**
   - Generate app-specific passwords for email
   - Revoke if compromised

4. **Separate email accounts (optional)**
   - Use a dedicated email for trading notifications
   - Reduce risk if credentials are compromised

5. **Monitor bot activity**
   - Check Telegram bot logs regularly
   - Revoke bot token if suspicious activity detected

---

## 10. Notification Workflow

```
4-Hour Rebalance Cycle:
┌─────────────────────────────────────────────┐
│  00:00 - Cron triggers rebalance            │
│  00:01 - Fetch data, compute signals        │
│  00:02 - Place limit orders (Stage 1)       │
│  00:02-01:00 - Monitor fills                │
│  01:00 - Sweep unfilled with market orders  │
│  01:01 - Log performance                    │
│  01:02 - Send Telegram alert ← YOU GET THIS │
└─────────────────────────────────────────────┘

Daily Email Summary:
┌─────────────────────────────────────────────┐
│  08:00 UTC - Daily email cron triggers      │
│  08:00 - Aggregate yesterday's rebalances   │
│  08:00 - Compute all-time statistics        │
│  08:01 - Send HTML email ← YOU GET THIS     │
└─────────────────────────────────────────────┘
```

---

## Need Help?

- Telegram API docs: https://core.telegram.org/bots/api
- Gmail app passwords: https://support.google.com/accounts/answer/185833
- SMTP troubleshooting: Check firewall, port, credentials

---

**You're all set! 🚀**

After setup, you'll receive:
- Telegram message after each 4h rebalance
- Daily email summary at 8 AM UTC

Happy trading!
