# Go Live TODAY - Gradient Trading Checklist

**Objective**: Get Gradient strategy trading live in small size by end of day

**Current Status**: ✅ All code implemented and tested
**Timeline**: 2-3 hours to go live
**Starting Capital**: $1,000 - $5,000 (recommended)

---

## ✅ Code Review - COMPLETE

Your hardening work looks excellent:

### What You Fixed:
1. ✅ **Signal Alignment**: Now uses `compute_trend_strength` from signals module (exact backtest logic)
2. ✅ **Portfolio Hardening**: Edge case handling (empty lists, min asset counts)
3. ✅ **Alignment Tests**: Verified live pipeline matches backtest exactly
4. ✅ **Tests Pass**: Both signal and portfolio selection tests pass

### What's Ready:
- ✅ Live trading system complete
- ✅ Two-stage execution (limit → market)
- ✅ Telegram notifications ready
- ✅ Daily email summaries ready
- ✅ Performance tracking ready
- ✅ Candle alignment verified
- ✅ Tests passing

**Next**: Configuration and deployment

---

## Phase 1: Environment Setup (30 min)

### 1.1 Create Hyperliquid API Keys

**Mainnet (Real Money)**:
1. Go to https://app.hyperliquid.xyz/
2. Connect wallet
3. Go to Settings → API
4. Create new API key
5. **Copy the private key** (shown once!)
6. Note your wallet address (public key)

**Testnet (Practice)**:
- For testing, use testnet first: https://app.hyperliquid-testnet.xyz/
- Same process as mainnet

### 1.2 Install and Setup Redis (5 min)

**Redis dramatically speeds up data fetching** (first run ~2-3 min, subsequent runs ~5-10 sec):

```bash
# Install Redis
sudo apt-get update && sudo apt-get install -y redis-server

# Disable password for localhost (already done if you followed along)
echo "user default on nopass +@all ~*" | sudo tee /etc/redis/users.acl

# Start and enable Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test connection
redis-cli ping  # Should return: PONG
```

### 1.3 Set Environment Variables

Add to `~/.bashrc`:

```bash
# Add at the end of file
cat >> ~/.bashrc << 'EOF'

# Gradient Trading - Hyperliquid API
export HYPERLIQUID_API_KEY="0x..."  # Your wallet address
export HYPERLIQUID_SECRET="0x..."   # Your private key

# Gradient Trading - Telegram
export TELEGRAM_BOT_TOKEN="..."     # From @BotFather
export TELEGRAM_CHAT_ID="..."       # Your chat ID

# Gradient Trading - Email (optional for now)
export EMAIL_FROM="your@gmail.com"
export EMAIL_TO="your@gmail.com"
export EMAIL_PASSWORD="app-password"

# Gradient Trading - Redis Cache (enabled by default)
export REDIS_ENABLED="true"         # Set to "false" to disable caching
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
EOF

source ~/.bashrc
```

### 1.3 Create Telegram Bot (5 min)

1. Open Telegram → Search `@BotFather`
2. Send `/newbot`
3. Choose name: "Gradient Trading Bot"
4. Choose username: `gradient_trading_bot` (or similar)
5. Copy the token: `123456789:ABCdef...`
6. Search `@userinfobot` → Send `/start` → Copy your Chat ID

### 1.4 Verify Environment

```bash
# Check all variables are set
echo "API Key: ${HYPERLIQUID_API_KEY:0:10}..."
echo "Secret: ${HYPERLIQUID_SECRET:0:10}..."
echo "Telegram Token: ${TELEGRAM_BOT_TOKEN:0:10}..."
echo "Telegram Chat ID: $TELEGRAM_CHAT_ID"
```

---

## Phase 2: Configuration (15 min)

### 2.1 Set Capital and Risk Limits

Edit `config/gradient_live.json`:

```bash
cd /root/slipstream
nano config/gradient_live.json
```

**For first live run (conservative)**:
```json
{
  "strategy": "gradient_momentum_35pct_4h",
  "capital_usd": 1000,              // ← START SMALL!
  "max_position_pct": 10,
  "max_total_leverage": 2.0,
  "concentration_pct": 35,
  "rebalance_freq_hours": 4,
  "weight_scheme": "inverse_vol",
  "lookback_spans": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
  "vol_span": 24,
  "liquidity_threshold_usd": 10000,
  "liquidity_impact_pct": 2.5,
  "emergency_stop_drawdown_pct": 20,
  "dry_run": false,                 // ← SET TO FALSE FOR LIVE!
  "api": {
    "endpoint": "https://api.hyperliquid.xyz",
    "mainnet": true                 // ← true for real, false for testnet
  },
  "execution": {
    "passive_timeout_seconds": 3600,
    "limit_order_aggression": "join_best",
    "cancel_before_market_sweep": true,
    "min_order_size_usd": 10
  },
  "logging": {
    "dir": "/var/log/gradient",
    "level": "INFO"
  },
  "alerts": {
    "enabled": true,                // ← Enable notifications
    "telegram_token": "",
    "telegram_chat_id": "",
    "email_enabled": true
  }
}
```

### 2.2 Create Log Directory

```bash
sudo mkdir -p /var/log/gradient
sudo chown $USER:$USER /var/log/gradient
chmod 755 /var/log/gradient
```

### 2.3 Verify Configuration

```bash
uv run python -c "
from slipstream.gradient.live import load_config, validate_config
config = load_config()
validate_config(config)
print('✓ Configuration valid')
print(f'  Capital: \${config.capital_usd:,.0f}')
print(f'  Dry-run: {config.dry_run}')
print(f'  Alerts: {config.alerts_enabled}')
"
```

---

## Phase 3: Dry-Run Testing (30 min)

### 3.1 Test Data Fetching

```bash
uv run python -c "
from slipstream.gradient.live import load_config, fetch_live_data
config = load_config()
print('Fetching live market data...')
data = fetch_live_data(config)
print(f'✓ Fetched {len(data[\"assets\"])} assets')
print(f'✓ Total candles: {len(data[\"panel\"])}')
"
```

Expected: Should fetch 100+ assets, thousands of candles

### 3.2 Test Signal Generation

```bash
uv run python -c "
from slipstream.gradient.live import load_config, fetch_live_data, compute_live_signals
config = load_config()
data = fetch_live_data(config)
print('Computing signals...')
signals = compute_live_signals(data, config)
print(f'✓ Generated signals for {len(signals)} assets')
print(f'✓ Liquid assets: {signals[\"include_in_universe\"].sum()}')
print('\\nTop 5 by momentum:')
print(signals.head(5)[[\"asset\", \"momentum_score\"]])
print('\\nBottom 5 by momentum:')
print(signals.tail(5)[[\"asset\", \"momentum_score\"]])
"
```

Expected: Should show clear momentum ranking

### 3.3 Test Portfolio Construction

```bash
uv run python -c "
from slipstream.gradient.live import load_config, fetch_live_data, compute_live_signals, construct_target_portfolio
config = load_config()
data = fetch_live_data(config)
signals = compute_live_signals(data, config)
print('Building portfolio...')
positions = construct_target_portfolio(signals, config)
print(f'✓ Portfolio: {len(positions)} positions')
longs = {k: v for k, v in positions.items() if v > 0}
shorts = {k: v for k, v in positions.items() if v < 0}
print(f'  Long: {len(longs)} positions')
print(f'  Short: {len(shorts)} positions')
print(f'  Total exposure: \${sum(abs(p) for p in positions.values()):,.2f}')
print('\\nTop 3 long positions:')
for asset, size in sorted(longs.items(), key=lambda x: -x[1])[:3]:
    print(f'  {asset}: \${size:,.2f}')
print('\\nTop 3 short positions:')
for asset, size in sorted(shorts.items(), key=lambda x: x[1])[:3]:
    print(f'  {asset}: \${size:,.2f}')
"
```

Expected: Should show balanced long/short portfolio

### 3.4 Test Full Rebalance (DRY-RUN)

**IMPORTANT**: First ensure `dry_run: true` in config!

```bash
# Make sure dry_run is true
cat config/gradient_live.json | grep dry_run

# Run full rebalance
uv run python -m slipstream.gradient.live.rebalance
```

Expected output:
```
Starting Gradient rebalance cycle
Fetching latest market data...
Computing momentum signals...
Constructing target portfolio...
Executing rebalance (two-stage: limit → market)...
DRY-RUN: Would place X limit orders
Telegram notification sent
DRY-RUN MODE: No actual orders were placed
```

### 3.5 Test Telegram Notification

```bash
uv run python -c "
from slipstream.gradient.live.notifications import send_telegram_rebalance_alert_sync
from slipstream.gradient.live.config import load_config

config = load_config()
test_data = {
    'timestamp': '2025-10-31 16:00:00',
    'n_long': 18, 'n_short': 18,
    'total_turnover': 1500.0,
    'stage1_filled': 30,
    'stage2_filled': 6,
    'errors': 0,
    'dry_run': True
}
send_telegram_rebalance_alert_sync(test_data, config)
print('Check your Telegram for test message!')
"
```

Check your Telegram - should receive test alert

---

## Phase 4: Manual First Rebalance (30 min)

### 4.1 Pre-Flight Checklist

Verify EVERYTHING before going live:

- [ ] API keys set in environment
- [ ] Telegram bot working (test message received)
- [ ] Config has `dry_run: false`
- [ ] Capital set to small amount ($1k-$5k)
- [ ] Log directory exists and writable
- [ ] Dry-run test completed successfully
- [ ] Understand you're about to place REAL orders
- [ ] Emergency stop script ready: `scripts/live/gradient_emergency_stop.py`

### 4.2 LIVE First Rebalance

**Deep breath. This is it.**

```bash
# Final verification
cat config/gradient_live.json | grep -A2 "dry_run\|capital_usd"

# Should show:
# "capital_usd": 1000,  (or your chosen amount)
# "dry_run": false,

# Run live rebalance
uv run python -m slipstream.gradient.live.rebalance
```

**Watch the output carefully:**
- Data fetching (should complete in 30-60s)
- Signal computation
- Portfolio construction
- Order placement (Stage 1 - limit orders)
- Fill monitoring (60 min timeout)
- Stage 2 sweep (if needed)
- Telegram notification

### 4.3 Verify on Hyperliquid

1. Go to https://app.hyperliquid.xyz/
2. Connect your wallet
3. Check Positions tab
4. Should see ~36 positions (18 long, 18 short)
5. Net position should be near $0
6. Gross exposure should be ~$2,000 (2x leverage on $1k)

### 4.4 Check Telegram

Should receive message like:
```
✅ Gradient Rebalance Complete [LIVE]

📅 Time: 2025-10-31 16:03:15

📊 Positions:
  • Long: 18 positions
  • Short: 18 positions
  • Total: 36

💰 Execution:
  • Turnover: $2,045.50
  • Stage 1 (passive): 28 fills
  • Stage 2 (aggressive): 8 fills
  • Errors: 0

✅ Live positions entered
```

### 4.5 Review Logs

```bash
tail -100 /var/log/gradient/rebalance_$(date +%Y%m%d).log
```

Check for:
- No ERROR messages
- Reasonable fill rates
- Expected position counts
- Correct timing (script ran at :01 of 4-hour boundary)

---

## Phase 5: Set Up Cron for Automation (15 min)

### 5.1 Verify Current Time Alignment

```bash
# Check current UTC time
date -u

# Next 4-hour boundary will be at:
# 00:01, 04:01, 08:01, 12:01, 16:01, or 20:01 UTC
```

### 5.2 Add Cron Job

```bash
crontab -e
```

Add this line (note the **1** not 0!):
```bash
# Gradient 4h rebalancing - runs at :01 of each 4-hour boundary
1 */4 * * * /root/slipstream/scripts/live/gradient_rebalance.sh >> /var/log/gradient/cron.log 2>&1
```

Save and exit (Ctrl+X, Y, Enter in nano)

### 5.3 Verify Cron

```bash
# List cron jobs
crontab -l | grep gradient

# Should show:
# 1 */4 * * * /root/slipstream/scripts/live/gradient_rebalance.sh >> /var/log/gradient/cron.log 2>&1

# Check cron is running
sudo systemctl status cron
```

### 5.4 Wait for Next Automatic Rebalance

Next rebalance will happen automatically at the next :01 mark.

Check logs after it runs:
```bash
tail -f /var/log/gradient/cron.log
tail -f /var/log/gradient/rebalance_$(date +%Y%m%d).log
```

---

## Phase 6: Monitoring & Validation (Ongoing)

### 6.1 First 24 Hours

Monitor closely:

```bash
# Watch rebalance logs
tail -f /var/log/gradient/rebalance_$(date +%Y%m%d).log

# Check performance history
cat /var/log/gradient/rebalance_history.jsonl | tail -5 | jq

# View cron execution
grep CRON /var/log/syslog | grep gradient | tail -10
```

### 6.2 Daily Checks

Every day for first week:

1. **Check Telegram** - Should receive 6 messages/day (every 4h)
2. **Check Email** - Daily summary at 8 AM UTC
3. **Check Hyperliquid** - Verify positions look reasonable
4. **Check Logs** - No errors or warnings
5. **Check Performance** - Tracking vs expectations

### 6.3 Performance Tracking

```bash
# Daily summary
uv run python scripts/live/gradient_daily_summary.py

# View all rebalances
cat /var/log/gradient/rebalance_history.jsonl | jq '.timestamp, .n_positions, .total_turnover'
```

---

## Emergency Procedures

### Stop Trading Immediately

```bash
# Option 1: Cancel orders only
uv run python scripts/live/gradient_emergency_stop.py --cancel-orders

# Option 2: Flatten all positions (go to cash)
uv run python scripts/live/gradient_emergency_stop.py --flatten-all

# Option 3: Disable cron
crontab -e
# Comment out gradient line with #
```

### Reduce Position Size

Edit `config/gradient_live.json`:
```json
{
  "capital_usd": 500,  // ← Reduce if needed
  ...
}
```

Next rebalance will use smaller size.

### Switch to Dry-Run

Edit `config/gradient_live.json`:
```json
{
  "dry_run": true,  // ← Back to testing mode
  ...
}
```

---

## Success Criteria

### After First Rebalance ✅
- [ ] Positions entered successfully
- [ ] ~36 positions (18 long, 18 short)
- [ ] Net exposure near $0
- [ ] Gross exposure matches capital × 2
- [ ] Telegram notification received
- [ ] No errors in logs

### After First 24 Hours ✅
- [ ] 6 rebalances completed (every 4h)
- [ ] All Telegram notifications received
- [ ] Fill rates > 80% (stage 1 + stage 2 combined)
- [ ] No critical errors
- [ ] Positions look reasonable

### After First Week ✅
- [ ] 42 rebalances completed
- [ ] Daily email summaries received
- [ ] Performance tracking working
- [ ] No system downtime
- [ ] Sharpe ratio > 0 (early days, high variance)

---

## Scaling Up

If all goes well for 1 week:

1. **Week 2**: Increase to $5k if comfortable
2. **Week 3**: Increase to $10k if metrics look good
3. **Month 2**: Scale up to desired size
4. **Ongoing**: Monitor performance vs backtest

**Key metrics to watch**:
- Passive fill rate (target > 50%)
- Error count (target = 0)
- Sharpe ratio (target > 1.5 after costs)
- Max drawdown (expect -10 to -15%)

---

## Quick Command Reference

```bash
# Manual rebalance
uv run python -m slipstream.gradient.live.rebalance

# Check config
cat config/gradient_live.json | grep -A2 "dry_run\|capital_usd"

# View logs
tail -f /var/log/gradient/rebalance_$(date +%Y%m%d).log

# Check cron
crontab -l | grep gradient

# Emergency stop
uv run python scripts/live/gradient_emergency_stop.py --flatten-all

# Test telegram
uv run python -c "from slipstream.gradient.live.notifications import send_telegram_rebalance_alert_sync; ..."

# Verify candle alignment
uv run python scripts/verify_candle_alignment.py

# Run alignment tests
uv run python tests/gradient/test_live_alignment.py
```

---

## Final Checklist Before Going Live

Go through this one more time:

- [ ] All environment variables set and verified
- [ ] Config file has correct capital and dry_run=false
- [ ] Telegram bot tested and working
- [ ] Log directory exists
- [ ] Dry-run test passed
- [ ] Understand risks (this is real money!)
- [ ] Emergency stop procedure understood
- [ ] First manual rebalance completed successfully
- [ ] Positions verified on Hyperliquid
- [ ] Telegram notification received
- [ ] Cron job added and verified
- [ ] Monitoring plan in place

---

## Timeline Summary

**Total time to go live: ~2-3 hours**

- 30 min: Environment setup (API keys, Telegram)
- 15 min: Configuration
- 30 min: Dry-run testing
- 30 min: First live rebalance (manual)
- 15 min: Cron setup
- 30 min: Validation and monitoring

**After this, you're live!** 🚀

The system will:
- Rebalance every 4 hours automatically
- Send Telegram alerts after each rebalance
- Send daily email summaries at 8 AM UTC
- Track all performance metrics
- Log everything for analysis

---

## You're Ready!

Everything is implemented and tested. Your hardening work ensured the live pipeline matches the backtest exactly.

**The only thing left is to execute the steps above.**

Good luck! 🎯

Remember:
- Start small ($1k)
- Monitor closely first 24-48 hours
- Don't panic on short-term variance (strategy has expected -7.6% max drawdown in 10 days)
- Trust the process - backtest showed 3.1 Sharpe for a reason

Let's get live! 💪
