# Gradient Live Trading - Quick Start

## 🚀 Immediate Next Steps (3-4 hours to go live)

### 1. Core Components

All live-trading modules are implemented and wired together:

- **`data.py`** fetches live 4-hour candles (with optional Redis caching) and computes momentum signals using the production liquidity filters and multi-span EWMA stack.
- **`portfolio.py`** ranks the signal panel, applies inverse-volatility sizing, enforces risk limits, and validates the resulting target book.
- **`execution.py`** manages rebalance execution with two-stage limit→market logic, complete with passive repricing, slippage aggregation, and Hyperliquid context helpers.
- **`rebalance.py`** is the cron entry point orchestrating config loading, data fetching, signal generation, execution, performance logging, and notifications.
- **`performance.py` / `notifications.py`** log telemetry (rebalance history, positions, signal tracking) and deliver Telegram/email alerts.

### 2. Set Up API Access (15 minutes)

```bash
# Export API credentials
export HYPERLIQUID_API_KEY="your_api_key_here"
export HYPERLIQUID_API_SECRET="your_api_secret_here"

# Test API connectivity
python -c "
import os
import requests
api_key = os.environ['HYPERLIQUID_API_KEY']
response = requests.post('https://api.hyperliquid.xyz/info', json={'type': 'clearinghouseState', 'user': api_key})
print('API Connected:', response.status_code == 200)
"
```

### 3. Configure Strategy (5 minutes)

Edit `config/gradient_live.json`:
- Set `capital_usd` to your starting capital
- Keep `dry_run: true` for initial testing
- Adjust `max_position_pct` if needed (default 10%)

### 4. Test Dry-Run (30 minutes)

```bash
# Run a single rebalance cycle (dry-run mode)
python -m slipstream.strategies.gradient.live.rebalance

# Check logs
tail -f /var/log/gradient/rebalance.log
```

Expected output:
```
[2025-01-15 12:00:00] Starting rebalance cycle...
[2025-01-15 12:00:05] Fetched data for 183 assets
[2025-01-15 12:00:06] Computed momentum signals
[2025-01-15 12:00:06] Selected 64 long, 64 short positions
[2025-01-15 12:00:06] Target portfolio: 128 positions, $5000 capital
[2025-01-15 12:00:06] DRY-RUN: Would place 128 orders
[2025-01-15 12:00:06] Rebalance complete (dry-run)
```

### 5. Go Live (5 minutes)

When ready to go live:

```json
// In config/gradient_live.json, change:
"dry_run": false
```

Then set up cron job:

```bash
# Edit crontab
crontab -e

# Add this line (runs every 4 hours):
0 */4 * * * cd /root/slipstream && /root/slipstream/.venv/bin/python -m slipstream.strategies.gradient.live.rebalance >> /var/log/gradient/rebalance.log 2>&1
```

## 📋 Implementation Checklist

Before going live, ensure:

- [ ] All 4 core files implemented (data.py, portfolio.py, execution.py, rebalance.py)
- [ ] API credentials configured
- [ ] Dry-run test successful
- [ ] Logging working correctly
- [ ] Cron job configured
- [ ] Emergency stop script ready
- [ ] Starting with small capital ($1k-$5k)

## 🛠️ Helpful Code Snippets

### Reuse Existing Code

Much of the logic already exists in `sensitivity.py`:

```python
# From sensitivity.py, reuse:
from slipstream.strategies.gradient.sensitivity import (
    compute_vol_normalized_returns,
    compute_multispan_momentum,
    filter_universe_by_liquidity,
)

# For data fetching, reuse from scripts/data_load.py:
from slipstream import fetch_candles  # (if you expose it in __init__)
```

### Hyperliquid API Reference

```python
import requests

# Get all perpetual markets
response = requests.post(
    "https://api.hyperliquid.xyz/info",
    json={"type": "meta"}
)
markets = response.json()["universe"]

# Get current positions
response = requests.post(
    "https://api.hyperliquid.xyz/info",
    json={
        "type": "clearinghouseState",
        "user": api_key
    }
)
positions = response.json()["assetPositions"]

# Place order (requires signing)
order = {
    "type": "order",
    "orders": [{
        "coin": "BTC",
        "is_buy": True,
        "sz": 0.1,
        "limit_px": 50000,
        "order_type": {"limit": {"tif": "Ioc"}}
    }]
}
# Note: Requires proper signing with API secret
```

## ⚠️ Safety Reminders

1. **Start Small**: Use $1k-$5k initially
2. **Dry-Run First**: Test for 1-2 days before live
3. **Monitor Closely**: Check logs after each rebalance for first week
4. **Have Emergency Stop**: Keep `scripts/strategies/gradient/live/emergency_stop.py` ready
5. **Check Fills**: Verify all orders fill at reasonable prices

## 📊 Expected Behavior

After going live, you should see:
- Rebalance every 4 hours
- ~128 positions (64 long, 64 short) at 35% concentration
- Near-zero net exposure (market neutral)
- Gradual portfolio growth if backtest performance holds

## 🆘 Emergency Stop

If something goes wrong:

```bash
# Flatten all positions immediately
python -m slipstream.strategies.gradient.live.emergency_stop --flatten-all

# Or manually via API
# (Keep a script ready for this)
```

## 📞 Support

Review docs:
- `docs/strategies/gradient/DEPLOYMENT.md` - Full deployment guide
- `docs/strategies/gradient/README.md` - Strategy overview and sensitivity analysis results

Good luck! 🚀
