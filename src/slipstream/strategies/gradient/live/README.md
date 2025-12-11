# Gradient Live Trading - Quick Start



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
