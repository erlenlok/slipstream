# Gradient Strategy - Setup Complete ✅

## Summary

The Gradient momentum strategy is now fully configured and ready for automated trading on Hyperliquid.

### What Was Fixed

**Primary Issue**: Wallet address confusion
- The system was querying the API vault (`0x998c0B58...`) for positions instead of the main wallet (`0xFd5cf66Cf037140A477419B89656E5F735fa82f4`)
- API vault is used only for signing trades - it holds no positions
- Main wallet is where your funds and positions actually live

**Solution**:
- Added `HYPERLIQUID_MAIN_WALLET` environment variable
- Updated `get_current_positions()` to use `clearinghouseState` API with main wallet
- Now correctly fetches positions from the right wallet

### Configuration

**Cron Schedule**: Every 4 hours at :01 (1:01, 5:01, 9:01, 13:01, 17:01, 21:01 UTC)

**Environment Variables** (configured in crontab):
```bash
HYPERLIQUID_API_SECRET=0xc414e72b... # Private key for signing
HYPERLIQUID_API_KEY=0x998c0B58193faca878B55aE29165b68167A1BD30  # API vault
HYPERLIQUID_MAIN_WALLET=0xFd5cf66Cf037140A477419B89656E5F735fa82f4  # Main wallet
REDIS_ENABLED=true
```

**Strategy Parameters** (`config/gradient_live.json`):
- Capital: $400 USD
- Concentration: 10% (top/bottom 10% of liquid assets)
- Max position: 10% per asset
- Max leverage: 2.0x
- Rebalance: Every 4 hours
- Weighting: Inverse volatility
- Dry-run: **FALSE** (live trading enabled)

**Execution Settings**:
- Passive timeout: 3600 seconds (1 hour for limit orders)
- Limit order aggression: join_best (top of book)
- Cancel before market sweep: true
- Minimum order size: $10 USD

### File Locations

**Code**:
- Main script: `/root/slipstream/src/slipstream/strategies/gradient/live/rebalance.py`
- Execution logic: `/root/slipstream/src/slipstream/strategies/gradient/live/execution.py` ✨ FIXED
- Configuration: `/root/slipstream/config/gradient_live.json`

**Logs**:
- Rebalance logs: `/var/log/gradient/rebalance_YYYYMMDD.log`
- Position history: `/var/log/gradient/positions_history.jsonl`
- Rebalance history: `/var/log/gradient/rebalance_history.jsonl`

**Test Scripts**:
- Full workflow test: `/root/slipstream/test_full_workflow.py`
- Documentation: `/root/slipstream/WALLET_CONFIGURATION.md`

### Verification

**Test the setup**:
```bash
# Run manual rebalance (will execute immediately)
export HYPERLIQUID_MAIN_WALLET="0xFd5cf66Cf037140A477419B89656E5F735fa82f4"
uv run python -m slipstream.strategies.gradient.live.rebalance

# Check logs
tail -f /var/log/gradient/rebalance_$(date +%Y%m%d).log

# View position history
cat /var/log/gradient/positions_history.jsonl | jq '.'
```

**Next cron execution**:
```bash
# Show next few runs
date && echo "Next runs: 1:01, 5:01, 9:01, 13:01, 17:01, 21:01 UTC"
```

**Monitor cron logs**:
```bash
# Watch for cron execution
grep CRON /var/log/syslog | grep gradient | tail -20
```

### Expected Behavior

Each rebalance cycle:
1. Fetches latest 4h candles for ~219 perpetual markets (with Redis caching)
2. Computes momentum signals across 10 lookback periods
3. Filters to liquid assets (>$10k trade size, <2.5% impact)
4. Selects top/bottom 10% by momentum
5. Constructs inverse-volatility weighted portfolio
6. Fetches current positions from **main wallet** ✅
7. Calculates position deltas
8. Places limit orders at top of book
9. Waits up to 1 hour for fills
10. Sweeps unfilled orders with market orders
11. Logs results and performance metrics

### Monitoring

**Check system status**:
```bash
# Verify cron is running
systemctl status cron

# View crontab
crontab -l

# Check Redis (cache)
redis-cli ping

# Test API connectivity
curl -s -X POST https://api.hyperliquid.xyz/info -H "Content-Type: application/json" -d '{"type": "meta"}' | jq '.universe | length'
```

**Performance tracking**:
- Position history: `/var/log/gradient/positions_history.jsonl`
- Rebalance history: `/var/log/gradient/rebalance_history.jsonl`
- Daily logs: `/var/log/gradient/rebalance_YYYYMMDD.log`

### Safety Features

✅ **Emergency stop**: Set `dry_run: true` in config to halt live trading
✅ **Position limits**: Max 10% per asset, 2.0x total leverage
✅ **Liquidity filter**: Only trades liquid assets (>$10k size)
✅ **Min order size**: $10 USD minimum to avoid dust
✅ **Rate limiting**: Built-in backoff for API 429 errors
✅ **Redis caching**: Reduces API load dramatically

### Next Steps

1. **Monitor first few rebalances** closely
2. **Check position history** to verify trades execute correctly
3. **Review logs** for any errors or warnings
4. **Adjust capital** in `config/gradient_live.json` if needed
5. **Set up Telegram alerts** (optional - add tokens to config)

### Emergency Procedures

**Stop trading immediately**:
```bash
# Method 1: Set dry-run mode
nano config/gradient_live.json  # Change "dry_run": true

# Method 2: Disable cron job
crontab -e  # Comment out the gradient line

# Method 3: Flatten all positions
python scripts/strategies/gradient/live/emergency_stop.py --flatten-all
```

**Check account status**:
```bash
# View current positions
export HYPERLIQUID_MAIN_WALLET="0xFd5cf66Cf037140A477419B89656E5F735fa82f4"
uv run python -c "
from slipstream.strategies.gradient.live.config import load_config
from slipstream.strategies.gradient.live.execution import get_current_positions
config = load_config()
positions = get_current_positions(config)
for asset, value in positions.items():
    print(f'{asset}: {'LONG' if value > 0 else 'SHORT'} \${abs(value):.2f}')
"
```

## Setup Status: ✅ COMPLETE

- [x] Environment variables configured
- [x] Cron job scheduled (every 4 hours at :01)
- [x] Main wallet authentication fixed
- [x] Test workflow verified
- [x] Logs directory created
- [x] Redis cache enabled
- [x] Configuration validated
- [x] Emergency procedures documented

**The system is ready for automated trading!** 🚀

Next rebalance will run at the next :01 minute of a 4-hour interval (1:01, 5:01, 9:01, 13:01, 17:01, 21:01 UTC).
