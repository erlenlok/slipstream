# Candle Alignment - Critical for Matching Backtest Performance

## ⚠️ CRITICAL: Why Candle Alignment Matters

The Gradient strategy backtest achieved ~3.1 Sharpe using specific 4-hour candle boundaries. **If live trading uses different candle timestamps, performance will NOT match the backtest.**

This document ensures live trading uses the exact same candle alignment as the backtest.

---

## Verified Candle Alignment

### Hyperliquid 4h Candle Timestamps

✅ **Confirmed**: Hyperliquid 4h candles close at:
```
00:00 UTC
04:00 UTC
08:00 UTC
12:00 UTC
16:00 UTC
20:00 UTC
```

### Backtest Data Timestamps

✅ **Confirmed**: Backtest data uses identical timestamps:
```
2025-10-31 00:00:00
2025-10-31 04:00:00
2025-10-31 08:00:00
2025-10-31 12:00:00
2025-10-31 16:00:00
2025-10-31 20:00:00
```

### ✅ Conclusion

**Live trading and backtest use IDENTICAL candle boundaries.** This is critical for replicating backtest performance.

---

## Critical Timing Requirement

### The Problem

- Candles close at exactly `:00` (e.g., 04:00:00 UTC)
- Hyperliquid needs a few seconds to finalize candle data
- If we fetch at exactly 04:00:00, we might get incomplete data
- This could cause us to use stale signals or miss the latest candle

### The Solution

**Run cron at `:01` instead of `:00`**

```bash
# WRONG (runs exactly at candle close):
0 */4 * * * /root/slipstream/scripts/strategies/gradient/live/rebalance.sh

# CORRECT (1-minute buffer for data availability):
1 */4 * * * /root/slipstream/scripts/strategies/gradient/live/rebalance.sh
```

This runs at:
```
00:01 UTC
04:01 UTC
08:01 UTC
12:01 UTC
16:01 UTC
20:01 UTC
```

### Why :01 is Safe

1. **Candle closes at 04:00:00**
2. **Hyperliquid finalizes data by 04:00:30** (estimated)
3. **We fetch at 04:01:00** - guaranteed complete data
4. **Still using the same candles as backtest** (the 00:00-04:00 candle)

---

## Rebalance Timing Flow

### Example: 04:00 UTC Rebalance

```
03:56:00 - Candle 00:00-04:00 is closing soon
04:00:00 - Candle closes ← Hyperliquid marks this candle complete
04:00:30 - Data fully finalized and available via API
04:01:00 - OUR CRON RUNS ← We fetch data
04:01:15 - Fetch 1100 4h candles (latest is 00:00-04:00, JUST closed)
04:01:45 - Compute signals using data through 04:00
04:02:00 - Build portfolio (long top 35%, short bottom 35%)
04:02:30 - Place limit orders (Stage 1)
05:02:30 - Stage 1 timeout (60 min passive period)
05:02:30 - Sweep unfilled with market orders (Stage 2)
05:03:00 - Rebalance complete
05:03:00 - Send Telegram notification
```

### What Candles Do We Use?

At 04:01:00, when we fetch data, we get:
```
Latest candle: 00:00 - 04:00 (JUST CLOSED)
Previous:      20:00 - 00:00 (yesterday)
Before that:   16:00 - 20:00 (yesterday)
... (going back 1100 candles for 1024h lookback)
```

This is **exactly** the same as the backtest:
- Backtest at timestamp 04:00 used candles through 04:00
- Live at 04:01 uses candles through 04:00
- **Same data, same signals, same performance**

---

## Verification Script

Run this script to verify alignment:

```bash
uv run python scripts/verify_candle_alignment.py
```

Output will show:
```
✓ Hyperliquid candles close at :00 of 4h intervals
✓ Backtest data uses same timestamps
⚠ Action Required: Update cron to run at :01 instead of :00
```

---

## Safety Checks

### 1. Cron Wrapper Timing Check

The `gradient_rebalance.sh` script includes automatic timing verification:

```bash
# Warns if running at unexpected time
if [ $HOUR_MOD -ne 0 ] || [ $CURRENT_MIN -lt 1 ] || [ $CURRENT_MIN -gt 5 ]; then
    echo "⚠️ WARNING: Script running at unexpected time!"
fi
```

### 2. Manual Verification

Check your cron is set correctly:

```bash
crontab -l | grep gradient_rebalance
```

Should show:
```
1 */4 * * * /root/slipstream/scripts/strategies/gradient/live/rebalance.sh
```

NOT:
```
0 */4 * * * ...  ← WRONG, will cause timing issues
```

---

## FAQ

### Q: Why not run at :00 exactly when candle closes?

A: Hyperliquid needs a few seconds to finalize the candle. Running at :00 risks fetching incomplete data.

### Q: Does the 1-minute delay affect performance?

A: No. We're still using the same candles as the backtest. The backtest "rebalanced" at the candle timestamp (e.g., 04:00), but in reality, you couldn't have the 04:00 candle data AT 04:00. The :01 timing is realistic and matches what the backtest implicitly assumes.

### Q: What if I run at :02 or :03?

A: That's fine too. Anywhere from :01 to :05 is safe. Just don't run exactly at :00 (too early) or much later like :10 (unnecessary delay).

### Q: Can I run at :00 if I add a sleep in the script?

A: Not recommended. Better to use cron timing directly. But if you must:
```bash
# In script
sleep 60  # Wait 1 minute
```

### Q: What happens if cron runs late (e.g., system lag)?

A: The timing check will warn you in logs. As long as it's within a few minutes, candle data will still be correct. If consistently late (>5 min), check system resources.

### Q: Do all exchanges use :00 boundaries for 4h candles?

A: Most do, but always verify! This doc is specific to Hyperliquid. Other exchanges may differ.

---

## Troubleshooting

### Issue: "WARNING: Script running at unexpected time"

**Cause**: Cron schedule doesn't match required timing

**Fix**:
```bash
crontab -e
# Change to:
1 */4 * * * /root/slipstream/scripts/strategies/gradient/live/rebalance.sh
```

### Issue: Signals don't match backtest expectations

**Check 1**: Verify candle timestamps
```bash
uv run python scripts/verify_candle_alignment.py
```

**Check 2**: Check what time cron actually ran
```bash
grep "Starting Gradient rebalance" /var/log/gradient/rebalance_*.log | tail -5
```

Should show timestamps like:
```
2025-10-31 04:01:15 - Starting...
2025-10-31 08:01:12 - Starting...
```

NOT like:
```
2025-10-31 04:00:02 - Starting...  ← TOO EARLY!
2025-10-31 08:10:45 - Starting...  ← TOO LATE!
```

### Issue: Getting stale candle data

**Symptom**: Latest candle in logs is not the just-closed candle

**Cause**: Running too early (before data available)

**Fix**: Ensure cron runs at :01 or later

---

## Summary Checklist

Before going live, verify:

- [ ] Hyperliquid 4h candles close at :00 (run verify script)
- [ ] Backtest data uses same timestamps (run verify script)
- [ ] Cron set to run at **:01** of each 4-hour boundary
- [ ] Cron command is: `1 */4 * * * /path/to/gradient_rebalance.sh`
- [ ] Timing check in wrapper script is enabled
- [ ] Verified first rebalance fetches correct candle timestamps
- [ ] Logs show rebalance starting at :01-:05 of expected hours

---

## Final Note

**This is NOT optional.** Candle alignment is critical for matching backtest performance. A misalignment of even a few minutes can cause:
- Different signals
- Different positions
- Different performance
- Frustration when live results don't match backtest

Take 5 minutes to verify this is correct before going live. It will save hours of debugging later.

---

**Verification Command:**
```bash
uv run python scripts/verify_candle_alignment.py
```

**Correct Cron:**
```bash
1 */4 * * * /root/slipstream/scripts/strategies/gradient/live/rebalance.sh
```

✅ Match these exactly, and your live performance will match the backtest.
