# Gradient Strategy - Live Deployment Roadmap

## 🎯 Optimal Configuration (From Sensitivity Analysis)

```
Concentration:     35% (top/bottom 35% of universe)
Rebalancing:       Every 4 hours (aligns with candle frequency)
Weighting:         Inverse-volatility
Expected Sharpe:   ~3.1 annualized (backtest upper bound)
```

## 📋 Deployment Checklist

### Phase 1: Pre-Launch Setup (1-2 hours)

- [ ] **API Access**
  - Create Hyperliquid API keys (mainnet)
  - Set up API key securely (environment variables or secrets manager)
  - Test API connectivity and permissions

- [ ] **Capital Allocation**
  - Decide initial capital (recommend starting small: $1k-$10k)
  - Calculate position sizes based on capital
  - Set maximum position size limits

- [ ] **Infrastructure**
  - Set up server/VPS with Python 3.11+
  - Install dependencies (`uv sync`)
  - Configure cron job for 4-hour execution
  - Set up logging directory

- [ ] **Risk Controls**
  - Define maximum drawdown threshold (e.g., -20%)
  - Set position size limits per asset
  - Configure emergency stop mechanism

### Phase 2: Core Trading Logic (2-3 hours)

**File: `src/slipstream/strategies/gradient/live.py`**

Components needed:
1. **Signal Generator**
   - Fetch latest 4h candles for all perps
   - Compute vol-normalized returns
   - Calculate multi-span EWMA momentum scores
   - Rank assets by momentum

2. **Portfolio Constructor**
   - Select top/bottom 35% by momentum
   - Apply liquidity filter (ADV check)
   - Calculate inverse-vol weights
   - Generate target positions

3. **Order Executor**
   - Get current positions from Hyperliquid
   - Calculate delta (target - current)
   - Place orders to rebalance
   - Handle partial fills and errors

4. **Risk Manager**
   - Check total portfolio value
   - Verify position size limits
   - Implement emergency stop
   - Log all actions

### Phase 3: Cron Job Setup (30 minutes)

**Cron Schedule:**
```bash
# Run every 4 hours (0:00, 4:00, 8:00, 12:00, 16:00, 20:00 UTC)
0 */4 * * * cd /root/slipstream && /root/slipstream/.venv/bin/python -m slipstream.strategies.gradient.live.rebalance >> /var/log/gradient/rebalance.log 2>&1
```

**Script: `src/slipstream/strategies/gradient/live/rebalance.py`**
- Entry point for cron job
- Orchestrates signal generation → portfolio construction → execution
- Comprehensive error handling and logging
- Safe exit on errors (no partial rebalances)

### Phase 4: Monitoring & Alerts (1 hour)

- [ ] **Logging**
  - Timestamped logs for every rebalance
  - Position changes, fills, errors
  - Performance metrics (PnL, positions, exposure)

- [ ] **Alerts**
  - Email/SMS on critical errors
  - Daily performance summary
  - Drawdown warnings

- [ ] **Dashboard** (optional)
  - Simple web dashboard for monitoring
  - Current positions, PnL, exposure
  - Recent rebalance history

## 🏗️ Implementation Plan

### Step 1: Create Live Trading Module

```
src/slipstream/strategies/gradient/live/
├── __init__.py          # Exports main functions
├── data.py              # Fetch candles, compute signals
├── portfolio.py         # Portfolio construction logic
├── execution.py         # Order placement and management
├── performance.py       # Tracking + history
├── rebalance.py         # Main cron job entry point
└── config.py            # Configuration (capital, limits, etc.)
```

### Step 2: Configuration File

**`config/gradient_live.json`:**
```json
{
  "capital_usd": 5000,
  "max_position_pct": 10,
  "max_total_leverage": 2.0,
  "concentration_pct": 35,
  "rebalance_freq_hours": 4,
  "weight_scheme": "inverse_vol",
  "lookback_spans": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
  "vol_span": 24,
  "liquidity_threshold_usd": 10000,
  "liquidity_impact_pct": 2.5,
  "emergency_stop_drawdown_pct": 20
}
```

Values merge with optional layered defaults:

- `config/global.(json|yml)` — shared settings across strategies
- `config/environments/<env>/...` — environment overlays (e.g. staging, prod)
- Run-time overrides supplied via the loader (tests or CLI)

### Step 3: API Client

Reuse Hyperliquid API client from existing codebase:
- Place market orders for immediate execution
- Use limit orders with aggressive pricing if preferred
- Handle rate limits and retries

### Step 4: Baseline Metrics

Capture a parity snapshot before and after the migration:

```bash
python scripts/strategies/gradient/capture_baseline.py \
  --returns-csv data/gradient/returns_4h.csv \
  --signals-csv data/gradient/signals/trend_strength.csv \
  --output-json output/gradient/baseline_metrics.json
```

Adjust the input paths to match your canonical return and signal panels. The JSON
report encodes period count, cumulative return, annualised Sharpe, max drawdown,
average gross exposure, active asset count, and turnover so you can diff against
the pre-migration state.

### Step 5: Testing Protocol

**Dry-Run Mode:**
- Test on paper trading account first
- Verify signal generation matches backtest
- Check order sizing and execution logic
- Monitor for 1-2 days before going live

**Go-Live Checklist:**
- [ ] Dry-run successful for 2+ rebalances
- [ ] All API endpoints working
- [ ] Logging and alerts functional
- [ ] Emergency stop mechanism tested
- [ ] Small capital amount allocated
- [ ] Monitoring dashboard accessible

## ⚠️ Risk Management

### Pre-Trade Checks

1. **Liquidity Filter**
   - Only trade assets where 10k USD < 2.5% of ADV
   - Skip illiquid assets even if momentum is strong

2. **Position Size Limits**
   - Max 10% of capital per position
   - Max 35% long, 35% short (70% gross exposure)
   - Net exposure should be ~0 (market neutral)

3. **Drawdown Protection**
   - Emergency stop at -20% drawdown
   - Reduce size if volatility spikes

### Post-Trade Monitoring

1. **Fill Quality**
   - Log all fills and slippage
   - Alert if slippage > 0.5%

2. **Position Drift**
   - Check position sizes vs targets
   - Alert if significant drift between rebalances

3. **Performance Tracking**
   - Daily PnL calculation
   - Compare to backtest expectations
   - Alert if underperforming by >2σ

## 🚀 Launch Timeline

### Day 0: Pre-Launch
- Complete Phase 1-3 implementation
- Run dry-run mode on paper account
- Set up monitoring and alerts

### Day 1-2: Paper Trading
- Monitor 2-3 rebalance cycles
- Verify signal generation and execution
- Check for any errors or edge cases

### Day 3: Go Live
- Allocate small capital ($1k-$5k)
- Execute first live rebalance
- Monitor closely for 24 hours

### Week 1: Ramp Up
- If performing well, increase capital gradually
- Monitor daily performance vs backtest
- Tune parameters if needed

### Ongoing: Optimization
- Monthly review of performance
- Quarterly re-optimization of parameters
- Continuous monitoring of market conditions

## 📊 Performance Expectations

**Based on backtest (upper bound):**
- 10-Day Return: +2.9% ± 5.6%
- Annualized Sharpe: ~3.1
- Max Observed Drawdown: -7.6% (in 10-day window)

**Realistic expectations (after costs):**
- Annualized Sharpe: 2.0-2.5
- Monthly return: 5-10%
- Drawdowns: 10-15% possible

## 🛠️ Tools & Dependencies

**Required:**
- Python 3.11+
- `uv` for dependency management
- Hyperliquid API access
- Server/VPS with stable internet

**Optional:**
- Telegram bot for alerts
- Grafana/dashboard for monitoring
- Database for historical tracking

## 📞 Emergency Procedures

### If Strategy Underperforms
1. Check if market regime has changed
2. Verify signal generation is working correctly
3. Review recent fills for slippage issues
4. Consider reducing size or pausing

### If System Fails
1. Emergency stop via manual API call
2. Flatten all positions
3. Debug issue before resuming
4. Keep emergency shutdown script ready

### Emergency Contact Script
```bash
# emergency_stop.sh
python -m slipstream.strategies.gradient.live.emergency_stop --flatten-all
```

## 🎓 Best Practices

1. **Start Small**: Begin with 1-5% of intended capital
2. **Monitor Closely**: Check daily for first week
3. **Keep It Simple**: Don't over-optimize, stick to backtest params
4. **Log Everything**: Comprehensive logs are crucial for debugging
5. **Have an Exit Plan**: Know when to stop (drawdown limits, time limits)
6. **Stay Disciplined**: Don't manually override the system

## 📁 Deliverables

1. `src/slipstream/strategies/gradient/live/` - Live trading module
2. `config/gradient_live.json` - Configuration file
3. `scripts/strategies/gradient/live/rebalance.sh` - Cron job wrapper (legacy alias: `scripts/live/gradient_rebalance.sh`)
4. `scripts/strategies/gradient/live/emergency_stop.py` - Emergency shutdown (legacy alias: `scripts/live/gradient_emergency_stop.py`)
5. `docs/GRADIENT_LIVE_OPERATIONS.md` - Operational manual

## Next Steps

1. Implement live trading module (2-3 hours)
2. Set up API keys and configuration (30 min)
3. Deploy to server and configure cron (30 min)
4. Run paper trading for 1-2 days
5. Go live with small capital
