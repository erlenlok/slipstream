# Gradient Live Trading Implementation Plan

**Goal**: Get the Gradient momentum strategy trading live in small size ASAP
**Strategy**: 35% concentration, 4h rebalancing, inverse-vol weighting
**Expected Performance**: ~3.1 annualized Sharpe (backtest upper bound)

## Backtest Results Summary

From 100 random 10-day periods:
- **Concentration**: 35% (top/bottom 35% of universe by momentum)
- **Rebalancing**: Every 4 hours (each candle)
- **Weighting**: Inverse-volatility
- **Mean 10-day return**: 2.90% ± 5.58%
- **Sharpe (10-day)**: 0.52 → **~3.1 annualized**
- **Max drawdown (10-day)**: -7.6%

Strategy characteristics:
- Long the top 35% highest momentum perps
- Short the bottom 35% lowest momentum perps
- Weight by inverse volatility (lower vol = higher weight)
- Rebalance every 4 hours

## Implementation Timeline: 6-8 Hours

### Phase 1: Core Dependencies (30 min)
- [ ] Install Hyperliquid Python SDK: `pip install hyperliquid`
- [ ] Set up API keys (mainnet)
- [ ] Test SDK connection and auth
- [ ] Update pyproject.toml with hyperliquid dependency

### Phase 2: Signal Generation (1 hour)
**File: `src/slipstream/gradient/live/data.py`**

Implement `fetch_live_data()` and `compute_live_signals()`:

```python
def fetch_live_data(config):
    """
    Fetch 4h candles for all perps from Hyperliquid.
    Need 1024+ candles for longest lookback window.

    Steps:
    1. Get all perpetual markets
    2. For each market, fetch last 1100 4h candles
    3. Return panel DataFrame
    """
    pass

def compute_live_signals(market_data, config):
    """
    Compute momentum signals from market data.

    Steps:
    1. Compute log returns
    2. Compute EWMA volatility (24h span)
    3. Compute vol-normalized returns
    4. Compute multi-span EWMA momentum (sum across all lookbacks)
    5. Compute ADV and filter universe (liquidity check)
    6. Rank by momentum score
    7. Return DataFrame: [asset, momentum_score, vol_24h, adv_usd]
    """
    pass
```

**Reuse existing code** from `src/slipstream/gradient/sensitivity.py`:
- `compute_log_returns()`
- `compute_ewma_vol()`
- `compute_adv_usd()`
- `filter_universe_by_liquidity()`
- Multi-span momentum logic

### Phase 3: Portfolio Construction (30 min)
**File: `src/slipstream/gradient/live/portfolio.py`**

Implement `construct_target_portfolio()`:

```python
def construct_target_portfolio(signals, config):
    """
    Select top/bottom 35% and apply inverse-vol weighting.

    Steps:
    1. Sort signals by momentum_score
    2. Select top 35% (long bucket)
    3. Select bottom 35% (short bucket)
    4. Apply inverse-vol weights: w_i = (1/vol_i) / sum(1/vol_j)
    5. Scale to USD positions based on config.capital_usd
    6. Apply position size limits (max 10% per asset)
    7. Return Dict[asset, position_size_usd]
    """
    pass
```

### Phase 4: Order Execution with Passive → Aggressive (3-4 hours)
**File: `src/slipstream/gradient/live/execution.py`**

This is the most complex part. Implement multi-stage execution:

#### 4.1: Get Current Positions (30 min)
```python
def get_current_positions(sdk):
    """
    Fetch current positions from Hyperliquid.

    Use SDK:
    from hyperliquid import HyperliquidSync
    positions = sdk.info.user_state(address)

    Return: Dict[asset, position_size_usd]
    """
    pass
```

#### 4.2: Two-Stage Execution Logic (2-3 hours)
```python
def execute_rebalance_with_stages(target, current, sdk, config):
    """
    Two-stage execution:
    Stage 1 (0-60 min): Place limit orders (passive, pay no fees)
    Stage 2 (60+ min): Sweep unfilled with market orders (aggressive)

    Timeline:
    00:00 - Compute signals and targets
    00:01 - Place all limit orders at mid/better prices
    00:01-01:00 - Monitor fills via WebSocket
    01:00 - Cancel unfilled orders, sweep with market orders

    Args:
        target: Target positions
        current: Current positions
        sdk: Hyperliquid SDK instance
        config: Configuration

    Returns:
        Execution summary dict
    """

    # Calculate deltas
    deltas = calculate_deltas(target, current)

    # Stage 1: Place limit orders
    stage1_orders = place_limit_orders(deltas, sdk, config)

    # Monitor fills for up to 60 minutes
    filled, unfilled = monitor_fills_with_timeout(
        orders=stage1_orders,
        sdk=sdk,
        timeout_seconds=3600  # 1 hour
    )

    # Stage 2: Sweep unfilled with market orders
    if unfilled and time_elapsed >= 3600:
        cancel_orders(unfilled, sdk)
        stage2_orders = place_market_orders(unfilled, sdk, config)

    return aggregate_results(filled, stage2_orders)
```

#### 4.3: Limit Order Placement
```python
def place_limit_orders(deltas, sdk, config):
    """
    Place limit orders at mid price or better.

    For each delta:
    1. Get current bid/ask from orderbook
    2. If buying: place limit at mid or bid (join best bid)
    3. If selling: place limit at mid or ask (join best ask)
    4. Use post-only orders to ensure no taker fees

    Use SDK:
    order = sdk.create_limit_order(
        symbol=f"{asset}/USDC:USDC",
        side="buy" or "sell",
        amount=size_coin,
        price=limit_price,
        params={"postOnly": True}  # Ensure passive order
    )
    """
    pass
```

#### 4.4: WebSocket Monitoring
```python
def monitor_fills_with_timeout(orders, sdk, timeout_seconds):
    """
    Monitor order fills via WebSocket with timeout.

    Use SDK WebSocket:
    from hyperliquid import HyperliquidWs
    ws = HyperliquidWs({})

    Loop:
    1. Subscribe to user fills
    2. Check fill status every 10 seconds
    3. Update filled/unfilled lists
    4. Exit after timeout or all filled

    Returns: (filled_orders, unfilled_orders)
    """
    pass
```

#### 4.5: Market Order Sweep
```python
def place_market_orders(unfilled_deltas, sdk, config):
    """
    Place market orders for remaining unfilled positions.

    For each unfilled delta:
    1. Place market order (aggressive, takes liquidity)
    2. Accept taker fees (~0.035%)
    3. Log fill and move on

    Use SDK:
    order = sdk.create_order(
        symbol=f"{asset}/USDC:USDC",
        type="market",
        side="buy" or "sell",
        amount=size_coin
    )
    """
    pass
```

### Phase 5: Main Orchestrator (30 min)
**File: `src/slipstream/gradient/live/rebalance.py`**

Already mostly implemented. Update to use new execution logic:

```python
def run_rebalance():
    # ... existing setup code ...

    # Initialize SDK
    from hyperliquid import HyperliquidSync
    sdk = HyperliquidSync({
        "apiKey": os.getenv("HYPERLIQUID_API_KEY"),
        "secret": os.getenv("HYPERLIQUID_SECRET")
    })

    # Fetch data and signals (reuse existing)
    market_data = fetch_live_data(config)
    signals = compute_live_signals(market_data, config)

    # Construct portfolio (reuse existing)
    target = construct_target_portfolio(signals, config)
    current = get_current_positions(sdk)

    # Execute with two-stage logic (NEW)
    results = execute_rebalance_with_stages(target, current, sdk, config)

    # Log results
    logger.info(f"Stage 1 fills: {results['stage1_filled']}")
    logger.info(f"Stage 2 fills: {results['stage2_filled']}")
    logger.info(f"Total turnover: ${results['total_turnover']}")
```

### Phase 6: Cron Setup (15 min)
**File: `scripts/live/gradient_rebalance.sh`**

Already created. Just needs testing.

Set up cron:
```bash
crontab -e

# Add this line (runs at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)
0 */4 * * * /root/slipstream/scripts/live/gradient_rebalance.sh >> /var/log/gradient/cron.log 2>&1
```

### Phase 7: Testing & Validation (2 hours)
- [ ] Test signal generation on recent data
- [ ] Verify momentum scores match backtest
- [ ] Test portfolio construction (35% top/bottom)
- [ ] Test inverse-vol weighting
- [ ] Dry-run full rebalance (no orders)
- [ ] Test limit order placement (small size)
- [ ] Test WebSocket monitoring
- [ ] Test market order fallback
- [ ] Verify logging works
- [ ] Test emergency stop script

### Phase 8: Go-Live (30 min)
- [ ] Set `dry_run: false` in config
- [ ] Set small capital ($1k-$5k)
- [ ] Manually run first rebalance
- [ ] Monitor closely for 1 hour
- [ ] Verify orders placed correctly
- [ ] Verify fills recorded
- [ ] Enable cron job

## Configuration

**File: `config/gradient_live.json`**

```json
{
  "strategy": "gradient_momentum_35pct_4h",
  "capital_usd": 5000,
  "max_position_pct": 10,
  "concentration_pct": 35,
  "rebalance_freq_hours": 4,
  "weight_scheme": "inverse_vol",
  "lookback_spans": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
  "vol_span": 24,
  "liquidity_threshold_usd": 10000,
  "liquidity_impact_pct": 2.5,

  "execution": {
    "passive_timeout_seconds": 3600,
    "limit_order_aggression": "join_best",
    "cancel_before_market_sweep": true,
    "min_order_size_usd": 10
  },

  "dry_run": true,
  "api": {
    "endpoint": "https://api.hyperliquid.xyz"
  },
  "logging": {
    "dir": "/var/log/gradient",
    "level": "INFO"
  }
}
```

## Environment Variables

```bash
# Add to ~/.bashrc or use secrets manager
export HYPERLIQUID_API_KEY="0x..."
export HYPERLIQUID_SECRET="your_secret_key"
```

## Risk Controls

### Pre-Trade Checks
1. **Liquidity filter**: Only trade if $10k < 2.5% of ADV
2. **Position limits**: Max 10% of capital per position
3. **Gross exposure**: ~70% (35% long + 35% short)
4. **Net exposure**: ~0% (market neutral)

### During Execution
1. **Stage 1 timeout**: Move to aggressive after 1 hour
2. **Partial fills accepted**: Don't roll back, just log
3. **Error handling**: Skip failed orders, continue to next

### Post-Trade Monitoring
1. **Check fill prices**: Alert if slippage > 1%
2. **Verify position sizes**: Alert if delta > 20% from target
3. **Log all actions**: Full audit trail

## Emergency Procedures

### Stop Trading Immediately
```bash
# Cancel all open orders
python scripts/live/gradient_emergency_stop.py --cancel-orders

# Flatten all positions (go to cash)
python scripts/live/gradient_emergency_stop.py --flatten-all
```

### Disable Cron
```bash
crontab -e
# Comment out the gradient line with #
```

## Expected Performance

**Backtest (100 random 10-day periods):**
- Mean return: +2.90% per 10 days
- Std dev: 5.58%
- Sharpe: 0.52 (10-day) → **3.1 annualized**
- Max drawdown: -7.6%

**Realistic after costs:**
- Limit order execution: ~50-70% passive (no fees)
- Market order execution: ~30-50% aggressive (0.035% taker fee)
- Estimated blended cost: ~0.01-0.02% per trade
- Turnover: ~70% of capital every 4h = ~420% daily
- Daily cost: 420% * 0.015% = 0.06% per day = 22% annualized

**Net expected Sharpe**: 2.0-2.5 after costs

## Files to Modify/Create

### Modify (already exist, need implementation)
1. `src/slipstream/gradient/live/data.py` - Signal generation
2. `src/slipstream/gradient/live/portfolio.py` - Portfolio construction
3. `src/slipstream/gradient/live/execution.py` - Order execution
4. `src/slipstream/gradient/live/rebalance.py` - Update to use SDK

### Create (new files)
1. `src/slipstream/gradient/live/websocket_monitor.py` - WebSocket fill monitoring
2. `tests/test_gradient_live.py` - Unit tests for live trading

### Update (configuration)
1. `pyproject.toml` - Add hyperliquid dependency
2. `config/gradient_live.json` - Add execution config

## Testing Checklist

### Unit Tests
- [ ] Signal computation matches backtest
- [ ] Portfolio construction selects correct 35%
- [ ] Inverse-vol weighting sums to 1.0 per side
- [ ] Delta calculation correct
- [ ] Position size limits enforced

### Integration Tests (Dry-Run)
- [ ] Fetch live data for all markets
- [ ] Compute signals for 100+ assets
- [ ] Construct portfolio (35% long, 35% short)
- [ ] Calculate deltas from current positions
- [ ] Simulate order placement (log only)
- [ ] Verify gross exposure ≈ 70% of capital

### Live Tests (Small Size)
- [ ] Place 1 limit order, verify it posts
- [ ] Monitor fill via WebSocket
- [ ] Cancel order after 30 seconds
- [ ] Place 1 market order, verify immediate fill
- [ ] Check position appears in account

### Full Cycle Test
- [ ] Run full rebalance with $500 capital
- [ ] Monitor for 1 hour (stage 1)
- [ ] Verify market sweep (stage 2)
- [ ] Check final positions match targets (±20%)
- [ ] Review logs for errors

## Success Criteria

**Go-live when:**
1. ✅ All unit tests pass
2. ✅ Dry-run completes without errors
3. ✅ Signals match backtest expectations
4. ✅ Small test orders execute correctly
5. ✅ Full cycle test succeeds
6. ✅ Logs capture all actions
7. ✅ Emergency stop works

**After 1 week:**
- No critical errors in logs
- Fill rate > 80% (stage 1 + stage 2)
- Position drift < 30% from targets
- Performance tracking vs backtest

**After 1 month:**
- Sharpe ratio > 1.0 (realistic floor)
- Max drawdown < 15%
- No system downtime
- Consider scaling capital

## Next Steps

1. **Install dependencies**: `pip install hyperliquid`
2. **Implement data.py**: Signal generation (1 hour)
3. **Implement portfolio.py**: Portfolio construction (30 min)
4. **Implement execution.py**: Two-stage execution (3-4 hours)
5. **Test thoroughly**: Dry-run + small live tests (2 hours)
6. **Go live**: Start with $1k-$5k (monitor closely)

**Total time to live trading: 6-8 hours of focused work**
