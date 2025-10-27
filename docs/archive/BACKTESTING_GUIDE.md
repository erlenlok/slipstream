## Backtesting Guide: Full Strategy Simulation

## Overview

The Slipstream backtesting framework implements the walk-forward simulation from `strategy_spec.md` Section 4.2, allowing you to test the complete strategy with:

- Beta-neutral portfolio optimization
- Transaction costs (fees + market impact)
- Discrete lot sizing
- Multiple holding periods H

## Quick Start

### 1. Prepare Trained Models

First, train joint alpha + funding models for your target H:

```bash
# Train models for H=8 hours
python scripts/find_optimal_H_joint.py --H 8 --n-bootstrap 1000
```

This generates:
- `data/features/joint_models/joint_model_H8.json`
- Trained alpha and funding model coefficients

### 2. Generate Predictions

Use the trained models to generate out-of-sample predictions on your backtest period.

### 3. Run Backtest

```python
from slipstream.portfolio import (
    run_backtest,
    BacktestConfig,
    TransactionCostModel,
)

# Configure backtest
config = BacktestConfig(
    H=8,  # 8-hour rebalancing
    start_date='2024-01-01',
    end_date='2024-12-31',
    initial_capital=1_000_000,
    leverage=1.0,
    use_costs=True,
    use_discrete_lots=False,
)

# Create cost model
cost_model = TransactionCostModel.create_default(n_assets=len(assets))

# Run simulation
result = run_backtest(
    config=config,
    alpha_price=alpha_predictions,  # DataFrame (timestamp, asset)
    alpha_funding=funding_predictions,  # DataFrame (timestamp, asset)
    beta=beta_exposures,  # DataFrame (timestamp, asset)
    S=covariance_dict,  # Dict[timestamp -> np.ndarray]
    realized_returns=actual_returns,
    realized_funding=actual_funding,
    cost_model=cost_model,
)

# View results
print(result.summary())
print(f"Sharpe Ratio: {result.sharpe_ratio():.2f}")
print(f"Max Drawdown: {result.max_drawdown():.2%}")

# Plot equity curve
result.equity_curve.plot(title='Backtest Equity Curve')
```

## Components

### Portfolio Optimization

The optimizer implements the closed-form solution from `strategy_spec.md` Section 2.4:

```python
from slipstream.portfolio import optimize_portfolio

# Cost-free optimization
w_optimal = optimize_portfolio(
    alpha=alpha_total,  # α_price - F_hat
    beta=beta_exposures,
    S=covariance_matrix,
    leverage=1.0,
)

# Verify constraints
assert abs(w_optimal @ beta_exposures) < 1e-6  # Beta neutral
assert abs(np.abs(w_optimal).sum() - 1.0) < 1e-3  # Leverage = 1
```

### Cost-Aware Optimization

For realistic simulations, include transaction costs:

```python
from slipstream.portfolio import optimize_portfolio_with_costs

w_optimal, info = optimize_portfolio_with_costs(
    alpha=alpha_total,
    beta=beta_exposures,
    S=covariance_matrix,
    w_old=current_weights,
    cost_linear=cost_model.fee_rate,  # 2 bps default
    cost_impact=cost_model.impact_coef,  # λ_i for impact
    leverage=1.0,
)

print(f"Expected return: {info['expected_return']:.4f}")
print(f"Transaction cost: {info['transaction_cost']:.4f}")
print(f"Turnover: {info['turnover']:.4f}")
```

### Transaction Cost Model

Costs follow `strategy_spec.md` Section 3.3:

```
C(Δw) = Σ |Δw_i| * fee_rate_i + Σ λ_i |Δw_i|^1.5
```

Create custom cost model from liquidity data:

```python
from slipstream.portfolio import TransactionCostModel

# Default (uniform costs)
cost_model = TransactionCostModel.create_default(n_assets=100)

# From liquidity metrics
cost_model = TransactionCostModel.from_liquidity_metrics(
    assets=['BTC', 'ETH', 'SOL', ...],
    liquidity_df=liquidity_data,
    base_fee_rate=0.0002,  # 2 bps
    impact_scale=0.001,
)
```

### Discrete Lot Sizing

For production trading, round to discrete lot sizes with beta repair:

```python
from slipstream.portfolio import round_to_lots

# Get ideal continuous weights
w_ideal = optimize_portfolio_with_costs(...)

# Round to discrete lots
w_discrete = round_to_lots(
    w_ideal=w_ideal,
    beta=beta_exposures,
    S=covariance_matrix,
    capital=1_000_000,
    min_trade_size=min_trade_dollars,  # Per-asset minimums
    beta_tolerance=0.01,
)

# Verify: still beta neutral
assert abs(w_discrete @ beta_exposures) < 0.01
```

## Backtest Workflow

### Step 1: Generate PCA Factors

```bash
# Generate timescale-matched PCA for H=8
python scripts/find_optimal_horizon.py --H 8 --K 30 --weight-method sqrt
```

### Step 2: Train Alpha + Funding Models

```bash
# Train joint models
python scripts/find_optimal_H_joint.py --H 8 --n-bootstrap 1000
```

### Step 3: Create Backtest Script

```python
# backtest_h8.py

import pandas as pd
import numpy as np
from slipstream.portfolio import run_backtest, BacktestConfig, TransactionCostModel
from slipstream.alpha import load_all_returns, load_all_funding

# Load data
returns = load_all_returns()
funding = load_all_funding()

# Load trained models
import json
with open('data/features/joint_models/joint_model_H8.json') as f:
    model = json.load(f)

alpha_coeffs = model['alpha_results']['coefficients']
funding_coeffs = model['funding_results']['coefficients']

# Generate predictions (simplified - full implementation needed)
# alpha_pred = X_alpha @ alpha_coeffs
# funding_pred = X_funding @ funding_coeffs

# Load PCA factors for beta
pca_data = pd.read_csv('data/features/pca_factor_H8_K30_sqrt.csv')
beta = pca_data.pivot(columns='asset', values='loading')

# Run backtest
config = BacktestConfig(
    H=8,
    start_date='2024-06-01',
    end_date='2025-10-01',
    leverage=1.0,
    use_costs=True,
)

result = run_backtest(
    config=config,
    alpha_price=alpha_pred,
    alpha_funding=funding_pred,
    beta=beta,
    S=covariance_dict,
    realized_returns=returns,
    realized_funding=funding,
)

# Analyze results
print(result.summary())
result.equity_curve.to_csv('backtest_results/equity_H8.csv')
result.trades.to_csv('backtest_results/trades_H8.csv')
```

## Performance Metrics

The `BacktestResult` object provides:

```python
# Summary statistics
metrics = result.summary()
# {
#   'total_return': 0.234,
#   'sharpe_ratio': 2.15,
#   'max_drawdown': -0.08,
#   'final_capital': 1_234_000,
#   'n_rebalances': 456,
#   'avg_turnover': 0.35,
#   'total_costs': 12_340,
#   'avg_n_positions': 42.3,
# }

# Time series
result.equity_curve  # Series of capital over time
result.returns  # Series of period returns
result.positions  # DataFrame of weights over time
result.trades  # DataFrame with each rebalance

# Compute custom metrics
annual_return = result.returns.mean() * 365.25 * 24 / config.H
annual_vol = result.returns.std() * np.sqrt(365.25 * 24 / config.H)
```

## Advanced: H* Optimization

Run backtests for multiple H values to find optimal holding period:

```python
H_values = [4, 8, 12, 24, 48]
results = {}

for H in H_values:
    config = BacktestConfig(H=H, ...)
    result = run_backtest(config, ...)
    results[H] = result

# Compare Sharpe ratios
for H, result in results.items():
    print(f"H={H:2d}: Sharpe={result.sharpe_ratio():.2f}, MDD={result.max_drawdown():.2%}")

# Plot H vs Sharpe
import matplotlib.pyplot as plt
sharpe_by_H = {H: r.sharpe_ratio() for H, r in results.items()}
plt.plot(sharpe_by_H.keys(), sharpe_by_H.values(), marker='o')
plt.xlabel('Holding Period H (hours)')
plt.ylabel('Sharpe Ratio')
plt.title('H* Optimization')
plt.show()
```

## Risk Analysis

### Portfolio Risk Decomposition

```python
from slipstream.portfolio.risk import decompose_risk

# At any timestamp
w = result.positions.iloc[-1].values
S = covariance_matrix

risk_decomp = decompose_risk(
    w=w,
    S=S,
    asset_names=assets,
)

# View top contributors
print(risk_decomp.sort_values('pct_contribution', ascending=False, key=abs).head(10))
```

### Factor Exposure Tracking

```python
# Track beta exposure over time
beta_exposure = (result.positions.values @ beta.T).diagonal()

import matplotlib.pyplot as plt
plt.plot(result.positions.index, beta_exposure)
plt.axhline(0, color='red', linestyle='--')
plt.ylabel('Market Beta Exposure')
plt.title('Beta Neutrality Check')
plt.show()
```

## Common Issues

### Issue: Optimization fails to converge

**Solution**: Increase `max_iter` or relax `ftol`:

```python
w, info = optimize_portfolio_with_costs(
    ...,
    max_iter=2000,
    ftol=1e-8,
)
```

### Issue: Beta constraint violated

**Solution**: Check beta values aren't extreme:

```python
print(f"Beta range: [{beta.min():.2f}, {beta.max():.2f}]")
print(f"Beta std: {beta.std():.2f}")

# Clip extreme betas
beta_clipped = np.clip(beta, -5, 5)
```

### Issue: Singular covariance matrix

**Solution**: Add ridge regularization in risk model:

```python
S_reg = S + 1e-6 * np.eye(n_assets)
```

### Issue: Unrealistic returns

**Check**:
1. Are predictions normalized correctly?
2. Is alpha_total = alpha_price - funding (not +)?
3. Are realized returns on the same scale?

```python
print(f"Alpha mean: {alpha_total.mean():.6f}")
print(f"Alpha std: {alpha_total.std():.6f}")
print(f"Returns mean: {realized_returns.mean():.6f}")
print(f"Returns std: {realized_returns.std():.6f}")
```

## Next Steps

1. **Implement full prediction pipeline**: Generate model predictions for backtest period
2. **Estimate cost parameters**: Use L2 orderbook data to calibrate λ_i
3. **Test multiple H values**: Find empirical H* via Sharpe optimization
4. **Add slippage**: Model bid-ask spread explicitly
5. **Implement live trading**: Adapt backtest framework for production

## References

- `strategy_spec.md` Section 4: Simulation & Optimization Workflow
- `JOINT_H_OPTIMIZATION.md`: Finding optimal holding period
- `src/slipstream/portfolio/`: Implementation details
