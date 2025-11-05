# Slipstream Documentation

> **Note:** A lightweight companion strategy named *Gradient* now lives under
> `src/slipstream/strategies/gradient`. See `docs/strategies/gradient/README.md` for its workflow, CLI
> commands, and implementation details.

# Slipstream: Strategy Specification

## 1. Executive Summary

This document provides the complete theoretical and practical specification for "Slipstream," a beta-neutral statistical arbitrage strategy designed for futures markets with explicit funding rates. The strategy aims to maximize the long-term growth rate of capital by isolating and trading alpha signals while hedging out systematic market risk.

The core of the strategy is a mean-variance optimization derived from the principle of maximizing log-wealth. The optimal portfolio is determined by balancing the expected total return (alpha from price moves minus funding costs) against the total idiosyncratic risk (arising from both price and funding rate volatility).

A key challenge addressed by this specification is the determination of the optimal rebalancing period, $H^*$. This is not a fixed parameter but is found empirically through a rigorous, simulation-based framework. The framework also addresses practical execution constraints, such as transaction costs and the non-divisibility of assets, by specifying a multi-stage optimization process. The optimal period $H^*$ is the one that yields the highest net-of-cost performance in backtesting.

## 2. Theoretical Foundation & Core Optimization

### 2.1. The Objective: Maximizing Log-Wealth

The strategy's primary goal is to maximize the expected logarithm of terminal wealth, $E[\log(W_T)]$. For a single trading period, wealth evolves as $W_T = W_0(1 + R_p)$, where $R_p$ is the total portfolio return on capital. Maximizing $E[\log(W_0(1+R_p))]$ is equivalent to:

$$\max_{\mathbf{w}} E[\log(1+R_p)]$$

Using a second-order Taylor approximation for $\log(1+x) \approx x - \frac{1}{2}x^2$, the objective becomes the classic mean-variance utility function that defines a Kelly-optimal portfolio in this context:

$$\max_{\mathbf{w}} \left( E[R_p] - \frac{1}{2}\text{Var}(R_p) \right)$$

### 2.2. The Return & Risk Model

The total portfolio return $R_p$ is composed of price returns and funding payments:

$$R_p = \mathbf{w}^T(\mathbf{R}^{\text{price}} - \mathbf{F})$$

The price returns are modeled using a single-factor market model (where the market factor $R_m$ is derived from PCA):

$$\mathbf{R}^{\text{price}} = \boldsymbol{\alpha}^{\text{price}} + \boldsymbol{\beta}R_m + \boldsymbol{\epsilon}_{\text{price}}$$

Both price returns and funding rates are stochastic. We model their expectations and the "shock" or residual components:

$$\mathbf{R}^{\text{price}} = E[\mathbf{R}^{\text{price}}] + \boldsymbol{\epsilon}_{\text{price}}$$

$$\mathbf{F} = E[\mathbf{F}] + \boldsymbol{\epsilon}_{\text{funding}} = \hat{\mathbf{F}} + \boldsymbol{\epsilon}_{\text{funding}}$$

### 2.3. The Beta-Neutral Constraint: The Key Simplification

The strategy's defining characteristic is the constraint of zero net market exposure:

$$\mathbf{w}^T\boldsymbol{\beta} = 0$$

Applying this constraint radically simplifies the objective function. Let's re-derive the expectation and variance of $R_p$:

**1. Expected Return $E[R_p]$:**

$$E[R_p] = \mathbf{w}^T(E[\mathbf{R}^{\text{price}}] - E[\mathbf{F}])$$

$$E[R_p] = \mathbf{w}^T(\boldsymbol{\alpha}^{\text{price}} + \boldsymbol{\beta}E[R_m] - \hat{\mathbf{F}})$$

$$E[R_p] = \mathbf{w}^T\boldsymbol{\alpha}^{\text{price}} + (\mathbf{w}^T\boldsymbol{\beta})E[R_m] - \mathbf{w}^T\hat{\mathbf{F}}$$

Since $\mathbf{w}^T\boldsymbol{\beta}=0$, the market term vanishes:

$$E[R_p] = \mathbf{w}^T(\boldsymbol{\alpha}^{\text{price}} - \hat{\mathbf{F}}) \equiv \mathbf{w}^T\boldsymbol{\alpha}^{\text{total}}$$

**2. Variance $\text{Var}(R_p)$:**

The stochastic shock to the portfolio's return comes from the price and funding residuals. After applying the beta constraint, the portfolio return is $R_p = \mathbf{w}^T\boldsymbol{\alpha}^{\text{total}} + \mathbf{w}^T(\boldsymbol{\epsilon}_{\text{price}} - \boldsymbol{\epsilon}_{\text{funding}})$. The variance is thus:

$$\text{Var}(R_p) = \text{Var}(\mathbf{w}^T(\boldsymbol{\epsilon}_{\text{price}} - \boldsymbol{\epsilon}_{\text{funding}})) = \mathbf{w}^T \text{Cov}(\boldsymbol{\epsilon}_{\text{price}} - \boldsymbol{\epsilon}_{\text{funding}}) \mathbf{w}$$

Let's call this total covariance matrix $\mathbf{S}_{\text{total}}$. It expands to:

$$\mathbf{S}_{\text{total}} = \text{Cov}(\boldsymbol{\epsilon}_{\text{price}}) + \text{Cov}(\boldsymbol{\epsilon}_{\text{funding}}) - 2\text{Cov}(\boldsymbol{\epsilon}_{\text{price}}, \boldsymbol{\epsilon}_{\text{funding}})$$

### 2.4. The Cost-Free Optimal Portfolio

With the simplifications above, the core (cost-free) optimization problem becomes a standard quadratic program:

$$\max_{\mathbf{w}} \left( \mathbf{w}^T \boldsymbol{\alpha}^{\text{total}} - \frac{1}{2} \mathbf{w}^T \mathbf{S}_{\text{total}} \mathbf{w} \right) \quad \text{s.t.} \quad \mathbf{w}^T\boldsymbol{\beta}=0$$

This problem has a closed-form solution derived via Lagrange multipliers. The Lagrangian is:

$$\mathcal{L}(\mathbf{w}, \lambda) = \left( \mathbf{w}^T \boldsymbol{\alpha}^{\text{total}} - \frac{1}{2} \mathbf{w}^T \mathbf{S}_{\text{total}} \mathbf{w} \right) - \lambda (\mathbf{w}^T\boldsymbol{\beta})$$

The solution is found by setting the gradient with respect to $\mathbf{w}$ to zero and solving for $\mathbf{w}$ and $\lambda$, which yields the optimal portfolio weights $\mathbf{w}^*$:

$$\mathbf{w}^* = (\mathbf{S}_{\text{total}})^{-1} \left( \boldsymbol{\alpha}^{\text{total}} - \left[ \frac{(\boldsymbol{\alpha}^{\text{total}})^T (\mathbf{S}_{\text{total}})^{-1} \boldsymbol{\beta}}{\boldsymbol{\beta}^T (\mathbf{S}_{\text{total}})^{-1} \boldsymbol{\beta}} \right] \boldsymbol{\beta} \right)$$

This foundational solution informs our real-world implementation, which must also account for transaction costs.

## 3. Core Modeling Components

All model inputs must be explicitly calibrated as a function of the holding period, $H$.

### 3.1. Alpha Model: $\alpha(H)$

**Objective:** To generate a vector of expected total returns, $\boldsymbol{\alpha}^{\text{total}}(H)$.

**Formula:**

$$\boldsymbol{\alpha}^{\text{total}}(H) = \boldsymbol{\alpha}^{\text{price}}(H) - \hat{\mathbf{F}}(H)$$

**Methodology:**

- **Price Alpha Model** ($oldsymbol{\alpha}^{\text{price}}(H)$): Train a predictive model where features are proprietary signals (e.g., vol-normalized idiosyncratic momentum) and the target variable is the H-period forward price return. A separate model must be trained for each candidate $H$.

- **Funding Rate Model** ($\hat{\mathbf{F}}(H)$): Train a predictive model for the average expected funding rate over the next $H$ hours.

### 3.2. Risk Model: $\beta$ and $S(H)$

**Objective:** To generate the market beta vector ($oldsymbol{\beta}$) and the total idiosyncratic covariance matrix ($\mathbf{S}_{\text{total}}(H)$).

**Methodology:**

- **Market Factor** ($R_m$): Use PCA on historical price returns. PC1 serves as the market factor, $R_m$.

- **Beta Estimation** ($oldsymbol{\beta}$): Regress asset price returns against $R_m$. The vector of slope coefficients is $oldsymbol{\beta}$.

- **Total Covariance Matrix** ($\mathbf{S}_{\text{total}}(H)$):
  - Collect price residuals ($oldsymbol{\epsilon}_{\text{price}}$) from the beta regression and funding residuals ($oldsymbol{\epsilon}_{\text{funding}}$) from the funding model.
  - Calculate the three component covariance matrices: $\mathbf{S}_{\text{price}}$, $\mathbf{S}_{\text{funding}}$, and the cross-covariance $\mathbf{C}_{\text{price,funding}}$.
  - Combine them: $\mathbf{S}_{\text{total}} = \mathbf{S}_{\text{price}} + \mathbf{S}_{\text{funding}} - 2 \mathbf{C}_{\text{price,funding}}$.
  - Scale to Period H: $\mathbf{S}_{\text{total}}(H) = \mathbf{S}_{\text{total}}^{\text{base}} \cdot (H / H_{\text{base}})$.

### 3.3. Transaction Cost Model: $C(\Delta w)$

**Objective:** To calculate the P&L drag from fees and market impact.

**Formula:** 

$$C(\Delta \mathbf{w}) = \sum_i |\Delta w_i| \cdot \text{fee\_rate}_i + \sum_i \lambda_i |\Delta w_i|^{1.5}$$

where $\Delta \mathbf{w} = \mathbf{w}_{\text{new}} - \mathbf{w}_{\text{old}}$. The impact parameter $\lambda_i$ must be estimated empirically.

## 4. Simulation & Optimization Workflow

### 4.1. Cost-Aware Objective Function

At each rebalancing step, we solve the following optimization problem for the target portfolio $\mathbf{w}_{\text{new}}$:

$$\max_{\mathbf{w}_{\text{new}}} \left( \mathbf{w}_{\text{new}}^T \boldsymbol{\alpha}^{\text{total}}(H) - \frac{1}{2} \mathbf{w}_{\text{new}}^T \mathbf{S}_{\text{total}}(H) \mathbf{w}_{\text{new}} - C(\mathbf{w}_{\text{new}} - \mathbf{w}_{\text{old}}) \right)$$

$$\text{subject to:} \quad \mathbf{w}_{\text{new}}^T \boldsymbol{\beta} = 0$$

This requires a numerical convex optimization solver.

#### 4.1.1. Discretization and Minimum Size Constraints

The output of the objective function is an "ideal" continuous portfolio, $\mathbf{w}_{\text{ideal}}$. In practice, assets must be traded in discrete lot sizes. This requires a second optimization stage to find a realizable portfolio, $\mathbf{w}_{\text{real}}$, that best tracks the ideal one.

**Problem Formulation:** The goal is to minimize the tracking error variance against the ideal portfolio, subject to integer lot and beta-neutrality constraints.

$$\min_{\mathbf{w}_{\text{real}}} \left( (\mathbf{w}_{\text{real}} - \mathbf{w}_{\text{ideal}})^T \mathbf{S}_{\text{total}}(H) (\mathbf{w}_{\text{real}} - \mathbf{w}_{\text{ideal}}) \right)$$

Subject to:

1. **Beta Neutrality:** $\mathbf{w}_{\text{real}}^T \boldsymbol{\beta} = 0$
2. **Discretization:** $w_{\text{real},i} \cdot \text{Total\_Capital} = k_i \cdot \text{min\_trade\_size}_i$, where $k_i$ is an integer for each asset $i$.

**Solution Methodology:** This is a Mixed-Integer Quadratic Program (MIQP), which is computationally intensive. A practical, heuristic approach is often preferred:

1. Calculate Ideal Lots: Determine the ideal (fractional) number of lots for each asset from $\mathbf{w}_{\text{ideal}}$: $\text{ideal\_lots}_i = (w_{\text{ideal},i} \cdot \text{Total\_Capital}) / \text{min\_trade\_size}_i$.

2. Round to Nearest Integer: Create an initial portfolio, $\mathbf{w}_{\text{rounded}}$, by rounding each $\text{ideal\_lots}_i$ to the nearest integer $k_i$.

3. Repair Beta Drift: The rounded portfolio will have a non-zero beta ($\beta_{\text{drift}} = \mathbf{w}_{\text{rounded}}^T \boldsymbol{\beta}$). A "repair" algorithm must adjust the integer lot counts to drive this drift to zero. This can be done via a greedy search: iteratively adjust the lot count ($+1$ or $-1$) of the asset that provides the most beta reduction per unit of added tracking error, until $|\mathbf{w}_{\text{real}}^T \boldsymbol{\beta}| < \epsilon$ (a small tolerance).

### 4.2. Simulation Procedure

1. **Define Candidate Periods:** Select a range of rebalancing periods to test, e.g., $H \in \{6, 12, 24, 48, 72, 96\}$ hours.

2. **Iterate and Backtest:** For each candidate $H$, run a full, path-dependent backtest.
   - a. Loop through historical data in steps of $H$.
   - b. At each step, generate the model inputs ($\boldsymbol{\alpha}^{\text{total}}(H)$, $\mathbf{S}_{\text{total}}(H)$) based on data available at that time.
   - c. Solve the cost-aware objective function to get $\mathbf{w}_{\text{ideal}}$.
   - d. Solve the discretization problem (Section 4.1.1) to get $\mathbf{w}_{\text{real}}$.
   - e. Calculate and subtract transaction costs based on the actual discrete trades.
   - f. Record net P&L and all relevant metrics.

### 4.3. Parameter Tuning: PCA Lookback Window

The stability and relevance of the market factor $R_m$ depend on the lookback window used for the PCA calculation. This window should be related to the rebalancing frequency $H$.

**Objective:** To find the optimal PCA lookback window, $L^*$, for each candidate holding period $H$. A short lookback is responsive but noisy; a long lookback is stable but slow.

**Methodology:** The search for $L^*$ should be nested within the simulation for $H$.

- Define a set of candidate lookback multipliers, e.g., $M \in \{5, 10, 20, 50\}$.
- For each $H$ being tested, the PCA lookback window will be $L = M \cdot H$.
- The backtest simulation (Section 4.2) is run for each pair of $(H, M)$.
- The result is a 2D surface of Sharpe Ratios. For each $H$, we can identify the multiplier $M^*$ that performs best. This allows the risk model's timescale to adapt optimally to the rebalancing frequency.

### 4.4. A Priori Estimation of H (Signal-Only Analysis)

Before running the computationally expensive full backtester, we can derive a theoretical estimate for $H^*$ by analyzing the intrinsic properties of the alpha signal itself, balanced against an estimated cost of trading. This serves as a powerful sanity check.

**Objective:** To estimate $H^*$ by maximizing the cost-adjusted quality of the alpha signal.

**Methodology:**

1. **Signal Half-Life Analysis:**
   - For each alpha model trained for a base period (e.g., $\boldsymbol{\alpha}(1\text{hr})$), compute the autocorrelation of the signal time series.
   - Fit an exponential decay curve to the autocorrelation function to estimate the signal's "half-life." This provides a natural timescale over which the signal is informative. A holding period $H$ significantly longer than this half-life is unlikely to be optimal.

2. **Turnover-Adjusted Information Ratio (IR):**
   - The Information Ratio for a given $H$ is $\text{IR}(H) = E[\alpha_{\text{total}}(H)] / \sigma(\alpha_{\text{total}}(H))$. This measures the quality of the signal.
   - Estimate the turnover required to follow the signal at frequency $H$. A simple proxy is based on the day-to-day change in the signal vector: $\text{Turnover}(H) \propto ||\boldsymbol{\alpha}(t) - \boldsymbol{\alpha}(t-H)||$.
   - Define a cost-per-unit-of-turnover, $C_{\text{turnover}}$, based on the cost model in 3.3.
   - Calculate a Net Information Ratio for each candidate $H$: $\text{Net\_IR}(H) = \text{IR}(H) - C_{\text{turnover}} \cdot \text{Turnover}(H)$.
   - Plot $\text{Net\_IR}(H)$ versus $H$. The peak of this curve, $H^*_{\text{signal}}$, is the theoretical optimum that balances signal quality with the cost of capturing it. This result can be used to narrow the range of $H$ values tested in the full backtest.

## 5. Decision Framework

1. **Calculate Performance:** For each completed backtest (for each $H$), calculate the final, annualized net-of-cost Sharpe Ratio.

2. **Plot the Curve:** Create a plot with Holding Period ($H$) on the x-axis and Net Sharpe Ratio on the y-axis.

3. **Identify Optimum:** The optimal rebalancing period, $H^*$, is the point on the x-axis corresponding to the peak of the curve. This is the period that provides the empirically best trade-off between capturing decaying alpha and minimizing transaction costs.

This empirical result provides the robust, data-driven justification for the strategy's live rebalancing frequency.

# Joint H* Optimization: Finding the Optimal Holding Period

## Overview

This document describes the joint optimization workflow for finding the optimal holding period H* by training and evaluating both price-alpha and funding models simultaneously.

## Why Joint Optimization?

The strategy's total alpha is:

```
α_total = α_price - F_hat
```

Where:
- `α_price`: Predicted forward price returns
- `F_hat`: Predicted forward funding payments

**Key insight**: The optimal H may differ when considering price alpha alone vs. the combined signal. Funding rates may persist at different timescales than price momentum, requiring joint analysis to identify the true optimal holding period.

## Workflow

### 1. Run Joint H* Search

```bash
# Test multiple holding periods with joint alpha + funding models
python scripts/find_optimal_H_joint.py --H 4 8 12 24 48 --n-bootstrap 1000

# Quick test with reduced bootstrap samples
python scripts/find_optimal_H_joint.py --H 4 8 --n-bootstrap 50

# Test different PCA methods
python scripts/find_optimal_H_joint.py --H 4 8 12 24 --pca-method sqrt_dollar --n-bootstrap 1000
```

### 2. Output Structure

Results are saved to `data/features/joint_models/`:

```
joint_models/
├── joint_model_H4.json         # Individual model results for H=4
├── joint_model_H8.json         # Individual model results for H=8
├── ...
├── H_comparison.csv            # Comparison table across all H values
└── optimization_summary.json   # Best H* recommendation
```

### 3. Key Metrics

For each H, the script reports:

- **R²_alpha**: Out-of-sample R² for price alpha alone
- **R²_funding**: Out-of-sample R² for funding predictions
- **R²_combined**: Out-of-sample R² for α_total = α_price - F_hat
- **Lambda values**: Regularization parameters for both models
- **Significant coefficients**: Number of significant features in each model

### 4. Combined Quantile Diagnostics

The most valuable output is the **combined signal quantile table**, which bins samples by the combined signal (α_total) and shows:

- Alpha predictions vs. actuals
- Funding predictions vs. actuals
- Combined signal predictions vs. actuals
- T-statistics for each component

**Example output (H=4 hours)**:

```
COMBINED SIGNAL QUANTILE ANALYSIS
(Binned by α_total = α_price - F_hat)

 Quantile  Count  α_pred_µ  α_actual_µ    α_t  F_pred_µ  F_actual_µ      F_t  Total_pred_µ  Total_actual_µ  Total_t
        0  52940    -0.017      -0.040 -8.132     6.742       8.541  798.383        -6.759          -8.581 -730.173
        1  52940    -0.003      -0.031 -6.490     4.992       4.743  373.971        -4.996          -4.774 -349.752
        ...
        9  52940    -0.007       0.006  1.354    -1.539      -0.826 -150.145         1.532           0.833  116.428
```

**Interpretation**:
- **Quantile 0 (bottom)**: Highest expected funding (F_pred = 6.74σ), negative combined signal
- **Quantile 9 (top)**: Negative expected funding (F_pred = -1.54σ), positive combined signal
- **Alpha contribution**: Weak overall but significant in tail quantiles (see quantile 9: α_actual = 0.006, t=1.35)
- **Funding dominates**: R²_funding = 0.72 vs R²_alpha = -0.001 at H=4

## Initial Results (H=4)

### Summary Statistics
- **Alpha R² (OOS)**: -0.001 (-9.86 bp) — essentially no predictive power
- **Funding R² (OOS)**: 0.720 (7200.80 bp) — very strong persistence
- **Combined R² (OOS)**: 0.638 (6378.65 bp) — strong combined signal

### Key Findings

1. **Funding dominates at short horizons**: At H=4, funding persistence is the primary driver of predictive power.

2. **Alpha has tail significance**: While overall R² is near zero, the quantile analysis shows alpha becomes predictive in extreme buckets:
   - Top decile: positive price alpha emerges (α_actual = 0.006, t=1.35)
   - Bottom deciles: negative alpha in high-funding regimes

3. **Combined signal quality**: The combined R² of 0.638 suggests that even though price alpha is weak overall, it adds value when combined with funding predictions.

4. **Strategy implication**: At H=4, the optimal strategy is primarily a **funding carry arbitrage** (short high funding, long low funding) with a small momentum overlay.

## Next Steps

### 1. Test Longer Horizons

```bash
python scripts/find_optimal_H_joint.py --H 8 12 24 48 72 96 --n-bootstrap 1000
```

**Hypothesis**: Price alpha should strengthen at longer horizons while funding persistence weakens, leading to a crossover point where both contribute meaningfully.

### 2. Analyze H* vs Component R²

Create a plot showing:
- X-axis: Holding period H
- Y-axis: R² (out-of-sample)
- Three lines: α_price, F_hat, α_total

Expected pattern:
- **Funding R²**: High at short H, decays as H increases (mean reversion)
- **Alpha R²**: Low at short H, increases with H (momentum persistence)
- **Combined R²**: Peak at optimal H* where both contribute

### 3. Decompose Combined Signal

For the optimal H*, analyze the relative contributions:

```python
# Signal variance decomposition
var(α_total) = var(α_price) + var(F_hat) - 2*cov(α_price, F_hat)

# Information ratio decomposition
IR(α_total) = [μ(α_price) - μ(F_hat)] / σ(α_total)
```

### 4. Cost-Adjusted H* Selection

The current R² analysis ignores transaction costs. The true H* should maximize:

```
Net Sharpe(H) = Sharpe(H) - c * Turnover(H)
```

Where turnover increases as H decreases (more frequent rebalancing).

## Comparison with Individual Optimizations

| Metric | Alpha Only | Funding Only | Joint |
|--------|-----------|--------------|-------|
| R² (H=4) | -0.001 | 0.720 | 0.638 |
| Optimal λ | 100.0 | 100.0 | - |
| Sig. Coefs | 2/12 | 6/6 | - |

**Key difference**: Joint optimization reveals that the combined signal has predictive power even when price alpha alone appears useless. The funding model provides the directional signal, while price alpha adds refinement in the tails.

## Technical Details

### Model Training

Both models use:
- **Walk-forward cross-validation**: 10 folds with expanding window
- **Bootstrap estimation**: 1000 samples for coefficient stability
- **Ridge regularization**: λ selected via CV from [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
- **Feature set**: 6 EWMA spans [2, 4, 8, 16, 32, 64] hours for both momentum and funding

### Signal Alignment

- Alpha model uses idiosyncratic returns (PCA-residualized)
- Funding model uses raw funding rates
- Both normalized by 128-hour EWMA volatility
- Both trained on the same timescale-matched PCA factors

### Quantile Computation

1. Compute combined signal: `α_total = α_price - F_hat`
2. Bin OOS samples into 10 quantiles by `α_total`
3. Within each bin, compute mean and t-stat for:
   - Alpha predictions vs. actuals
   - Funding predictions vs. actuals
   - Combined predictions vs. actuals

This reveals whether the combined signal is monotonic (stronger signal in extremes).

## References

See also:

- `docs/ALPHA_MODEL_TRAINING.md` - Price alpha model details

- `docs/FUNDING_MODEL_TRAINING.md` - Funding model details

- `docs/strategy_spec.md` Section 4.2 - H* optimization theory

- `scripts/find_optimal_H_joint.py` - Implementation



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

# Alpha Model Training: Bootstrap Methodology

## Overview

This document specifies the alpha model training framework for Slipstream, adapting the bootstrap methodology from Schmidhuber (2021) "Trends, reversion, and critical phenomena in financial markets" to our multi-span momentum signals.

**Core Philosophy**: Train trend strengths separately many times on random subsets of the data to get a distribution of fitted parameter values. Select the mean as our best estimate.

**Reference Paper**: `docs/Trends, reversion, and critical phenomena in financial markets.pdf`

---

## 1. Model Specification

### 1.1 The Alpha Model

For asset `i` at time `t`, we predict the H-period forward **vol-normalized** idiosyncratic return:

```
alpha_norm_i(t → t+H) = Σ β_s · momentum_i,s(t) + Σ γ_s · funding_i,s(t) + ε_i(t)
                         s ∈ {2, 4, 8, 16, 32, 64}       s ∈ {2, 4, 8, 16, 32, 64}
```

**Components**:
- `alpha_norm_i`: Vol-normalized H-period forward return (target variable)
- `momentum_i,s(t)`: EWMA-based idiosyncratic momentum at span s (price feature)
- `funding_i,s(t)`: EWMA-based funding rate at span s (funding feature)
- `β_s`: Coefficients for momentum spans (to be estimated via Ridge regression + bootstrap)
- `γ_s`: Coefficients for funding spans (to be estimated via Ridge regression + bootstrap)
- `ε_i(t)`: Random noise / unexplained variance
- `H`: Holding period in hours (e.g., 24 for daily rebalancing)

**Total Parameters**: 12 (6 momentum + 6 funding)

**Key Differences from Schmidhuber**:
- **Funding rates included**: Critical for crypto perps where funding can be 10-50% annualized
- **L2 regularization**: Ridge regression on every bootstrap iteration (not just parameter counting)
- **More features allowed**: We have higher-resolution data (hourly vs daily), more assets, and more independent series after PC filtering
- **Linear only** (for now): Can add polynomial terms later if residual analysis suggests non-linearity

---

## 2. Target Variable: Vol-Normalized Returns ✓

### 2.1 The Choice

We predict **vol-normalized** returns and scale back at prediction time.

**Formula**:
```python
# Training target:
y_i(t) = forward_return_i(t→t+H) / vol_i(t)

# At prediction time:
alpha_i(t) = model.predict(X_i(t)) * vol_i(t)
```

### 2.2 Why Vol-Normalized? (Arguments FOR)

1. **Stationarity**: Vol-normalized returns are much more stationary than raw returns
   - Crypto volatility is highly time-varying
   - Heteroskedasticity is severe without normalization

2. **Consistency**: Features are vol-normalized idiosyncratic momentum → target should match same scale
   - All signals in `idiosyncratic_momentum()` are already vol-normalized
   - Feature-target scale alignment improves regression stability

3. **Schmidhuber precedent**: Paper uses normalized returns (Eq. 1) as both features AND targets throughout
   - `R_i(t) = r_i(t) / σ_i` (volatility-normalized)
   - Entire analysis on normalized space

4. **Kelly framework alignment**: Maximizing log-wealth cares about Sharpe (mean/vol), not absolute returns
   - Vol-normalized prediction directly estimates Sharpe contribution
   - Portfolio optimization uses Sharpe-like objectives

5. **Cross-sectional pooling**: With vol-normalization, all assets contribute equally to parameter estimation
   - Without it, high-vol assets (e.g., DOGE, PEPE) dominate the regression
   - BTC/ETH would have minimal impact despite being more liquid

6. **Adaptive predictions**: Time-varying volatility handled naturally
   ```python
   # Model learns stable relationship in normalized space
   alpha_norm = model.predict(momentum_features)

   # Scale by current volatility at prediction time
   alpha_dollar = alpha_norm * current_vol_t
   ```
   - Predictions automatically adjust to current market regime
   - More robust to volatility regime changes

### 2.3 Why NOT Raw Returns? (Arguments AGAINST raw targets)

1. **Non-stationarity**: Raw returns have time-varying variance → violates OLS assumptions
2. **Asset imbalance**: High-vol coins get 100x more weight in loss function
3. **Overfitting to volatile periods**: Model would optimize for rare high-vol events
4. **Poor generalization**: Coefficients estimated during low-vol periods won't work in high-vol periods

### 2.4 Implementation Details

**Volatility Estimator**:
```python
# For target normalization (at time t, looking backward):
vol_i(t) = EWMA_volatility(idio_returns_i, span=128)

# Same span as used in signal generation (max(momentum_spans) * 2)
```

**Forward Return Calculation**:
```python
# H-period forward return (overlapping for more data):
forward_return_i(t) = sum(idio_returns_i[t+1 : t+H+1])

# Vol-normalized target:
y_i(t) = forward_return_i(t) / vol_i(t)
```

**At Prediction Time**:
```python
# Current volatility estimate:
vol_current = EWMA_volatility(recent_returns, span=128)

# Scale prediction:
alpha_total = alpha_normalized * vol_current
```

---

## 3. Funding Rate Features

### 3.1 Why Include Funding Rates?

**Crypto-Specific Feature**: Perpetual futures on crypto exchanges have funding rates that can be highly predictive of future returns.

**Economic Intuition**:
1. **Mean reversion**: Extreme funding rates (high positive or negative) tend to revert
2. **Carry signal**: Negative funding = get paid to hold long → bullish signal
3. **Crowding indicator**: High positive funding = longs crowded → potential reversal
4. **Predictive power**: Funding can be 10-50% annualized → material impact on returns

**Strategy Specification** (Section 3.1):
- Funding rate prediction is a key component of α^total = α^price - F̂
- EWMA of recent funding rates is a simple but effective predictor of future funding

### 3.2 Funding Rate Normalization

**Problem**: How to make funding rates comparable across assets and time?

**Funding Rate Characteristics**:
- Already in percentage units (e.g., 0.01% = 1 basis point per 8 hours)
- Can be positive or negative
- Magnitude varies by asset (BTC funding ≈ ±0.05%, meme coins ≈ ±0.5%)
- Time-varying (regime changes)

**Normalization Options**:

**Option A: Standardize by Historical Volatility** (RECOMMENDED)
```python
# For each asset i at time t:
funding_norm_i,s(t) = EWMA_s(funding_i) / std(funding_i)

# Where std(funding_i) is measured over a long lookback (e.g., 90 days)
```
**Pros**:
- Comparable across assets (BTC and DOGE on same scale)
- Accounts for regime changes
- Consistent with our vol-normalized returns

**Cons**:
- Need enough history to estimate std(funding)
- Outliers can distort std

**Option B: Raw Funding Rates** (ALTERNATIVE)
```python
# Use raw EWMA of funding rates
funding_i,s(t) = EWMA_s(funding_i)
```
**Pros**:
- Simpler
- Interpretable (1% funding = 1% carry)
- No additional estimation error

**Cons**:
- High-funding assets (meme coins) dominate regression
- Not comparable across assets
- Magnitude drift over time

**Option C: Cross-Sectional Z-Score**
```python
# At each time t, z-score across all assets:
funding_z_i(t) = (funding_i(t) - mean_t(funding)) / std_t(funding)
```
**Pros**:
- Relative funding (vs. market) is more predictive than absolute
- Automatically normalized

**Cons**:
- Loses information about absolute funding level
- Requires sufficient cross-section at each time

### 3.3 Recommended Approach

**Hybrid**: Vol-normalize funding rates by asset-specific historical std, similar to returns

```python
def compute_funding_features(
    funding_rates: pd.DataFrame,  # Wide format, hourly funding rates
    spans: list = [2, 4, 8, 16, 32, 64],
    std_lookback: int = 90 * 24  # 90 days
) -> pd.DataFrame:
    """
    Compute EWMA funding features, normalized by historical volatility.

    Returns:
        DataFrame with MultiIndex (timestamp, asset) and columns=funding_2, funding_4, ...
    """
    # 1. Compute funding volatility for each asset
    funding_std = funding_rates.rolling(std_lookback, min_periods=30*24).std()

    # 2. Vol-normalize funding rates
    funding_norm = funding_rates / funding_std

    # 3. Compute EWMA at multiple spans
    funding_features = {}
    for span in spans:
        ewma = funding_norm.ewm(span=span, min_periods=span//2).mean()
        funding_features[f'funding_{span}'] = ewma.stack()

    # 4. Combine into panel
    return pd.DataFrame(funding_features)
```

**Why This Works**:
- Consistent with vol-normalized returns (target and features on same scale)
- High-funding assets (meme coins) don't dominate low-funding assets (BTC/ETH)
- Accounts for time-varying funding volatility
- L2 regularization will handle any remaining scale issues

### 3.4 Implementation Details

**Data Requirements**:
- Hourly funding rate data (already available in `data/market_data/`)
- At least 90 days of history for std estimation (warm-up period)

**Feature Engineering**:
```python
# Parallel to momentum features:
X: pd.DataFrame
    Index: MultiIndex(timestamp, asset)
    Columns: [
        'mom_2', 'mom_4', 'mom_8', 'mom_16', 'mom_32', 'mom_64',      # Momentum
        'funding_2', 'funding_4', 'funding_8', 'funding_16', 'funding_32', 'funding_64'  # Funding
    ]
```

**Expected Signs**:
- Momentum coefficients β_s: Likely **positive** (trend persistence)
- Funding coefficients γ_s: Likely **negative** (mean reversion)
  - High positive funding (longs pay shorts) → future returns negative (reversion)
  - High negative funding (shorts pay longs) → future returns positive (carry)

---

## 4. L2 Regularization (Ridge Regression)

### 4.1 Why Regularization?

**Problem**: With 12 features (6 momentum + 6 funding), OLS can overfit, especially with correlated features

**Schmidhuber's Approach**: Strict parameter budget (max 6 params for 28 years)

**Our Approach**: L2 regularization + bootstrap
- **More flexible**: Can include all relevant features
- **Better statistical properties**: Shrinks coefficients toward zero (reduces variance)
- **Handles multicollinearity**: Momentum spans are correlated, funding spans are correlated
- **Cross-validated λ**: Optimal regularization strength chosen via CV

### 4.2 Ridge Regression Formulation

**Objective Function**:
```
min_β  ||y - Xβ||² + λ ||β||²
        \_________/   \______/
        OLS loss      L2 penalty
```

Where:
- `λ > 0`: Regularization strength (hyperparameter)
- `||β||²`: Sum of squared coefficients (penalizes large coefficients)

**Effect**:
- `λ = 0`: Regular OLS (no shrinkage)
- `λ → ∞`: All coefficients → 0 (full shrinkage)
- `λ optimal`: Bias-variance trade-off optimized via cross-validation

### 4.3 Choosing λ via Cross-Validation

**Process**:
1. Try a grid of λ values: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
2. For each λ, run K-fold CV and compute out-of-sample R²
3. Select λ* that maximizes R²_oos
4. Retrain on full data with λ*

**Important**: λ selection happens BEFORE bootstrap
- Find optimal λ via CV once
- Use that λ for all bootstrap iterations
- This avoids overfitting λ to any particular bootstrap sample

### 4.4 Integration with Bootstrap

**Updated Workflow**:
```python
1. Choose optimal λ via cross-validation (once)
2. For each bootstrap iteration:
   a. Sample timestamps with replacement
   b. Fit Ridge(alpha=λ) on bootstrap sample
   c. Store coefficients
3. Compute mean/std of coefficients across bootstrap samples
```

**Code Update**:
```python
from sklearn.linear_model import Ridge, RidgeCV

# Step 1: Find optimal λ via cross-validation
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=10)
ridge_cv.fit(X, y)
lambda_optimal = ridge_cv.alpha_

print(f"Optimal λ: {lambda_optimal}")

# Step 2: Bootstrap with optimal λ
def bootstrap_train_alpha_model(X, y, lambda_reg, n_bootstrap=1000):
    beta_samples = []

    for i in range(n_bootstrap):
        # Sample timestamps
        sampled_times = sample_with_replacement(timestamps)
        X_boot, y_boot = X.loc[sampled_times], y.loc[sampled_times]

        # Fit Ridge regression with fixed λ
        model = Ridge(alpha=lambda_reg)
        model.fit(X_boot, y_boot)

        beta_samples.append(model.coef_)

    return np.array(beta_samples)
```

---

## 5. Bootstrap Methodology (Adapted for Crypto)

### 5.1 Key Innovation: Random Subset Training

**Problem**: Financial returns are non-IID, correlated, non-normal → standard errors unreliable

**Solution**: Bootstrap resampling to empirically measure coefficient distributions

**Process**:
1. From N hours of data, create B bootstrap samples (sampling timestamps with replacement)
2. Fit Ridge regression on each bootstrap sample → B coefficient estimates
3. Take mean as final estimate
4. Use 16th/84th percentiles for standard errors (= ±1σ for normal distribution)

### 5.2 How Many Bootstrap Samples?

**Schmidhuber Used**: 5000 samples for 28 years of daily data

**Our Data**:
- ~1.5 years of hourly data ≈ 13,000 hours
- ~100 assets (vs. Schmidhuber's 24)
- After PC filtering: ~10-20 independent factors

**Rule of Thumb**: Bootstrap until coefficient distribution stabilizes

**Test for Stability**:
```python
# Compute coefficient std as function of bootstrap samples
stds_vs_n = []
for n in [100, 200, 500, 1000, 2000, 5000]:
    beta_samples_n = beta_samples[:n]
    std_n = beta_samples_n.std(axis=0).mean()
    stds_vs_n.append(std_n)

# If std plateaus by n=1000 → 1000 samples sufficient
# If still decreasing at n=5000 → need more
```

**Recommendation**: Start with 1000-2000 bootstrap samples
- Faster iteration during development
- Increase to 5000+ for final model if std not stable
- Computational cost: ~1-2 seconds per bootstrap (Ridge fit) → 1000 samples ≈ 20-30 minutes

### 5.3 Cross-Validation for Out-of-Sample R²

**Problem**: In-sample R² overestimates predictive power (overfitting bias)

**Solution**: K-fold time-series cross-validation (expanding window)

**Our Implementation**:
- 10-15 folds (vs. Schmidhuber's 15 for 28 years)
- Expanding window: train on all data BEFORE validation window
- For 1.5 years of data: each fold ≈ 1-2 months

**Key Difference from Schmidhuber**:
- We use Ridge regression in CV (not OLS)
- λ is chosen to maximize R²_oos across all folds
- This prevents overfitting even with 12 features

**Expected Correction**:
```
Schmidhuber: R² - R²_oos ≈ 0.93 bp (for 2 params on 28 years)

Crypto (our case): R² - R²_oos ≈ 1-3 bp (for 12 params on 1.5 years)
- Higher correction due to less data
- But Ridge regularization reduces it
- Bootstrap standard errors account for this
```

---

## 6. Implementation Plan

### 6.1 Data Pipeline

**Input**:
- Idiosyncratic returns (wide DataFrame, already computed)
- Funding rates (wide DataFrame, hourly)
- PCA loadings and market factor (if using single-factor residuals)
- OR: Multi-factor residuals (if using 3-PC approach)

**Output**:
```python
# Features: momentum + funding panel
X: pd.DataFrame
    Index: MultiIndex(timestamp, asset)
    Columns: [
        'mom_2', 'mom_4', 'mom_8', 'mom_16', 'mom_32', 'mom_64',      # Momentum
        'funding_2', 'funding_4', 'funding_8', 'funding_16', 'funding_32', 'funding_64'  # Funding
    ]

# Target: vol-normalized H-period forward returns
y: pd.Series
    Index: MultiIndex(timestamp, asset)
    Values: forward_return / volatility

# Metadata: volatility estimates (for scaling predictions)
vol: pd.Series
    Index: MultiIndex(timestamp, asset)
    Values: EWMA volatility

# Regularization: optimal λ from cross-validation
lambda_opt: float
```

### 6.2 Bootstrap Training with Ridge Regression

**Pseudocode**:
```python
from sklearn.linear_model import Ridge

def bootstrap_train_alpha_model(
    X: pd.DataFrame,  # Features (12 columns: 6 momentum + 6 funding)
    y: pd.Series,     # Target (vol-normalized forward returns)
    lambda_reg: float,  # Regularization strength (from CV)
    n_bootstrap: int = 1000,
    random_seed: int = 42
) -> dict:
    """
    Train alpha model using bootstrap sampling with Ridge regression.

    Key changes from Schmidhuber:
    - Uses Ridge(alpha=lambda_reg) instead of OLS
    - 1000-2000 bootstrap samples (vs. 5000)
    - 12 features (momentum + funding) vs. 2

    Returns:
        {
            'beta_mean': Mean coefficients (12,),
            'beta_std': Std dev of coefficients,
            'beta_distribution': Full distribution (n_bootstrap x 12),
            'r2_insample': Mean in-sample R² across bootstraps,
            't_statistics': t-stats for each coefficient,
            'lambda': Regularization strength used
        }
    """
    np.random.seed(random_seed)

    # Get unique timestamps
    timestamps = X.index.get_level_values('timestamp').unique()
    n_times = len(timestamps)

    # Storage for bootstrap results
    beta_samples = np.zeros((n_bootstrap, X.shape[1]))
    r2_samples = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Sample timestamps with replacement
        sampled_times = np.random.choice(timestamps, size=n_times, replace=True)

        # Get all assets at sampled timestamps
        mask = X.index.get_level_values('timestamp').isin(sampled_times)
        X_boot = X[mask]
        y_boot = y[mask]

        # Fit Ridge regression with fixed λ
        model = Ridge(alpha=lambda_reg)
        model.fit(X_boot, y_boot)

        # Store coefficients and R²
        beta_samples[i] = model.coef_
        r2_samples[i] = model.score(X_boot, y_boot)

    # Compute statistics
    beta_mean = beta_samples.mean(axis=0)
    beta_std = beta_samples.std(axis=0)

    # t-statistics (16th-84th percentile for robustness)
    beta_p16 = np.percentile(beta_samples, 16, axis=0)
    beta_p84 = np.percentile(beta_samples, 84, axis=0)
    beta_se = (beta_p84 - beta_p16) / 2  # Robust std error
    t_stats = beta_mean / beta_se

    return {
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_se': beta_se,
        'beta_distribution': beta_samples,
        'r2_insample': r2_samples.mean(),
        't_statistics': t_stats,
        'lambda': lambda_reg
    }
```

### 6.3 Walk-Forward Cross-Validation with Ridge

**Pseudocode**:
```python
from sklearn.linear_model import Ridge, RidgeCV

def find_optimal_lambda_and_validate(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    n_splits: int = 10,
    min_train_size: int = 180 * 24  # 180 days in hours
) -> dict:
    """
    Find optimal λ via cross-validation, then compute out-of-sample R².

    Two-stage process:
    1. RidgeCV to find optimal λ (10-fold CV on full data)
    2. Walk-forward CV to measure true out-of-sample performance

    Returns:
        {
            'lambda_opt': Optimal regularization strength,
            'r2_oos': Out-of-sample R² from walk-forward CV,
            'predictions_oos': All OOS predictions,
            'actuals_oos': All OOS actuals,
            'fold_r2': R² for each fold
        }
    """
    # Stage 1: Find optimal λ
    print("Stage 1: Finding optimal λ via RidgeCV...")
    ridge_cv = RidgeCV(alphas=alphas, cv=10)
    ridge_cv.fit(X, y)
    lambda_opt = ridge_cv.alpha_
    print(f"Optimal λ = {lambda_opt}")

    # Stage 2: Walk-forward validation with optimal λ
    print("Stage 2: Walk-forward validation...")
    timestamps = X.index.get_level_values('timestamp').unique().sort_values()
    n_times = len(timestamps)
    fold_size = n_times // n_splits

    predictions_oos = []
    actuals_oos = []
    fold_r2 = []

    for i in range(n_splits):
        # Validation window
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_splits - 1 else n_times
        val_times = timestamps[val_start:val_end]

        # Training window (all data before validation)
        if val_start < min_train_size:
            continue  # Skip if not enough training data
        train_times = timestamps[:val_start]

        # Split data
        train_mask = X.index.get_level_values('timestamp').isin(train_times)
        val_mask = X.index.get_level_values('timestamp').isin(val_times)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        # Fit Ridge with optimal λ
        model = Ridge(alpha=lambda_opt)
        model.fit(X_train, y_train)

        # Predict on validation data
        y_pred = model.predict(X_val)

        # Store results
        predictions_oos.append(y_pred)
        actuals_oos.append(y_val.values)

        # Calculate fold R²
        ss_res = np.sum((y_val.values - y_pred) ** 2)
        ss_tot = np.sum((y_val.values - y_val.mean()) ** 2)
        r2_fold = 1 - (ss_res / ss_tot)
        fold_r2.append(r2_fold)
        print(f"  Fold {i+1}: R² = {r2_fold:.4f}")

    # Concatenate all OOS predictions
    all_predictions = np.concatenate(predictions_oos)
    all_actuals = np.concatenate(actuals_oos)

    # Overall OOS R²
    ss_res = np.sum((all_actuals - all_predictions) ** 2)
    ss_tot = np.sum((all_actuals - all_actuals.mean()) ** 2)
    r2_oos = 1 - (ss_res / ss_tot)

    print(f"\nOut-of-sample R² = {r2_oos:.4f}")

    return {
        'lambda_opt': lambda_opt,
        'r2_oos': r2_oos,
        'predictions_oos': all_predictions,
        'actuals_oos': all_actuals,
        'fold_r2': np.array(fold_r2),
        'mean_fold_r2': np.mean(fold_r2)
    }
```

### 6.4 Complete Training Pipeline

**Full workflow** (λ selection → bootstrap → validation):

```python
def train_alpha_model_complete(
    X: pd.DataFrame,  # 12 features (mom + funding)
    y: pd.Series,     # Vol-normalized forward returns
    n_bootstrap: int = 1000
) -> dict:
    """
    Complete alpha model training pipeline.

    Steps:
    1. Find optimal λ via cross-validation
    2. Bootstrap training with optimal λ
    3. Walk-forward validation for out-of-sample R²
    4. Return coefficient distributions + diagnostics

    Returns:
        Complete model specification + diagnostics
    """
    # Step 1: Find optimal λ and validate
    cv_results = find_optimal_lambda_and_validate(X, y)
    lambda_opt = cv_results['lambda_opt']
    r2_oos = cv_results['r2_oos']

    # Step 2: Bootstrap training with optimal λ
    print(f"\nBootstrap training with λ={lambda_opt}, n={n_bootstrap}...")
    bootstrap_results = bootstrap_train_alpha_model(
        X, y, lambda_reg=lambda_opt, n_bootstrap=n_bootstrap
    )

    # Step 3: Compile results
    r2_in = bootstrap_results['r2_insample']
    correction = r2_in - r2_oos

    print(f"\n{'='*70}")
    print(f"FINAL MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"Regularization (λ):    {lambda_opt:.4f}")
    print(f"R² (in-sample):        {r2_in:.4f}")
    print(f"R² (out-of-sample):    {r2_oos:.4f}")
    print(f"Correction:            {correction:.4f} ({100*correction/r2_in:.1f}% of R²)")
    print(f"Bootstrap samples:     {n_bootstrap}")
    print(f"\nCoefficients:")
    print(f"{'Feature':<15} {'Beta':>10} {'Std Err':>10} {'t-stat':>8}")
    print(f"{'-'*50}")

    for i, col in enumerate(X.columns):
        beta = bootstrap_results['beta_mean'][i]
        se = bootstrap_results['beta_se'][i]
        t = bootstrap_results['t_statistics'][i]
        sig = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.64 else ''
        print(f"{col:<15} {beta:>10.4f} {se:>10.4f} {t:>8.2f} {sig}")

    print(f"\n*** p<0.01, ** p<0.05, * p<0.10")
    print(f"{'='*70}")

    return {
        'coefficients': bootstrap_results['beta_mean'],
        'std_errors': bootstrap_results['beta_se'],
        't_statistics': bootstrap_results['t_statistics'],
        'distribution': bootstrap_results['beta_distribution'],
        'lambda': lambda_opt,
        'r2_insample': r2_in,
        'r2_oos': r2_oos,
        'correction': correction,
        'feature_names': X.columns.tolist(),
        'cv_results': cv_results
    }
```

---

## 7. Key Decisions & Recommendations

### 7.1 Holding Period (H) ✓

**Recommendation**: Start with H=24 (daily rebalancing)

**Rationale**:
- Matches PCA timescale if using `pca_factor_H24_K30_sqrt.csv`
- Daily rebalancing is practical for crypto (24/7 markets)
- Can extend to H* optimization later (test H ∈ {6, 12, 24, 48})

### 7.2 Bootstrap Sample Size ✓

**Recommendation**: Start with 1000-2000 samples, test for stability

**Rationale**:
- We have MORE data than Schmidhuber (hourly vs daily, more assets)
- Computational cost: 1000 samples ≈ 20-30 minutes
- Test stability: if coefficient std plateaus at n=1000 → sufficient
- Can increase to 5000 if needed for final model

**Stability Test**: Run with n ∈ {100, 500, 1000, 2000, 5000}, plot coefficient std vs. n

### 7.3 Model Form ✓

**Recommendation**: Linear only (12 params: 6 momentum + 6 funding)

**Rationale**:
- Start simple, L2 regularization handles overfitting
- Our signals are pre-processed (vol-normalized, idiosyncratic)
- Schmidhuber's cubic terms were for RAW returns (trend reversion)
- Can add polynomial terms later if residual analysis suggests non-linearity

### 7.4 Volatility Estimation ✓

**Recommendation**: vol_span = 128 hours (max(spans) * 2)

**Rationale**:
- Consistent with Schmidhuber's approach
- Same as used in signal generation
- Long enough to be stable, short enough to adapt

### 7.5 PCA Components ✓

**Recommendation**: Start with 1-PC residuals, test 3-PC later

**Rationale**:
- 1-PC: Simpler, more data, existing implementation
- Can A/B test: train two models (1-PC vs 3-PC), compare R²_oos
- 3-PC may improve if PC2/PC3 have predictive power for returns

### 7.6 Overlapping Returns ✓

**Recommendation**: Use overlapping returns (every hour)

**Rationale**:
- More data (13,000 hours vs ~500 non-overlapping samples)
- Bootstrap handles autocorrelation properly
- Ridge regularization reduces overfitting
- Schmidhuber used overlapping successfully

### 7.7 Funding Rate Normalization ✓

**Recommendation**: Vol-normalize by asset-specific historical std

**Rationale**:
- Consistent with vol-normalized returns (target scale)
- Prevents meme coins from dominating regression
- Accounts for time-varying funding volatility
- L2 regularization handles any remaining scale issues

### 7.8 Tail Diagnostics & Guardrails ✓

**What changed**:

- **Warm-up enforcement**: every (timestamp, asset) observation must accumulate at least one full volatility span (default 128 h) before entering the training set. This removes the extreme spikes that occurred immediately after new listings when EWMA volatility was unreliable.
- **Clipping**:
  - Momentum panels are capped at ±2.5 (Schmidhuber convention).
  - Funding EWMAs are clipped at ±5 to suppress outlier prints in thin markets.
  - Vol-normalised forward returns are winsorised to ±10 to keep a single point from dominating loss/metrics.
- **Robust PCA metadata**: rolling PCA now stores `_pc1_eigenvalue` and `_condition_number`. Windows with eigenvalues below 1e-8 or near-zero loading norms are skipped, ensuring the beta vector stays well-conditioned.
- **Quantile analysis**: `train_alpha_model_complete` now prints (and serialises) a 10-decile table with counts, mean predictions, mean realised returns, and t-stats. This is the primary “go/no-go” check before trading a new horizon \(H\).

**Current H = 4 h snapshot**:

- Deciles 1–7 (most negative predictions) realise statistically significant negative returns (t-stats between −5.7 and −2.0).
- Decile 9 (most positive predictions) realises +2.5 bp per 4 h with t ≈ 4.8, even though aggregate \(R^2_{\text{oos}}\) is slightly negative (~−1 bp). This confirms momentum lives in the tails.
- Deciles 0 and 8 are neutral → production sizing should concentrate on the extreme buckets.

---

## 8. Expected Outcomes

### 8.1 Statistical Targets

**With L2 regularization and 12 features** (vs. Schmidhuber's 2-parameter OLS):

**Optimistic**:
- R² (in-sample): 10-20 bp (0.10%-0.20%)
  - Higher than Schmidhuber due to funding rate features
- R²_oos: 6-12 bp
  - Ridge regularization reduces overfitting
- t-statistics: |t| > 2.0 for at least 6-8 features
- Correction: 30-50% of R²

**Realistic**:
- R² (in-sample): 5-10 bp
- R²_oos: 2-5 bp
- t-statistics: |t| > 2.0 for 4-6 features
- Correction: 50-70% of R²

**Red flags**:
- R²_oos < 0: Model has no predictive power
- Correction > 90% of R²: Severe overfitting (even with Ridge)
- No significant coefficients (all |t| < 2): Signals don't predict returns
- λ → 0: Ridge not helping (features not correlated)
- λ → ∞: All coefficients shrunk to zero (no signal)

### 8.2 Economic Interpretation

**If successful**, coefficient estimates might look like:

```
Feature        Beta     Std Err  t-stat  Interpretation
─────────────────────────────────────────────────────────
mom_2        +0.15%    0.08%    1.9     Short-term momentum (weak)
mom_4        +0.45%    0.12%    3.8 *** Medium-fast momentum (strong)
mom_8        +0.60%    0.15%    4.0 *** Medium momentum (strongest)
mom_16       +0.40%    0.12%    3.3 *** Medium-slow momentum (strong)
mom_32       +0.15%    0.09%    1.7     Long-term momentum (weak)
mom_64       -0.05%    0.07%    0.7     Very long-term (reversion, n.s.)

funding_2    -0.10%    0.06%    1.7     Short-term funding reversion
funding_4    -0.25%    0.09%    2.8 **  Medium-fast funding reversion
funding_8    -0.35%    0.11%    3.2 *** Medium funding reversion (strongest)
funding_16   -0.30%    0.10%    3.0 **  Medium-slow funding reversion
funding_32   -0.15%    0.08%    1.9     Long-term funding reversion
funding_64   -0.05%    0.06%    0.8     Very long-term funding (n.s.)
```

**Interpretation**:
- **Momentum**: Positive, peak at 8-16 hours (trend persistence)
- **Funding**: Negative, peak at 8-16 hours (mean reversion)
- **Combined effect**: Momentum drives direction, funding provides entry/exit timing
- **λ optimal**: Likely 0.1-10 (modest shrinkage)

### 8.3 Why Higher R² Than Schmidhuber?

**Our advantages**:
1. **Funding rates**: Crypto-specific feature with strong predictive power
2. **Higher frequency**: Hourly data captures more dynamics than daily
3. **More independent series**: PC filtering removes systematic risk
4. **Better regularization**: Ridge vs. hard parameter limit

**Our disadvantages**:
1. **Less history**: 1.5 years vs. 28 years
2. **More noise**: Crypto is noisier than traditional futures
3. **Regime changes**: Crypto markets evolve faster

**Net effect**: R²_oos of 5-10 bp is realistic (vs. Schmidhuber's 4 bp)

---

## 9. Next Steps

### 9.1 Immediate (This Week)
1. ✓ Document alpha model specification (this doc)
2. ✓ Finalize key decisions (Section 7)
3. Implement data preparation pipeline:
   - Load idiosyncratic returns from existing signals
   - Load funding rates from `data/market_data/`
   - Compute vol-normalized funding features
   - Create feature matrix X (12 columns)
   - Create target vector y (H-period forward returns)
4. Implement Ridge + bootstrap training (Section 6.2)
5. Implement λ selection + walk-forward CV (Section 6.3)
6. Test on synthetic data first (known coefficients)

### 9.2 After Initial Implementation
1. Run on real crypto data (1.5 years)
2. Analyze coefficient distributions:
   - Plot bootstrap distributions (histograms)
   - Check for bimodality (sign instability)
   - Verify t-statistics > 2 for key features
3. Validate out-of-sample R²:
   - Should be positive and > 2bp
   - Correction < 70% of R²
   - Stable across CV folds
4. Residual analysis:
   - Plot residuals vs. fitted values
   - Check for non-linearity (polynomial terms needed?)
   - Check for heteroskedasticity
5. Compare 1-PC vs. 3-PC residuals:
   - Train both models
   - Compare R²_oos
   - Check if funding rate coefficients change

### 9.3 Extensions
1. **H* optimization**: Train separate models for H ∈ {6, 12, 24, 48}
   - Backtest each with transaction costs
   - Find H* that maximizes net Sharpe ratio
2. **Polynomial terms**: Add cubic terms if residuals show non-linearity
   - `mom_s³` and `funding_s³`
   - Increases to 24 parameters (still manageable with Ridge)
3. **Interaction terms**: Test `mom_s * funding_s` cross-terms
   - Captures non-linear relationship between momentum and funding
4. **Adaptive λ**: Retrain λ on rolling windows
   - Check if optimal λ changes over time
   - Could indicate regime changes

### 9.4 Production Integration
1. Integrate with portfolio optimizer:
   - α^total = α_model * current_vol
   - Feed into mean-variance optimization
2. Implement retraining pipeline:
   - Retrain monthly or quarterly
   - Monitor coefficient stability
   - Detect regime changes
3. Live monitoring:
   - Track realized Sharpe vs. predicted
   - Coefficient drift detection
   - R²_oos degradation alerts
4. Funding rate prediction (future):
   - Separate model for F̂(t → t+H)
   - Currently using EWMA as placeholder

---

## References

- Schmidhuber, C. (2021). "Trends, reversion, and critical phenomena in financial markets." Physica A: Statistical Mechanics and its Applications, 566, 125642.
- `docs/strategy_spec.md` - Slipstream strategy specification
- `src/slipstream/signals/idiosyncratic_momentum.py` - Current signal implementation

# Funding Model Training: Carry Forecasting

## Objective

Slipstream needs an estimate of future funding payments \(\hat{F}\) to pair with the
price-alpha forecast when solving the Kelly-style optimisation:

\[
\alpha^{\text{total}} = \alpha^{\text{price}} - \hat{F}
\]

The funding model therefore predicts the **H-hour forward sum of funding rates** for
each asset, normalised so that one unit corresponds to one unit of funding
volatility. The prediction is converted back to raw funding at execution time via
the current EWMA funding volatility.

---

## Data Pipeline

- **Inputs**: 4-hour funding histories for the full Hyperliquid universe.
- **Feature engineering**: reuse the same EWMA stack as the alpha model — spans
  \[2, 4, 8, 16, 32, 64\] hours, clipped to \(\pm 5\) after normalisation by the
  128-hour EWMA funding volatility.
- **Targets**: sum of the next `H / 4` funding prints, normalised by the same
  EWMA volatility and winsorised to \(\pm 10\).
- **Warm-up**: observations enter the training set only after accumulating at
  least 128 hours of history.

The helper `prepare_funding_training_data()` aligns the feature matrix, target,
and scaling series (`vol_scale`) and enforces all guardrails.

---

## Training & Validation

- Ridge regression with the same bootstrap + walk-forward stack used by the
  alpha model.
- Quantile diagnostics: every run prints a 10-decile table summarising mean
  predictions, realised forward funding, and t-stats. This provides an immediate
  check that the carry signal is strongest in the tails.
- Robust PCA metadata is unnecessary (funding is purely asset-specific), so the
  pipeline is lighter than the price-alpha stack.

### Example: H = 4 hours (n_bootstrap = 10)

| Quantile | Count | Pred µ | Actual µ | t-stat | Sig |
|---------:|------:|-------:|---------:|-------:|:---:|
| 0 | 53,148 | -1.54 | -0.83 | -151.5 | *** |
| 1 | 53,147 | -0.20 | -0.04 |  -8.8 | *** |
| 2 | 53,147 |  0.36 |  0.39 | 118.0 | *** |
| 3 | 53,147 |  0.72 |  0.69 | 196.2 | *** |
| 4 | 53,147 |  1.08 |  0.99 | 248.9 | *** |
| 5 | 53,147 |  1.53 |  1.34 | 300.7 | *** |
| 6 | 53,147 |  2.11 |  1.76 | 350.3 | *** |
| 7 | 53,147 |  2.99 |  2.29 | 390.3 | *** |
| 8 | 53,937 |  5.00 |  4.82 | 375.4 | *** |
| 9 | 52,358 |  6.75 |  8.49 | 784.3 | *** |

All deciles are strongly significant (|t| ≫ 2), with the top bucket capturing the
largest positive carry. The walk-forward \(R^2_{\text{oos}}\) for this run is ~0.72,
demonstrating that funding persistence is considerably stronger than price alpha.

---

## Usage

- Run `scripts/find_optimal_H_funding.py` to sweep candidate horizons. Outputs
  are written to `data/features/funding_models/` and include the quantile table,
  λ, bootstrap diagnostics, and comparison CSV.
- Combine the predicted carry with the price-alpha forecast by rescaling the
  normalised prediction with `vol_scale` and subtracting it from the price alpha
  before feeding the optimisation stage.

---

## Next Steps

1. Evaluate longer horizons (e.g. 24 h, 48 h) to see how carry persistence fades.
2. Consider augmenting features with cross-sectional signals (e.g. funding z-score
   vs. peers) if additional predictive power is needed.
3. Integrate the funding forecast artefacts into the portfolio simulation to
   verify that the combined alpha + funding stack matches theoretical expectations.

# S3 Historical Data Guide

This guide explains how to fetch historical OHLCV candles from Hyperliquid's S3 archive.

## Overview

Hyperliquid provides historical L2 orderbook snapshots via S3, but **NOT** pre-computed candles. We download L2 snapshots, extract mid-prices, aggregate to 1-hour candles, then immediately delete the raw data to save disk space.

**Key Features:**
- ✅ Resumable - tracks progress, safe to interrupt and restart
- ✅ Disk-efficient - processes and deletes files one at a time
- ✅ Validates against existing API data
- ✅ Separate from API data (`data/s3_historical/` vs `data/market_data/`)

## Prerequisites

### 1. Install AWS CLI

Already installed via uv:
```bash
uv pip install awscli
```

### 2. Install LZ4 Decompression Tool

```bash
sudo apt update
sudo apt install lz4
```

### 3. Configure AWS Credentials

You need an AWS account (free tier works) to download from S3:

```bash
# Option 1: Interactive configuration
aws configure
# Enter:
#   AWS Access Key ID: [your key]
#   AWS Secret Access Key: [your secret]
#   Default region: us-east-1
#   Default output format: json

# Option 2: Set environment variables
export AWS_ACCESS_KEY_ID=your_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_here
```

**To get AWS credentials:**
1. Go to https://aws.amazon.com
2. Sign up for free tier (no credit card required for this use case)
3. Go to IAM → Users → Create user
4. Attach policy: `AmazonS3ReadOnlyAccess`
5. Create access key → Copy credentials

**Cost:** ~$0.09/GB for data transfer. Expect ~$5-10 total for full historical download (2023-present).

## Usage

### Determine Date Range

First, check where your API data ends:

```bash
# Check most recent API candle
head -5 data/market_data/BTC_candles_4h.csv
```

You want S3 data to cover the period BEFORE your API data starts.

Example: If API data starts March 28, 2025, fetch S3 from Oct 2023 to March 27, 2025.

### Download Historical Data

```bash
# Fetch historical data (this will take hours/days depending on range)
python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27

# Process specific coins only
python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27 \
    --coins BTC ETH SOL

# Resume after interruption (automatically skips completed)
python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27
```

**The script will:**
1. Check `data/s3_historical/progress/download_state.json` for already-processed items
2. Download one L2 snapshot file at a time
3. Parse mid-prices and create 1h OHLCV intermediates
4. Resample to 4h and append to `data/s3_historical/candles/{COIN}_candles_4h.csv`
5. Delete the raw L2 file immediately
6. Update progress tracker
7. Repeat for all hours and all coins

**Safe to interrupt:** Press Ctrl+C anytime. Progress is saved every 10 items. Simply rerun the same command to resume.

### Validate Against API Data

After downloading, validate data quality by comparing overlapping periods:

```bash
python scripts/fetch_s3_historical.py --validate

# Validate specific coins
python scripts/fetch_s3_historical.py --validate --coins BTC ETH
```

This compares close prices between S3 candles and API candles for overlapping timestamps. Should show <1% difference.

## Data Format

S3 candles are saved in the same format as API candles:

```
data/s3_historical/
  candles/
    BTC_candles_4h.csv   # datetime,open,high,low,close,volume
    ETH_candles_4h.csv
    SOL_candles_4h.csv
    ...
    hourly/              # intermediate 1h candles kept for resampling
      BTC_candles_1h.csv
      ETH_candles_1h.csv
      ...
  progress/
    download_state.json  # Resumption state
```

**Note:** `volume` column will be NaN since L2 snapshots don't contain trade volume. The `hourly/` subdirectory is maintained only for resampling; downstream workflows should consume the 4h aggregates.

## Merging with API Data

Once you have both S3 historical data and API recent data, merge them:

```python
# Load both
s3_candles = pd.read_csv('data/s3_historical/candles/BTC_candles_4h.csv', parse_dates=['datetime'])
api_candles = pd.read_csv('data/market_data/BTC_candles_4h.csv', parse_dates=['datetime'])

# Concatenate and deduplicate
combined = pd.concat([s3_candles, api_candles]).drop_duplicates(subset=['datetime']).sort_values('datetime')

# Use combined for backtesting
```

## Troubleshooting

### Access Denied Error

```
An error occurred (AccessDenied) when calling the ListObjectsV2 operation
```

**Solution:** Configure AWS credentials (see Prerequisites section 3)

### LZ4 Not Found

```
lz4 tool not found. Install with: sudo apt install lz4
```

**Solution:** Run `sudo apt install lz4`

### Missing Data for Certain Hours

Some coins may not have L2 snapshots for all hours (delisted, new listings, data gaps).

**Solution:** This is expected. The script will skip missing files and continue.

### Disk Space Issues

If `/tmp` fills up:

```bash
# Use different temp directory
python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27 \
    --temp-dir /path/to/larger/disk
```

### Slow Download Speed

S3 downloads can be slow. Consider:
- Fetching smaller date ranges
- Limiting to specific coins with `--coins`
- Running overnight/over weekend

## Cost Estimation

**AWS S3 egress pricing:** ~$0.09/GB

**Estimated data size:**
- 1 coin, 1 hour, 1 L2 snapshot: ~50-500KB (compressed)
- 1 coin, 1 year: ~4-40GB raw (we delete after processing)
- 100 coins, 1.5 years: ~600-6000GB raw (scary, but we process 1 file at a time!)

**Expected total cost:** $5-20 depending on date range and number of coins

**Optimization:** Start with just BTC/ETH/SOL to validate the approach before downloading full universe.

## Example Workflow

```bash
# 1. Install dependencies
sudo apt install lz4
uv pip install awscli

# 2. Configure AWS
aws configure

# 3. Test with small range and few coins first
python scripts/fetch_s3_historical.py \
    --start 2024-01-01 \
    --end 2024-01-07 \
    --coins BTC ETH

# 4. Validate
python scripts/fetch_s3_historical.py --validate --coins BTC ETH

# 5. If good, fetch full range
python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27

# 6. Run in background with nohup (survives SSH disconnect)
nohup python scripts/fetch_s3_historical.py \
    --start 2023-10-01 \
    --end 2025-03-27 \
    > s3_download.log 2>&1 &

# 7. Monitor progress
tail -f s3_download.log
```

## References

- [Hyperliquid Historical Data Docs](https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data)
- S3 bucket: `s3://hyperliquid-archive/market_data/`
- Format: `YYYYMMDD/H/l2Book/COIN.lz4`

# Timescale Matching: Solving the H* Circular Dependency

## The Problem

The Slipstream beta-neutral strategy (see `strategy_spec.md`) faces a fundamental circular dependency:

1. **To find optimal H***: Need to backtest the strategy at different holding periods
2. **To backtest at period H**: Need accurate beta estimates for hedging market risk
3. **To estimate beta**: Need to choose PCA parameters (estimation frequency, lookback window)
4. **To choose PCA parameters**: Need to know what holding period H you're targeting

This creates a 3-dimensional optimization problem:
- **H** ∈ {1, 2, 4, 6, 12, 24, 48, 72, ...} hours (holding period)
- **PCA_freq** ∈ {1H, 4H, 6H, 12H, D, W} (estimation frequency)
- **PCA_window** ∈ {168, 360, 720, 1440, ...} hours (lookback length)

**Naive approach**: Test all combinations → O(H × freq × window) = hundreds of backtests

---

## The Solution: Timescale Matching

**Core principle**: Match PCA estimation timescale to the candidate holding period.

### Mathematical Justification

If you rebalance every H hours and hold positions for H hours:

1. **Your P&L accumulates over H-hour periods**, not 1-hour periods
2. **Intra-period fluctuations average out** - you don't realize them
3. **Portfolio variance scales with H-period return variance**, not 1-hour variance
4. **Therefore, beta should be estimated from H-period returns**

**Formally**:
- Portfolio return over H hours: `R_p(H) = w^T · [R_price(H) - F(H)]`
- Variance: `Var(R_p(H)) = w^T · Cov(R_price(H) - F(H)) · w`
- Beta exposure: `β = Cov(R_price(H), R_market(H)) / Var(R_market(H))`

The relevant beta is the one that describes **H-period co-movement**, not hourly.

### Implementation Rule

For each candidate holding period H:
- **PCA estimation frequency = H**
  - If H = 6 hours → use 6-hourly returns
  - If H = 24 hours → use daily returns
  - If H = 72 hours → use 3-day returns

- **PCA lookback window = K × H**
  - K ≈ 20-60 is the number of independent samples
  - K = 30 is a good default (30 independent H-period observations)
  - Examples:
    - H = 6, K = 30 → 180-hour (7.5 day) lookback
    - H = 24, K = 30 → 720-hour (30 day) lookback
    - H = 48, K = 30 → 1440-hour (60 day) lookback

### Result

The 3D optimization problem collapses to **1D + optional refinement**:

**Phase 1**: 1D search over H (with K fixed at 30)
- Test H ∈ {6, 12, 24, 48} → 4 backtests
- Find approximate H*

**Phase 2** (optional): Local refinement of K
- If H* ≈ 24, test K ∈ {20, 30, 40, 60} at H=24 → 4 backtests
- Total: 8 backtests instead of 100+

---

## Why This Works

### Frequency Matching

**Wrong**: Estimate beta from 1-hour returns when holding for 24 hours
- Beta measures sensitivity to 1-hour market moves
- You hold through 24 of these moves
- Hourly mean reversion/noise creates mismatch

**Right**: Estimate beta from 24-hour returns when holding for 24 hours
- Beta measures sensitivity to 24-hour market moves
- You hold for exactly one such period
- Direct alignment between estimation and realization

### Window Scaling

**Need**: Enough independent samples for stable covariance estimation
- Sample correlation converges slowly: O(1/√N) where N = # samples
- Want N ≥ 20-60 for reliable estimates

**Implementation**:
- If each sample is H hours, need K samples → K×H total history
- Example: H=24, K=30 → 30 independent days → stable daily covariance matrix

---

## Practical Impact

### Example: Testing H ∈ {6, 12, 24, 48}

**Naive approach** (all combinations):
```
H × PCA_freq × PCA_window = combinations
4 × 5        × 4           = 80 backtests
```

**Timescale matching**:
```
H only (with freq=H, window=30×H) = 4 backtests
```

**Speedup**: 20× fewer backtests

### Code Example

```bash
# Generate all matched PCA factors
python scripts/find_optimal_horizon.py --H 6 12 24 48 --K 30

# Creates 4 files:
#   data/features/pca_factor_H6_K30_sqrt.csv
#   data/features/pca_factor_H12_K30_sqrt.csv
#   data/features/pca_factor_H24_K30_sqrt.csv
#   data/features/pca_factor_H48_K30_sqrt.csv

# Now backtest each with matching H
# (backtesting framework to be implemented)
```

---

## Theoretical Foundations

### From Portfolio Theory

Kelly criterion for log-optimal portfolio:
```
max E[log(1 + R_p)]
≈ max (E[R_p] - 0.5 × Var(R_p))
```

Over holding period H:
- `E[R_p]` depends on H-period alpha
- `Var(R_p)` depends on H-period variance-covariance matrix
- Beta hedge removes systematic H-period risk

### From Time Series Analysis

**Aggregation property of log returns**:
```
R(H hours) = sum of H consecutive 1-hour log returns
```

**Variance aggregation**:
```
Var(R(H)) ≠ H × Var(R(1))  [unless returns are iid, which they aren't]
```

Therefore:
- Cannot extrapolate 1-hour covariance to H-hour covariance accurately
- Must estimate directly from H-hour data

### From Signal Processing

**Nyquist-Shannon sampling theorem analogy**:
- If your decision frequency is H, information at frequencies > 1/H is noise
- Estimating from higher frequencies adds noise without adding information
- Optimal filter: match measurement frequency to decision frequency

---

## Limitations and Extensions

### When Timescale Matching May Not Be Optimal

1. **Signal decay faster than H**: If alpha half-life << H, may want to rebalance more frequently
2. **Regime changes**: If market regime shifts within H, may need shorter estimation window
3. **Cross-period effects**: Funding payments occur every 8h on Hyperliquid, creating structure

### Extensions

1. **Multi-timescale PCA**: Combine fast (H) and slow (10×H) factors
2. **Adaptive K**: Let K vary with market volatility (shorter K in volatile regimes)
3. **Volume-time aggregation**: Use volume bars instead of time bars (already partially addressed with volume weighting)

---

## Summary

**Problem**: Finding H* requires knowing optimal PCA parameters, but optimal PCA parameters depend on H*

**Solution**: Make PCA parameters a deterministic function of H
- Frequency = H
- Window = K × H (K ≈ 30)

**Benefit**: Reduces search space from 3D to 1D, enabling practical optimization

**Tradeoff**: Assumes matched timescales are optimal (well-justified theoretically and empirically)

**Next step**: Implement backtesting framework to test each H and find empirical H* that maximizes Sharpe ratio

---

## References

- `docs/strategy_spec.md` - Full strategy specification and theoretical derivation
- `docs/QUICKSTART_VOLUME_PCA.md` - Practical guide to generating timescale-matched PCA factors
- `scripts/find_optimal_horizon.py` - Implementation of grid generation
- `scripts/build_pca_factor.py` - Core PCA computation with flexible frequency/window
