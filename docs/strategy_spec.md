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

- **Price Alpha Model** ($\boldsymbol{\alpha}^{\text{price}}(H)$): Train a predictive model where features are proprietary signals (e.g., vol-normalized idiosyncratic momentum) and the target variable is the H-period forward price return. A separate model must be trained for each candidate $H$.

- **Funding Rate Model** ($\hat{\mathbf{F}}(H)$): Train a predictive model for the average expected funding rate over the next $H$ hours.

### 3.2. Risk Model: $\beta$ and $S(H)$

**Objective:** To generate the market beta vector ($\boldsymbol{\beta}$) and the total idiosyncratic covariance matrix ($\mathbf{S}_{\text{total}}(H)$).

**Methodology:**

- **Market Factor** ($R_m$): Use PCA on historical price returns. PC1 serves as the market factor, $R_m$.

- **Beta Estimation** ($\boldsymbol{\beta}$): Regress asset price returns against $R_m$. The vector of slope coefficients is $\boldsymbol{\beta}$.

- **Total Covariance Matrix** ($\mathbf{S}_{\text{total}}(H)$):
  - Collect price residuals ($\boldsymbol{\epsilon}_{\text{price}}$) from the beta regression and funding residuals ($\boldsymbol{\epsilon}_{\text{funding}}$) from the funding model.
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