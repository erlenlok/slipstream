# Idiosyncratic Statistical Arbitrage System Specification

## 1. Context & Philosophy
This system is designed to harvest risk premia (Momentum, Mean Reversion, Funding Carry) in the crypto perpetual market (Hyperliquid). It operates by stripping out market factors (BTC/ETH) to isolate idiosyncratic returns, optimizing a portfolio in this "idio-space," and then executing a hedged portfolio in "physical space."

## 2. Technical Requirements
- **Language**: Python 3.10+
- **Core Libraries**: pandas, numpy, cvxpy, sklearn, statsmodels
- **Execution Venue**: Hyperliquid (via API)
- **Frequency**: Daily (00:00 UTC Close)

### 2.1 Dynamic Universe Protocol (Robustness)
- **Liquidity Filter**: Re-evaluated Daily. Assets with `30d_Avg_Vol < $10M` are marked inactive.
- **Minimum History**: Assets must have > 30 days of price history to enter the active universe (required for Rolling OLS).
- **Handling Dropouts**: If an asset transitions active -> inactive, its target weight is forced to 0 by the Optimizer/Bridge.
- **Handling New Entrants**: If an asset transitions inactive -> active, it enters the covariance matrix with $w_{prev} = 0$.

### 2.2 Operational Robustness (Execution Safety)
- **Beta Leakage Control (Atomic Hedging)**:
  - The Beta Hedge (BTC/ETH) is executed strictly based on confirmed fills of the altcoin legs.
  - NEVER hedge based on "Projected" or "Open" orders.
  - The `HedgeManager` runs a continuous loop checking `Net_Portfolio_Delta`. If `abs(Delta) > Threshold`, it executes a market hedge.
- **Execution Timing (Predictive Rebalancing)**:
  - **Phase 1 (Front-Run)**: Execution begins at 23:50 UTC using "Projected Close" prices (Live Mid).
  - **Phase 2 (Correction)**: At 00:01 UTC, re-run pipeline with Official Close. Update remaining TWAP orders to target the final quantity.
- **Whipsaw Policy**: If Signals flip between 23:50 and 00:00, the system accepts the impact cost to correct the position. Convergence to the model portfolio takes precedence over transaction costs in this scenario.
- **Data Grace Period**: Pipeline waits 120 seconds after daily close before fetching OHLCV to ensure exchange data finality (for Phase 2).

## 3. Module Definitions

### Module A: Factor Engine (`factor_model.py`)
- **Responsibility**: Decompose asset returns into Market Beta and Idiosyncratic Residuals.
- **Input**: Daily OHLCV Dataframes for Universe + BTC + ETH.
- **Process**:
  - **Calculate Log Returns (Daily)**.
  - **Funding Data**: Calculate `Daily_Funding_Yield` as the SUM of the past 24 hourly funding rates.
  - **Universe Masking**: For each date $t$, select only assets satisfying Protocol 2.1.
  - **Perform Rolling OLS** (Window=30 periods / 30 days) for each Asset $i$:
    $$r_{i} = \alpha + \beta_{BTC} r_{BTC} + \beta_{ETH} r_{ETH} + \epsilon_{i}$$
- **Output**:
  - `betas`: DataFrame of hedge ratios (Masked NaN if insufficient history).
  - `residuals`: DataFrame of $\epsilon$ (idiosyncratic daily returns).
  - `idio_vol`: Rolling standard deviation of $\epsilon$ (e.g., 30-day window).

### Module B: Signal Factory (`signals.py`)
- **Responsibility**: Generate standardized risk factor scores from residuals.
- **Input**: `residuals`, `daily_funding_yield`, `betas`, `idio_vol`.
- **Process**:
  - **Warmup Handling**: If residuals contain NaN (due to new listing), signals output NaN until full lookback (e.g., 10 days) is available.
  - **Idio-Carry**: Calculate hedged funding yield (annualized), normalize by `idio_vol` (annualized).
  - **Idio-Momentum**: $(EMA_{fast=3d} - EMA_{slow=10d})$ of residuals, normalize by `idio_vol`.
  - **Idio-MeanRev**: Negative deviation from short-term mean ($SMA_{5d}$), normalize by `idio_vol`.
    - Math: $-1 \times (\epsilon_{t} - SMA(\epsilon, 5d)) / \sigma_{\epsilon}$.
  - **Standardization**: Apply Cross-Sectional Z-Score (winsorized at $\pm 3$) to all signals. Ignore NaNs in Z-score calculation.
- **Output**: `signal_matrix` (Time x Assets x 3_Factors).

### Module C: Dynamic Weighting (`ridge_weighting.py`)
- **Responsibility**: Determine the weight of each risk factor using Rolling Pooled Ridge Regression.
- **Input**: `signal_matrix`, `residuals` (Forward shifted 1 day).
- **Target Definition**: $Y = \epsilon_{t+1} / \sigma_{\epsilon, t}$ (Next Day Residual Return normalized by Vol).
- **Process**:
  - **Frequency**: Retrain model Daily.
  - **Stacking (Robust)**: Flatten the Time x Asset matrices. Mask indices where Target or Signals are NaN.
  - **Pooling**: Pool data from all assets over Lookback (e.g., 60 periods).
  - **Ridge Regression**: Fit `Ridge(alpha=opt, positive=True, fit_intercept=False)`.
  - **Extract coefficients** $[w_{mom}, w_{mr}, w_{carry}]$.
- **Output**: Global weights vector for the current timestamp.

### Module D: Robust Optimizer (`optimizer.py`)
- **Responsibility**: Generate optimal idiosyncratic portfolio weights.
- **Input**: Composite Alpha ($\sum w_{factor} S_{factor}$), residuals history, `previous_weights_map` (Dict).
- **Process**:
  - **Universe Alignment**: Identify `current_assets`. Construct $w_{prev}$ (fill 0.0 for new entrants).
  - **Cost Vector ($\lambda$)**: Construct vector $\lambda$ where $\lambda_i$ is proportional to the asset's rolling volatility or spread (proxy for liquidity cost).
  - **Covariance**: Compute Sample Covariance of residuals (Daily) for `current_assets` only $\to$ Apply Ledoit-Wolf Shrinkage.
  - **Optimization (CVXPY)**:
    - Obj: Maximize $\mu^T w - \gamma w^T \Sigma w - || \lambda \odot (w - w_{prev}) ||_1$. (Using Element-wise multiplication for cost).
  - **Constraints**:
    - $\sum |w| \le \text{TargetIdioLeverage}$ (e.g., 1.0).
    - Note: Physical Gross Leverage will be $\approx \text{TargetIdioLeverage} \times (1 + \text{AvgBeta})$.
    - $|w_i| \le \text{MaxSinglePos}$.
    - $|w_i| \le \text{LiquidityLimit}_i$.
- **Output**: `target_idio_weights` (Vector, indexed by `current_assets`).

### Module E: Execution Bridge (`execution.py`)
- **Responsibility**: Convert idio-weights to tradeable orders with Strict Beta Neutrality and Two-Stage Timing.
- **Input**: `target_idio_weights`, `betas`, `account_equity`, `current_positions`, `live_prices` (Websockets).
- **Process (The Loop)**:
  - **Trigger 23:50 UTC (Projected Phase)**:
    1. Fetch `live_prices` (mid). Treat as $P_{close}$.
    2. Run Pipeline (Modules A->D) to get $w_{projected}$.
    3. Generate `TWAP_Orders_Projected` (Duration: 15 mins).
    4. Start Execution.
  - **Trigger 00:01 UTC (Correction Phase)**:
    1. Fetch Official $P_{close}$ (OHLCV).
    2. Run Pipeline (Modules A->D) to get $w_{final}$.
    3. Calculate Delta: $Qty_{correction} = Qty_{final} - Qty_{filled\_so\_far}$.
    4. Update remaining TWAP orders to target $Qty_{correction}$ over the final ~5 mins.
  - **Hedge Calculation (Continuous)**:
    - Monitor `current_positions` (Live Fills).
    - Calculate: $\text{Delta}_{BTC} = \sum (Qty_{i} \cdot \beta_{i, BTC})$ and $\text{Delta}_{ETH}$.
    - If $|\text{Net\_Delta}| > \text{Hedge\_Threshold}$, Market Sell/Buy BTC/ETH to neutralize.
- **Output**: Final list of Execution Tasks.

## 4. Implementation Guidelines (Sprint Plan)

### Sprint 1: The Engine Room
- **Goal**: Get daily data in, residuals out.
- **Robustness**: Ensure Rolling OLS handles `min_periods` correctly and outputs NaNs for short history. Verify `Daily_Funding_Yield` is a Sum.

### Sprint 2: The Alphas & Ridge
- **Goal**: Generate the 3 signals and the Ridge weigher.
- **Robustness**: Implement the Stacking/Flattening logic. Ensure Target is Residual Return.

### Sprint 3: The Optimizer
- **Goal**: Build the CVXPY module.
- **Robustness**: Implement Vectorized Cost Penalty ($\lambda$).

### Sprint 4: The Bridge (Two-Stage)
- **Goal**: End-to-end simulation with live data simulation.
- **Robustness**: Create a `MockWebsocket` class. Verify that Beta Hedge only reacts to Fills, not Targets.