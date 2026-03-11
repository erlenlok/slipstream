# TDD Sprints for Spectrum Strategy Implementation

This document outlines the Test-Driven Development sprints for implementing the Spectrum strategy in the Slipstream multi-strategy platform. Following the specification in `spectrum_spec.md`, these sprints will add Spectrum as the third strategy alongside Brawler and Gradient.

## Overview
- **Strategy Name**: Spectrum - Idiosyncratic Statistical Arbitrage System
- **Current Strategies**: Brawler, Gradient
- **Target**: Brawler, Gradient, Spectrum
- **Data Alignment**: All components align with existing Slipstream data structures

## Sprint 1: The Engine Room (Factor Engine Implementation)

### Goal
Implement Module A: Factor Engine with OLS regression to decompose asset returns into BTC/ETH Betas and Idiosyncratic Residuals.

### Requirements from spec
- Rolling OLS (Window=30 periods) for each Asset i: `r_i = α + β_BTC*r_BTC + β_ETH*r_ETH + ε_i`
- Apply Dynamic Universe Protocol: Liquidity filter (30d_Avg_Vol < $10M inactive), Minimum History (>30 days)
- Output: `betas`, `residuals`, `idio_vol`

### Reusable Components
- Data loading from `scripts/data_load.py` (existing API data structure)
- Data validation utilities from `slipstream.core.common`
- Existing pandas workflow patterns

### Sprint Implementation Plan
1. Create `src/slipstream/strategies/spectrum/` directory with basic structure
2. Implement `factor_model.py` with OLS regression
3. Add universe filtering logic
4. Write comprehensive unit tests validating:
   - OLS regression accuracy
   - Universe masking
   - Rolling window behavior
   - NaN handling for short history assets

### Success Criteria
- Module correctly decomposes returns into betas and residuals
- Handles assets with insufficient history correctly (NaN output)
- Passes all unit tests
- Performance benchmarked against expected speed

## Sprint 2: The Alphas & Ridge (Signal Factory + Dynamic Weighting)

### Goal
Implement Module B: Signal Factory and Module C: Dynamic Weighting to generate standardized risk factor scores and determine factor weights via pooled Ridge regression.

### Requirements from spec
- Module B: Generate Idio-Carry, Idio-Momentum, Idio-MeanRev signals
- Module C: Rolling Pooled Ridge Regression to weight factors
- Input: residuals, daily_funding_yield, betas, idio_vol
- Standardization: Cross-Sectional Z-Score (winsorized at ±3)

### Reusable Components
- Existing Ridge regression examples from `legacy/scripts/find_optimal_H_joint.py`
- Signal normalization utilities from `slipstream.core.signals.idiosyncratic_momentum`
- Cross-sectional z-scoring from existing codebase
- Bootstrap sampling patterns

### Sprint Implementation Plan
1. Create `signals.py` with Idio-Carry, Idio-Momentum, Idio-MeanRev functions
2. Implement Ridge regression weighting in `ridge_weighting.py`
3. Add signal standardization and winsorization
4. Write tests validating:
   - Signal calculations (momentum, mean reversion, carry)
   - Ridge regression coefficient stability
   - Cross-sectional z-score correctness
   - Target alignment for regression

### Success Criteria
- Signals correctly computed from residuals and funding
- Ridge weighting produces stable coefficients
- All signals properly standardized
- Factor weights update daily as specified

## Sprint 3: The Optimizer (Robust Optimizer)

### Goal
Implement Module D: Robust Optimizer using CVXPY to generate optimal idiosyncratic portfolio weights.

### Requirements from spec
- Input: Composite Alpha, residuals history, previous_weights_map
- Constraints: TargetIdioLeverage, MaxSinglePos, LiquidityLimit
- Cost model with lambda vector proportional to volatility/spread
- Ledoit-Wolf covariance shrinkage
- CVXPY optimization with transaction cost penalty

### Reusable Components
- CVXPY optimization from `slipstream.core.optimizer` (existing code may need adaptation)
- Ledoit-Wolf shrinkage from `slipstream.core.common.risk`
- Transaction cost modeling patterns
- Portfolio constraint utilities from existing portfolio module

### Sprint Implementation Plan
1. Create `optimizer.py` with CVXPY-based optimization
2. Implement cost vector calculation
3. Add Ledoit-Wolf covariance estimation
4. Implement all specified constraints
5. Write tests validating:
   - Optimization convergence
   - Constraint satisfaction
   - Cost penalty effectiveness
   - Leverage constraint enforcement

### Success Criteria
- Optimizer produces valid portfolio weights
- All constraints respected
- Costs properly penalized
- Leverages controlled as specified

## Sprint 4: The Bridge (Execution System)

### Goal
Implement Module E: Execution Bridge for two-stage timing with beta hedging.

### Requirements from spec
- Two-stage execution: 23:50 (projected) and 00:01 (correction)
- Beta hedging based on confirmed fills (not targets)
- Convert idio-weights to tradeable orders
- Handle new entrants and dropouts

### Reusable Components
- Strategy template from `src/slipstream/strategies/template/`
- Brawler execution patterns for hedging
- Configuration system from `slipstream.core.config`
- Risk management patterns from existing strategies
- API connection patterns from Brawler

### Sprint Implementation Plan
1. Create strategy structure following template pattern
2. Implement two-stage execution timing
3. Add beta hedging logic
4. Implement universe change handling (entrants/dropouts)
5. Write integration tests validating:
   - Two-stage timing accuracy
   - Beta hedging behavior
   - Position management
   - Order execution patterns

### Success Criteria
- Two-stage execution works as specified
- Beta hedging only on confirmed fills
- Proper handling of universe changes
- Correct position tracking

## Sprint 5: Integration and Testing

### Goal
Complete integration of Spectrum strategy with the Slipstream platform.

### Requirements from spec
- End-to-end validation matching spec requirements
- Integration with existing strategy management system
- Backtesting alignment with existing data structures
- Performance metrics aligned with other strategies

### Reusable Components
- Strategy registration system from `slipstream.strategies.__init__`
- Backtesting framework from `slipstream.core.portfolio.backtest`
- Configuration loading from `slipstream.core.config`
- Performance metrics from existing analytics modules

### Sprint Implementation Plan
1. Register Spectrum in the strategy registry
2. Create CLI entry points following existing patterns
3. Integrate with existing backtesting framework
4. Add comprehensive system tests validating:
   - End-to-end pipeline functionality
   - Data alignment with existing structures
   - Performance against other strategies
   - Risk management effectiveness

### Success Criteria
- Spectrum integrates seamlessly with existing architecture
- Backtesting works with existing data structures
- Performance metrics comparable to other strategies
- All configuration and CLI patterns consistent with existing strategies

## Data Alignment Strategy

All implementations will align with existing Slipstream data structures:
- Daily OHLCV data in `data/market_data/1d/` from `hl-load --interval 1d` command
- Daily funding data in `data/market_data/1d/` from `hl-load --interval 1d` command
- All dataframes indexed by datetime with asset columns
- Return calculations consistent with existing modules
- Volume and liquidity data from existing market data files

The Spectrum strategy specification calls for daily data processing, which aligns perfectly with the available daily data in the existing Slipstream system. Daily data is already available from the `hl-load` command using the `--interval 1d` option:
- OHLCV data: Daily candles already in `data/market_data/1d/*.csv` files
- Funding data: Daily funding rates already in `data/market_data/1d/*.csv` files
- Returns: Daily returns calculated from daily closes