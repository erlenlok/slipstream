# Optimal Quoting Implementation Roadmap

This document outlines the Test Driven Development (TDD) sprints required to transition `brawler` from a static configuration to a mathematically optimal, budget-aware quoting engine.

## Core Objective
Maximize `OrderQuality` (responsiveness to price moves) subject to `RequestBudget > 0`.
Mechanism: Dynamically adjust `reprice_tolerance` based on available request credits, and automatically "reload" credits via wash trading when critical.

---

## Sprint 1: Observability & Economics (The Meter)
**Goal:** The strategy must know its precise economic state (Volume vs Requests) in real-time.

### 1.1 Model the "Request Purse"
- **Test:** Create `RequestPurse` class.
    - `test_purse_initialization`: Starts with 0 or query values.
    - `test_purse_deduction`: API call -> decrements budget.
    - `test_purse_income`: Trade fill $X -> increments budget by $X.
- **Implementation:**
    - New component `slipstream.strategies.brawler.economics.RequestPurse`.
    - Tracks `cumulative_volume` and `request_count`.

### 1.2 Exchange Synchronization
- **Test:** Integration test fetching `userState` from Hyperliquid.
    - `test_fetch_rate_limits`: Returns `total_volume`, `requests_sent`.
    - `test_rate_limit_parsing`: Correctly handles the "penalty box" response format.
- **Implementation:**
    - Add `HyperliquidInfoClient.get_user_state()`.
    - Engine loop periodically syncs `RequestPurse` with truthful exchange data.

### 1.3 Shadow Price Calculation
- **Test:** Calculate $\lambda$ (Shadow Price).
    - `test_shadow_price_calc`: Given spread+fee config, returns cost per request.
- **Implementation:**
    - Configurable `cost_per_request_usd` (default ~$0.00035).

---

## Sprint 2: Adaptive Tolerance (The Controller)
**Goal:** Automatically dilate quoting tolerance when poor, tighten when rich.

### 2.1 The Tolerance Function
- **Test:** `ToleranceController`.
    - `test_tolerance_rich`: Budget > 10k -> returns min tolerance (1 tick).
    - `test_tolerance_poor`: Budget < 1k -> returns dilated tolerance (e.g. 10 ticks).
    - `test_tolerance_bankrupt`: Budget < 0 -> returns survival tolerance (50+ ticks).
- **Implementation:**
    - `slipstream.strategies.brawler.economics.ToleranceController`.
    - Formula: $T = \max(T_{min}, \frac{K}{B})$

### 2.2 Engine Integration
- **Test:** `BrawlerEngine` respects dynamic tolerance.
    - `test_quote_decision_uses_dynamic_tolerance`: Mock controller returning high tolerance -> Engine returns `None` for small price moves.
- **Implementation:**
    - Wire `ToleranceController` into `_build_quote_decision`.
    - Replace static `cfg.quote_reprice_tolerance_ticks` with dynamic call.

---

## Sprint 3: The Reloader (The Pay-to-Play Module)
**Goal:** Automatically purchase request credits when dangerously low.

### 3.1 Reloader Logic
- **Test:** `ReloaderAgent`.
    - `test_needs_reload`: Returns True when Budget < CriticalThreshold.
    - `test_reload_plan`: Calculates correct trade size $Z$ to gain $N$ credits.
- **Implementation:**
    - `slipstream.strategies.brawler.reloader.ReloaderAgent`.
    - Identify liquid target pair (e.g. BTC via config).

### 3.2 Execution (Round Trip)
- **Test:** Execution safety.
    - `test_reload_execution`: Post Taker Buy -> Wait -> Post Taker Sell.
    - `test_reload_safety`: Abort if spread is too wide (> 5bps).
- **Implementation:**
    - `perform_reload_cycle(amount_usd)`.
    - Uses `HyperliquidExecutionClient` to execute immediate IOC orders.

---

## Sprint 4: Integration & Optimization
**Goal:** Full closed-loop operation.

### 4.1 Integration Tests
- **Test:** Simulation.
    - `test_full_cycle`: Start low budget -> Tolerance High -> Trigger Reload -> Budget High -> Tolerance Low.
- **Implementation:**
    - Run backtest/dry-run with mocked exchange to verify state transitions.

### 4.2 Dashboard/Logging
- **Task:** enhance logs to show:
    - `[ECON] Budget: -500 | ShadowPrice: $0.00035 | Tolerance: 55 ticks | Status: BANKRUPT`
- **Implementation:**
    - Add `RequestPurse` stats to `_log_status_summary`.

---
