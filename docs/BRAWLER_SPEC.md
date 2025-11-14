# Strategy Specification: Brawler a Passive CEX-Anchored Market Maker

## 1. Strategy Name

**Passive CEX-Anchored Market Maker (BRAWLER)**

## 2. Core Concept & Philosophy

- Passive, risk-averse market maker explicitly pricing in a known latency and hardware disadvantage.
- Focuses on collecting a risk premium by providing stable, wide liquidity anchored to a high-liquidity CEX price feed.
- Spreads widen aggressively during volatility to avoid adverse selection from faster participants; this is a "slow money" liquidity provider, not an HFT bot.

## 3. Core Problem & Constraints

- **Problem:** Profitably make markets on a fast, thin venue (e.g., Hyperliquid) despite significant latency and hardware disadvantages.
- **Constraints:**
  - Bot will always be slower than HFTs and arbitrageurs.
  - Must remain robust to high volatility and potential manipulation on the thin local orderbook.

## 4. Key Assumptions

- A high-liquidity CEX (e.g., Binance, Bybit) serves as the global price source of truth.
- The local venue may sit at a persistent but stable basis relative to the CEX.
- Profit comes from capturing a wide spread and volatility premium rather than speed.
- Losses from adverse selection are part of the business; ensure collected premium \( \delta \) outweighs those losses.

## 5. Required Data Feeds

### CEX Data Feed

- `CEX_Best_Bid`: Current best bid.
- `CEX_Best_Ask`: Current best ask.
- `CEX_Last_Trade_Price`: Optional for volatility if BBO feed is unavailable.

### Local (Hyperliquid) Data Feed

- `HL_Best_Bid`: Local best bid.
- `HL_Best_Ask`: Local best ask.

### Bot State

- `Current_Inventory (I)`: Current position (positive long, negative short).

## 6. Core Parameters

| Parameter | Description |
| --- | --- |
| `BASE_SPREAD (δ_base)` | Minimum spread percentage (e.g., 0.1%) that covers fees and base margin. |
| `VOLATILITY_LOOKBACK_PERIOD (n)` | Number of CEX mid-price samples for volatility (e.g., 60 seconds). |
| `RISK_AVERSION_PARAM (k)` | Volatility multiplier; higher values widen the spread faster. |
| `BASIS_SMOOTHING_ALPHA (α)` | EMA smoothing factor for the fair basis (e.g., 0.05). |
| `MAX_INVENTORY (I_max)` | Maximum absolute position size (e.g., 1.5 contracts). |
| `INVENTORY_AVERSION_PARAM (λ)` | Dollar-value multiplier for inventory skew. |
| `ORDER_SIZE` | Quote size per order (e.g., 0.1 contracts). |
| `MAX_VOLATILITY_THRESHOLD` | Ceiling on \( \sigma_{cex} \) before halting quotes. |
| `MAX_BASIS_DEVIATION` | Tolerance between instantaneous basis and \( B_{fair} \). |
| `TICK_SIZE` | Venue tick size used to snap quotes to valid price levels. |
| `QUOTE_REPRICE_TOLERANCE_TICKS` | Number of ticks required before cancel/replace triggers. |
| `PORTFOLIO_MAX_GROSS` | Aggregate absolute inventory cap across all assets. |
| `PORTFOLIO_HALT_RATIO` | Gross ratio (vs cap) where all assets enter portfolio suspension. |
| `PORTFOLIO_REDUCE_ONLY_RATIO` | Ratio where only inventory-reducing orders remain active. |
| `PORTFOLIO_TAPER_START` | Ratio where order-size tapering begins. |
| `PORTFOLIO_MIN_ORDER_RATIO` | Floor multiplier applied when tapering reaches 100%. |

## 7. Mathematical Formulation

Run the following steps on each data tick (or fixed interval, e.g., every 500 ms).

### Step 1: Fair Value \( P_{fair} \)

CEX mid-price:

$$
P_{cex} = \frac{CEX\_Best\_Bid + CEX\_Best\_Ask}{2}
$$

Local mid-price:

$$
P_{hl} = \frac{HL\_Best\_Bid + HL\_Best\_Ask}{2}
$$

Instantaneous basis:

$$
B_t = P_{hl} - P_{cex}
$$

Fair basis EMA:

- On initialize: \( B_{fair} = B_t \)
- On tick: \( B_{fair} = \alpha \cdot B_t + (1 - \alpha) \cdot B_{fair,prev} \)

Fair value anchor:

$$
P_{fair} = P_{cex} + B_{fair}
$$

### Step 2: Risk-Adjusted Spread \( \delta \)

Compute CEX volatility:

1. Collect rolling window of the last \( n \) CEX mid-prices.
2. Log returns \( r_t = \log \left(\frac{P_{cex,t}}{P_{cex,t-1}}\right) \).
3. Standard deviation:

$$
\sigma_{cex} = \text{StDev}(r_1, r_2, \dots, r_n)
$$

Total spread (absolute dollars):

$$
\delta = P_{fair} \cdot \delta_{base} + k \cdot \sigma_{cex} \cdot P_{fair}
$$

### Step 3: Inventory Skew \( \gamma \)

Inventory ratio (clamp between -1 and +1):

$$
I_{ratio} = \frac{I}{I_{max}}
$$

Inventory skew (dollar value, with \( \lambda \) set in dollars):

$$
\gamma = \lambda \cdot I_{ratio}
$$

### Step 4: Final Quotes

Half-spread:

$$
\text{half\_spread} = \frac{\delta}{2}
$$

Quoted prices:

$$
Ask_{quote} = P_{fair} + \text{half\_spread} - \gamma
$$

$$
Bid_{quote} = P_{fair} - \text{half\_spread} - \gamma
$$

## 8. Execution Logic

### Initialization

- Load configurable parameters.
- Connect to CEX and Hyperliquid feeds.
- Fetch initial position for \( I \).
- Run the tick logic once to seed \( B_{fair} \) and place initial quotes.

### OnTick Loop

1. Receive new CEX/HL data.
2. Update \( I \) based on fills.
3. Execute Steps 1–4 above.

### Quote Placement & Cancellation

- Maintain at most one resting bid and ask.
- On each tick, compare new quotes with active orders (respect tick-size tolerance).
- If materially different, cancel old orders and place the new ones (cancel/replace loop).

### Order Sizing

- Use configurable `ORDER_SIZE`.
- Optional extension: quote tapering that reduces size as inventory grows (V2+).

### Portfolio-Level Guardrails

- Track gross inventory \( G = \sum_i |I_i| \) and net inventory \( N = \sum_i I_i \).
- If \( G / G_{max} \geq \text{halt\_ratio} \), suspend quoting for every asset until ratio falls below `resume_ratio`.
- If \( G / G_{max} \geq \text{reduce_only_ratio} \) but below halt, only allow orders that reduce the sign of \( N \) (e.g., net long → only sell/ask quotes). Bid/ask sizes taper linearly between `taper_start_ratio` and full halt, with a configurable floor.

## 9. Core Risk Management Rules

- **Max Inventory Kill Switch:** If \( |I| > I_{max} \), cancel all orders and switch to reduce-only until back within bounds.
- **Volatility Kill Switch:** If \( \sigma_{cex} > MAX\_VOLATILITY\_THRESHOLD \), cancel all orders and halt quoting.
- **CEX/HL Disconnect:** If either feed is stale beyond \( N \) seconds, cancel all orders immediately.
- **Basis De-Peg:** If \( |B_t - B_{fair}| > MAX\_BASIS\_DEVIATION \), cancel orders; the fair-value anchor is invalid.

## 10. Candidate Discovery & Screening

Before enabling an instrument in the live `assets` map, Brawler should verify that the venue relationship matches “passive CEX-anchored, wide local spread” assumptions. A **good candidate** satisfies all of the following over the lookback window \( T \):

1. **Tight Basis Alignment**
   - Mean absolute basis \( \mathbb{E}[|HL\_mid - CEX\_mid|] \) stays below a configurable fraction of the HL tick size (default: \( < 2 \times tick\_size \)).
   - Basis drift standard deviation is capped (`basis_std_max`) so the EMA anchor remains stable between restarts.

2. **Volatility Parity**
   - Realized sigma ratio \( \sigma_{hl} / \sigma_{cex} \in [0.8, 1.2] \); large divergences imply data quality issues or fat-tail local moves that undermine the anchor.

3. **Spread Edge**
   - Average HL top-of-book spread exceeds CEX spread by a `min_spread_ratio` (e.g., HL is \( \ge 2.5 \times \) wider), creating enough room for latency-buffered quotes.
   - Optional enhancement: measure “effective spread” by blending top-of-book width with top-tier depth (e.g., price level covering `order_size` notional).

4. **Liquidity & Stability**
   - HL depth at `order_size` is sufficient to avoid excessive slippage (`depth_multiple >= 3` of our resting size).
   - Funding rate variance and abnormal bursts in basis are penalized to avoid listings with structural carry shocks.

### Scoring Heuristic

For screening runs, compute per-asset statistics over recent recordings (Binance + Hyperliquid websockets or historical dumps) and derive a scalar suitability score:

```
spread_edge   = hl_spread / max(cex_spread, tick_size)
basis_penalty = max(0, (abs(basis_mean) - basis_target) / basis_target)
vol_penalty   = abs(log(sigma_hl / sigma_cex))
depth_penalty = max(0, depth_target - depth_multiple)
funding_risk  = funding_std / funding_target

score = w1*spread_edge - w2*basis_penalty - w3*vol_penalty - w4*depth_penalty - w5*funding_risk
```

Thresholds (`basis_target`, `depth_target`, `funding_target`, weights `w*`) live in a new `candidate_screening` config block. Assets must exceed `score_min` and satisfy each hard constraint before being added to `assets`.

### Workflow Integration

1. **Offline Scan:** Run `uv run python -m slipstream.strategies.brawler.tools.candidate_scan --hl-pattern ... --cex-pattern ... [--hl-depth-pattern ...] [--funding-pattern ...] --symbols ...` to replay captured feeds, compute the metrics above (including depth multiples and funding volatility when provided), and emit a ranked table plus YAML snippets for promising coins. Use `uv run python scripts/strategies/brawler/watchlist_report.py ...` in cron/CI to publish the same ranking into timestamped CSV/JSON/Markdown artifacts for the ops “watchlist” channel.
2. **Spec Compliance Check:** During onboarding, the ops checklist requires attaching the scan output to the PR that registers a new asset, proving that basis/vol parity hold.
3. **Runtime Enforcement (Optional):** Live metrics can continue to compute the same ratios; falling below `score_min` in production can trigger auto-suspension flags or alerts so operators know when an asset drifts out of profile.
4. **Data Capture Prerequisite:** All of the above assumes we are recording HL BBO/depth and Binance BBO/funding streams to disk. Ship the recorder CLI (`scripts/strategies/brawler/record_bbo.py`) as a daemon/cron target so it tails those feeds and writes per-symbol CSV/Parquet files; document how often it runs, where files land, and the retention cadence next to the automation runbook.

Additional heuristics worth exploring:

- **Funding Alignment:** Favor listings where HL funding stays near Binance futures funding to reduce “carry shocks” that inventory management can’t hedge.
- **Latency Budget:** Track mean and worst-case websocket latency per venue; candidates with persistently higher CEX lag should be deprioritized.
- **Tick/Size Economics:** Penalize instruments whose tick size is too coarse relative to the desired spread (quoting edge disappears if HL tick > half-spread).
- **Volume Gap Screener:** Compare 24h notional volumes between Hyperliquid and Binance using the REST endpoints (HL `metaAndAssetCtxs` `dayNtlVlm` vs. Binance `/fapi/v1/ticker/24hr quoteVolume`). Normalize by BTC/ETH ratios so listings with HL/Bin ratios < 30% of the baseline jump to the top of the onboarding queue. The helper CLI (`scripts/strategies/brawler/volume_gap_screener.py`) automates this check.

Capturing these metrics in the spec ensures Brawler grows the `assets` list deliberately rather than reactively, keeping the passive-MM thesis intact as multi-instrument support arrives.
