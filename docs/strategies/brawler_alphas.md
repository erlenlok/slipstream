# Technical Specification: Hyperliquid L1 Microstructure Alphas
**Version**: 1.1
**Target Asset Class**: Low-Liquidity Perpetuals ("Shitcoins")
**Infrastructure Requirement**: Hyperliquid L1 Read-Node or "StreamBlocks" gRPC subscription.

## 1. System Overview
The goal is to build a **Rapid Taker Bot** that exploits information asymmetry in the Hyperliquid L1 block data. Unlike standard bots that react to price updates (L2 Data), this system reacts to **Intent** and **Actor Profiling** found in the transaction payloads (L1 Data).

### Core Philosophy
*   **Latency**: Strategies must execute within the same block or next block (~200ms window).
*   **Data Source**: `replica_cmds` (L1 Block Stream). Do not use the standard trades websocket channel for signal generation, as it lacks `tif` and `cloid` metadata.

---

## 2. Data Ingestion Layer

### 2.1 Input Data Source
*   **Protocol**: Hyperliquid L1 `stream_blocks` (gRPC) or `debug_traceTransaction` (RPC polling if low freq, but Stream is required for Alpha 3).
*   **Target Payload**: The `signed_action_bundles` array within a Block.

### 2.2 Critical Data Schema
The implementation must extract the following fields from the raw Block JSON:

| Field | Path in JSON | Type | Purpose |
| :--- | :--- | :--- | :--- |
| Asset ID | `action.orders[].a` | int | Filtering for target shitcoin. |
| Is Buy | `action.orders[].b` | bool | Directionality. |
| Size | `action.orders[].s` | float | Volume sizing. |
| Limit Price | `action.orders[].p` | float | Price aggression. |
| Time In Force | `action.orders[].t.limit.tif` | str | **Critical for Alpha 3**. Detects `Ioc` (Immediate-or-Cancel). |
| Client Order ID | `action.orders[].c` | hex | **Critical for Alpha 3**. Profiling bot sophistication. |
| Signer Address | `signed_actions[].signature` (derived) | hex | **Critical for Alpha 1**. Mapping wallet IDs. |

---

## 3. Alpha Specifications

### Alpha 1: The L1 Wallet Shadow (Insider Tracking)
**Logic**: Detect accumulation by "Insider" wallets that precedes price expansion.

**Data Structure**:
*   `WalletProfile`: Map `{ wallet_address: { 'score': float, 'last_seen': timestamp } }`
*   `AccumulationMetric`: Rolling sum of Net Delta for specific wallets.

**Algorithm**:
1.  Ingest every Limit Order Place action.
2.  Tag the wallet address.
3.  If wallet has history of buying > $10k notional < 5 mins before a > 5% pump: Tag as **INSIDER**.

**Signal**:
*   IF **INSIDER** wallet adds > $5k bids within 1% of Mid Price: **Trigger LONG**.
*   IF **INSIDER** wallet places large Ask wall: **Trigger EXIT**.

### Alpha 2: Order Book Replenishment Velocity (The "Fear" Metric)
**Logic**: Measure Market Maker (MM) confidence by tracking the time it takes for liquidity to return to a price level after it is consumed.

**Metric**: `ReplenishmentLatency` (ms).

**Parameters**:
*   `RecoveryThreshold`: 0.8 (Liquidity must return to 80% of pre-trade size).
*   `Timeout`: 5000ms.

**Algorithm**:
1.  Monitor L2 Book updates.
2.  **Event Trigger**: Detect a Trade event that fully or partially consumes a specific Bid/Ask price level.
3.  **Timer Start**: Record timestamp $t_{start}$ of the consumption event.
4.  **Monitor Level**: Watch updates for that specific price tick.
5.  **Timer Stop**: Record timestamp $t_{end}$ when volume at that level $\ge$ PreTradeVolume * RecoveryThreshold.
6.  **Calculate**: $\Delta t = t_{end} - t_{start}$.

**Interpretation**:
*   **Fast Replenishment (< 200ms)**: MMs are confident, running standard auto-refill logic. Trend is stable.
*   **Slow/No Replenishment (> 2000ms)**: MMs have turned off auto-refill or widened spreads. They suspect toxic flow. **Fear Signal**.

**Signal**:
*   IF Sell Flow is active AND ReplenishmentLatency spikes > 2s: **Market Sell** (Front-run the liquidity pull).

### Alpha 3: Sophisticated Order Flow Imbalance (The "Rapid Taker")
**Logic**: Calculate buy/sell pressure weighted by the sophistication of the order placement mechanism.

**Formula**: Time-Decayed Weighted Sum.
$$OFI_t = OFI_{t-1} \cdot e^{-\lambda \Delta t} + (W_{buy} \cdot Vol_{buy}) - (W_{sell} \cdot Vol_{sell})$$

**Parameters**:
*   `HalfLife`: 2.0 seconds (Aggressive decay).
*   `Weights` ($W$):
    *   **Weight=5.0**: If `TIF == 'Ioc'` (Immediate-or-Cancel).
    *   **Weight=3.0**: If `cloid` is High-Entropy Hex (Bot).
    *   **Weight=0.5**: If `cloid` is null (Retail GUI).

**Execution Trigger**:
*   IF `OFI_t` > Threshold (e.g., +10,000 weighted units) AND Price is stable: **Market BUY**.
*   **Note**: You are betting that the IOC buyer knows something the Limit sellers don't.

---

## 4. Execution Strategy (Implementation Notes)
*   **Loop**: Run the Parser in an infinite loop consuming the Block Stream.
*   **State**: Maintain a local `OrderBook` object and `AlphaState` object.
*   **Concurrency**:
    *   Thread A: Network Ingestion (buffer blocks).
    *   Thread B: Parser & Alpha Calculation (compute signals).
    *   Thread C: Execution (send orders).
*   **Risk Management**:
    *   **Staleness Check**: If Block Stream latency > 500ms, disable Alpha 3 (too risky).
    *   **Inventory**: Hard cap on max position size regardless of signal strength.

---

## 5. Assessment & Feasibility

### Technical Challenges
1.  **L1 Data Access**:
    *   Hyperliquid does not expose a public gRPC block stream for *users* easily. The "L1" data is usually only available if you run a node or rely on the `debug_traceTransaction` (slow) or reverse-engineering the consensus mechanism.
    *   **Workaround**: The `replica_cmds` are sometimes broadcasted or can be inferred from the `l2Book` updates + User Fills, but `TIF` and `CLOID` visible to *other* users is generally **private information** on most exchanges.
    *   **Assumption Check**: Does Hyperliquid actually publish the `TIF` and `ClientOrderId` of *other participants* in the public block data? If not, Alpha 3 is impossible. Alpha 1 relies on "Signer Address", which is definitely public in L1 blockchains but might be hashed or obscured in HL's specific L1 implementation (which is optimized for speed).
    *   **Verdict**: **High Technical Risk**. We need to verify if we can actually get this data from the public endpoint (`https://api.hyperliquid.xyz/info` -> `l2Book` or `userFills` does NOT have this). We might need to connect to the underlying tendon/consensus layer if accessible.

2.  **Latency Budget**:
    *   Parsing raw blocks in Python might be too slow for "Same Block" execution. Rust or C++ is recommended for the ingestion layer.
    *   For Python `slipstream`, we can prototype, but expect slippage.

3.  **Alpha 2 Feasibility**:
    *   This is purely L2 Book data. **High Feasibility**. We already have `HyperliquidQuoteStream`. We just need to track "Level Restoration".

### Recommendation
1.  **Start with Alpha 2**: It uses existing WebSocket infrastructure (`l2Book`). It fits the current `Brawler` architecture perfectly.
2.  **Investigate Alpha 1/3 Data**: Before coding, write a script to fetch a raw block/transaction bundle from HL to verify if `tif`, `cloid`, and `signer` are actually visible.