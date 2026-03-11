#!/usr/bin/env python3
"""
Utility script to sanity-check passive limit execution on Hyperliquid.

Workflow:
1. Places a passive (join-best) limit buy on BTC sized at ~USD 10.
2. Waits up to a configured timeout for the order to rest and fill.
3. If filled, places an opposite passive limit sell for the filled notional.
4. Emits a summary of fills and any residual position.

This script reuses the same execution helpers that power the live rebalance
pipeline, so it is a convenient smoke test for the passive entry logic.

Requirements:
- Live credentials must be configured via environment variables or config file.
- Run from the repo root: `python scripts/strategies/gradient/test_passive_btc_fill.py`
  (legacy path `scripts/test_passive_btc_fill.py` remains available for now).
"""

from __future__ import annotations

import argparse
import time
import os
from typing import Dict, Any, List, Tuple, Optional

from slipstream.strategies.gradient.live.config import load_config, validate_config
from slipstream.strategies.gradient.live.execution import (
    _prepare_hyperliquid_context,
    _calculate_slippage_bps,
    _build_asset_meta,
    _fetch_meta_and_asset_ctxs,
    _price_tick,
    _resolve_main_wallet,
    place_limit_orders,
    monitor_fills_with_timeout,
    cancel_orders,
    get_current_positions,
)

USD_NOTIONAL = 10.0
ASSET = "BTC"
DEFAULT_TIMEOUT = 300  # seconds
MONITOR_SLICE_SECONDS = 20
MIN_REST_BEFORE_REPRICE = 6
REPRICE_TICK_THRESHOLD = 1
REPRICE_MIN_SPREAD_BPS = 0.5
STABILITY_DELAY_SECONDS = 1.0
MAX_REPRICES = 6
CANCEL_COOLDOWN_SECONDS = 0.5
POSITION_SETTLE_TIMEOUT = 20
POSITION_POLL_INTERVAL = 2
FINAL_FLAT_TOLERANCE_USD = 2.0


def _format_order(order: Dict[str, Any]) -> str:
    return (
        f"{order.get('asset')} {order.get('side')} "
        f"${order.get('fill_usd', order.get('size_usd', 0.0)):.2f} "
        f"@ {order.get('execution_px', order.get('limit_px'))}"
    )


def _sum_fill_coin(fills: List[Dict[str, Any]]) -> float:
    total = 0.0
    for fill in fills:
        coin = fill.get("size_coin")
        if coin is None and fill.get("fill_usd") and fill.get("execution_px"):
            try:
                coin = float(fill["fill_usd"]) / float(fill["execution_px"])
            except (TypeError, ValueError, ZeroDivisionError):
                coin = None
        if coin is None and fill.get("size_usd") and fill.get("execution_px"):
            try:
                coin = float(fill["size_usd"]) / float(fill["execution_px"])
            except (TypeError, ValueError, ZeroDivisionError):
                coin = None
        if coin is None:
            continue
        total += abs(float(coin))
    return total


def _filled_notional_usd(fills: List[Dict[str, Any]]) -> float:
    total = 0.0
    for fill in fills:
        fill_usd = fill.get("fill_usd")
        if fill_usd is None:
            fill_usd = fill.get("size_usd")
        if fill_usd is None:
            continue
        total += abs(float(fill_usd))
    return total


def _wait_for_position_snapshot(
    config,
    expected_direction: str,
    expected_usd: float,
) -> Optional[float]:
    """
    Poll Hyperliquid main wallet for the expected BTC position.

    Returns:
        Signed USD position if detected within the timeout, else None.
    """
    deadline = time.time() + POSITION_SETTLE_TIMEOUT
    direction = expected_direction.lower()
    min_detectable = max(1.0, expected_usd * 0.4)

    while time.time() < deadline:
        snapshot = get_current_positions(config)
        position_usd = snapshot.get(ASSET)
        if position_usd is not None:
            if direction == "long" and position_usd > min_detectable:
                return position_usd
            if direction == "short" and position_usd < -min_detectable:
                return position_usd
        time.sleep(POSITION_POLL_INTERVAL)

    return None


def _fetch_open_orders(config, context) -> List[Dict[str, Any]]:
    addresses: List[str] = []
    try:
        addresses.append(_resolve_main_wallet(config))
    except Exception:
        pass

    api_vault = context.account_address or os.getenv("HYPERLIQUID_API_KEY")
    if api_vault:
        addresses.append(api_vault)

    seen_orders: List[Dict[str, Any]] = []
    for address in addresses:
        try:
            orders = context.info.open_orders(address)
        except Exception as exc:  # pragma: no cover - network call
            print(f"⚠️  Failed to fetch open orders for {address}: {exc}")
            continue

        if isinstance(orders, dict) and "data" in orders:
            orders = orders["data"]
        if not isinstance(orders, list):
            continue
        seen_orders.extend(orders)

    return seen_orders


def _matching_open_orders(orders: List[Dict[str, Any]], is_buy: bool) -> List[Dict[str, Any]]:
    matching = []
    for entry in orders:
        if not isinstance(entry, dict):
            continue
        if entry.get("coin") != ASSET:
            continue
        side = str(entry.get("side", "")).upper()
        if side in ("A", "B"):
            if is_buy and side != "B":
                continue
            if not is_buy and side != "A":
                continue
        matching.append(entry)
    return matching


def _await_flat_position(config) -> float:
    """
    Poll until BTC exposure shrinks within tolerance or timeout reached.

    Returns:
        Final USD exposure (may be > tolerance if timeout hit).
    """
    deadline = time.time() + POSITION_SETTLE_TIMEOUT
    latest = 0.0
    while time.time() < deadline:
        snapshot = get_current_positions(config)
        latest = snapshot.get(ASSET, 0.0)
        if abs(latest) <= FINAL_FLAT_TOLERANCE_USD:
            return latest
        time.sleep(POSITION_POLL_INTERVAL)
    return latest


def _passive_round_trip(
    context,
    config,
    usd_notional: float,
    timeout: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    asset_meta = context.asset_meta.get(ASSET) or _refresh_asset_meta(context)

    print(f"➡️  Submitting passive buy for {ASSET} (${usd_notional:.2f})")
    filled_buy = _execute_passive_leg(
        context=context,
        config=config,
        usd_notional=usd_notional,
        timeout=timeout,
        is_buy=True,
    )
    buy_coin = _sum_fill_coin(filled_buy)
    print(f"    Filled coin: {buy_coin:.8f}")
    print(f"✅  Buy filled: {', '.join(_format_order(o) for o in filled_buy)}")

    # Refresh positions to get the precise USD delta for the exit leg.
    filled_notional = buy_coin * (asset_meta.mid_px or filled_buy[0].get("execution_px") or 0.0)
    current_btc = _wait_for_position_snapshot(
        config=config,
        expected_direction="long",
        expected_usd=filled_notional,
    )
    if current_btc is None:
        print("⚠️  Position snapshot missing after buy; using fill notional for exit leg.")
        current_btc = filled_notional
    asset_meta = _refresh_asset_meta(context)
    price_basis = asset_meta.mid_px or filled_buy[0].get("execution_px")
    if not price_basis or price_basis <= 0:
        price_basis = usd_notional / max(buy_coin, 1e-9)
    position_coin = abs(current_btc) / max(price_basis, 1e-9)
    print(f"    Snapshot USD: {current_btc:.2f} (implied coin {position_coin:.8f})")
    if position_coin > buy_coin * 1.05:
        exit_coin = position_coin
    else:
        exit_coin = buy_coin
    exit_usd = exit_coin * (asset_meta.mid_px or filled_buy[0].get("execution_px") or 0.0)
    print(f"    Target exit coin: {exit_coin:.8f} (~${exit_usd:.2f})")

    print(f"➡️  Submitting passive sell for {ASSET} (${current_btc:.2f})")
    filled_sell = _execute_passive_leg(
        context=context,
        config=config,
        usd_notional=abs(exit_usd),
        timeout=timeout,
        is_buy=False,
    )
    sell_coin = _sum_fill_coin(filled_sell)
    print(f"    Sold coin: {sell_coin:.8f}")
    print(f"✅  Sell filled: {', '.join(_format_order(o) for o in filled_sell)}")
    return filled_buy, filled_sell


def _summarize_fills(label: str, fills: List[Dict[str, Any]]) -> None:
    total = sum(abs(o.get("fill_usd", 0.0)) for o in fills)
    slippages = [
        _calculate_slippage_bps(o.get("execution_px"), o.get("reference_mid"), o.get("side") == "buy")
        for o in fills
        if o.get("execution_px") and o.get("reference_mid")
    ]
    avg_slip = (
        sum(slippages) / len(slippages)
        if slippages
        else None
    )
    print(f"{label}: ${total:.2f} across {len(fills)} fills", end="")
    if avg_slip is not None:
        print(f" (avg slippage {avg_slip:+.3f} bps)")
    else:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usd",
        type=float,
        default=USD_NOTIONAL,
        help="USD notional for each leg (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Passive fill timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    config = load_config()
    if config.dry_run:
        raise RuntimeError("Config is in dry_run mode; provide live credentials.")

    config.execution = dict(config.execution)
    config.execution.setdefault("limit_order_aggression", "join_best")
    config.execution["passive_timeout_seconds"] = args.timeout

    min_usd = float(config.execution.get("min_order_size_usd", 10.0))
    # Provide a small buffer so rounding to exchange precision doesn't drop below the minimum.
    effective_usd = max(args.usd, min_usd * 1.15)
    config.execution["min_order_size_usd"] = min_usd

    validate_config(config)

    context = _prepare_hyperliquid_context(config)
    if ASSET not in context.asset_meta:
        _refresh_asset_meta(context)

    entry_fills, exit_fills = _passive_round_trip(
        context,
        config,
        effective_usd,
        args.timeout,
    )

    _summarize_fills("Entry", entry_fills)

    asset_meta = context.asset_meta.get(ASSET) or _refresh_asset_meta(context)
    final_btc = _await_flat_position(config)
    _summarize_fills("Exit", exit_fills)

    final_coin_est = (
        final_btc / asset_meta.mid_px if asset_meta.mid_px else float("nan")
    )
    print(f"Final BTC exposure: ${final_btc:.2f} (~{final_coin_est:.8f} coin)")
    if abs(final_btc) > 1.0:
        print("⚠️  Residual BTC exposure detected! Consider flattening manually.")


def _refresh_asset_meta(context) -> Any:
    """Fetch a fresh asset metadata snapshot so repricing uses live book data."""
    meta_payload, asset_ctxs = _fetch_meta_and_asset_ctxs(context.info)
    refreshed = _build_asset_meta(context.info, meta_payload, asset_ctxs)
    context.asset_meta.update(refreshed)
    asset_meta = context.asset_meta.get(ASSET)
    if not asset_meta:
        raise RuntimeError(f"Missing asset metadata for {ASSET}")
    return asset_meta


def _best_price(meta, is_buy: bool) -> Optional[float]:
    return meta.impact_bid if is_buy else meta.impact_ask


def _spread_bps(meta) -> Optional[float]:
    bid = meta.impact_bid
    ask = meta.impact_ask
    if not bid or not ask or bid <= 0 or ask <= bid:
        return None
    mid = meta.mid_px or (bid + ask) / 2.0
    if not mid or mid <= 0:
        return None
    return ((ask - bid) / mid) * 1e4


def _should_reprice(meta, current_price: float, is_buy: bool, last_reprice_ts: float, require_rest: bool = True) -> bool:
    if current_price is None:
        return False
    now = time.time()
    if require_rest and (now - last_reprice_ts) < MIN_REST_BEFORE_REPRICE:
        return False
    best_px = _best_price(meta, is_buy)
    if not best_px or best_px <= 0:
        return False
    tick = max(_price_tick(meta), 1e-8)
    guard = max(tick * REPRICE_TICK_THRESHOLD, current_price * 1e-4 * 0.5)
    if is_buy:
        if best_px <= current_price + guard:
            return False
    else:
        if best_px >= current_price - guard:
            return False
    spread = _spread_bps(meta)
    if spread is not None and spread < REPRICE_MIN_SPREAD_BPS:
        return False
    return True


def _execute_passive_leg(
    context,
    config,
    usd_notional: float,
    timeout: int,
    is_buy: bool,
) -> List[Dict[str, Any]]:
    """Place a passive order and judiciously rejoin BBO without getting walked."""
    deadline = time.time() + timeout
    if ASSET not in context.asset_meta:
        _refresh_asset_meta(context)
    delta = {ASSET: usd_notional if is_buy else -usd_notional}

    orders, errors = place_limit_orders(
        delta,
        config,
        context.info,
        context.asset_meta,
        context.exchange,
    )
    if errors:
        raise RuntimeError(f"Failed to place {'buy' if is_buy else 'sell'} order: {errors}")
    if not orders:
        raise RuntimeError("No orders placed.")

    current_orders = orders
    current_quote_px = orders[0].get("limit_px")
    last_reprice_ts = time.time()
    reprices = 0
    fills: List[Dict[str, Any]] = []

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            cancel_orders(current_orders, config, context.exchange)
            raise RuntimeError(f"{'Buy' if is_buy else 'Sell'} leg did not fill before timeout.")

        window = int(min(max(5, MONITOR_SLICE_SECONDS), max(5, remaining)))
        filled, unfilled = monitor_fills_with_timeout(
            current_orders,
            window,
            config,
            context.info,
        )

        if filled:
            open_orders_snapshot = _fetch_open_orders(config, context)
            if open_orders_snapshot:
                residual = _matching_open_orders(open_orders_snapshot, is_buy=is_buy)
                if residual:
                    print(
                        "ℹ️  Detected resting orders on book despite missing from poll; "
                        "continuing to wait for actual fills."
                    )
                    for entry in residual:
                        try:
                            oid = entry.get("oid")
                            sz = entry.get("sz")
                            px = entry.get("limitPx") or entry.get("limit_px")
                            print(f"    Resting order oid={oid} sz={sz} px={px}")
                        except Exception:
                            continue
                    # Preserve current orders so we keep monitoring them.
                    current_orders = current_orders
                    time.sleep(POSITION_POLL_INTERVAL)
                    continue

            fills.extend(filled)
            return fills

        if not unfilled:
            # No outstanding orders but no fills observed; re-poll quickly.
            continue

        meta = _refresh_asset_meta(context)
        if reprices >= MAX_REPRICES:
            continue

        if not _should_reprice(meta, current_quote_px, is_buy, last_reprice_ts, require_rest=True):
            continue

        stability_sleep = min(STABILITY_DELAY_SECONDS, max(0.0, deadline - time.time()))
        if stability_sleep > 0:
            time.sleep(stability_sleep)
        meta_confirm = _refresh_asset_meta(context)
        if not _should_reprice(meta_confirm, current_quote_px, is_buy, last_reprice_ts, require_rest=False):
            continue

        best_side = "bid" if is_buy else "ask"
        best_px = _best_price(meta_confirm, is_buy)
        print(
            f"🔁 Repricing passive {'buy' if is_buy else 'sell'}: top {best_side} now {best_px:.6f}, "
            f"previous quote {current_quote_px:.6f}"
        )

        cancel_errors = cancel_orders(unfilled, config, context.exchange)
        if cancel_errors:
            print(f"⚠️  Cancel errors: {cancel_errors}")
        if CANCEL_COOLDOWN_SECONDS > 0:
            time.sleep(CANCEL_COOLDOWN_SECONDS)

        new_orders, new_errors = place_limit_orders(
            delta,
            config,
            context.info,
            context.asset_meta,
            context.exchange,
        )
        if new_errors:
            raise RuntimeError(f"Failed to place {'buy' if is_buy else 'sell'} order after repricing: {new_errors}")
        if not new_orders:
            raise RuntimeError("No orders placed after repricing.")

        current_orders = new_orders
        current_quote_px = new_orders[0].get("limit_px")
        last_reprice_ts = time.time()
        reprices += 1


if __name__ == "__main__":
    main()
