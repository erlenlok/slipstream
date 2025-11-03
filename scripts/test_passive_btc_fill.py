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
- Run from the repo root: `python scripts/test_passive_btc_fill.py`.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Any, List, Tuple

from slipstream.gradient.live.config import load_config, validate_config
from slipstream.gradient.live.execution import (
    _prepare_hyperliquid_context,
    _calculate_slippage_bps,
    place_limit_orders,
    monitor_fills_with_timeout,
    cancel_orders,
    get_current_positions,
)

USD_NOTIONAL = 10.0
ASSET = "BTC"
DEFAULT_TIMEOUT = 300  # seconds


def _format_order(order: Dict[str, Any]) -> str:
    return (
        f"{order.get('asset')} {order.get('side')} "
        f"${order.get('fill_usd', order.get('size_usd', 0.0)):.2f} "
        f"@ {order.get('execution_px', order.get('limit_px'))}"
    )


def _passive_round_trip(
    context,
    config,
    usd_notional: float,
    timeout: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    buy_delta = {ASSET: usd_notional}
    print(f"➡️  Submitting passive buy for {ASSET} (${usd_notional:.2f})")
    buy_orders, buy_errors = place_limit_orders(
        buy_delta,
        config,
        context.info,
        context.asset_meta,
        context.exchange,
    )

    if buy_errors:
        raise RuntimeError(f"Failed to place buy order: {buy_errors}")
    if not buy_orders:
        raise RuntimeError("No buy orders placed.")

    filled_buy, unfilled_buy = monitor_fills_with_timeout(
        buy_orders, timeout, config, context.info
    )

    if unfilled_buy:
        print(f"⚠️  Buy order not filled after {timeout}s, cancelling.")
        cancel_orders(unfilled_buy, config, context.exchange)
        raise RuntimeError("Buy leg did not fill.")

    filled_notional = sum(abs(order.get("fill_usd", 0.0)) for order in filled_buy)
    print(f"✅  Buy filled: {', '.join(_format_order(o) for o in filled_buy)}")

    # Refresh positions to get the precise USD delta for the exit leg.
    positions_after_buy = get_current_positions(config)
    current_btc = positions_after_buy.get(ASSET, 0.0)
    if current_btc <= 0:
        raise RuntimeError(
            f"Unexpected BTC position after buy: {current_btc}. Aborting sell leg."
        )

    sell_delta = {ASSET: -current_btc}
    print(f"➡️  Submitting passive sell for {ASSET} (${current_btc:.2f})")
    sell_orders, sell_errors = place_limit_orders(
        sell_delta,
        config,
        context.info,
        context.asset_meta,
        context.exchange,
    )

    if sell_errors:
        raise RuntimeError(f"Failed to place sell order: {sell_errors}")
    if not sell_orders:
        raise RuntimeError("No sell orders placed.")

    filled_sell, unfilled_sell = monitor_fills_with_timeout(
        sell_orders, timeout, config, context.info
    )

    if unfilled_sell:
        print(f"⚠️  Sell order not filled after {timeout}s, cancelling.")
        cancel_orders(unfilled_sell, config, context.exchange)
        raise RuntimeError("Sell leg did not fill.")

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

    entry_fills, exit_fills = _passive_round_trip(
        context,
        config,
        effective_usd,
        args.timeout,
    )

    _summarize_fills("Entry", entry_fills)

    # Give the chain a moment before checking the final position snapshot.
    time.sleep(3)
    final_positions = get_current_positions(config)
    final_btc = final_positions.get(ASSET, 0.0)
    _summarize_fills("Exit", exit_fills)

    print(f"Final BTC exposure: ${final_btc:.2f}")
    if abs(final_btc) > 1.0:
        print("⚠️  Residual BTC exposure detected! Consider flattening manually.")


if __name__ == "__main__":
    main()
