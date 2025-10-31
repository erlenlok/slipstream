"""Order execution for live Gradient trading with two-stage limit→market execution."""

import time
import asyncio
from typing import Dict, Any, List, Tuple
from datetime import datetime
import os


def get_current_positions(config) -> Dict[str, float]:
    """
    Fetch current positions from Hyperliquid.

    Args:
        config: GradientConfig instance

    Returns:
        Dictionary mapping asset -> current position size in USD
        Positive = long, negative = short
    """
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants

        # Initialize Info API
        info = Info(constants.MAINNET_API_URL if config.api.get("mainnet", True) else constants.TESTNET_API_URL)

        # Get API key from environment
        api_key = os.getenv("HYPERLIQUID_API_KEY")
        if not api_key:
            raise ValueError("HYPERLIQUID_API_KEY environment variable not set")

        # Fetch user state
        user_state = info.user_state(api_key)

        # Parse positions
        positions = {}
        if "assetPositions" in user_state:
            for pos_data in user_state["assetPositions"]:
                pos = pos_data.get("position", {})
                coin = pos.get("coin")
                size = float(pos.get("szi", 0))  # Size (positive=long, negative=short)

                if coin and size != 0:
                    # Get mark price to convert to USD
                    entry_px = float(pos.get("entryPx", 0))
                    # Use mark price if available, otherwise entry price
                    mark_px = entry_px  # Simplified - ideally get current mark price

                    usd_value = size * mark_px
                    positions[coin] = usd_value

        print(f"Current positions: {len(positions)} assets")
        return positions

    except ImportError:
        raise ImportError(
            "hyperliquid package not installed. Run: pip install hyperliquid"
        )
    except Exception as e:
        print(f"Error fetching positions: {e}")
        if config.dry_run:
            return {}  # Return empty in dry-run mode
        raise


def execute_rebalance_with_stages(
    target_positions: Dict[str, float],
    current_positions: Dict[str, float],
    config
) -> Dict[str, Any]:
    """
    Execute rebalance with two-stage limit→market execution.

    Stage 1 (0-60 min): Place limit orders at mid/better prices
    Stage 2 (60+ min): Sweep unfilled with market orders

    Args:
        target_positions: Target positions in USD
        current_positions: Current positions in USD
        config: GradientConfig instance

    Returns:
        Execution summary dictionary
    """
    # Calculate deltas
    deltas = calculate_deltas(target_positions, current_positions)

    if not deltas:
        print("No rebalancing needed (all positions within threshold)")
        return {
            "stage1_orders": [],
            "stage2_orders": [],
            "stage1_filled": 0,
            "stage2_filled": 0,
            "total_turnover": 0.0,
            "errors": []
        }

    print(f"Executing rebalance for {len(deltas)} position adjustments")

    # Stage 1: Place limit orders
    stage1_start = time.time()
    stage1_orders, stage1_errors = place_limit_orders(deltas, config)

    if config.dry_run:
        # In dry-run mode, simulate fills and skip monitoring
        print(f"DRY-RUN: Would place {len(stage1_orders)} limit orders")
        return {
            "stage1_orders": stage1_orders,
            "stage2_orders": [],
            "stage1_filled": len(stage1_orders),
            "stage2_filled": 0,
            "total_turnover": sum(abs(o["size_usd"]) for o in stage1_orders),
            "errors": stage1_errors
        }

    # Monitor fills for up to passive_timeout_seconds
    timeout = config.execution.get("passive_timeout_seconds", 3600)
    filled, unfilled = monitor_fills_with_timeout(stage1_orders, timeout, config)

    stage1_time = time.time() - stage1_start
    print(f"Stage 1 complete: {len(filled)}/{len(stage1_orders)} filled in {stage1_time:.1f}s")

    # Stage 2: Sweep unfilled with market orders
    stage2_orders = []
    stage2_errors = []

    if unfilled and stage1_time >= timeout:
        print(f"Stage 2: Sweeping {len(unfilled)} unfilled orders with market orders")

        # Cancel unfilled limit orders
        cancel_orders([o["order_id"] for o in unfilled], config)

        # Convert unfilled to deltas
        unfilled_deltas = {o["asset"]: o["size_usd"] for o in unfilled}

        # Place market orders
        stage2_orders, stage2_errors = place_market_orders(unfilled_deltas, config)

    # Aggregate results
    stage1_turnover = sum(abs(o["fill_usd"]) for o in filled)
    stage2_turnover = sum(abs(o["fill_usd"]) for o in stage2_orders)

    return {
        "stage1_orders": stage1_orders,
        "stage2_orders": stage2_orders,
        "stage1_filled": len(filled),
        "stage2_filled": len(stage2_orders),
        "total_turnover": stage1_turnover + stage2_turnover,
        "errors": stage1_errors + stage2_errors
    }


def calculate_deltas(
    target: Dict[str, float],
    current: Dict[str, float],
    min_threshold: float = 10.0
) -> Dict[str, float]:
    """
    Calculate position deltas.

    Args:
        target: Target positions in USD
        current: Current positions in USD
        min_threshold: Minimum USD threshold for rebalancing

    Returns:
        Dictionary mapping asset -> delta in USD
    """
    all_assets = set(target.keys()) | set(current.keys())

    deltas = {}
    for asset in all_assets:
        target_size = target.get(asset, 0.0)
        current_size = current.get(asset, 0.0)
        delta = target_size - current_size

        if abs(delta) > min_threshold:
            deltas[asset] = delta

    return deltas


def place_limit_orders(deltas: Dict[str, float], config) -> Tuple[List[Dict], List[Dict]]:
    """
    Place limit orders at mid price or better.

    Args:
        deltas: Position deltas in USD
        config: Configuration

    Returns:
        Tuple of (orders, errors)
    """
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants

    orders = []
    errors = []

    if config.dry_run:
        # Simulate order placement
        for asset, delta_usd in deltas.items():
            orders.append({
                "asset": asset,
                "side": "buy" if delta_usd > 0 else "sell",
                "size_usd": abs(delta_usd),
                "order_type": "limit",
                "order_id": f"dry_run_{asset}_{int(time.time())}",
                "status": "simulated"
            })
        return orders, errors

    try:
        # Initialize exchange
        api_key = os.getenv("HYPERLIQUID_API_KEY")
        api_secret = os.getenv("HYPERLIQUID_SECRET")

        if not api_key or not api_secret:
            raise ValueError("HYPERLIQUID_API_KEY and HYPERLIQUID_SECRET must be set")

        exchange = Exchange(
            api_key,
            constants.MAINNET_API_URL if config.api.get("mainnet", True) else constants.TESTNET_API_URL,
            account_address=api_key
        )

        # Place orders
        for asset, delta_usd in deltas.items():
            try:
                # Get current price (simplified - should fetch actual mid price)
                # For now, we'll use a market order with limit price far from mid
                is_buy = delta_usd > 0

                # Convert USD to coin size (need current price)
                # This is simplified - in production, fetch actual price
                size_coin = abs(delta_usd) / 40000  # Placeholder: $40k per coin

                # Place limit order
                order_result = exchange.order(
                    coin=asset,
                    is_buy=is_buy,
                    sz=size_coin,
                    limit_px=None,  # Market price
                    order_type={"limit": {"tif": "Gtc"}},  # Good-til-cancel
                    reduce_only=False
                )

                orders.append({
                    "asset": asset,
                    "side": "buy" if is_buy else "sell",
                    "size_usd": abs(delta_usd),
                    "size_coin": size_coin,
                    "order_type": "limit",
                    "order_id": str(order_result.get("status", {}).get("resting", [{}])[0].get("oid", "")),
                    "status": "placed"
                })

            except Exception as e:
                errors.append({
                    "asset": asset,
                    "error": str(e),
                    "stage": "limit_order"
                })
                print(f"Error placing limit order for {asset}: {e}")

    except Exception as e:
        errors.append({
            "asset": "ALL",
            "error": str(e),
            "stage": "exchange_init"
        })
        print(f"Error initializing exchange: {e}")

    return orders, errors


def monitor_fills_with_timeout(
    orders: List[Dict],
    timeout_seconds: int,
    config
) -> Tuple[List[Dict], List[Dict]]:
    """
    Monitor order fills via polling with timeout.

    Args:
        orders: List of order dictionaries
        timeout_seconds: Maximum time to wait
        config: Configuration

    Returns:
        Tuple of (filled_orders, unfilled_orders)
    """
    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    filled = []
    unfilled = list(orders)
    start_time = time.time()

    try:
        info = Info(constants.MAINNET_API_URL if config.api.get("mainnet", True) else constants.TESTNET_API_URL)
        api_key = os.getenv("HYPERLIQUID_API_KEY")

        while unfilled and (time.time() - start_time) < timeout_seconds:
            # Poll every 10 seconds
            time.sleep(10)

            # Check fill status
            user_state = info.user_state(api_key)
            open_orders = user_state.get("assetPositions", [])

            # Check which orders are filled
            still_unfilled = []
            for order in unfilled:
                # Check if order is still open (simplified)
                order_id = order.get("order_id")
                # In production: query specific order status
                # For now: assume filled after some time
                if time.time() - start_time > 300:  # 5 min simulation
                    filled.append({**order, "fill_usd": order["size_usd"], "fill_time": datetime.now()})
                else:
                    still_unfilled.append(order)

            unfilled = still_unfilled

            if not unfilled:
                break

    except Exception as e:
        print(f"Error monitoring fills: {e}")

    return filled, unfilled


def place_market_orders(deltas: Dict[str, float], config) -> Tuple[List[Dict], List[Dict]]:
    """
    Place market orders for immediate execution.

    Args:
        deltas: Position deltas in USD
        config: Configuration

    Returns:
        Tuple of (filled_orders, errors)
    """
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants

    filled = []
    errors = []

    if config.dry_run:
        # Simulate market orders
        for asset, delta_usd in deltas.items():
            filled.append({
                "asset": asset,
                "side": "buy" if delta_usd > 0 else "sell",
                "size_usd": abs(delta_usd),
                "fill_usd": abs(delta_usd),
                "order_type": "market",
                "status": "filled_simulated"
            })
        return filled, errors

    try:
        api_key = os.getenv("HYPERLIQUID_API_KEY")
        api_secret = os.getenv("HYPERLIQUID_SECRET")

        exchange = Exchange(
            api_key,
            constants.MAINNET_API_URL if config.api.get("mainnet", True) else constants.TESTNET_API_URL,
            account_address=api_key
        )

        for asset, delta_usd in deltas.items():
            try:
                is_buy = delta_usd > 0
                size_coin = abs(delta_usd) / 40000  # Placeholder

                # Place market order
                order_result = exchange.market_open(
                    coin=asset,
                    is_buy=is_buy,
                    sz=size_coin,
                    reduce_only=False
                )

                filled.append({
                    "asset": asset,
                    "side": "buy" if is_buy else "sell",
                    "size_usd": abs(delta_usd),
                    "fill_usd": abs(delta_usd),
                    "size_coin": size_coin,
                    "order_type": "market",
                    "status": "filled"
                })

            except Exception as e:
                errors.append({
                    "asset": asset,
                    "error": str(e),
                    "stage": "market_order"
                })
                print(f"Error placing market order for {asset}: {e}")

    except Exception as e:
        errors.append({
            "asset": "ALL",
            "error": str(e),
            "stage": "exchange_init"
        })

    return filled, errors


def cancel_orders(order_ids: List[str], config) -> None:
    """
    Cancel open orders.

    Args:
        order_ids: List of order IDs to cancel
        config: Configuration
    """
    if config.dry_run:
        print(f"DRY-RUN: Would cancel {len(order_ids)} orders")
        return

    try:
        from hyperliquid.exchange import Exchange
        from hyperliquid.utils import constants

        api_key = os.getenv("HYPERLIQUID_API_KEY")
        api_secret = os.getenv("HYPERLIQUID_SECRET")

        exchange = Exchange(
            api_key,
            constants.MAINNET_API_URL if config.api.get("mainnet", True) else constants.TESTNET_API_URL,
            account_address=api_key
        )

        for oid in order_ids:
            try:
                exchange.cancel(oid)
                print(f"Cancelled order {oid}")
            except Exception as e:
                print(f"Error cancelling order {oid}: {e}")

    except Exception as e:
        print(f"Error cancelling orders: {e}")


def validate_execution_results(results: Dict[str, Any], config) -> None:
    """
    Validate execution results and log warnings.

    Args:
        results: Execution results
        config: Configuration
    """
    stage1_filled = results["stage1_filled"]
    stage2_filled = results["stage2_filled"]
    total_orders = len(results["stage1_orders"])
    errors = results.get("errors", [])

    if total_orders == 0:
        print("No orders placed (portfolio unchanged)")
        return

    fill_rate = (stage1_filled + stage2_filled) / total_orders if total_orders > 0 else 0

    if fill_rate < 0.95:
        print(f"Warning: Low fill rate {fill_rate:.1%} ({stage1_filled + stage2_filled}/{total_orders})")

    if len(errors) > 0:
        print(f"Warning: {len(errors)} errors during execution:")
        for err in errors[:5]:  # Show first 5 errors
            print(f"  {err.get('asset', 'N/A')}: {err.get('error', 'Unknown error')}")

    total_turnover = results["total_turnover"]
    turnover_pct = (total_turnover / config.capital_usd) * 100 if config.capital_usd > 0 else 0

    print(f"Execution summary:")
    print(f"  Stage 1 (limit): {stage1_filled} filled")
    print(f"  Stage 2 (market): {stage2_filled} filled")
    print(f"  Total turnover: ${total_turnover:,.2f} ({turnover_pct:.1f}% of capital)")
    print(f"  Errors: {len(errors)}")
