"""Order execution for live Gradient trading with two-stage limit→market execution."""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Iterable


@dataclass
class AssetMeta:
    """Per-asset metadata required for sizing and pricing orders."""

    name: str
    sz_decimals: int
    mid_px: Optional[float]
    impact_bid: Optional[float]
    impact_ask: Optional[float]


@dataclass
class HyperliquidContext:
    """Container for Hyperliquid clients and metadata used during execution."""

    info: Any
    exchange: Optional[Any]
    asset_meta: Dict[str, AssetMeta]
    account_address: Optional[str] = None
    meta_timestamp: Dict[str, float] = field(default_factory=dict)


def _load_hyperliquid_modules():
    """Lazy-import Hyperliquid SDK components and return them."""
    try:
        from hyperliquid.info import Info  # type: ignore
        from hyperliquid.exchange import Exchange  # type: ignore
        from hyperliquid.utils import constants  # type: ignore
        from eth_account import Account  # type: ignore
    except ImportError as exc:  # pragma: no cover - surfaced to caller
        raise ImportError(
            "hyperliquid package not installed. Run: pip install hyperliquid"
        ) from exc

    return Info, Exchange, constants, Account


def _resolve_base_url(config, constants_module) -> str:
    """Resolve which Hyperliquid base URL to talk to."""
    if getattr(config, "api_endpoint", None):
        return config.api_endpoint.rstrip("/")

    api_cfg = getattr(config, "api", {}) or {}
    explicit = api_cfg.get("endpoint")
    if explicit:
        return explicit.rstrip("/")

    mainnet = api_cfg.get("mainnet", getattr(config, "mainnet", True))
    return (
        constants_module.MAINNET_API_URL
        if mainnet
        else constants_module.TESTNET_API_URL
    )


def _parse_float(value: Any) -> Optional[float]:
    """Safely convert API strings/numbers into floats."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        # Remove whitespace first (e.g. "110 650")
        cleaned = cleaned.replace(" ", "")
        # If only commas are present, treat them as decimal separators.
        if cleaned.count(",") and cleaned.count(".") == 0:
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_down(value: float, decimals: int) -> float:
    """Round a float downwards to the supported decimal precision."""
    if decimals <= 0:
        return math.floor(value)
    factor = 10**decimals
    return math.floor(value * factor) / factor


def _round_price(price: float, sz_decimals: int = 0) -> float:
    """
    Round price following Hyperliquid's official formula.

    Hyperliquid uses:
    1. 5 significant figures (via .5g format)
    2. Then rounds to (6 - sz_decimals) decimal places for perps

    This ensures prices respect the exchange's tick size requirements.

    Args:
        price: Price to round
        sz_decimals: Asset's szDecimals from metadata (default 0)

    Returns:
        Properly rounded price that respects Hyperliquid tick sizes

    Examples:
        BTC (sz_decimals=5): 6-5=1 decimal → $107485.0
        ETH (sz_decimals=4): 6-4=2 decimals → $3709.60
        PENDLE (sz_decimals=0): 6-0=6 decimals → $2.798800
    """
    # Round to 5 significant figures first, then to proper decimal places
    decimal_places = 6 - sz_decimals
    return round(float(f"{price:.5g}"), decimal_places)


def _price_tick(meta: AssetMeta) -> float:
    """Return the minimum price increment implied by sz_decimals."""
    price_decimals = max(0, 6 - int(meta.sz_decimals))
    return 10 ** (-price_decimals) if price_decimals > 0 else 1.0


def _ensure_passive_price(price: float, meta: AssetMeta, is_buy: bool) -> float:
    """Adjust price so that rounding keeps the order on the passive side."""
    tick_size = _price_tick(meta)
    bid = meta.impact_bid if meta.impact_bid and meta.impact_bid > 0 else None
    ask = meta.impact_ask if meta.impact_ask and meta.impact_ask > 0 else None

    candidate = max(price, tick_size)

    for _ in range(50):
        rounded = _round_price(candidate, meta.sz_decimals)
        if rounded <= 0:
            return tick_size

        if is_buy:
            if ask and rounded >= ask:
                candidate -= tick_size
                if candidate <= 0:
                    return tick_size
                continue
            return rounded

        if bid and rounded <= bid:
            candidate += tick_size
            continue
        return rounded

    # Fallback: return best-effort rounded price after iterations
    return _round_price(max(price, tick_size), meta.sz_decimals)


def _resolve_main_wallet(config) -> str:
    """
    Resolve the main wallet address where balances and resting orders live.

    Prefers environment variable override, falls back to config.api.main_wallet.
    """
    main_wallet = os.getenv("HYPERLIQUID_MAIN_WALLET")
    if not main_wallet:
        api_cfg = getattr(config, "api", {}) or {}
        main_wallet = api_cfg.get("main_wallet")
    if not main_wallet:
        raise ValueError(
            "HYPERLIQUID_MAIN_WALLET environment variable (or api.main_wallet in config) must be set "
            "to fetch positions and open orders; this should point at the MAIN wallet, not the API vault."
        )
    return main_wallet


def _calculate_slippage_bps(
    execution_px: Optional[float],
    mid_px: Optional[float],
    is_buy: bool,
) -> Optional[float]:
    """Return slippage in bps versus the prevailing mid price."""
    if not execution_px or execution_px <= 0:
        return None
    if not mid_px or mid_px <= 0:
        return None

    if is_buy:
        return ((execution_px - mid_px) / mid_px) * 1e4
    return ((mid_px - execution_px) / mid_px) * 1e4


# Passive order management heuristics
PASSIVE_MONITOR_SLICE_SECONDS = 20
PASSIVE_MIN_REST_BEFORE_REPRICE = 6
PASSIVE_REPRICE_TICK_THRESHOLD = 1
PASSIVE_REPRICE_MIN_SPREAD_BPS = 0.5
PASSIVE_STABILITY_DELAY_SECONDS = 1.0
PASSIVE_MAX_REPRICES = 6
PASSIVE_CANCEL_COOLDOWN_SECONDS = 0.5
PASSIVE_META_REFRESH_SECONDS = 15.0
PASSIVE_STALE_TICK_MULTIPLIER = 3
PASSIVE_STALE_BPS_THRESHOLD = 3.0  # Require quote to be within 3 bps unless max reprices exceeded


def _refresh_asset_meta_for_assets(context, assets: Optional[Iterable[str]] = None) -> Dict[str, AssetMeta]:
    """Refresh asset metadata and optionally return a subset.

    To avoid hammering the API, only refresh if the cached snapshot is stale.
    """
    now = time.time()

    if assets is not None:
        assets = list(assets)
        stale_assets = [
            asset
            for asset in assets
            if context.meta_timestamp.get(asset, 0.0) + PASSIVE_META_REFRESH_SECONDS <= now
        ]
        if not stale_assets:
            return {asset: context.asset_meta.get(asset) for asset in assets}
    else:
        stale_assets = None
        if (
            context.meta_timestamp
            and all(now - ts <= PASSIVE_META_REFRESH_SECONDS for ts in context.meta_timestamp.values())
        ):
            return context.asset_meta

    meta_payload, asset_ctxs = _fetch_meta_and_asset_ctxs(context.info)
    refreshed = _build_asset_meta(context.info, meta_payload, asset_ctxs)
    context.asset_meta.update(refreshed)
    refreshed_ts = time.time()
    context.meta_timestamp.update({name: refreshed_ts for name in refreshed})

    if assets is None:
        return context.asset_meta
    return {asset: context.asset_meta.get(asset) for asset in assets}


def _best_price(meta: AssetMeta, is_buy: bool) -> Optional[float]:
    return meta.impact_bid if is_buy else meta.impact_ask


def _spread_bps_from_meta(meta: AssetMeta) -> Optional[float]:
    bid = meta.impact_bid
    ask = meta.impact_ask
    if not bid or not ask or bid <= 0 or ask <= bid:
        return None
    mid = meta.mid_px or (bid + ask) / 2.0
    if not mid or mid <= 0:
        return None
    return ((ask - bid) / mid) * 1e4


def _should_reprice_passive(
    meta: Optional[AssetMeta],
    current_price: Optional[float],
    is_buy: bool,
    last_reprice_ts: float,
    require_rest: bool = True,
) -> bool:
    if meta is None or current_price is None or current_price <= 0:
        return False
    now = time.time()
    if require_rest and (now - last_reprice_ts) < PASSIVE_MIN_REST_BEFORE_REPRICE:
        return False
    best_px = _best_price(meta, is_buy)
    if not best_px or best_px <= 0:
        return False
    tick = max(_price_tick(meta), 1e-8)
    guard = max(tick * PASSIVE_REPRICE_TICK_THRESHOLD, current_price * 1e-4 * 0.5)
    if is_buy:
        if best_px <= current_price + guard:
            return False
    else:
        if best_px >= current_price - guard:
            return False
    spread = _spread_bps_from_meta(meta)
    if spread is not None and spread < PASSIVE_REPRICE_MIN_SPREAD_BPS:
        return False
    return True


def _quote_is_stale(
    meta: Optional[AssetMeta],
    current_price: Optional[float],
    is_buy: bool,
) -> bool:
    if meta is None or current_price is None or current_price <= 0:
        return False
    best_px = _best_price(meta, is_buy)
    if not best_px or best_px <= 0:
        return False
    tick = max(_price_tick(meta), 1e-8)
    tick_threshold = tick * PASSIVE_STALE_TICK_MULTIPLIER
    bps_threshold = current_price * PASSIVE_STALE_BPS_THRESHOLD * 1e-4
    threshold = max(tick_threshold, bps_threshold)
    if is_buy:
        return best_px - current_price > threshold
    return current_price - best_px > threshold


def _aggregate_slippage_metrics(orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate slippage metrics for a collection of filled orders."""
    total_usd = 0.0
    weighted_sum = 0.0
    count = 0
    avg_sum = 0.0

    for order in orders:
        slippage = order.get("slippage_bps")
        notional = abs(order.get("fill_usd") or order.get("size_usd") or 0.0)
        if slippage is None or notional <= 0:
            continue
        total_usd += notional
        weighted_sum += slippage * notional
        avg_sum += slippage
        count += 1

    weighted_bps = weighted_sum / total_usd if total_usd > 0 else None
    avg_bps = avg_sum / count if count > 0 else None

    return {
        "count": count,
        "total_usd": total_usd,
        "weighted_bps": weighted_bps,
        "avg_bps": avg_bps,
    }


def _combine_slippage_stats(stats_a: Dict[str, Any], stats_b: Dict[str, Any]) -> Dict[str, Any]:
    """Combine two slippage stats dictionaries."""
    total_usd = (stats_a.get("total_usd", 0.0) or 0.0) + (stats_b.get("total_usd", 0.0) or 0.0)
    count = (stats_a.get("count", 0) or 0) + (stats_b.get("count", 0) or 0)

    weighted_sum = 0.0
    if stats_a.get("weighted_bps") is not None and stats_a.get("total_usd", 0) > 0:
        weighted_sum += stats_a["weighted_bps"] * stats_a["total_usd"]
    if stats_b.get("weighted_bps") is not None and stats_b.get("total_usd", 0) > 0:
        weighted_sum += stats_b["weighted_bps"] * stats_b["total_usd"]

    avg_sum = 0.0
    if stats_a.get("avg_bps") is not None and stats_a.get("count", 0) > 0:
        avg_sum += stats_a["avg_bps"] * stats_a["count"]
    if stats_b.get("avg_bps") is not None and stats_b.get("count", 0) > 0:
        avg_sum += stats_b["avg_bps"] * stats_b["count"]

    weighted_bps = weighted_sum / total_usd if total_usd > 0 else None
    avg_bps = avg_sum / count if count > 0 else None

    return {
        "count": count,
        "total_usd": total_usd,
        "weighted_bps": weighted_bps,
        "avg_bps": avg_bps,
    }


def _enrich_orders_with_actual_fills(
    info_client,
    account_address: Optional[str],
    filled_stage1: List[Dict[str, Any]],
    stage2_orders: List[Dict[str, Any]],
    start_timestamp: float,
) -> None:
    """Fetch actual fills and update order records with executed pricing."""
    if not account_address:
        return

    try:
        start_ms = int(max(0, (start_timestamp - 120) * 1000))  # include 2-minute buffer
        end_ms = int(time.time() * 1000) + 1000
        fills = info_client.user_fills_by_time(
            account_address,
            start_ms,
            end_ms,
            aggregate_by_time=True,
        )
    except Exception as exc:
        print(f"Warning: Failed to fetch user fills for slippage enrichment: {exc}")
        return

    if not isinstance(fills, list):
        return

    order_lookup: Dict[str, Dict[str, Any]] = {}
    fill_accum: Dict[str, Dict[str, Any]] = {}

    for order in filled_stage1 + stage2_orders:
        oid = order.get("order_id")
        if oid is None:
            continue
        order_lookup[str(oid)] = order

    for entry in fills:
        oid = entry.get("oid")
        if oid is None:
            continue
        order = order_lookup.get(str(oid))
        if not order:
            continue

        size = _parse_float(entry.get("sz"))
        price = _parse_float(entry.get("px"))
        timestamp = entry.get("time")

        if size is None or price is None or abs(size) <= 0 or price <= 0:
            continue

        record = fill_accum.setdefault(
            str(oid),
            {"total_sz": 0.0, "weighted_px": 0.0, "latest_time": 0},
        )
        abs_size = abs(size)
        record["total_sz"] += abs_size
        record["weighted_px"] += price * abs_size
        if isinstance(timestamp, (int, float)):
            record["latest_time"] = max(record["latest_time"], int(timestamp))

    for oid, accum in fill_accum.items():
        order = order_lookup.get(oid)
        if not order:
            continue

        total_sz = accum["total_sz"]
        if total_sz <= 0:
            continue

        avg_px = accum["weighted_px"] / total_sz
        order["execution_px"] = avg_px
        order["fill_price"] = avg_px
        order["size_coin"] = total_sz
        order["fill_usd"] = avg_px * total_sz
        order["size_usd"] = order.get("size_usd") or order["fill_usd"]

        reference_mid = order.get("reference_mid")
        is_buy = order.get("side") == "buy"
        order["slippage_bps"] = _calculate_slippage_bps(avg_px, reference_mid, is_buy)

        if accum["latest_time"]:
            order["fill_time"] = datetime.fromtimestamp(accum["latest_time"] / 1000.0)


def _fetch_meta_and_asset_ctxs(info_client) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Fetch universe metadata plus per-asset context (mid, impact prices, etc.)."""
    response = info_client.meta_and_asset_ctxs()

    if isinstance(response, list) and len(response) >= 2:
        meta_payload = response[0]
        asset_ctxs = response[1]
    elif isinstance(response, dict):
        meta_payload = response
        asset_ctxs = response.get("assetCtxs", [])
    else:
        raise RuntimeError("Unexpected response format from metaAndAssetCtxs")

    if isinstance(asset_ctxs, dict) and "assetCtxs" in asset_ctxs:
        asset_ctxs = asset_ctxs["assetCtxs"]

    if not isinstance(asset_ctxs, list):
        raise RuntimeError("Unexpected asset context payload from Hyperliquid API")

    return meta_payload, asset_ctxs


def _build_asset_meta(
    info_client,
    meta_payload: Dict[str, Any],
    asset_ctxs: List[Dict[str, Any]],
) -> Dict[str, AssetMeta]:
    """Combine universe metadata + asset contexts into a quick-lookup map."""
    universe = meta_payload.get("universe", [])
    asset_meta: Dict[str, AssetMeta] = {}

    for idx, entry in enumerate(universe):
        name = entry.get("name")
        if not name:
            continue

        ctx = asset_ctxs[idx] if idx < len(asset_ctxs) else {}
        impact_raw = ctx.get("impactPxs", [])
        bid_px = None
        ask_px = None
        if isinstance(impact_raw, (list, tuple)):
            bid_px = _parse_float(impact_raw[0]) if len(impact_raw) > 0 else None
            ask_px = _parse_float(impact_raw[1]) if len(impact_raw) > 1 else None

        asset_meta[name] = AssetMeta(
            name=name,
            sz_decimals=int(entry.get("szDecimals", 6)),
            mid_px=_parse_float(
                ctx.get("markPx") or ctx.get("midPx") or ctx.get("oraclePx")
            ),
            impact_bid=bid_px,
            impact_ask=ask_px,
        )

    # Fill missing mids using all_mids() fallback if required.
    missing = [meta for meta in asset_meta.values() if not meta.mid_px]
    if missing:
        mids = info_client.all_mids()
        for meta in missing:
            coin_idx = info_client.name_to_coin.get(meta.name)
            if coin_idx is not None and coin_idx < len(mids):
                meta.mid_px = _parse_float(mids[coin_idx])

    return asset_meta


def _prepare_hyperliquid_context(config) -> HyperliquidContext:
    """Instantiate Info/Exchange clients and fetch per-asset metadata."""
    Info, Exchange, constants, Account = _load_hyperliquid_modules()
    base_url = _resolve_base_url(config, constants)

    info_client = Info(base_url)
    meta_payload, asset_ctxs = _fetch_meta_and_asset_ctxs(info_client)
    asset_meta = _build_asset_meta(info_client, meta_payload, asset_ctxs)

    exchange_client = None
    account_address: Optional[str] = None
    if not config.dry_run:
        api_secret = os.getenv("HYPERLIQUID_API_SECRET") or config.api_secret
        api_key = os.getenv("HYPERLIQUID_API_KEY") or config.api_key
        if not api_secret:
            raise ValueError(
                "HYPERLIQUID_API_SECRET must be set when dry_run is false"
            )
        if not api_key:
            raise ValueError(
                "HYPERLIQUID_API_KEY must be set when dry_run is false"
            )

        wallet = Account.from_key(api_secret)
        if wallet.address.lower() != api_key.lower():
            raise ValueError(
                "HYPERLIQUID_API_KEY does not match address derived from secret"
            )

        exchange_client = Exchange(
            wallet=wallet,
            base_url=base_url,
            meta=meta_payload,
            account_address=wallet.address,
        )
        account_address = wallet.address

    now = time.time()

    return HyperliquidContext(
        info=info_client,
        exchange=exchange_client,
        asset_meta=asset_meta,
        account_address=account_address,
        meta_timestamp={name: now for name in asset_meta},
    )


def _fetch_clearinghouse_state(base_url: str, wallet_address: str, dex: Optional[str] = None) -> Dict[str, Any]:
    """Fetch clearinghouse state via raw HTTP POST (SDK doesn't expose this method).

    Args:
        base_url: API base URL
        wallet_address: Main wallet address
        dex: Optional perp DEX name

    Returns:
        Clearinghouse state dict with assetPositions, marginSummary, etc.
    """
    import requests

    body: Dict[str, Any] = {
        "type": "clearinghouseState",
        "user": wallet_address
    }
    if dex:
        body["dex"] = dex

    response = requests.post(f"{base_url}/info", json=body, timeout=10)
    response.raise_for_status()
    return response.json()


def _collect_user_states(base_url: str, wallet_address: str) -> List[Dict[str, Any]]:
    """Fetch clearinghouse state for all relevant perp dexes.

    Args:
        base_url: API base URL
        wallet_address: Main wallet address where positions live (not API key vault)
    """
    states: List[Dict[str, Any]] = []
    # Always include default dex (empty string)
    try:
        # Use raw HTTP to fetch clearinghouse state from main wallet
        default_state = _fetch_clearinghouse_state(base_url, wallet_address)
        states.append(default_state)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise RuntimeError(f"Failed to fetch clearinghouse_state for primary dex: {exc}") from exc

    # Note: Multi-dex support omitted for now since we only need main DEX
    # Could add perp_dexs() query here if needed in the future

    return states


def _walk_position_nodes(node: Any, seen: set[int]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Depth-first search yielding (container, position) pairs."""
    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    stack: List[Any] = [node]

    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        if isinstance(current, dict):
            pos = current.get("position")
            if isinstance(pos, dict):
                pos_id = id(pos)
                if pos_id not in seen:
                    seen.add(pos_id)
                    results.append((current, pos))
            if {"coin", "szi"} <= current.keys():
                results.append((current, current))  # Dict already represents a position

            for value in current.values():
                if isinstance(value, (dict, list, tuple)):
                    stack.append(value)

        elif isinstance(current, (list, tuple)):
            stack.extend(current)

    return results


def _iter_position_entries(user_state: Dict[str, Any]):
    """Iterate over all position dictionaries within a user_state payload."""
    seen: set[int] = set()
    keys_to_scan = [
        "assetPositions",
        "assetPositionsPerDex",
        "portfolioAssetPositions",
        "crossAssetPositions",
        "positions",
    ]

    any_found = False
    for key in keys_to_scan:
        container = user_state.get(key)
        if container is None:
            continue
        any_found = True
        for holder, position in _walk_position_nodes(container, seen):
            yield holder, position

    if not any_found:
        for holder, position in _walk_position_nodes(user_state, seen):
            yield holder, position


def _compute_limit_price(
    meta: AssetMeta,
    is_buy: bool,
    aggression: str,
) -> float:
    """Compute limit price based on aggression setting and available meta."""
    if not meta.mid_px or meta.mid_px <= 0:
        raise ValueError(f"Missing mid price for {meta.name}")

    aggression = (aggression or "join_best").lower()
    offset_bps = 0.0001  # 1 bp guard used when we need to fallback without book data

    bid = meta.impact_bid if meta.impact_bid and meta.impact_bid > 0 else None
    ask = meta.impact_ask if meta.impact_ask and meta.impact_ask > 0 else None
    tick_size = _price_tick(meta)

    if aggression == "join_best":
        price = meta.mid_px

        if is_buy:
            if bid:
                price = bid
            elif ask:
                price = max(ask - tick_size, tick_size)
            else:
                price = meta.mid_px * (1 - offset_bps)
        else:
            if ask:
                price = ask
            elif bid:
                price = bid + tick_size
            else:
                price = meta.mid_px * (1 + offset_bps)

        return _ensure_passive_price(price, meta, is_buy)

    if aggression == "mid":
        return _round_price(meta.mid_px, meta.sz_decimals)

    if aggression == "aggressive":
        candidate = meta.impact_ask if is_buy else meta.impact_bid
        if candidate and candidate > 0:
            return _round_price(candidate, meta.sz_decimals)
        factor = 1 + offset_bps if is_buy else 1 - offset_bps
        return _round_price(meta.mid_px * factor, meta.sz_decimals)

    # Fallback: treat as mid if unknown aggression keyword supplied.
    return _round_price(meta.mid_px, meta.sz_decimals)


def _compute_size_coin(delta_usd: float, meta: AssetMeta) -> Tuple[float, float]:
    """Convert a USD delta into coin size respecting asset precision."""
    if not meta.mid_px or meta.mid_px <= 0:
        raise ValueError(f"Missing price for {meta.name}")

    raw_size = abs(delta_usd) / meta.mid_px
    size_coin = _round_down(raw_size, meta.sz_decimals)
    return size_coin, meta.mid_px


def _extract_order_id(response: Any) -> str:
    """Best-effort extraction of order id from Hyperliquid responses."""
    if isinstance(response, dict):
        status = response.get("status")
        if status == "err":
            raise RuntimeError(response.get("error", "Unknown order error"))
        if "error" in response and status is None:
            raise RuntimeError(response["error"])

        # Some responses nest inside "response" then "data" → "statuses" → ...
        for key in (
            "response",
            "data",
            "resting",
            "filled",
            "orders",
            "statuses",
            "status",
        ):
            if key in response:
                try:
                    return _extract_order_id(response[key])
                except RuntimeError:
                    raise
                except Exception:
                    continue

        oid = response.get("oid")
        if oid is not None:
            return str(oid)

    elif isinstance(response, list):
        for item in response:
            try:
                oid = _extract_order_id(item)
            except RuntimeError:
                raise
            if oid:
                return oid

    elif isinstance(response, (int, float)):
        return str(response)

    return ""


def _collect_open_order_ids(info_client, account_address: str) -> List[str]:
    """Fetch the current set of open order ids for an account."""
    try:
        open_orders = info_client.frontend_open_orders(account_address)
    except Exception:
        return []

    if isinstance(open_orders, dict) and "data" in open_orders:
        open_orders = open_orders["data"]

    order_ids: List[str] = []
    if isinstance(open_orders, list):
        for entry in open_orders:
            if isinstance(entry, dict):
                oid = entry.get("oid")
                if oid is not None:
                    order_ids.append(str(oid))

            # Some responses store child orders inside "children"
            children = entry.get("children") if isinstance(entry, dict) else None
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        child_oid = child.get("oid")
                        if child_oid is not None:
                            order_ids.append(str(child_oid))

    return order_ids


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
        Info, _, constants, _ = _load_hyperliquid_modules()
    except ImportError:
        raise ImportError(
            "hyperliquid package not installed. Run: pip install hyperliquid"
        )

    base_url = _resolve_base_url(config, constants)

    # Get main wallet address (where positions live)
    # On Hyperliquid: API key is a vault that trades FOR the main wallet
    # Positions live on the main wallet, not the API key vault
    main_wallet = _resolve_main_wallet(config)

    user_states = _collect_user_states(base_url, main_wallet)
    positions: Dict[str, float] = {}
    dedupe_keys: set[Tuple[str, float, float]] = set()

    for state in user_states:
        if not isinstance(state, dict):
            continue

        for container, pos in _iter_position_entries(state):
            coin = pos.get("coin")
            size = _parse_float(pos.get("szi"))

            if not coin or size is None or abs(size) < 1e-10:
                continue

            entry_px = _parse_float(pos.get("entryPx"))
            mark_px = (
                _parse_float(pos.get("markPx"))
                or _parse_float(container.get("markPx") if isinstance(container, dict) else None)
            )
            position_value = _parse_float(pos.get("positionValue"))
            if position_value is None and isinstance(container, dict):
                position_value = _parse_float(container.get("positionValue"))
            if position_value is None:
                fallback_px = mark_px or entry_px
                if fallback_px is None:
                    continue
                position_value = abs(size) * fallback_px

            key = (
                coin,
                round(size, 12),
                round(entry_px if entry_px is not None else 0.0, 6),
            )
            if key in dedupe_keys:
                continue
            dedupe_keys.add(key)

            usd_value = position_value if size > 0 else -position_value
            positions[coin] = positions.get(coin, 0.0) + usd_value

    print(f"Current positions: {len(positions)} assets across {len(user_states)} dex snapshots")
    return positions


def execute_rebalance_with_stages(
    target_positions: Dict[str, float],
    current_positions: Dict[str, float],
    config,
) -> Dict[str, Any]:
    """
    Execute rebalance with two-stage limit→market execution.

    Stage 1 (0-60 min): Place limit orders at mid/better prices
    Stage 2 (60+ min): Sweep unfilled with market orders
    """
    min_threshold = config.execution.get("min_order_size_usd", 2.0)
    deltas = calculate_deltas(target_positions, current_positions, min_threshold)

    if not deltas:
        print("No rebalancing needed (all positions within threshold)")
        return {
            "stage1_orders": [],
            "stage2_orders": [],
            "stage1_filled": 0,
            "stage2_filled": 0,
            "total_turnover": 0.0,
            "errors": [],
        }

    print(f"Executing rebalance for {len(deltas)} position adjustments")
    sample = dict(list(deltas.items())[:5])
    print(f"Sample deltas (USD): {sample}")

    context = _prepare_hyperliquid_context(config)
    min_order_usd = config.execution.get("min_order_size_usd", 2.0)
    stage1_start = time.time()

    stage1_orders, stage1_errors = place_limit_orders(
        deltas,
        config,
        context.info,
        context.asset_meta,
        context.exchange,
    )

    if config.dry_run:
        print("DRY-RUN: Skipping live monitoring and market sweep")
        return {
            "stage1_orders": stage1_orders,
            "stage2_orders": [],
            "stage1_filled": len(stage1_orders),
            "stage2_filled": 0,
            "total_turnover": sum(abs(o["size_usd"]) for o in stage1_orders),
            "errors": stage1_errors,
        }

    timeout = config.execution.get("passive_timeout_seconds", 3600)
    deadline = stage1_start + timeout
    monitor_window = int(
        config.execution.get("passive_monitor_slice_seconds", PASSIVE_MONITOR_SLICE_SECONDS)
    )
    monitor_window = max(5, monitor_window)

    all_stage1_orders: List[Dict[str, Any]] = list(stage1_orders)
    filled: List[Dict[str, Any]] = []
    outstanding_orders: List[Dict[str, Any]] = list(stage1_orders)
    filled_usd_by_asset: Dict[str, float] = defaultdict(float)
    target_usd_by_asset: Dict[str, float] = {asset: abs(size) for asset, size in deltas.items()}
    side_by_asset: Dict[str, bool] = {asset: (size > 0) for asset, size in deltas.items()}
    last_reprice_ts: Dict[str, float] = defaultdict(lambda: stage1_start)
    reprice_counts: Dict[str, int] = defaultdict(int)

    while outstanding_orders and time.time() < deadline:
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            break
        window = int(min(max(5, monitor_window), max(5, remaining_time)))

        window_filled, window_outstanding = monitor_fills_with_timeout(
            outstanding_orders,
            window,
            config,
            context.info,
        )

        for order in window_filled:
            filled.append(order)
            fill_usd = order.get("fill_usd")
            if fill_usd is None:
                fill_usd = order.get("size_usd", 0.0)
            filled_usd_by_asset[order.get("asset")] += abs(fill_usd or 0.0)

        if not window_outstanding:
            outstanding_orders = []
            break

        outstanding_by_asset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for order in window_outstanding:
            outstanding_by_asset[order.get("asset")].append(order)

        assets_to_consider = [asset for asset, orders in outstanding_by_asset.items() if orders]
        if assets_to_consider:
            _refresh_asset_meta_for_assets(context, assets_to_consider)

        candidate_assets: List[str] = []
        for asset, orders in outstanding_by_asset.items():
            if not orders:
                continue
            meta = context.asset_meta.get(asset)
            current_price = orders[0].get("limit_px")
            is_buy = side_by_asset.get(asset, True)
            total_target = target_usd_by_asset.get(asset, 0.0)
            remaining_target = max(total_target - filled_usd_by_asset.get(asset, 0.0), 0.0)
            min_order_usd = config.execution.get("min_order_size_usd", 2.0)
            if remaining_target < max(min_order_usd * 0.9, 1.0):
                continue
            reached_limit = reprice_counts[asset] >= PASSIVE_MAX_REPRICES
            stale_quote = _quote_is_stale(meta, current_price, is_buy)
            if reached_limit and not stale_quote:
                continue
            if _should_reprice_passive(meta, current_price, is_buy, last_reprice_ts[asset], require_rest=True):
                candidate_assets.append(asset)

        confirmed_assets: List[str] = []
        if candidate_assets:
            stability_sleep = min(PASSIVE_STABILITY_DELAY_SECONDS, max(0.0, deadline - time.time()))
            if stability_sleep > 0:
                time.sleep(stability_sleep)
            _refresh_asset_meta_for_assets(context, candidate_assets)
            for asset in candidate_assets:
                orders = outstanding_by_asset.get(asset)
                if not orders:
                    continue
                meta = context.asset_meta.get(asset)
                current_price = orders[0].get("limit_px")
                is_buy = side_by_asset.get(asset, True)
                reached_limit = reprice_counts[asset] >= PASSIVE_MAX_REPRICES
                stale_quote = _quote_is_stale(meta, current_price, is_buy)
                if reached_limit and not stale_quote:
                    continue
                if _should_reprice_passive(meta, current_price, is_buy, last_reprice_ts[asset], require_rest=False):
                    confirmed_assets.append(asset)

        assets_to_reprice: List[str] = []
        if confirmed_assets:
            min_order_usd = config.execution.get("min_order_size_usd", 2.0)
            for asset in confirmed_assets:
                total_target = target_usd_by_asset.get(asset, 0.0)
                remaining_target = max(total_target - filled_usd_by_asset.get(asset, 0.0), 0.0)
                if remaining_target < max(min_order_usd * 0.9, 1.0):
                    continue
                assets_to_reprice.append(asset)

        if assets_to_reprice:
            to_cancel = [
                order for order in window_outstanding if order.get("asset") in assets_to_reprice
            ]
            cancel_errors = cancel_orders(to_cancel, config, context.exchange)
            stage1_errors.extend(cancel_errors)
            if PASSIVE_CANCEL_COOLDOWN_SECONDS > 0:
                time.sleep(PASSIVE_CANCEL_COOLDOWN_SECONDS)

            reprice_deltas: Dict[str, float] = {}
            timestamp_now = time.time()
            for asset in assets_to_reprice:
                total_target = target_usd_by_asset.get(asset, 0.0)
                remaining_target = max(total_target - filled_usd_by_asset.get(asset, 0.0), 0.0)
                if remaining_target <= 0:
                    continue
                is_buy = side_by_asset.get(asset, True)
                current_orders = outstanding_by_asset.get(asset, [])
                current_price = current_orders[0].get("limit_px") if current_orders else None
                meta = context.asset_meta.get(asset)
                best_px = _best_price(meta, is_buy) if meta else None
                side_label = "buy" if is_buy else "sell"
                top_label = "bid" if is_buy else "ask"
                if current_price and best_px:
                    print(
                        f"🔁 Repricing {asset} {side_label}: top {top_label} {best_px:.6f}, "
                        f"previous quote {current_price:.6f}"
                    )
                reprice_deltas[asset] = remaining_target if is_buy else -remaining_target
                last_reprice_ts[asset] = timestamp_now
                reprice_counts[asset] += 1

            new_orders, new_errors = place_limit_orders(
                reprice_deltas,
                config,
                context.info,
                context.asset_meta,
                context.exchange,
            )
            stage1_errors.extend(new_errors)
            all_stage1_orders.extend(new_orders)
            outstanding_orders = [
                order for order in window_outstanding if order.get("asset") not in assets_to_reprice
            ] + new_orders
            continue

        outstanding_orders = list(window_outstanding)

    stage1_time = time.time() - stage1_start
    target_asset_count = len(target_usd_by_asset)
    total_target_usd = sum(target_usd_by_asset.values())
    epsilon = max(1e-6, min_order_usd * 0.02)

    stage1_asset_fills = 0
    stage1_filled_notional = 0.0
    for asset, target_usd in target_usd_by_asset.items():
        filled_usd = filled_usd_by_asset.get(asset, 0.0)
        clipped_filled = min(filled_usd, target_usd)
        stage1_filled_notional += clipped_filled
        if target_usd > 0 and clipped_filled + epsilon >= target_usd:
            stage1_asset_fills += 1

    print(
        f"Stage 1 complete: {stage1_asset_fills}/{target_asset_count} assets filled in "
        f"{stage1_time:.1f}s"
    )

    stage1_orders = all_stage1_orders
    unfilled = outstanding_orders

    stage2_orders: List[Dict[str, Any]] = []
    stage2_errors: List[Dict[str, Any]] = []

    if unfilled and context.exchange is not None:
        if config.execution.get("cancel_before_market_sweep", True):
            cancel_errors = cancel_orders(unfilled, config, context.exchange)
            stage1_errors.extend(cancel_errors)

        unfilled_deltas = {
            order["asset"]: order["size_usd"]
            if order["side"] == "buy"
            else -order["size_usd"]
            for order in unfilled
        }

        stage2_orders, stage2_errors = place_market_orders(
            unfilled_deltas,
            config,
            context.info,
            context.asset_meta,
            context.exchange,
        )

    if not config.dry_run:
        _enrich_orders_with_actual_fills(
            context.info,
            context.account_address,
            filled,
            stage2_orders,
            stage1_start,
        )

    stage1_turnover = sum(abs(o.get("fill_usd", 0.0)) for o in filled)
    stage2_turnover = sum(abs(o.get("fill_usd", 0.0)) for o in stage2_orders)
    stage2_asset_fills = len({order.get("asset") for order in stage2_orders if order.get("asset")})

    passive_stats = _aggregate_slippage_metrics(filled)
    aggressive_stats = _aggregate_slippage_metrics(stage2_orders)
    total_stats = _combine_slippage_stats(passive_stats, aggressive_stats)

    target_order_count = target_asset_count
    passive_fill_rate = (
        stage1_filled_notional / total_target_usd if total_target_usd > 0 else 0.0
    )
    aggressive_fill_rate = (
        stage2_turnover / total_target_usd if total_target_usd > 0 else 0.0
    )

    stage2_highlights = sorted(
        [
            {
                "asset": order.get("asset"),
                "side": order.get("side"),
                "notional_usd": abs(order.get("fill_usd", 0.0)),
                "slippage_bps": order.get("slippage_bps"),
            }
            for order in stage2_orders
        ],
        key=lambda item: item["notional_usd"],
        reverse=True,
    )[:5]

    return {
        "stage1_orders": stage1_orders,
        "stage2_orders": stage2_orders,
        "stage1_filled": len(filled),
        "stage2_filled": len(stage2_orders),
        "total_turnover": stage1_turnover + stage2_turnover,
        "errors": stage1_errors + stage2_errors,
        "passive_slippage": passive_stats,
        "aggressive_slippage": aggressive_stats,
        "total_slippage": total_stats,
        "passive_fill_rate": passive_fill_rate,
        "aggressive_fill_rate": aggressive_fill_rate,
        "stage2_highlights": stage2_highlights,
        "target_order_count": target_order_count,
        "total_target_usd": total_target_usd,
        "stage1_fill_notional": stage1_filled_notional,
        "stage2_fill_notional": stage2_turnover,
        "stage1_asset_fills": stage1_asset_fills,
        "stage2_asset_fills": stage2_asset_fills,
    }


def calculate_deltas(
    target: Dict[str, float],
    current: Dict[str, float],
    min_threshold: float = 10.0,
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

        if abs(delta) >= min_threshold:
            deltas[asset] = delta

    return deltas


def place_limit_orders(
    deltas: Dict[str, float],
    config,
    info_client,
    asset_meta: Dict[str, AssetMeta],
    exchange_client=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Place limit orders at mid price or better.

    Args:
        deltas: Position deltas in USD
        config: Configuration
        info_client: Hyperliquid Info client
        asset_meta: Mapping of asset -> metadata
        exchange_client: Hyperliquid Exchange client (None in dry-run)

    Returns:
        Tuple of (orders, errors)
    """
    orders: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    aggression = config.execution.get("limit_order_aggression", "join_best")
    min_order_usd = config.execution.get("min_order_size_usd", 2.0)

    for asset, delta_usd in deltas.items():
        meta = asset_meta.get(asset)
        if not meta:
            errors.append(
                {"asset": asset, "error": "Missing asset metadata", "stage": "limit"}
            )
            continue

        is_buy = bool(delta_usd > 0)

        try:
            size_coin, reference_px = _compute_size_coin(delta_usd, meta)
        except Exception as exc:
            errors.append(
                {"asset": asset, "error": str(exc), "stage": "sizing"}
            )
            continue

        if size_coin <= 0:
            errors.append(
                {
                    "asset": asset,
                    "error": "Order size truncated below minimum precision",
                    "stage": "sizing",
                }
            )
            continue

        size_usd = size_coin * reference_px
        if size_usd < min_order_usd:
            errors.append(
                {
                    "asset": asset,
                    "error": f"Size ${size_usd:.2f} below min_order_size_usd",
                    "stage": "sizing",
                }
            )
            continue

        try:
            limit_px = _compute_limit_price(meta, is_buy, aggression)
        except Exception as exc:
            errors.append(
                {"asset": asset, "error": str(exc), "stage": "pricing"}
            )
            continue

        reference_mid = meta.mid_px
        slippage_bps = _calculate_slippage_bps(limit_px, reference_mid, is_buy)
        spread_bps = None
        if meta.impact_bid and meta.impact_ask and meta.impact_bid > 0:
            spread_bps = ((meta.impact_ask - meta.impact_bid) / meta.impact_bid) * 1e4

        order_record = {
            "asset": asset,
            "side": "buy" if is_buy else "sell",
            "requested_usd": abs(delta_usd),
            "size_usd": size_usd,
            "size_coin": size_coin,
            "order_type": "limit",
            "limit_px": limit_px,
            "status": "simulated" if config.dry_run else "placed",
            "execution_px": limit_px,
            "reference_mid": reference_mid,
            "reference_bid": meta.impact_bid,
            "reference_ask": meta.impact_ask,
            "slippage_bps": slippage_bps,
            "spread_bps": spread_bps,
        }

        if config.dry_run or exchange_client is None:
            order_record["order_id"] = f"dry_run_{asset}_{int(time.time())}"
            orders.append(order_record)
            continue

        try:
            response = exchange_client.order(
                name=asset,
                is_buy=is_buy,
                sz=size_coin,
                limit_px=limit_px,
                order_type={"limit": {"tif": "Gtc"}},  # Good-til-cancelled for reliable fills
                reduce_only=False,
            )
            order_id = _extract_order_id(response)

            # Log first response for debugging
            if len(orders) == 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"First order response for {asset}:")
                logger.info(f"  Type: {type(response)}")
                logger.info(f"  Response: {response}")
                logger.info(f"  Extracted order_id: {order_id}")

            if not order_id:
                print(f"WARNING: Failed to extract order_id for {asset}")
                print(f"  Response type: {type(response)}")
                print(f"  Response: {response}")
                order_record["status"] = "unknown"
                order_record["raw_response"] = response
            else:
                order_record["order_id"] = order_id
            orders.append(order_record)
        except Exception as exc:
            errors.append(
                {"asset": asset, "error": str(exc), "stage": "limit_order"}
            )

    if errors:
        print(f"place_limit_orders encountered {len(errors)} errors")

    if orders:
        print(f"Successfully placed {len(orders)} limit orders")
        # Log sample order details for debugging
        if len(orders) > 0:
            sample = orders[0]
            print(f"  Sample order: {sample['asset']} {sample['side']} "
                  f"{sample['size_coin']:.4f} @ ${sample['limit_px']:.6f}")

    return orders, errors


def monitor_fills_with_timeout(
    orders: List[Dict[str, Any]],
    timeout_seconds: int,
    config,
    info_client,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Monitor order fills via polling with timeout.

    Args:
        orders: List of order dictionaries
        timeout_seconds: Maximum time to wait
        config: Configuration
        info_client: Hyperliquid Info client

    Returns:
        Tuple of (filled_orders, unfilled_orders)
    """
    if not orders:
        return [], []

    if config.dry_run:
        filled = []
        for order in orders:
            filled.append(
                {
                    **order,
                    "fill_usd": order["size_usd"],
                    "fill_time": datetime.utcnow(),
                    "fill_price": order.get("execution_px"),
                }
            )
        return filled, []

    polling_addresses: List[str] = []
    try:
        polling_addresses.append(_resolve_main_wallet(config))
    except ValueError:
        pass

    api_vault = os.getenv("HYPERLIQUID_API_KEY") or config.api_key
    if api_vault:
        polling_addresses.append(api_vault)

    if not polling_addresses:
        raise ValueError(
            "Unable to determine address for open order monitoring. "
            "Set HYPERLIQUID_MAIN_WALLET and HYPERLIQUID_API_KEY."
        )

    start_time = time.time()
    outstanding: Dict[str, Dict[str, Any]] = {
        order.get("order_id"): order
        for order in orders
        if order.get("order_id")
    }

    print(f"Monitoring {len(outstanding)}/{len(orders)} orders with valid order_ids")
    if len(outstanding) == 0:
        print("WARNING: No orders have order_ids - monitoring will exit immediately!")
        return [], orders

    filled: List[Dict[str, Any]] = []
    poll_interval = min(10, max(5, timeout_seconds // 12 or 5))
    polls = 0
    print(f"Poll interval: {poll_interval}s, timeout: {timeout_seconds}s")

    while outstanding and (time.time() - start_time) < timeout_seconds:
        time.sleep(poll_interval)
        polls += 1
        open_order_ids: set[str] = set()
        for address in polling_addresses:
            open_order_ids.update(_collect_open_order_ids(info_client, address))

        newly_filled = 0
        for order_id in list(outstanding.keys()):
            if order_id not in open_order_ids:
                order = outstanding.pop(order_id)
                filled.append(
                    {
                        **order,
                        "fill_usd": order["size_usd"],
                        "fill_time": datetime.utcnow(),
                        "fill_price": order.get("execution_px"),
                    }
                )
                newly_filled += 1

        if newly_filled > 0:
            elapsed = time.time() - start_time
            print(f"  Poll #{polls} ({elapsed:.1f}s): {newly_filled} new fills, "
                  f"{len(filled)} total filled, {len(outstanding)} still open")

    unfilled = list(outstanding.values())
    if filled:
        print(f"Fill monitoring complete: {len(filled)}/{len(filled) + len(unfilled)} filled")
    return filled, unfilled


def place_market_orders(
    deltas: Dict[str, float],
    config,
    info_client,
    asset_meta: Dict[str, AssetMeta],
    exchange_client=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Place market orders for immediate execution.

    Args:
        deltas: Position deltas in USD
        config: Configuration
        info_client: Hyperliquid Info client
        asset_meta: Asset metadata map
        exchange_client: Hyperliquid Exchange client

    Returns:
        Tuple of (filled_orders, errors)
    """
    filled: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if not deltas:
        return filled, errors

    for asset, delta_usd in deltas.items():
        meta = asset_meta.get(asset)
        if not meta:
            errors.append(
                {"asset": asset, "error": "Missing asset metadata", "stage": "market"}
            )
            continue

        is_buy = delta_usd > 0

        try:
            size_coin, reference_px = _compute_size_coin(delta_usd, meta)
        except Exception as exc:
            errors.append(
                {"asset": asset, "error": str(exc), "stage": "market_sizing"}
            )
            continue

        if size_coin <= 0:
            errors.append(
                {
                    "asset": asset,
                    "error": "Order size below minimum precision",
                    "stage": "market_sizing",
                }
            )
            continue

        execution_px = meta.impact_ask if is_buy else meta.impact_bid
        if not execution_px or execution_px <= 0:
            execution_px = reference_px
        fill_usd = size_coin * execution_px
        slippage_bps = _calculate_slippage_bps(execution_px, meta.mid_px, is_buy)

        order_record = {
            "asset": asset,
            "side": "buy" if is_buy else "sell",
            "requested_usd": abs(delta_usd),
            "size_usd": fill_usd,
            "size_coin": size_coin,
            "order_type": "market",
            "status": "filled_simulated" if config.dry_run else "filled",
            "execution_px": execution_px,
            "reference_mid": meta.mid_px,
            "reference_bid": meta.impact_bid,
            "reference_ask": meta.impact_ask,
            "slippage_bps": slippage_bps,
        }
        order_record["fill_usd"] = fill_usd

        if config.dry_run or exchange_client is None:
            filled.append(order_record)
            continue

        try:
            response = exchange_client.market_open(
                name=asset,
                is_buy=is_buy,
                sz=size_coin,
                px=meta.mid_px,
            )
            order_id = _extract_order_id(response)
            if order_id:
                order_record["order_id"] = order_id
            filled.append(order_record)
        except Exception as exc:
            errors.append(
                {"asset": asset, "error": str(exc), "stage": "market_order"}
            )

    return filled, errors


def cancel_orders(
    orders: List[Dict[str, Any]],
    config,
    exchange_client,
) -> List[Dict[str, Any]]:
    """
    Cancel open orders.

    Args:
        orders: Orders to cancel
        config: Configuration
        exchange_client: Hyperliquid Exchange client

    Returns:
        List of cancellation errors (empty on success)
    """
    if not orders:
        return []

    if config.dry_run or exchange_client is None:
        print(f"DRY-RUN: Would cancel {len(orders)} orders")
        return []

    errors: List[Dict[str, Any]] = []

    for order in orders:
        order_id = order.get("order_id")
        asset = order.get("asset")

        if not order_id or not asset:
            continue

        try:
            exchange_client.cancel(name=asset, oid=int(order_id))
            print(f"Cancelled order {order_id} on {asset}")
        except Exception as exc:
            errors.append(
                {"asset": asset, "error": str(exc), "stage": "cancel"}
            )

    return errors


def validate_execution_results(results: Dict[str, Any], config) -> None:
    """
    Validate execution results and log warnings.

    Args:
        results: Execution results
        config: Configuration
    """
    stage1_filled = results["stage1_filled"]
    stage2_filled = results["stage2_filled"]
    stage1_asset_fills = results.get("stage1_asset_fills", stage1_filled)
    stage2_asset_fills = results.get("stage2_asset_fills", stage2_filled)
    total_orders = results.get("target_order_count", len(results["stage1_orders"]))
    errors = results.get("errors", [])

    if total_orders == 0:
        print("No orders placed (portfolio unchanged)")
        return

    total_target_usd = results.get("total_target_usd")
    stage1_fill_notional = results.get("stage1_fill_notional", 0.0) or 0.0
    stage2_fill_notional = results.get("stage2_fill_notional", 0.0) or 0.0
    if total_target_usd:
        fill_rate = (stage1_fill_notional + stage2_fill_notional) / total_target_usd
    else:
        fill_rate = (
            (stage1_asset_fills + stage2_asset_fills) / total_orders
            if total_orders > 0
            else 0
        )

    if fill_rate < 0.95:
        print(
            f"Warning: Low fill rate {fill_rate:.1%} "
            f"({stage1_asset_fills + stage2_asset_fills}/{total_orders})"
        )

    if len(errors) > 0:
        print(f"Warning: {len(errors)} errors during execution:")
        for err in errors[:5]:
            print(f"  {err.get('asset', 'N/A')}: {err.get('error', 'Unknown error')}")

    total_turnover = results["total_turnover"]
    turnover_pct = (
        (total_turnover / config.capital_usd) * 100 if config.capital_usd > 0 else 0
    )

    print("Execution summary:")
    print(
        f"  Stage 1 (limit): {stage1_asset_fills}/{total_orders} assets "
        f"({stage1_fill_notional:,.2f}/{(total_target_usd or 0):,.2f} USD)"
    )
    print(
        f"  Stage 2 (market): {stage2_asset_fills}/{total_orders} assets "
        f"({stage2_fill_notional:,.2f} USD)"
    )
    print(f"  Total turnover: ${total_turnover:,.2f} ({turnover_pct:.1f}% of capital)")
    print(f"  Errors: {len(errors)}")
