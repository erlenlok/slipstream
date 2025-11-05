from types import SimpleNamespace

import pytest

from slipstream.strategies.gradient.live import execution


def _build_asset_meta():
    return {
        "BTC": execution.AssetMeta(
            name="BTC",
            sz_decimals=5,
            mid_px=20000.0,
            impact_bid=19999.0,
            impact_ask=20001.0,
        ),
        "ETH": execution.AssetMeta(
            name="ETH",
            sz_decimals=4,
            mid_px=1500.0,
            impact_bid=1499.0,
            impact_ask=1501.0,
        ),
    }


def test_execute_rebalance_with_stages_dry_run(monkeypatch):
    config = SimpleNamespace(
        dry_run=True,
        capital_usd=1000.0,
        execution={
            "min_order_size_usd": 10.0,
            "limit_order_aggression": "join_best",
            "passive_timeout_seconds": 10,
        },
    )

    def fake_prepare(cfg):
        return execution.HyperliquidContext(
            info=object(),
            exchange=None,
            asset_meta=_build_asset_meta(),
        )

    monkeypatch.setattr(execution, "_prepare_hyperliquid_context", fake_prepare)

    target_positions = {"BTC": 120.0, "ETH": -80.0, "UNI": 5.0}
    current_positions = {"BTC": 20.0}

    result = execution.execute_rebalance_with_stages(
        target_positions,
        current_positions,
        config,
    )

    # UNI delta (5 USD) should be ignored because it is below threshold.
    assert all(order["asset"] in {"BTC", "ETH"} for order in result["stage1_orders"])
    assert result["stage1_filled"] == len(result["stage1_orders"]) == 2
    assert result["stage2_orders"] == []
    assert result["stage2_filled"] == 0
    assert result["errors"] == []

    order_map = {order["asset"]: order for order in result["stage1_orders"]}

    btc_order = order_map["BTC"]
    eth_order = order_map["ETH"]

    assert btc_order["side"] == "buy"
    assert eth_order["side"] == "sell"
    # Requested notionals should align with deltas (within rounding tolerance)
    assert btc_order["requested_usd"] == pytest.approx(100.0, rel=1e-6)
    assert eth_order["requested_usd"] == pytest.approx(80.0, rel=1e-6)

    total_turnover = sum(abs(order["size_usd"]) for order in result["stage1_orders"])
    assert result["total_turnover"] == pytest.approx(total_turnover, rel=1e-9)


def test_execute_rebalance_with_stages_triggers_stage2(monkeypatch):
    config = SimpleNamespace(
        dry_run=False,
        capital_usd=1000.0,
        execution={
            "min_order_size_usd": 10.0,
            "limit_order_aggression": "join_best",
            "passive_timeout_seconds": 0,  # force immediate fallback
            "cancel_before_market_sweep": True,
        },
    )

    asset_meta = _build_asset_meta()
    context = execution.HyperliquidContext(
        info=object(),
        exchange=object(),  # non-None to enable stage 2
        asset_meta=asset_meta,
    )
    monkeypatch.setattr(execution, "_prepare_hyperliquid_context", lambda cfg: context)

    captured_cancelled = []
    captured_market_deltas = {}

    stage1_orders = [
        {
            "asset": "BTC",
            "side": "buy",
            "size_usd": 120.0,
            "order_id": "btc-1",
        },
        {
            "asset": "ETH",
            "side": "sell",
            "size_usd": 80.0,
            "order_id": "eth-1",
        },
    ]

    def fake_place_limit_orders(deltas, cfg, info, meta, exchange):
        assert deltas == {"BTC": 120.0, "ETH": -80.0}
        return (
            [
                {
                    **stage1_orders[0],
                    "requested_usd": 120.0,
                    "status": "placed",
                    "limit_px": 20000.0,
                },
                {
                    **stage1_orders[1],
                    "requested_usd": 80.0,
                    "status": "placed",
                    "limit_px": 1500.0,
                },
            ],
            [],
        )

    def fake_cancel_orders(orders, cfg, exchange):
        captured_cancelled.extend(orders)
        return []

    def fake_place_market_orders(deltas, cfg, info, meta, exchange):
        captured_market_deltas.update(deltas)
        return (
            [
                {
                    "asset": asset,
                    "side": "buy" if delta > 0 else "sell",
                    "fill_usd": abs(delta),
                    "slippage_bps": 2.0,
                }
                for asset, delta in deltas.items()
            ],
            [],
        )

    monkeypatch.setattr(execution, "place_limit_orders", fake_place_limit_orders)
    monkeypatch.setattr(execution, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(execution, "place_market_orders", fake_place_market_orders)
    monkeypatch.setattr(execution, "_enrich_orders_with_actual_fills", lambda *args, **kwargs: None)

    target_positions = {"BTC": 120.0, "ETH": -80.0}
    current_positions = {"BTC": 0.0, "ETH": 0.0}

    result = execution.execute_rebalance_with_stages(
        target_positions,
        current_positions,
        config,
    )

    # Stage 1 should produce two orders, but none filled due to zero timeout
    assert result["stage1_filled"] == 0
    assert result["stage2_filled"] == 2
    assert result["stage2_orders"]
    assert result["errors"] == []

    # Ensure market sweep used expected deltas
    assert captured_market_deltas == {"BTC": 120.0, "ETH": -80.0}
    # Cancel should have been invoked on both outstanding orders
    assert {order["order_id"] for order in captured_cancelled} == {"btc-1", "eth-1"}

    # Total turnover combines stage 2 fills
    assert result["total_turnover"] == pytest.approx(200.0, rel=1e-9)
    assert result["passive_fill_rate"] == pytest.approx(0.0, abs=1e-12)
    assert result["aggressive_fill_rate"] == pytest.approx(1.0, abs=1e-12)
