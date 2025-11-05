import pytest

from slipstream.strategies.gradient.live.execution import (
    AssetMeta,
    _compute_limit_price,
    _price_tick,
    _calculate_slippage_bps,
    _aggregate_slippage_metrics,
    _extract_order_id,
)
from slipstream.strategies.gradient.live.performance import compute_signal_tracking_metrics


@pytest.mark.parametrize(
    "bid,ask,mid,sz_decimals",
    [
        (2.345001, 2.34501, 2.345005, 0),
        (10500.0, 10500.8, 10500.4, 5),
    ],
)
def test_join_best_buy_stays_passive(bid, ask, mid, sz_decimals):
    meta = AssetMeta(
        name="ALT",
        sz_decimals=sz_decimals,
        mid_px=mid,
        impact_bid=bid,
        impact_ask=ask,
    )

    limit_price = _compute_limit_price(meta, is_buy=True, aggression="join_best")
    tick = _price_tick(meta)

    passive_guard = max(tick * 1e-3, 1e-9)
    assert limit_price < ask - passive_guard
    assert limit_price > 0


@pytest.mark.parametrize(
    "bid,ask,mid,sz_decimals",
    [
        (2.345001, 2.34501, 2.345005, 0),
        (10500.0, 10500.8, 10500.4, 5),
    ],
)
def test_join_best_sell_stays_passive(bid, ask, mid, sz_decimals):
    meta = AssetMeta(
        name="ALT",
        sz_decimals=sz_decimals,
        mid_px=mid,
        impact_bid=bid,
        impact_ask=ask,
    )

    limit_price = _compute_limit_price(meta, is_buy=False, aggression="join_best")
    tick = _price_tick(meta)

    passive_guard = max(tick * 1e-3, 1e-9)
    assert limit_price > bid + passive_guard


def test_join_best_handles_missing_book_data():
    meta = AssetMeta(
        name="NODATA",
        sz_decimals=3,
        mid_px=1.234,
        impact_bid=None,
        impact_ask=None,
    )

    price = _compute_limit_price(meta, is_buy=True, aggression="join_best")
    assert price > 0
    price = _compute_limit_price(meta, is_buy=False, aggression="join_best")
    assert price > 0


def test_calculate_slippage_bps_buy_sell_symmetry():
    mid = 100.0
    buy_price = 100.5
    sell_price = 99.5

    buy_slip = _calculate_slippage_bps(buy_price, mid, is_buy=True)
    sell_slip = _calculate_slippage_bps(sell_price, mid, is_buy=False)

    assert buy_slip == pytest.approx(50.0)
    assert sell_slip == pytest.approx(50.0)


def test_aggregate_slippage_metrics_weighted_average():
    orders = [
        {"fill_usd": 1000.0, "slippage_bps": 2.0},
        {"fill_usd": 2000.0, "slippage_bps": 4.0},
        {"fill_usd": 0.0, "slippage_bps": 10.0},
        {"fill_usd": 500.0, "slippage_bps": None},
    ]

    stats = _aggregate_slippage_metrics(orders)

    assert stats["count"] == 2
    assert stats["total_usd"] == pytest.approx(3000.0)
    assert stats["weighted_bps"] == pytest.approx((1000 * 2 + 2000 * 4) / 3000)


def test_compute_signal_tracking_metrics():
    records = [
        {"forecast": 0.01, "realized": 0.015},
        {"forecast": -0.02, "realized": -0.01},
        {"forecast": 0.0, "realized": 0.003},
    ]

    metrics = compute_signal_tracking_metrics(records)

    assert metrics["mae"] == pytest.approx(0.0060, rel=0, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(0.0066833, rel=0, abs=1e-6)
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["pearson_corr"] == pytest.approx(0.9860819)


def test_compute_signal_tracking_metrics_empty():
    metrics = compute_signal_tracking_metrics([])

    assert metrics["mae"] is None
    assert metrics["rmse"] is None
    assert metrics["hit_rate"] is None
    assert metrics["pearson_corr"] is None


def test_extract_order_id_handles_nested_response():
    response = {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {
                "statuses": [
                    {
                        "filled": {
                            "totalSz": "0.045",
                            "avgPx": "463.83",
                            "oid": 221329737449,
                        }
                    }
                ]
            },
        },
    }

    assert _extract_order_id(response) == "221329737449"
