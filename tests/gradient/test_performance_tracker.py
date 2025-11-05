from datetime import datetime
from types import SimpleNamespace

import pytest

from slipstream.strategies.gradient.live.performance import PerformanceTracker


def test_performance_tracker_daily_and_all_time_summary(tmp_path):
    tracker = PerformanceTracker(log_dir=tmp_path)
    config = SimpleNamespace(capital_usd=1000.0, dry_run=False)

    ts1 = datetime(2024, 1, 1, 4)
    ts2 = datetime(2024, 1, 1, 8)

    exec_a = {
        "stage1_filled": 2,
        "stage2_filled": 1,
        "stage1_asset_fills": 2,
        "stage2_asset_fills": 1,
        "target_order_count": 3,
        "stage1_orders": [],
        "total_turnover": 250.0,
        "errors": [],
        "passive_slippage": {"count": 2, "total_usd": 200.0, "weighted_bps": 1.0},
        "aggressive_slippage": {"count": 1, "total_usd": 50.0, "weighted_bps": 6.0},
        "total_slippage": {"count": 3, "total_usd": 250.0, "weighted_bps": 2.5},
        "stage1_fill_notional": 200.0,
        "stage2_fill_notional": 50.0,
        "total_target_usd": 250.0,
    }
    exec_b = {
        "stage1_filled": 2,
        "stage2_filled": 1,
        "stage1_asset_fills": 2,
        "stage2_asset_fills": 1,
        "target_order_count": 3,
        "stage1_orders": [],
        "total_turnover": 240.0,
        "errors": [],
        "passive_slippage": {"count": 2, "total_usd": 180.0, "weighted_bps": 1.5},
        "aggressive_slippage": {"count": 1, "total_usd": 60.0, "weighted_bps": 4.0},
        "total_slippage": {"count": 3, "total_usd": 240.0, "weighted_bps": 2.2},
        "stage1_fill_notional": 180.0,
        "stage2_fill_notional": 60.0,
        "total_target_usd": 240.0,
    }

    tracker.log_rebalance(
        ts1,
        {"BTC": 125.0, "ETH": -125.0},
        exec_a,
        config,
    )
    tracker.log_rebalance(
        ts2,
        {"BTC": 120.0, "ETH": -120.0},
        exec_b,
        config,
    )

    daily = tracker.get_daily_summary(ts1)
    assert daily["n_rebalances"] == 2
    assert daily["total_turnover"] == pytest.approx(490.0)
    assert daily["avg_turnover_per_rebalance"] == pytest.approx(245.0)
    assert daily["total_orders"] == 6
    assert daily["passive_fills"] == 4
    assert daily["aggressive_fills"] == 2

    expected_passive_rate = (200.0 + 180.0) / (250.0 + 240.0)
    expected_aggressive_rate = (50.0 + 60.0) / (250.0 + 240.0)
    assert daily["passive_fill_rate"] == pytest.approx(expected_passive_rate)
    assert daily["aggressive_fill_rate"] == pytest.approx(expected_aggressive_rate)

    passive_weighted = 200.0 * 1.0 + 180.0 * 1.5
    aggressive_weighted = 50.0 * 6.0 + 60.0 * 4.0
    total_slip_weighted = passive_weighted + aggressive_weighted

    assert daily["passive_slippage_bps"] == pytest.approx(passive_weighted / 380.0)
    assert daily["aggressive_slippage_bps"] == pytest.approx(aggressive_weighted / 110.0)
    assert daily["total_slippage_bps"] == pytest.approx(total_slip_weighted / 490.0)

    recent = tracker.get_recent_rebalance()
    assert recent["timestamp"].startswith("2024-01-01T08:00")

    summary = tracker.get_all_time_summary()
    assert summary["total_rebalances"] == 2
    assert summary["total_turnover"] == pytest.approx(490.0)
    assert summary["passive_fill_rate"] == pytest.approx(expected_passive_rate)
    assert summary["aggressive_fill_rate"] == pytest.approx(expected_aggressive_rate)
    assert summary["passive_slippage_bps"] == pytest.approx(passive_weighted / 380.0)
    assert summary["aggressive_slippage_bps"] == pytest.approx(aggressive_weighted / 110.0)
    assert summary["total_slippage_bps"] == pytest.approx(total_slip_weighted / 490.0)
    assert summary["days_running"] == 1
