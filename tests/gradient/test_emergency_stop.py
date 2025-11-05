import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_emergency_stop_module():
    path = Path(__file__).resolve().parents[2] / "scripts/strategies/gradient/live/emergency_stop.py"
    spec = importlib.util.spec_from_file_location("gradient_emergency_stop", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flatten_all_positions_places_opposite_orders(monkeypatch):
    positions_queue = [
        {"BTC": 150.0, "ETH": -50.0},
        {},
    ]
    snapshot_calls = []

    def fake_get_current_positions(config):
        snapshot_calls.append(dict(positions_queue[0] if positions_queue else {}))
        return positions_queue.pop(0) if positions_queue else {}

    captured_deltas = {}

    def fake_prepare_context(config):
        return SimpleNamespace(info=object(), exchange=None, asset_meta={})

    def fake_place_market_orders(deltas, config, info, asset_meta, exchange):
        captured_deltas.update(deltas)
        orders = [
            {
                "asset": asset,
                "side": "buy" if delta > 0 else "sell",
                "fill_usd": abs(delta),
            }
            for asset, delta in deltas.items()
        ]
        return orders, []

    emergency_stop = _load_emergency_stop_module()

    monkeypatch.setattr(emergency_stop, "get_current_positions", fake_get_current_positions)
    monkeypatch.setattr(emergency_stop, "_prepare_hyperliquid_context", fake_prepare_context)
    monkeypatch.setattr(emergency_stop, "place_market_orders", fake_place_market_orders)

    config = SimpleNamespace(dry_run=True)

    emergency_stop.flatten_all_positions(config)

    assert captured_deltas == {"BTC": -150.0, "ETH": 50.0}
    # Should have polled twice: before flattening and after.
    assert len(snapshot_calls) >= 2
