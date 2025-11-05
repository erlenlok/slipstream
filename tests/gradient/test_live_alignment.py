import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from slipstream.strategies.gradient.live.data import compute_live_signals
from slipstream.strategies.gradient.live.portfolio import construct_target_portfolio
from slipstream.strategies.gradient.portfolio import construct_gradient_portfolio
from slipstream.strategies.gradient.signals import compute_trend_strength


def _build_config():
    return SimpleNamespace(
        liquidity_threshold_usd=10_000.0,
        liquidity_impact_pct=2.5,
        vol_span=24,
        lookback_spans=[2, 4, 8, 16],
        concentration_pct=40.0,
        weight_scheme="inverse_vol",
        capital_usd=1000.0,
        max_position_pct=100.0,
        max_total_leverage=4.0,
        dry_run=True,
    )


def _build_panel(n_periods: int = 256, assets=None) -> pd.DataFrame:
    if assets is None:
        assets = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE"]

    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=n_periods, freq="4h", tz="UTC")

    rows = []
    for asset in assets:
        returns = rng.normal(0.0, 0.01, size=n_periods)
        price = 100 * np.exp(np.cumsum(returns))
        volume = rng.uniform(500_000, 1_000_000, size=n_periods)

        for ts, px, ret, vol in zip(index, price, returns, volume):
            rows.append(
                {
                    "timestamp": ts,
                    "asset": asset,
                    "open": px * np.exp(-ret / 2),
                    "high": px * 1.01,
                    "low": px * 0.99,
                    "close": px,
                    "volume": vol,
                }
            )

    panel = pd.DataFrame(rows)
    return panel


class LiveGradientAlignmentTests(unittest.TestCase):
    def test_live_signals_align_with_backtest_scores(self):
        config = _build_config()
        panel = _build_panel()
        signals = compute_live_signals({"panel": panel}, config)

        prices = panel.pivot(index="timestamp", columns="asset", values="close")
        log_returns = np.log(prices / prices.shift(1))
        expected = compute_trend_strength(
            log_returns, lookbacks=config.lookback_spans
        ).iloc[-1]

        live_scores = signals.set_index("asset")["momentum_score"].sort_index()
        expected_scores = expected.reindex(live_scores.index)

        pd.testing.assert_series_equal(
            live_scores,
            expected_scores,
            check_names=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_live_portfolio_matches_backtest_selection(self):
        config = _build_config()
        panel = _build_panel()

        signals = compute_live_signals({"panel": panel}, config)
        liquid_assets = signals.loc[signals["include_in_universe"], "asset"].tolist()

        prices = panel.pivot(index="timestamp", columns="asset", values="close")
        log_returns = np.log(prices / prices.shift(1))[liquid_assets]
        trend_strength = compute_trend_strength(
            log_returns, lookbacks=config.lookback_spans
        )[liquid_assets]

        n_liquid = len(liquid_assets)
        n_select = max(
            1,
            min(
                n_liquid // 2,
                int(np.ceil(n_liquid * config.concentration_pct / 100.0)),
            ),
        )

        backtest_weights = construct_gradient_portfolio(
            trend_strength,
            log_returns,
            top_n=n_select,
            bottom_n=n_select,
            vol_span=config.vol_span,
            target_side_dollar_vol=1.0,
        )
        latest = backtest_weights.iloc[-1]
        expected_longs = {asset for asset, weight in latest.items() if weight > 0}
        expected_shorts = {asset for asset, weight in latest.items() if weight < 0}

        live_positions = construct_target_portfolio(signals, config)
        live_longs = {asset for asset, size in live_positions.items() if size > 0}
        live_shorts = {asset for asset, size in live_positions.items() if size < 0}

        self.assertEqual(live_longs, expected_longs)
        self.assertEqual(live_shorts, expected_shorts)


if __name__ == "__main__":
    unittest.main()
