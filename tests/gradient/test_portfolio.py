import numpy as np
import pandas as pd
import pytest

from slipstream.common import ewm_volatility
from slipstream.gradient.portfolio import construct_gradient_portfolio


def test_construct_gradient_portfolio_balances_dollar_vol():
    rng = np.random.default_rng(123)
    index = pd.date_range("2024-01-01", periods=64, freq="h")
    returns = pd.DataFrame(
        rng.normal(scale=0.01, size=(64, 2)),
        index=index,
        columns=["A", "B"],
    )

    trend_strength = pd.DataFrame(
        {
            "A": np.linspace(-1.0, 1.0, len(index)),
            "B": np.linspace(1.0, -1.0, len(index)),
        },
        index=index,
    )

    target_vol = 2.0
    vol_span = 8
    weights = construct_gradient_portfolio(
        trend_strength,
        returns,
        top_n=1,
        bottom_n=1,
        vol_span=vol_span,
        target_side_dollar_vol=target_vol,
    )

    # Use the same volatility estimate as the allocator
    vol_est = ewm_volatility(returns, span=vol_span, min_periods=vol_span)

    # Focus on later rows once the EWMA is initialized
    late_idx = index[-1]
    vol_row = vol_est.loc[late_idx]
    weight_row = weights.loc[late_idx]

    valid_assets = [asset for asset in weight_row.index if weight_row[asset] != 0]
    assert len(valid_assets) == 2

    long_asset = max(valid_assets, key=lambda a: weight_row[a])
    short_asset = min(valid_assets, key=lambda a: weight_row[a])

    long_contribution = abs(weight_row[long_asset]) * vol_row[long_asset]
    short_contribution = abs(weight_row[short_asset]) * vol_row[short_asset]

    assert pytest.approx(long_contribution, rel=1e-3) == target_vol / 1  # Only one long
    assert pytest.approx(short_contribution, rel=1e-3) == target_vol / 1  # Only one short
    assert np.sign(weight_row[long_asset]) == 1
    assert np.sign(weight_row[short_asset]) == -1
