import pandas as pd
import numpy as np
import pytest

from slipstream.alpha.data_prep import compute_forward_returns, BASE_INTERVAL_HOURS
from scripts.find_optimal_H_alpha import compute_market_factor_from_loadings


def make_returns(rows: int = 6, assets: int = 2) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq=f"{BASE_INTERVAL_HOURS}H", tz="UTC")
    data = np.arange(rows * assets, dtype=float).reshape(rows, assets) / 100.0
    columns = [f"asset_{i}" for i in range(assets)]
    return pd.DataFrame(data, index=index, columns=columns)


def test_forward_returns_requires_multiple_of_base():
    returns = make_returns()
    with pytest.raises(ValueError):
        compute_forward_returns(returns, H=6)  # 6 is not divisible by 4


def test_forward_returns_aggregates_correct_steps():
    returns = make_returns(rows=6, assets=1)
    forward, volatility = compute_forward_returns(returns, H=8, vol_span=8)  # two base intervals
    assert len(forward) > 0
    first_timestamp = forward.index.get_level_values("timestamp")[0]
    first_asset = forward.index.get_level_values("asset")[0]
    position = returns.index.get_loc(first_timestamp)
    steps = 8 // BASE_INTERVAL_HOURS
    col_idx = returns.columns.get_loc(first_asset)
    expected = returns.iloc[position + 1 : position + 1 + steps, col_idx].sum()
    norm_value = forward.loc[(first_timestamp, first_asset)]
    vol_value = volatility.loc[(first_timestamp, first_asset)]
    assert pytest.approx(norm_value * vol_value) == expected


def test_market_factor_projection_matches_dot_product():
    returns_index = pd.date_range("2024-01-01", periods=4, freq="4H", tz="UTC")
    returns = pd.DataFrame(
        [
            [0.01, 0.02],
            [0.03, -0.01],
            [0.04, 0.00],
            [0.05, 0.01],
        ],
        index=returns_index,
        columns=["BTC", "ETH"],
    )

    # Daily PCA loadings that will be forward-filled to 4H grid
    loadings_index = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC")
    loadings = pd.DataFrame(
        [[0.6, 0.8], [0.6, 0.8]],
        index=loadings_index,
        columns=["BTC", "ETH"],
    )

    factor = compute_market_factor_from_loadings(loadings, returns)
    # For the first timestamp, normalization denominator sqrt(0.6^2 + 0.8^2) = 1
    expected_first = 0.6 * 0.01 + 0.8 * 0.02
    assert pytest.approx(factor.iloc[0]) == expected_first
