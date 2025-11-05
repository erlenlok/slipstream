import numpy as np
import pandas as pd

from slipstream.strategies.gradient.signals import compute_trend_strength


def test_compute_trend_strength_returns_components():
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=256, freq="h")
    returns = pd.DataFrame(
        rng.normal(loc=0.0, scale=0.01, size=(256, 3)),
        index=index,
        columns=["BTC", "ETH", "SOL"],
    )

    aggregate, components = compute_trend_strength(
        returns,
        lookbacks=[2, 4, 8, 16],
        return_components=True,
    )

    assert aggregate.shape == returns.shape
    assert set(components.keys()) == {2, 4, 8, 16}
    # Ensure at least one late observation has valid signal
    assert aggregate.iloc[-1].notna().any()
