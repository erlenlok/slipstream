import numpy as np
import pandas as pd

from slipstream.strategies.gradient.backtest import run_gradient_backtest


def test_run_gradient_backtest_produces_outputs():
    rng = np.random.default_rng(99)
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    returns = pd.DataFrame(
        rng.normal(scale=0.01, size=(200, 4)),
        index=index,
        columns=["BTC", "ETH", "SOL", "XRP"],
    )

    result = run_gradient_backtest(
        returns,
        lookbacks=[2, 4, 8],
        top_n=2,
        bottom_n=2,
        vol_span=16,
    )

    assert result.weights.shape == returns.shape
    assert len(result.portfolio_returns) == len(returns)
    assert result.portfolio_returns.notna().any()

    sharpe = result.annualized_sharpe(periods_per_year=24 * 365)
    assert isinstance(sharpe, float)
