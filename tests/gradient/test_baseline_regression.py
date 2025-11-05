import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.strategies.gradient.capture_baseline import compute_metrics
from slipstream.strategies.gradient.backtest import run_gradient_backtest


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_gradient_backtest_matches_baseline_snapshot():
    returns_path = FIXTURE_DIR / "baseline_returns.json"
    metrics_path = FIXTURE_DIR / "baseline_metrics.json"

    returns = pd.read_json(returns_path, orient="split")
    expected = json.loads(metrics_path.read_text())

    result = run_gradient_backtest(
        returns,
        lookbacks=[2, 4, 8],
        top_n=2,
        bottom_n=2,
        vol_span=16,
        target_side_dollar_vol=1.0,
    )
    metrics = compute_metrics(result)

    for key, expected_value in expected.items():
        actual = metrics[key]
        if isinstance(expected_value, (int, float)):
            assert actual == pytest.approx(expected_value, rel=1e-7, abs=1e-9)
        else:
            assert actual == expected_value
