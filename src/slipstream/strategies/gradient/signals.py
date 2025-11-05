"""
Signal construction for the Gradient strategy.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from slipstream.common import vol_normalize_returns


DEFAULT_LOOKBACKS = [2 ** i for i in range(1, 11)]  # 2, 4, 8, ..., 1024


def compute_trend_strength(
    log_returns: pd.DataFrame,
    lookbacks: Iterable[int] = DEFAULT_LOOKBACKS,
    normalization_method: str = "ewm",
    return_components: bool = False,
) -> Tuple[pd.DataFrame, dict[int, pd.DataFrame]] | pd.DataFrame:
    """Aggregate multi-horizon trend strength measures.

    For each lookback window ``L`` we:
      1. Volatility-normalize the raw log returns using ``vol_normalize_returns``.
      2. Compute the rolling momentum (sum of normalized returns) over ``L``.
      3. Sum the contributions across all lookbacks to obtain the final score.

    Args:
        log_returns: Wide DataFrame of log returns (timestamp index, asset columns).
        lookbacks: Iterable of lookback windows (bars) to evaluate.
        normalization_method: Passed through to ``vol_normalize_returns``.
        return_components: If ``True`` also return the per-lookback components.

    Returns:
        Either the aggregated trend strength DataFrame, or a tuple of
        ``(aggregate, components_dict)`` where ``components_dict[L]`` contains the
        momentum contribution for lookback ``L``.
    """
    if log_returns.empty:
        raise ValueError("log_returns is empty")

    lookbacks = list(lookbacks)
    if any(lb <= 0 for lb in lookbacks):
        raise ValueError("All lookbacks must be positive integers.")

    aggregate = pd.DataFrame(
        0.0,
        index=log_returns.index,
        columns=log_returns.columns,
    )
    contribution_counts = pd.DataFrame(
        0,
        index=log_returns.index,
        columns=log_returns.columns,
        dtype=int,
    )
    components: dict[int, pd.DataFrame] = {}

    for lookback in lookbacks:
        normalized = vol_normalize_returns(
            log_returns,
            window=lookback,
            method=normalization_method,
            min_periods=lookback,
        )

        momentum = normalized.rolling(
            window=lookback,
            min_periods=lookback,
        ).sum()

        components[lookback] = momentum

        aggregate = aggregate.add(momentum.fillna(0.0), fill_value=0.0)
        contribution_counts = contribution_counts.add(
            momentum.notna().astype(int),
            fill_value=0,
        )

    aggregate = aggregate.where(contribution_counts > 0, np.nan)

    if return_components:
        return aggregate, components

    return aggregate

