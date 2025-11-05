"""
Universe selection helpers for the Gradient strategy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


def select_top_bottom_assets(
    trend_strength: pd.DataFrame,
    top_n: int = 5,
    bottom_n: Optional[int] = None,
    min_abs_strength: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify long and short candidate masks from trend strength scores.

    Args:
        trend_strength: Wide DataFrame of trend scores.
        top_n: Number of strongest assets to consider for the long side.
        bottom_n: Number of weakest assets for the short side (defaults to ``top_n``).
        min_abs_strength: Optional absolute-strength filter to avoid trading tiny signals.

    Returns:
        Tuple ``(long_mask, short_mask)`` where each mask is a boolean DataFrame
        with the same shape as ``trend_strength``.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    bottom_n = bottom_n or top_n
    if bottom_n <= 0:
        raise ValueError("bottom_n must be positive")

    ranks_desc = trend_strength.rank(axis=1, method="first", ascending=False)
    ranks_asc = trend_strength.rank(axis=1, method="first", ascending=True)

    long_mask = ranks_desc <= top_n
    short_mask = ranks_asc <= bottom_n

    if min_abs_strength > 0:
        long_mask &= trend_strength >= min_abs_strength
        short_mask &= trend_strength <= -min_abs_strength

    return long_mask, short_mask

