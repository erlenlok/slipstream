"""
Gradient strategy components.

The Gradient strategy targets a balanced long/short portfolio using
multi-horizon trend strength signals derived from volatility-normalized
returns.
"""

from .signals import DEFAULT_LOOKBACKS, compute_trend_strength  # noqa: F401
from .portfolio import construct_gradient_portfolio  # noqa: F401
from .backtest import run_gradient_backtest  # noqa: F401

__all__ = [
    "DEFAULT_LOOKBACKS",
    "compute_trend_strength",
    "construct_gradient_portfolio",
    "run_gradient_backtest",
]

