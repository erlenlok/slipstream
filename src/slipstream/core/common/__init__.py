"""
Common utilities shared across Slipstream strategies.

This package centralizes helpers for return normalization, volatility
estimation, risk management (VAR, RIE covariance), and other small utilities
that are useful for both the original Slipstream strategy and the Gradient variant.
"""

from .returns import vol_normalize_returns  # noqa: F401
from .volatility import ewm_volatility, annualize_volatility  # noqa: F401
from .risk import (  # noqa: F401
    compute_daily_returns,
    estimate_covariance_rie,
    compute_portfolio_var,
)

__all__ = [
    "vol_normalize_returns",
    "ewm_volatility",
    "annualize_volatility",
    "compute_daily_returns",
    "estimate_covariance_rie",
    "compute_portfolio_var",
]

