"""
Common utilities shared across Slipstream strategies.

This package centralizes helpers for return normalization, volatility
estimation, and other small utilities that are useful for both the
original Slipstream strategy and the upcoming Gradient variant.
"""

from .returns import vol_normalize_returns  # noqa: F401
from .volatility import ewm_volatility, annualize_volatility  # noqa: F401

__all__ = [
    "vol_normalize_returns",
    "ewm_volatility",
    "annualize_volatility",
]

