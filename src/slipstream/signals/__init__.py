"""
Signal generation for return predictions.

This module provides the core signal computation functions for the Slipstream strategy.
All signals are pure DataFrame transformations that can be imported into notebooks and
used as the single source of truth for research and production.
"""

from slipstream.signals.pca_momentum import (
    idiosyncratic_momentum,
    compute_idiosyncratic_returns,
)
from slipstream.signals.utils import (
    align_signals_to_universe,
    normalize_signal_cross_sectional,
    compute_signal_autocorrelation,
)

__all__ = [
    "idiosyncratic_momentum",
    "compute_idiosyncratic_returns",
    "align_signals_to_universe",
    "normalize_signal_cross_sectional",
    "compute_signal_autocorrelation",
]
