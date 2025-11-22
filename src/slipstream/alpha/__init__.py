"""
Alpha model framework.

This module provides data preparation functions for predictive models
on vol-normalized idiosyncratic returns. Some components have been moved to
legacy for historical reference.

Key components:
- data_prep: Feature engineering (momentum + funding) and target construction
- Legacy training components are in the legacy folder

Reference: legacy documentation for historical H* optimization
"""

from slipstream.alpha.data_prep import (
    prepare_alpha_training_data,
    compute_funding_features,
    compute_forward_returns,
)

# Legacy training functions have been moved to legacy folder
# See legacy/src/slipstream/alpha/training.py for historical reference

__all__ = [
    'prepare_alpha_training_data',
    'compute_funding_features',
    'compute_forward_returns',
]
