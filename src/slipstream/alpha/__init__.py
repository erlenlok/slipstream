"""
Alpha model training framework.

This module implements the bootstrap methodology for training predictive alpha models
on vol-normalized idiosyncratic returns, adapted from Schmidhuber (2021).

Key components:
- data_prep: Feature engineering (momentum + funding) and target construction
- training: Ridge regression with bootstrap for coefficient estimation
- validation: Walk-forward cross-validation for out-of-sample performance

Reference: docs/ALPHA_MODEL_TRAINING.md
"""

from slipstream.alpha.data_prep import (
    prepare_alpha_training_data,
    compute_funding_features,
    compute_forward_returns,
)

from slipstream.alpha.training import (
    find_optimal_lambda,
    bootstrap_train_alpha_model,
    train_alpha_model_complete,
)

__all__ = [
    'prepare_alpha_training_data',
    'compute_funding_features',
    'compute_forward_returns',
    'find_optimal_lambda',
    'bootstrap_train_alpha_model',
    'train_alpha_model_complete',
]
