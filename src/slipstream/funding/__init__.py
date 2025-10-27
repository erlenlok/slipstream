"""
Funding rate prediction utilities.

Provides data preparation helpers for forecasting future funding payments
within the Slipstream framework.
"""

from slipstream.funding.data_prep import (
    compute_forward_funding,
    prepare_funding_training_data,
)

__all__ = [
    "compute_forward_funding",
    "prepare_funding_training_data",
]
