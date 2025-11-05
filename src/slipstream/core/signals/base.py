"""
Base interfaces and types for signal generation.

This module defines the standard data structures and validation functions
used across all signal computation modules.
"""

from typing import Protocol, runtime_checkable
import pandas as pd
import numpy as np


@runtime_checkable
class SignalFunction(Protocol):
    """Protocol for signal computation functions.

    All signal functions should accept DataFrames and return a signal DataFrame
    with (timestamp, asset) index and 'signal' column.
    """

    def __call__(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """Compute signal from returns data.

        Args:
            returns: DataFrame with timestamp index and asset columns
            **kwargs: Additional signal-specific parameters

        Returns:
            DataFrame with MultiIndex (timestamp, asset) and 'signal' column
        """
        ...


def validate_returns_dataframe(df: pd.DataFrame, name: str = "returns") -> None:
    """Validate that a DataFrame matches expected returns format.

    Args:
        df: DataFrame to validate
        name: Name of the DataFrame for error messages

    Raises:
        ValueError: If DataFrame doesn't match expected format
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have DatetimeIndex")

    if df.empty:
        raise ValueError(f"{name} DataFrame is empty")

    if not np.issubdtype(df.values.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")


def validate_signal_dataframe(df: pd.DataFrame, name: str = "signal") -> None:
    """Validate that a DataFrame matches expected signal format.

    Expected format: MultiIndex (timestamp, asset) with 'signal' column.

    Args:
        df: DataFrame to validate
        name: Name of the DataFrame for error messages

    Raises:
        ValueError: If DataFrame doesn't match expected format
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"{name} must have MultiIndex (timestamp, asset)")

    if df.index.nlevels != 2:
        raise ValueError(f"{name} must have 2-level MultiIndex")

    if not isinstance(df.index.get_level_values(0), pd.DatetimeIndex):
        raise ValueError(f"{name} level 0 must be DatetimeIndex")

    if 'signal' not in df.columns:
        raise ValueError(f"{name} must have 'signal' column")

    if df.empty:
        raise ValueError(f"{name} DataFrame is empty")
