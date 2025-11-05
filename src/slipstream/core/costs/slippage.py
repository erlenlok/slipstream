"""
Liquidity-sensitive transaction cost model.

Estimates one-way execution cost (entry OR exit) based on candle features.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union


class TransactionCostModel:
    """
    Transaction cost estimator trained on L2 orderbook data.

    Usage:
        model = TransactionCostModel.load()
        cost = model.estimate_cost(candle_df, position_usd=10000)
    """

    def __init__(self, coefficients: dict, intercept: float, features: list):
        self.coefficients = coefficients
        self.intercept = intercept
        self.features = features

    @classmethod
    def load(cls, model_path: Union[str, Path] = None):
        """Load trained model from JSON."""
        if model_path is None:
            # Default path
            model_path = Path(__file__).parent.parent.parent.parent / "data/features/transaction_cost_model.json"

        with open(model_path, 'r') as f:
            model_data = json.load(f)

        return cls(
            coefficients=model_data['coefficients'],
            intercept=model_data['intercept'],
            features=model_data['features']
        )

    def _compute_features(self, candle: pd.Series, position_usd: float) -> dict:
        """
        Compute features from candle data.

        Args:
            candle: Series with OHLCV columns
            position_usd: Position size in USD

        Returns:
            Dict of feature values
        """
        # Dollar volume
        dollar_volume = candle['close'] * candle['volume']

        # Spread proxy
        spread_proxy = (candle['high'] - candle['low']) / candle['close']

        # Candle range
        candle_range_pct = (candle['high'] - candle['low']) / candle['close']

        # Log return
        log_return = np.log(candle['close'] / candle['open'])
        abs_log_return = np.abs(log_return)

        # Relative position size (square root for market impact)
        relative_position = position_usd / dollar_volume if dollar_volume > 0 else 0
        sqrt_relative_position = np.sqrt(relative_position)

        # Base features
        features = {
            'spread_proxy': spread_proxy,
            'candle_range_pct': candle_range_pct,
            'abs_log_return': abs_log_return,
            'relative_position': relative_position,
            'sqrt_relative_position': sqrt_relative_position,
        }

        # Polynomial features (squared terms)
        features['spread_proxy_sq'] = spread_proxy ** 2
        features['candle_range_pct_sq'] = candle_range_pct ** 2
        features['abs_log_return_sq'] = abs_log_return ** 2
        features['sqrt_relative_position_sq'] = sqrt_relative_position ** 2

        return features

    def estimate_cost(self, candle: Union[pd.Series, pd.DataFrame], position_usd: float) -> Union[float, pd.Series]:
        """
        Estimate one-way transaction cost.

        Args:
            candle: Series or DataFrame with OHLCV data
            position_usd: Position size in USD (absolute value)

        Returns:
            Cost as decimal (e.g., 0.0005 = 5 bps = 0.05%)
        """
        if isinstance(candle, pd.DataFrame):
            # Vectorized computation for DataFrame
            return candle.apply(lambda row: self.estimate_cost(row, position_usd), axis=1)

        # Compute features
        feature_values = self._compute_features(candle, position_usd)

        # Linear combination
        cost = self.intercept
        for feat in self.features:
            cost += self.coefficients[feat] * feature_values[feat]

        # Floor at zero (no negative costs)
        return max(0.0, cost)

    def estimate_cost_bps(self, candle: Union[pd.Series, pd.DataFrame], position_usd: float) -> Union[float, pd.Series]:
        """Convenience method returning cost in basis points."""
        cost_decimal = self.estimate_cost(candle, position_usd)
        if isinstance(cost_decimal, pd.Series):
            return cost_decimal * 10000
        return cost_decimal * 10000


# Convenience function
def estimate_transaction_cost(candle: Union[pd.Series, pd.DataFrame], position_usd: float) -> Union[float, pd.Series]:
    """
    Quick function to estimate transaction cost.

    Args:
        candle: OHLCV candle data
        position_usd: Position size in USD

    Returns:
        One-way cost as decimal (e.g., 0.0005 = 5 bps)

    Example:
        >>> candle = pd.Series({'open': 100, 'high': 102, 'low': 99, 'close': 101, 'volume': 1000})
        >>> cost = estimate_transaction_cost(candle, position_usd=10000)
        >>> print(f"Cost: {cost*10000:.2f} bps")
    """
    model = TransactionCostModel.load()
    return model.estimate_cost(candle, position_usd)
