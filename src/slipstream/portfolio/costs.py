"""
Transaction cost modeling for portfolio optimization.

Implements the cost model from strategy_spec.md Section 3.3:
C(Δw) = Σ |Δw_i| * fee_rate_i + Σ λ_i |Δw_i|^1.5
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransactionCostModel:
    """
    Transaction cost parameters for each asset.

    Attributes:
        fee_rate: Fixed fee rate (e.g., 0.0002 for 2 basis points)
        impact_coef: Market impact coefficient λ_i
        min_trade_size: Minimum trade size in dollars
    """

    fee_rate: np.ndarray  # (N,) - linear cost per unit traded
    impact_coef: np.ndarray  # (N,) - market impact λ_i
    min_trade_size: np.ndarray  # (N,) - minimum trade size in dollars

    def __post_init__(self):
        """Validate dimensions."""
        N = len(self.fee_rate)
        if len(self.impact_coef) != N or len(self.min_trade_size) != N:
            raise ValueError("All cost arrays must have same length")

    @classmethod
    def from_liquidity_metrics(
        cls,
        assets: list[str],
        liquidity_df: pd.DataFrame,
        base_fee_rate: float = 0.0002,  # 2 bps
        impact_scale: float = 0.001,  # Scale factor for impact
    ) -> TransactionCostModel:
        """
        Create cost model from liquidity metrics.

        Args:
            assets: List of asset symbols
            liquidity_df: DataFrame with columns [asset, avg_volume_24h, avg_spread_bps]
            base_fee_rate: Base exchange fee rate (default 2 bps)
            impact_scale: Scaling factor for market impact

        Returns:
            TransactionCostModel instance
        """
        N = len(assets)

        fee_rate = np.full(N, base_fee_rate)
        impact_coef = np.zeros(N)
        min_trade_size = np.full(N, 10.0)  # Default $10 min trade

        # Map liquidity metrics to cost parameters
        for i, asset in enumerate(assets):
            if asset in liquidity_df.index:
                metrics = liquidity_df.loc[asset]

                # Higher spread → higher fee
                spread_bps = metrics.get('avg_spread_bps', 2.0)
                fee_rate[i] = base_fee_rate + spread_bps / 10000

                # Lower volume → higher impact
                volume_24h = metrics.get('avg_volume_24h', 1e6)
                # λ_i ∝ 1 / sqrt(volume)
                impact_coef[i] = impact_scale / np.sqrt(volume_24h + 1e4)

                # Set min trade size based on liquidity
                min_trade_size[i] = max(10.0, volume_24h / 10000)

        return cls(
            fee_rate=fee_rate,
            impact_coef=impact_coef,
            min_trade_size=min_trade_size,
        )

    @classmethod
    def create_default(cls, n_assets: int) -> TransactionCostModel:
        """
        Create default cost model with reasonable assumptions.

        Args:
            n_assets: Number of assets

        Returns:
            TransactionCostModel with default parameters
        """
        return cls(
            fee_rate=np.full(n_assets, 0.0002),  # 2 bps
            impact_coef=np.full(n_assets, 0.0001),  # Small impact
            min_trade_size=np.full(n_assets, 10.0),  # $10 min
        )


def compute_transaction_costs(
    w_new: np.ndarray,
    w_old: np.ndarray,
    cost_model: TransactionCostModel,
    capital: float = 1.0,
) -> dict:
    """
    Compute transaction costs for portfolio rebalance.

    Args:
        w_new: New portfolio weights (N,)
        w_old: Old portfolio weights (N,)
        cost_model: Transaction cost parameters
        capital: Total portfolio capital

    Returns:
        Dictionary with cost breakdown
    """
    dw = w_new - w_old
    dollar_trade = np.abs(dw) * capital

    # Linear fee component
    fee_cost = np.sum(np.abs(dw) * cost_model.fee_rate)

    # Market impact component (power = 1.5)
    impact_cost = np.sum(cost_model.impact_coef * np.abs(dw) ** 1.5)

    # Total cost
    total_cost = fee_cost + impact_cost

    # Per-asset breakdown
    asset_costs = (
        np.abs(dw) * cost_model.fee_rate +
        cost_model.impact_coef * np.abs(dw) ** 1.5
    )

    return {
        'total_cost': total_cost,
        'fee_cost': fee_cost,
        'impact_cost': impact_cost,
        'turnover': np.abs(dw).sum(),
        'dollar_turnover': dollar_trade.sum(),
        'n_trades': np.sum(np.abs(dw) > 1e-6),
        'asset_costs': asset_costs,
    }


def estimate_liquidity_adjusted_costs(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    lookback: int = 30 * 24,  # 30 days in hours
) -> pd.DataFrame:
    """
    Estimate cost parameters from historical returns and volume.

    Args:
        returns: Historical returns (wide format, hourly)
        volume: Historical volume (wide format, hourly)
        lookback: Lookback period in hours

    Returns:
        DataFrame with columns [asset, avg_volume_24h, avg_spread_bps, liquidity_score]
    """
    # Compute rolling 24-hour volume
    volume_24h = volume.rolling(24).sum()

    # Estimate spread from volatility (Hasbrouck 2009 model)
    volatility = returns.rolling(24).std()
    spread_bps = volatility * 10000 * 2  # Rough bid-ask spread estimate

    # Aggregate metrics
    results = []
    for asset in returns.columns:
        avg_vol = volume_24h[asset].iloc[-lookback:].mean()
        avg_spread = spread_bps[asset].iloc[-lookback:].mean()

        # Liquidity score: higher is more liquid
        liquidity_score = avg_vol / (avg_spread + 1e-6)

        results.append({
            'asset': asset,
            'avg_volume_24h': avg_vol,
            'avg_spread_bps': avg_spread,
            'liquidity_score': liquidity_score,
        })

    df = pd.DataFrame(results).set_index('asset')
    return df


__all__ = [
    "TransactionCostModel",
    "compute_transaction_costs",
    "estimate_liquidity_adjusted_costs",
]
