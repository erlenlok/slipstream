"""
Portfolio optimization and backtesting for Slipstream strategy.

This module implements the beta-neutral portfolio optimization with
transaction costs and discretization constraints.
"""

from .optimizer import (
    optimize_portfolio,
    optimize_portfolio_with_costs,
)
from .costs import (
    compute_transaction_costs,
    TransactionCostModel,
)
from .backtest import (
    run_backtest,
    BacktestResult,
    BacktestConfig,
)

__all__ = [
    "optimize_portfolio",
    "optimize_portfolio_with_costs",
    "compute_transaction_costs",
    "TransactionCostModel",
    "run_backtest",
    "BacktestResult",
    "BacktestConfig",
]
