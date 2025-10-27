"""
Portfolio optimization and backtesting for Slipstream strategy.

This module implements the beta-neutral portfolio optimization with
transaction costs and discretization constraints.
"""

from slipstream.portfolio.optimizer import (
    optimize_portfolio,
    optimize_portfolio_with_costs,
)
from slipstream.portfolio.costs import (
    compute_transaction_costs,
    TransactionCostModel,
)
from slipstream.portfolio.backtest import (
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
