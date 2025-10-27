"""
Backtesting framework for Slipstream strategy.

Implements walk-forward simulation from strategy_spec.md Section 4.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

from slipstream.alpha.data_prep import BASE_INTERVAL_HOURS

from slipstream.portfolio.optimizer import (
    optimize_portfolio,
    optimize_portfolio_with_costs,
    round_to_lots,
)
from slipstream.portfolio.costs import (
    TransactionCostModel,
    compute_transaction_costs,
)


@dataclass
class BacktestConfig:
    """Configuration for backtest simulation."""

    H: int  # Rebalancing period in hours
    start_date: str  # Start date (YYYY-MM-DD)
    end_date: str  # End date (YYYY-MM-DD)
    initial_capital: float = 1e6  # Starting capital ($1M default)
    leverage: float = 1.0  # Target leverage
    use_costs: bool = True  # Include transaction costs
    use_discrete_lots: bool = False  # Round to discrete lots
    min_history_hours: int = 720  # Min history before first trade (30 days)
    beta_tolerance: float = 0.01  # Max acceptable |w^T β|


@dataclass
class BacktestResult:
    """Results from backtest simulation."""

    config: BacktestConfig
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame  # Time series of weights
    trades: pd.DataFrame  # Record of each rebalance
    metrics: dict  # Summary statistics

    def sharpe_ratio(self, annualization_factor: float = 365.25 * 24) -> float:
        """Compute annualized Sharpe ratio."""
        mean_ret = self.returns.mean() * annualization_factor
        std_ret = self.returns.std() * np.sqrt(annualization_factor)
        return mean_ret / std_ret if std_ret > 0 else 0.0

    def max_drawdown(self) -> float:
        """Compute maximum drawdown."""
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax
        return drawdown.min()

    def total_return(self) -> float:
        """Compute total return."""
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1

    def summary(self) -> dict:
        """Generate summary statistics."""
        return {
            'total_return': self.total_return(),
            'sharpe_ratio': self.sharpe_ratio(),
            'max_drawdown': self.max_drawdown(),
            'final_capital': self.equity_curve.iloc[-1],
            'n_rebalances': len(self.trades),
            'avg_turnover': self.trades['turnover'].mean(),
            'total_costs': self.trades['cost'].sum(),
            'avg_n_positions': self.positions.apply(lambda x: (np.abs(x) > 1e-4).sum(), axis=1).mean(),
        }


def run_backtest(
    config: BacktestConfig,
    alpha_price: pd.DataFrame,  # (timestamp, asset) -> predicted return
    alpha_funding: pd.DataFrame,  # (timestamp, asset) -> predicted funding
    beta: pd.DataFrame,  # (timestamp, asset) -> market beta
    S: pd.DataFrame,  # (timestamp, asset, asset) -> covariance matrix OR dict[timestamp -> np.ndarray]
    realized_returns: pd.DataFrame,  # (timestamp, asset) -> actual returns
    realized_funding: pd.DataFrame,  # (timestamp, asset) -> actual funding
    cost_model: Optional[TransactionCostModel] = None,
    vol_scale: Optional[pd.DataFrame] = None,  # For rescaling normalized predictions
) -> BacktestResult:
    """
    Run walk-forward backtest simulation.

    Args:
        config: Backtest configuration
        alpha_price: Predicted price returns (normalized)
        alpha_funding: Predicted funding rates (normalized)
        beta: Market beta exposures
        S: Idiosyncratic covariance matrix (or dict mapping timestamp to matrix)
        realized_returns: Actual realized returns
        realized_funding: Actual realized funding payments
        cost_model: Transaction cost parameters
        vol_scale: Volatility scaling factors for denormalizing predictions

    Returns:
        BacktestResult with full simulation history
    """
    # Combine alpha: α_total = α_price - F_hat
    alpha_total = alpha_price - alpha_funding

    # Get timestamps and assets
    timestamps = pd.to_datetime(alpha_total.index).unique()
    timestamps = timestamps[(timestamps >= config.start_date) & (timestamps <= config.end_date)]
    assets = alpha_total.columns

    N = len(assets)

    # Initialize
    if cost_model is None:
        cost_model = TransactionCostModel.create_default(N)

    capital = config.initial_capital
    w_current = np.zeros(N)  # Start with no positions

    # Storage
    equity_history = []
    returns_history = []
    positions_history = []
    trades_history = []

    print(f"\n{'='*70}")
    print(f"BACKTEST SIMULATION (H={config.H} hours)")
    print(f"{'='*70}")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Initial capital: ${config.initial_capital:,.0f}")
    print(f"Leverage: {config.leverage}")
    print(f"Rebalancing frequency: {config.H} hours")
    print(f"Include costs: {config.use_costs}")
    print(f"Discrete lots: {config.use_discrete_lots}")
    print(f"{'='*70}\n")

    # Walk forward
    rebalance_timestamps = timestamps
    prev_timestamp = None

    for i, t in enumerate(rebalance_timestamps):
        if i < config.min_history_hours // config.H:
            # Skip until we have enough history
            continue

        try:
            # Get current signals
            alpha_t = alpha_total.loc[t].values
            beta_t = beta.loc[t].values

            # Get covariance matrix
            if isinstance(S, dict):
                S_t = S[t]
            else:
                # Assume S is pre-computed per timestamp
                # This is a simplification - in practice would compute from residuals
                S_t = np.diag(realized_returns.loc[:t].iloc[-720:].std().values ** 2)

            # Optimize portfolio
            if config.use_costs and prev_timestamp is not None:
                w_new, opt_info = optimize_portfolio_with_costs(
                    alpha=alpha_t,
                    beta=beta_t,
                    S=S_t,
                    w_old=w_current,
                    cost_linear=cost_model.fee_rate,
                    cost_impact=cost_model.impact_coef,
                    leverage=config.leverage,
                )
            else:
                w_new = optimize_portfolio(
                    alpha=alpha_t,
                    beta=beta_t,
                    S=S_t,
                    leverage=config.leverage,
                )
                opt_info = {'transaction_cost': 0.0}

            # Apply discretization if requested
            if config.use_discrete_lots:
                w_new = round_to_lots(
                    w_ideal=w_new,
                    beta=beta_t,
                    S=S_t,
                    capital=capital,
                    min_trade_size=cost_model.min_trade_size,
                    beta_tolerance=config.beta_tolerance,
                )

            # Compute realized P&L over next H hours
            if prev_timestamp is not None:
                # Get returns over holding period
                period_returns = realized_returns.loc[prev_timestamp:t].sum()
                period_funding = realized_funding.loc[prev_timestamp:t].sum()

                # P&L from positions
                pnl_returns = w_current @ period_returns.reindex(assets, fill_value=0).values
                pnl_funding = -w_current @ period_funding.reindex(assets, fill_value=0).values  # Paying funding

                # Transaction costs from rebalance
                cost_dict = compute_transaction_costs(w_new, w_current, cost_model, capital)
                cost_total = cost_dict['total_cost'] if config.use_costs else 0.0

                # Net P&L
                net_pnl = pnl_returns + pnl_funding - cost_total

                # Update capital
                capital += net_pnl * capital
                period_return = net_pnl

                # Record
                returns_history.append(period_return)
                equity_history.append(capital)

                # Record trade
                trades_history.append({
                    'timestamp': t,
                    'capital': capital,
                    'pnl_gross': (pnl_returns + pnl_funding) * capital,
                    'cost': cost_total * capital,
                    'pnl_net': net_pnl * capital,
                    'turnover': cost_dict['turnover'],
                    'n_positions': np.sum(np.abs(w_new) > 1e-4),
                    'beta_exposure': w_new @ beta_t,
                    'leverage': np.abs(w_new).sum(),
                })

            else:
                # First rebalance - no P&L yet
                equity_history.append(capital)
                returns_history.append(0.0)

            # Update position
            w_current = w_new
            positions_history.append(w_new.copy())

            prev_timestamp = t

            # Progress
            if (i + 1) % 50 == 0:
                sharpe = np.mean(returns_history) / (np.std(returns_history) + 1e-10) * np.sqrt(365.25 * 24 / config.H)
                print(f"  {t.date()} | Capital: ${capital:,.0f} | Sharpe: {sharpe:.2f} | Positions: {np.sum(np.abs(w_current) > 1e-4)}")

        except Exception as e:
            print(f"⚠ Error at {t}: {e}")
            continue

    # Create result
    result_timestamps = rebalance_timestamps[:len(equity_history)]

    result = BacktestResult(
        config=config,
        equity_curve=pd.Series(equity_history, index=result_timestamps),
        returns=pd.Series(returns_history, index=result_timestamps),
        positions=pd.DataFrame(positions_history, index=result_timestamps, columns=assets),
        trades=pd.DataFrame(trades_history),
        metrics={},
    )

    result.metrics = result.summary()

    print(f"\n{'='*70}")
    print("BACKTEST RESULTS")
    print(f"{'='*70}")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"{key:.<30} {value:>15.2f}")
        else:
            print(f"{key:.<30} {value:>15}")
    print(f"{'='*70}\n")

    return result


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
]
