"""
Portfolio optimization with beta-neutrality constraint.

Implements the closed-form solution from strategy_spec.md Section 2.4 and
the cost-aware numerical optimization from Section 4.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Tuple


def optimize_portfolio(
    alpha: np.ndarray,
    beta: np.ndarray,
    S: np.ndarray,
    leverage: float = 1.0,
) -> np.ndarray:
    """
    Compute optimal beta-neutral portfolio (cost-free closed-form solution).

    From strategy_spec.md Section 2.4:

    w* = S^{-1} (α - [(α^T S^{-1} β) / (β^T S^{-1} β)] β)

    Args:
        alpha: Expected returns (N,) - this is α_total = α_price - F_hat
        beta: Market betas (N,)
        S: Total covariance matrix (N, N) - idiosyncratic + funding
        leverage: Portfolio leverage multiplier (default 1.0)

    Returns:
        Optimal weights (N,) summing to leverage, with w^T β = 0
    """
    N = len(alpha)

    if len(beta) != N or S.shape != (N, N):
        raise ValueError(f"Dimension mismatch: alpha={len(alpha)}, beta={len(beta)}, S={S.shape}")

    # Add small ridge to ensure invertibility
    S_reg = S + 1e-8 * np.eye(N)

    try:
        S_inv = np.linalg.inv(S_reg)
    except np.linalg.LinAlgError:
        # If still singular, use pseudo-inverse
        S_inv = np.linalg.pinv(S_reg)

    # Compute lagrange multiplier for beta constraint
    # λ = (α^T S^{-1} β) / (β^T S^{-1} β)
    S_inv_beta = S_inv @ beta
    numerator = alpha @ S_inv_beta
    denominator = beta @ S_inv_beta

    if abs(denominator) < 1e-10:
        # Beta constraint is degenerate - all betas near zero
        # Just use unconstrained solution
        lambda_constraint = 0.0
    else:
        lambda_constraint = numerator / denominator

    # Compute optimal weights: w* = S^{-1} (α - λ β)
    w = S_inv @ (alpha - lambda_constraint * beta)

    # Scale to target leverage
    current_leverage = np.abs(w).sum()
    if current_leverage > 1e-10:
        w = w * (leverage / current_leverage)

    return w


def optimize_portfolio_with_costs(
    alpha: np.ndarray,
    beta: np.ndarray,
    S: np.ndarray,
    w_old: np.ndarray,
    cost_linear: np.ndarray,
    cost_impact: np.ndarray,
    leverage: float = 1.0,
    max_iter: int = 1000,
    ftol: float = 1e-9,
) -> Tuple[np.ndarray, dict]:
    """
    Optimize portfolio with transaction costs and beta constraint.

    From strategy_spec.md Section 4.1:

    max_w [ w^T α - 0.5 w^T S w - C(w - w_old) ]
    subject to: w^T β = 0

    Where C(Δw) = Σ |Δw_i| * c_i + Σ λ_i |Δw_i|^1.5

    Args:
        alpha: Expected returns (N,)
        beta: Market betas (N,)
        S: Total covariance matrix (N, N)
        w_old: Current portfolio weights (N,)
        cost_linear: Linear cost coefficients (N,) - fee rates
        cost_impact: Market impact coefficients (N,) - λ_i
        leverage: Target portfolio leverage
        max_iter: Maximum iterations for optimizer
        ftol: Function tolerance for convergence

    Returns:
        Tuple of (optimal_weights, info_dict)
    """
    N = len(alpha)

    # Start from cost-free solution as initial guess
    w_init = optimize_portfolio(alpha, beta, S, leverage=leverage)

    def objective(w):
        """Negative of objective function (we minimize instead of maximize)."""
        # Expected return term
        ret = w @ alpha

        # Risk term (variance)
        risk = 0.5 * w @ S @ w

        # Transaction cost
        dw = w - w_old
        cost = (
            np.sum(np.abs(dw) * cost_linear) +
            np.sum(cost_impact * np.abs(dw) ** 1.5)
        )

        # Return negative (since we minimize)
        return -(ret - risk - cost)

    def gradient(w):
        """Gradient of negative objective."""
        # d/dw [w^T α] = α
        grad_ret = alpha

        # d/dw [0.5 w^T S w] = S w
        grad_risk = S @ w

        # d/dw [C(w - w_old)]
        dw = w - w_old
        sign_dw = np.sign(dw)
        grad_cost = (
            sign_dw * cost_linear +
            1.5 * cost_impact * sign_dw * np.abs(dw) ** 0.5
        )

        # Return negative gradient
        return -(grad_ret - grad_risk - grad_cost)

    # Constraint: w^T β = 0
    constraints = {
        'type': 'eq',
        'fun': lambda w: w @ beta,
        'jac': lambda w: beta
    }

    # Leverage constraint: ||w||_1 = leverage
    leverage_constraint = {
        'type': 'eq',
        'fun': lambda w: np.abs(w).sum() - leverage,
        'jac': lambda w: np.sign(w)
    }

    # Solve
    result = minimize(
        objective,
        w_init,
        method='SLSQP',
        jac=gradient,
        constraints=[constraints, leverage_constraint],
        options={'maxiter': max_iter, 'ftol': ftol}
    )

    if not result.success:
        print(f"⚠ Optimization warning: {result.message}")

    w_opt = result.x

    # Compute final metrics
    ret = w_opt @ alpha
    risk = 0.5 * w_opt @ S @ w_opt
    dw = w_opt - w_old
    cost = (
        np.sum(np.abs(dw) * cost_linear) +
        np.sum(cost_impact * np.abs(dw) ** 1.5)
    )

    info = {
        'success': result.success,
        'message': result.message,
        'iterations': result.nit,
        'expected_return': ret,
        'variance': 2 * risk,  # Convert back from 0.5 * var
        'transaction_cost': cost,
        'net_objective': ret - risk - cost,
        'beta_exposure': w_opt @ beta,
        'leverage': np.abs(w_opt).sum(),
        'turnover': np.abs(dw).sum(),
        'n_positions': np.sum(np.abs(w_opt) > 1e-6),
    }

    return w_opt, info


def round_to_lots(
    w_ideal: np.ndarray,
    beta: np.ndarray,
    S: np.ndarray,
    capital: float,
    min_trade_size: np.ndarray,
    beta_tolerance: float = 0.01,
    max_iterations: int = 100,
) -> np.ndarray:
    """
    Round ideal continuous weights to discrete lots with beta repair.

    From strategy_spec.md Section 4.1.1:
    1. Round to nearest integer lots
    2. Repair beta drift via greedy search

    Args:
        w_ideal: Ideal continuous weights (N,)
        beta: Market betas (N,)
        S: Covariance matrix for tracking error calculation
        capital: Total portfolio capital
        min_trade_size: Minimum trade size per asset (N,) in dollars
        beta_tolerance: Maximum acceptable |w^T β|
        max_iterations: Maximum repair iterations

    Returns:
        Discrete weights (N,) satisfying beta constraint
    """
    N = len(w_ideal)

    # Step 1: Convert weights to lots
    dollar_positions = w_ideal * capital
    ideal_lots = dollar_positions / min_trade_size

    # Step 2: Round to nearest integer
    lots = np.round(ideal_lots).astype(int)

    # Step 3: Convert back to weights
    w_rounded = (lots * min_trade_size) / capital

    # Step 4: Repair beta drift
    beta_exposure = w_rounded @ beta

    iteration = 0
    while abs(beta_exposure) > beta_tolerance and iteration < max_iterations:
        # Find asset that gives best beta reduction per tracking error
        best_asset = None
        best_score = -np.inf
        best_delta = 0

        for i in range(N):
            # Try adding or removing one lot
            for delta_lots in [-1, 1]:
                # Skip if it would create a position in wrong direction
                if lots[i] + delta_lots == 0:
                    continue

                # Compute change in beta
                delta_w = delta_lots * min_trade_size[i] / capital
                delta_beta = delta_w * beta[i]
                new_beta = beta_exposure + delta_beta

                # Check if this moves us closer to zero
                if abs(new_beta) >= abs(beta_exposure):
                    continue

                # Compute tracking error increase
                w_delta = np.zeros(N)
                w_delta[i] = delta_w
                tracking_error = np.sqrt(w_delta @ S @ w_delta)

                # Score: beta reduction per unit tracking error
                beta_reduction = abs(beta_exposure) - abs(new_beta)
                score = beta_reduction / (tracking_error + 1e-10)

                if score > best_score:
                    best_score = score
                    best_asset = i
                    best_delta = delta_lots

        if best_asset is None:
            break

        # Apply best adjustment
        lots[best_asset] += best_delta
        w_rounded = (lots * min_trade_size) / capital
        beta_exposure = w_rounded @ beta

        iteration += 1

    if abs(beta_exposure) > beta_tolerance:
        print(f"⚠ Beta repair: Could not achieve target tolerance. Final |β| = {abs(beta_exposure):.4f}")

    return w_rounded


__all__ = [
    "optimize_portfolio",
    "optimize_portfolio_with_costs",
    "round_to_lots",
]
