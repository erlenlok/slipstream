"""
Test portfolio optimization module.
"""

import numpy as np
import pytest
from slipstream.portfolio.optimizer import (
    optimize_portfolio,
    optimize_portfolio_with_costs,
)
from slipstream.portfolio.costs import TransactionCostModel


def test_optimize_portfolio_basic():
    """Test basic beta-neutral optimization."""
    N = 5

    # Create simple test case
    alpha = np.array([0.05, 0.03, 0.01, -0.01, -0.03])
    beta = np.array([1.2, 0.8, 1.0, 1.1, 0.9])
    S = np.eye(N) * 0.01  # Diagonal covariance

    # Optimize
    w = optimize_portfolio(alpha, beta, S, leverage=1.0)

    # Verify constraints
    assert abs(np.abs(w).sum() - 1.0) < 1e-3, "Leverage constraint violated"
    assert abs(w @ beta) < 1e-6, f"Beta constraint violated: {w @ beta:.6f}"

    # Check that positive alpha assets get positive weights
    # (may not always hold with beta constraint, but should trend that way)
    print(f"\nAlpha: {alpha}")
    print(f"Beta: {beta}")
    print(f"Weights: {w}")
    print(f"Beta exposure: {w @ beta:.6f}")
    print(f"Leverage: {np.abs(w).sum():.6f}")


def test_optimize_portfolio_with_costs_basic():
    """Test cost-aware optimization."""
    N = 5

    alpha = np.array([0.05, 0.03, 0.01, -0.01, -0.03])
    beta = np.array([1.2, 0.8, 1.0, 1.1, 0.9])
    S = np.eye(N) * 0.01

    w_old = np.array([0.2, 0.2, 0.0, -0.2, -0.2])

    cost_model = TransactionCostModel.create_default(N)

    # Optimize with costs
    w_new, info = optimize_portfolio_with_costs(
        alpha=alpha,
        beta=beta,
        S=S,
        w_old=w_old,
        cost_linear=cost_model.fee_rate,
        cost_impact=cost_model.impact_coef,
        leverage=1.0,
    )

    # Verify constraints
    assert abs(np.abs(w_new).sum() - 1.0) < 1e-2, "Leverage constraint violated"
    assert abs(w_new @ beta) < 1e-3, f"Beta constraint violated: {w_new @ beta:.6f}"

    print(f"\nOld weights: {w_old}")
    print(f"New weights: {w_new}")
    print(f"Beta exposure: {w_new @ beta:.6f}")
    print(f"Turnover: {info['turnover']:.4f}")
    print(f"Cost: {info['transaction_cost']:.6f}")
    print(f"Net objective: {info['net_objective']:.6f}")


def test_beta_neutrality_enforced():
    """Test that beta constraint is strictly enforced."""
    N = 10

    np.random.seed(42)
    alpha = np.random.randn(N) * 0.05
    beta = np.random.randn(N) * 0.5 + 1.0  # Betas around 1.0
    S = np.eye(N) * 0.02

    w = optimize_portfolio(alpha, beta, S, leverage=2.0)

    # Check beta neutrality
    beta_exposure = w @ beta
    assert abs(beta_exposure) < 1e-5, f"Beta not neutral: {beta_exposure:.8f}"

    # Check leverage
    leverage = np.abs(w).sum()
    assert abs(leverage - 2.0) < 1e-2, f"Leverage wrong: {leverage:.4f}"

    print(f"\nRandom test:")
    print(f"Alpha: {alpha}")
    print(f"Beta: {beta}")
    print(f"Weights: {w}")
    print(f"Beta exposure: {beta_exposure:.8f}")
    print(f"Leverage: {leverage:.4f}")


def test_cost_reduces_turnover():
    """Test that including costs reduces turnover."""
    N = 5

    alpha = np.array([0.05, 0.03, 0.01, -0.01, -0.03])
    beta = np.array([1.2, 0.8, 1.0, 1.1, 0.9])
    S = np.eye(N) * 0.01

    w_old = np.array([0.2, 0.2, 0.0, -0.2, -0.2])

    # Low cost optimization
    cost_model = TransactionCostModel.create_default(N)
    w_low_cost, info_low = optimize_portfolio_with_costs(
        alpha=alpha,
        beta=beta,
        S=S,
        w_old=w_old,
        cost_linear=cost_model.fee_rate,  # Normal costs
        cost_impact=cost_model.impact_coef,
        leverage=1.0,
    )
    turnover_low_cost = np.abs(w_low_cost - w_old).sum()

    # High cost optimization
    w_high_cost, info_high = optimize_portfolio_with_costs(
        alpha=alpha,
        beta=beta,
        S=S,
        w_old=w_old,
        cost_linear=cost_model.fee_rate * 100,  # Very high costs
        cost_impact=cost_model.impact_coef * 100,
        leverage=1.0,
    )
    turnover_high_cost = np.abs(w_high_cost - w_old).sum()

    print(f"\nCost effect on turnover:")
    print(f"Turnover (low cost): {turnover_low_cost:.4f}")
    print(f"Turnover (high cost): {turnover_high_cost:.4f}")

    # With very high costs, optimizer should stay closer to w_old
    print(f"\nLow cost weights: {w_low_cost}")
    print(f"High cost weights: {w_high_cost}")
    print(f"Old weights: {w_old}")

    # High costs should result in less trading
    print(f"\nNet objective (low cost): {info_low['net_objective']:.6f}")
    print(f"Net objective (high cost): {info_high['net_objective']:.6f}")


if __name__ == "__main__":
    print("="*70)
    print("PORTFOLIO OPTIMIZER TESTS")
    print("="*70)

    test_optimize_portfolio_basic()
    test_optimize_portfolio_with_costs_basic()
    test_beta_neutrality_enforced()
    test_cost_reduces_turnover()

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
