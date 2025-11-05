"""
Risk model utilities for portfolio optimization.

Implements covariance estimation and beta calculation from strategy_spec.md Section 3.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

from ..signals import compute_idiosyncratic_returns


def compute_idiosyncratic_covariance(
    returns: pd.DataFrame,
    loadings: pd.Series,
    market_factor: pd.Series,
    lookback: int = 720,  # 30 days @ 4H
    min_periods: int = 180,  # 7.5 days minimum
) -> pd.DataFrame:
    """
    Compute covariance matrix of idiosyncratic returns.

    Args:
        returns: Historical returns (wide format)
        loadings: PCA loadings (long format with MultiIndex)
        market_factor: Market factor returns
        lookback: Rolling window in hours
        min_periods: Minimum periods for valid estimate

    Returns:
        Covariance matrix of idiosyncratic returns
    """
    # Compute idiosyncratic returns
    idio_returns = compute_idiosyncratic_returns(returns, loadings, market_factor)

    # Convert to wide format
    idio_wide = idio_returns.unstack(level='asset')

    # Compute rolling covariance on recent window
    recent = idio_wide.iloc[-lookback:]

    if len(recent) < min_periods:
        raise ValueError(f"Insufficient data: {len(recent)} < {min_periods}")

    # Sample covariance with Ledoit-Wolf shrinkage
    cov_matrix = recent.cov()

    # Shrinkage toward diagonal
    variance = np.diag(cov_matrix)
    target = np.diag(variance)
    shrinkage = 0.1  # 10% shrinkage

    cov_shrunk = (1 - shrinkage) * cov_matrix + shrinkage * target

    return cov_shrunk


def compute_funding_covariance(
    funding_rates: pd.DataFrame,
    lookback: int = 720,
    min_periods: int = 180,
) -> pd.DataFrame:
    """
    Compute covariance matrix of funding rate innovations.

    Args:
        funding_rates: Historical funding rates (wide format)
        lookback: Rolling window in hours
        min_periods: Minimum periods for valid estimate

    Returns:
        Covariance matrix of funding innovations
    """
    # Use recent data
    recent = funding_rates.iloc[-lookback:]

    if len(recent) < min_periods:
        raise ValueError(f"Insufficient data: {len(recent)} < {min_periods}")

    # Compute innovations (first differences)
    innovations = recent.diff()

    # Sample covariance
    cov_matrix = innovations.cov()

    # Shrinkage
    variance = np.diag(cov_matrix)
    target = np.diag(variance)
    shrinkage = 0.1

    cov_shrunk = (1 - shrinkage) * cov_matrix + shrinkage * target

    return cov_shrunk


def compute_total_covariance(
    returns: pd.DataFrame,
    funding_rates: pd.DataFrame,
    loadings: pd.Series,
    market_factor: pd.Series,
    lookback: int = 720,
) -> pd.DataFrame:
    """
    Compute total covariance matrix: S_total = S_price + S_funding - 2*C_cross.

    From strategy_spec.md Section 3.2.

    Args:
        returns: Historical returns (wide format)
        funding_rates: Historical funding rates (wide format)
        loadings: PCA loadings (long format)
        market_factor: Market factor returns
        lookback: Rolling window in hours

    Returns:
        Total covariance matrix for optimization
    """
    # Compute individual covariances
    S_price = compute_idiosyncratic_covariance(
        returns, loadings, market_factor, lookback
    )

    S_funding = compute_funding_covariance(
        funding_rates, lookback
    )

    # Compute cross-covariance
    # (simplified - assumes independence for now)
    C_cross = np.zeros_like(S_price)

    # Total covariance: S_total = S_price + S_funding - 2*C_cross
    S_total = S_price.values + S_funding.values - 2 * C_cross

    return pd.DataFrame(S_total, index=S_price.index, columns=S_price.columns)


def estimate_beta_from_loadings(
    loadings: pd.DataFrame,
    window: int = 720,
) -> pd.DataFrame:
    """
    Estimate market beta from PCA loadings.

    In the PCA framework, β_i = loading_i is the beta exposure.

    Args:
        loadings: PCA loadings (wide format, timestamp x asset)
        window: Rolling window for averaging betas

    Returns:
        Beta estimates (wide format)
    """
    # Use rolling average of loadings as beta estimate
    beta = loadings.rolling(window, min_periods=180).mean()

    return beta


def compute_portfolio_variance(
    w: np.ndarray,
    S: np.ndarray,
) -> float:
    """
    Compute portfolio variance: w^T S w.

    Args:
        w: Portfolio weights (N,)
        S: Covariance matrix (N, N)

    Returns:
        Portfolio variance
    """
    return w @ S @ w


def compute_portfolio_volatility(
    w: np.ndarray,
    S: np.ndarray,
    annualization_factor: float = np.sqrt(365.25 * 24),
) -> float:
    """
    Compute annualized portfolio volatility.

    Args:
        w: Portfolio weights (N,)
        S: Covariance matrix (N, N) at base frequency
        annualization_factor: Factor for annualization

    Returns:
        Annualized volatility
    """
    variance = compute_portfolio_variance(w, S)
    return np.sqrt(variance) * annualization_factor


def decompose_risk(
    w: np.ndarray,
    S: np.ndarray,
    asset_names: list[str],
) -> pd.DataFrame:
    """
    Decompose portfolio risk by asset contribution.

    Marginal contribution to risk (MCR):
        MCR_i = (S w)_i / sqrt(w^T S w)

    Component contribution:
        Component_i = w_i * MCR_i

    Args:
        w: Portfolio weights (N,)
        S: Covariance matrix (N, N)
        asset_names: Asset identifiers

    Returns:
        DataFrame with risk decomposition
    """
    N = len(w)

    # Total portfolio volatility
    portfolio_var = w @ S @ w
    portfolio_vol = np.sqrt(portfolio_var)

    if portfolio_vol < 1e-10:
        # Zero portfolio
        return pd.DataFrame({
            'asset': asset_names,
            'weight': w,
            'mcr': np.zeros(N),
            'contribution': np.zeros(N),
            'pct_contribution': np.zeros(N),
        })

    # Marginal contribution to risk
    S_w = S @ w
    mcr = S_w / portfolio_vol

    # Component contribution
    contribution = w * mcr

    # Percentage contribution
    pct_contribution = contribution / portfolio_vol * 100

    return pd.DataFrame({
        'asset': asset_names,
        'weight': w,
        'mcr': mcr,
        'contribution': contribution,
        'pct_contribution': pct_contribution,
    }).sort_values('pct_contribution', ascending=False, key=abs)


__all__ = [
    "compute_idiosyncratic_covariance",
    "compute_funding_covariance",
    "compute_total_covariance",
    "estimate_beta_from_loadings",
    "compute_portfolio_variance",
    "compute_portfolio_volatility",
    "decompose_risk",
]
