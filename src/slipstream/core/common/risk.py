"""
Risk management utilities for VAR-based portfolio construction.

Implements parametric VAR with RIE (Rotationally-Invariant Estimator) covariance
cleaning based on Random Matrix Theory.

References:
    Bun, J., Bouchaud, J.P., and Potters, M. (2016).
    "Cleaning large correlation matrices: tools from Random Matrix Theory"
    arXiv:1610.08104
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg


def compute_daily_returns(
    returns_4h: pd.DataFrame,
    window: int = 6,
) -> pd.DataFrame:
    """
    Resample 4-hour log returns to daily frequency.

    Args:
        returns_4h: Wide DataFrame of 4-hour log returns (timestamp index, asset columns)
        window: Number of 4h periods per day (default 6, since 6*4h = 24h)

    Returns:
        DataFrame of daily log returns (sum of consecutive 4h returns)
    """
    return returns_4h.rolling(window=window).sum()


def estimate_covariance_rie(
    returns: pd.DataFrame,
    lookback_days: int = 60,
    fallback_diagonal: bool = True,
) -> pd.DataFrame:
    """
    Estimate covariance matrix using debiased RIE from Random Matrix Theory.

    This implementation follows the algorithm from Bun, Bouchaud & Potters (2016).
    For regime where q = N/T approaches 1, empirical eigenvalues are heavily
    corrupted by Marchenko-Pastur noise. RIE debiases eigenvalues to produce
    optimal shrinkage estimators.

    Args:
        returns: Wide DataFrame of returns (T rows, N columns)
        lookback_days: Number of days to use for estimation
        fallback_diagonal: If True, use diagonal covariance when T < N

    Returns:
        DataFrame of cleaned covariance matrix (N x N)
    """
    # Use recent data
    recent = returns.iloc[-lookback_days:].dropna(axis=1, how="all")

    T, N = recent.shape

    # Fallback to diagonal if insufficient data
    if T < N and fallback_diagonal:
        variances = recent.var()
        cov_matrix = pd.DataFrame(
            np.diag(variances),
            index=recent.columns,
            columns=recent.columns,
        )
        return cov_matrix

    # Compute empirical correlation matrix (standardize first)
    standardized = (recent - recent.mean()) / recent.std()
    C_emp = standardized.T @ standardized / T

    # Apply RIE eigenvalue cleaning
    eigenvalues, eigenvectors = linalg.eigh(C_emp.values)

    # Compute q = N/T ratio
    q = N / T

    # Clean eigenvalues using RIE debiasing
    eigenvalues_clean = _debias_eigenvalues_rie(eigenvalues, q)

    # Reconstruct cleaned correlation matrix
    C_clean = eigenvectors @ np.diag(eigenvalues_clean) @ eigenvectors.T

    # Convert back to covariance (rescale by standard deviations)
    std = recent.std()
    D = np.diag(std)
    cov_clean = D @ C_clean @ D

    # Ensure symmetry (numerical stability)
    cov_clean = (cov_clean + cov_clean.T) / 2

    return pd.DataFrame(cov_clean, index=recent.columns, columns=recent.columns)


def _debias_eigenvalues_rie(eigenvalues: np.ndarray, q: float) -> np.ndarray:
    """
    Debias eigenvalues using simplified RIE from Random Matrix Theory.

    This uses a practical shrinkage approach based on the Marchenko-Pastur
    distribution rather than the full RIE inverse transform.

    Args:
        eigenvalues: Empirical eigenvalues (sorted)
        q: Ratio N/T where N=assets, T=time samples

    Returns:
        Cleaned eigenvalues
    """
    # Marchenko-Pastur edges
    lambda_plus = (1 + np.sqrt(q)) ** 2
    lambda_minus = max((1 - np.sqrt(q)) ** 2, 0)  # Can't be negative

    cleaned = np.zeros_like(eigenvalues)

    for i, lam in enumerate(eigenvalues):
        if lam <= lambda_minus:
            # Below MP lower edge: pure noise, set to zero
            cleaned[i] = 0.0
        elif lambda_minus < lam <= lambda_plus:
            # Inside MP bulk: shrink toward zero (noise-dominated)
            # Use linear shrinkage: λ_clean = λ_emp * (λ_emp - λ_minus) / (λ_plus - λ_minus)
            # This progressively shrinks eigenvalues, with those near λ_minus → 0
            shrinkage_factor = (lam - lambda_minus) / (lambda_plus - lambda_minus)
            cleaned[i] = lam * shrinkage_factor * 0.5  # Additional dampening
        else:
            # Above MP upper edge: signal eigenvalue
            # Apply inverse Marchenko-Pastur mapping
            # λ_true ≈ λ_emp / (1 + sqrt(q))
            # More conservative than the full inverse to avoid instability
            cleaned[i] = lam / (1 + np.sqrt(q))

    return cleaned


def compute_portfolio_var(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame,
    confidence: float = 0.95,
) -> float:
    """
    Compute parametric Value-at-Risk for a portfolio.

    Uses the parametric formula: VAR = z_α * sqrt(w^T Σ w)

    Args:
        weights: Portfolio weights (array or dict matching cov_matrix columns)
        cov_matrix: Covariance matrix (N x N DataFrame)
        confidence: Confidence level (default 0.95 for 95% VAR)

    Returns:
        One-day VAR at specified confidence level
    """
    # Convert weights to array if dict
    if isinstance(weights, dict):
        weight_array = np.array([weights.get(col, 0.0) for col in cov_matrix.columns])
    else:
        weight_array = np.asarray(weights)

    # Compute portfolio variance
    portfolio_variance = weight_array.T @ cov_matrix.values @ weight_array

    # Portfolio standard deviation
    portfolio_std = np.sqrt(portfolio_variance)

    # Critical value for normal distribution (one-tailed)
    if confidence == 0.95:
        z_alpha = 1.645
    elif confidence == 0.99:
        z_alpha = 2.326
    else:
        # General case using inverse CDF
        from scipy import stats

        z_alpha = stats.norm.ppf(confidence)

    # Parametric VAR
    var = z_alpha * portfolio_std

    return var
