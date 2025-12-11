"""
Robust Optimizer for Spectrum Strategy: CVXPY-based Portfolio Optimization

This module implements the CVXPY optimization for generating optimal idiosyncratic 
portfolio weights as specified in the Spectrum strategy specification.

Reference: spectrum_spec.md - Module D: Robust Optimizer
"""
import pandas as pd
import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, Optional
from sklearn.covariance import LedoitWolf
from datetime import datetime


def compute_cost_vector(
    volatilities: pd.Series,
    spread_proxy: Optional[pd.Series] = None,
    base_cost: float = 0.0002  # 2 bps base cost
) -> pd.Series:
    """
    Construct cost vector where lambda_i is proportional to asset's rolling volatility or spread.
    
    Args:
        volatilities: Series with asset names as index containing rolling volatilities
        spread_proxy: Optional Series with asset names as index containing spread proxy
        base_cost: Base cost coefficient (default 2 bps)
    
    Returns:
        Series with asset names as index containing cost coefficients
    """
    if spread_proxy is not None:
        # Use spread proxy if available
        cost_vector = spread_proxy.fillna(base_cost)
    else:
        # Use volatility as proxy for liquidity costs
        # Higher volatility assets may have higher costs due to slippage
        cost_vector = volatilities * base_cost * 10  # Scale by volatility
    
    # Make sure all assets have a cost value
    cost_vector = cost_vector.fillna(base_cost)
    
    # Ensure all values are positive and reasonable
    cost_vector = cost_vector.clip(lower=base_cost/10, upper=base_cost*100)
    
    return cost_vector


def compute_ledoit_wolf_covariance(
    residuals: pd.DataFrame,
    shrinkage_target: str = 'constant_correlation'
) -> np.ndarray:
    """
    Compute sample covariance of residuals with Ledoit-Wolf shrinkage.
    
    Args:
        residuals: DataFrame with datetime index and asset columns containing idiosyncratic returns
        shrinkage_target: Type of shrinkage target ('constant_correlation', 'single_factor', 'constant_variance')
    
    Returns:
        2D numpy array of covariance matrix
    """
    # Remove assets with all NaN values
    valid_assets = residuals.columns[~residuals.isna().all()]
    if len(valid_assets) == 0:
        raise ValueError("No valid assets with non-NaN returns")
    
    residuals_clean = residuals[valid_assets].dropna(how='all')
    if residuals_clean.empty:
        # If all data is NaN, return identity matrix scaled by base volatility
        n_assets = len(valid_assets)
        return np.eye(n_assets) * 0.02**2  # Default 2% daily vol squared
    
    # Clean data by removing rows with any NaN
    X = residuals_clean.dropna().values
    
    if X.shape[0] < 2 or X.shape[1] < 2:
        # If insufficient data, return diagonal matrix
        if X.shape[0] < 2:
            # Just use default volatilities
            vol_diag = np.ones(X.shape[1]) * 0.02  # Default 2% daily vol
            return np.diag(vol_diag**2)
        else:
            # Compute per-asset variances, set covariances to 0
            variances = np.var(X, axis=0, ddof=1)  # Sample variance
            variances = np.where(variances > 0, variances, 0.02**2)  # Default to 2%^2 if 0
            return np.diag(variances)
    
    # Apply Ledoit-Wolf shrinkage
    try:
        lw = LedoitWolf()
        cov_matrix = lw.fit(X).covariance_
        
        # Ensure the matrix is positive semi-definite
        # If not, add small diagonal adjustment
        eigenvals = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvals < 1e-12):
            # Add small regularization if not PSD
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        
        return cov_matrix
    except Exception:
        # Fallback to basic sample covariance if Ledoit-Wolf fails
        cov_matrix = np.cov(X.T)
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            # Use diagonal matrix as ultimate fallback
            stds = np.std(X, axis=0)
            stds = np.where(stds > 0, stds, 0.02)  # Default to 2% if 0
            return np.diag(stds**2)
        
        # Ensure positive semi-definiteness
        eigenvals = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvals < 1e-12):
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        
        return cov_matrix


def compute_robust_optimization(
    alpha: np.ndarray,
    cov_matrix: np.ndarray,
    betas: np.ndarray,
    w_prev: np.ndarray,
    cost_vector: np.ndarray,
    target_leverage: float = 1.0,
    max_single_pos: float = 0.1,  # 10% max single position
    liquidity_limits: Optional[np.ndarray] = None,
    gamma: float = 1.0,  # Risk aversion parameter
    transaction_cost_weight: float = 1.0
) -> Tuple[np.ndarray, Dict]:
    """
    Perform robust optimization with CVXPY.
    
    Objective: Maximize μ^T w - γ w^T Σ w - || λ ⊙ (w - w_prev) ||_1
    Subject to:
    - sum(|w|) <= TargetIdioLeverage
    - |w_i| <= MaxSinglePos
    - |w_i| <= LiquidityLimit_i (if provided)
    - w^T β = 0 (beta neutrality)
    
    Args:
        alpha: 1D array of expected returns (alpha predictions)
        cov_matrix: 2D covariance matrix
        betas: 1D array of asset betas
        w_prev: 1D array of previous portfolio weights
        cost_vector: 1D array of transaction cost coefficients
        target_leverage: Target total leverage (default 1.0)
        max_single_pos: Maximum absolute weight per asset (default 0.1)
        liquidity_limits: Optional 1D array of per-asset liquidity limits
        gamma: Risk aversion parameter (default 1.0)
        transaction_cost_weight: Weight for transaction costs (default 1.0)
    
    Returns:
        Tuple of (optimal_weights, info_dict)
    """
    n_assets = len(alpha)
    
    if n_assets == 0:
        return np.array([]), {'error': 'No assets to optimize'}
    
    # Define optimization variables
    w = cp.Variable(n_assets)
    turnover = cp.Variable(n_assets)  # Auxiliary variables for L1 transaction costs
    
    # Objective function: μ^T w - γ w^T Σ w - λ^T turnover
    # where turnover >= w - w_prev and turnover >= -(w - w_prev)
    expected_return = alpha.T @ w
    risk = cp.quad_form(w, cov_matrix)
    transaction_cost = cost_vector.T @ turnover
    
    objective = cp.Maximize(expected_return - gamma * risk - transaction_cost_weight * transaction_cost)
    
    # Constraints
    constraints = []
    
    # Absolute value constraints for turnover: turnover >= |w - w_prev|
    constraints.append(turnover >= w - w_prev)  # turnover >= w - w_prev
    constraints.append(turnover >= -(w - w_prev))  # turnover >= -(w - w_prev)
    
    # Leverage constraint: sum of absolute weights <= target_leverage
    # This is approximated using a slack variable approach in CVXPY
    abs_w = cp.Variable(n_assets)
    constraints.append(abs_w >= w)
    constraints.append(abs_w >= -w)
    constraints.append(cp.sum(abs_w) <= target_leverage)
    
    # Single position size constraints
    constraints.append(w <= max_single_pos)
    constraints.append(w >= -max_single_pos)
    
    # Liquidity limits if provided
    if liquidity_limits is not None and len(liquidity_limits) == n_assets:
        constraints.append(abs_w <= liquidity_limits)
    
    # Beta neutrality constraint: w^T β = 0
    constraints.append(betas.T @ w == 0)
    
    # Solve the optimization problem
    try:
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)  # Use CLARABEL as default
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # Fallback: solve a simpler problem if complex one fails
            # Just maximize alpha with leverage constraint and beta neutrality
            simple_objective = cp.Maximize(alpha.T @ w - gamma * risk)
            simple_constraints = [
                cp.sum(cp.abs(w)) <= target_leverage,
                betas.T @ w == 0
            ]
            simple_prob = cp.Problem(simple_objective, simple_constraints)
            simple_prob.solve(solver=cp.CLARABEL, verbose=False)
            
            if simple_prob.status not in ["optimal", "optimal_inaccurate"]:
                # Ultimate fallback: simple risk-adjusted allocation
                w_opt = alpha / (gamma * np.diag(cov_matrix))
                w_opt = w_opt / np.sum(np.abs(w_opt)) * target_leverage
                # Project onto beta neutrality constraint
                if np.sum(np.abs(betas)) > 1e-10:  # If betas are not all zero
                    # Project w_opt to satisfy w^T * beta = 0
                    w_opt = w_opt - (w_opt.T @ betas) / (betas.T @ betas) * betas
            else:
                w_opt = w.value
        else:
            w_opt = w.value
        
        if w_opt is None:
            # If optimization failed completely, return simple equal weight
            w_opt = np.zeros(n_assets)
        
        # Clean the results to ensure they meet constraints
        w_opt = np.nan_to_num(w_opt, nan=0.0)  # Replace NaN with 0
        
        # Verify beta neutrality
        beta_exposure = w_opt.T @ betas if len(betas) == len(w_opt) else 0
        if abs(beta_exposure) > 1e-6:
            # Project onto beta neutrality constraint
            if len(betas) == len(w_opt) and np.sum(betas**2) > 1e-10:
                w_opt = w_opt - (w_opt.T @ betas) / (betas.T @ betas) * betas
        
        # Calculate additional metrics
        actual_leverage = np.sum(np.abs(w_opt))
        turnover_vec = np.abs(w_opt - w_prev) if w_prev is not None else np.abs(w_opt)
        transaction_cost_actual = cost_vector.T @ turnover_vec if len(cost_vector) == len(turnover_vec) else 0
        expected_return_actual = alpha.T @ w_opt if len(alpha) == len(w_opt) else 0
        risk_actual = w_opt.T @ cov_matrix @ w_opt if len(w_opt) == cov_matrix.shape[0] else 0
        
        info = {
            'status': prob.status if 'prob' in locals() else 'fallback',
            'success': True,
            'expected_return': float(expected_return_actual),
            'risk': float(risk_actual),
            'transaction_cost': float(transaction_cost_actual),
            'actual_leverage': float(actual_leverage),
            'beta_exposure': float(beta_exposure),
            'turnover': float(np.sum(turnover_vec)),
            'n_positions': int(np.sum(np.abs(w_opt) > 1e-6))
        }
        
        return w_opt, info
        
    except Exception as e:
        # Fallback if CVXPY fails
        print(f"CVXPY optimization failed: {e}")
        
        # Simple fallback: risk-adjusted allocation based on alpha and covariances
        if len(alpha) > 0 and cov_matrix.shape[0] == len(alpha):
            # Inverse volatility weighting as approximation
            vol_diag = np.sqrt(np.diag(cov_matrix))
            vol_diag = np.where(vol_diag > 0, vol_diag, 1.0)  # Avoid division by zero
            w_opt = alpha / (gamma * vol_diag**2)
            
            # Normalize to target leverage
            current_leverage = np.sum(np.abs(w_opt))
            if current_leverage > 1e-10:
                w_opt = w_opt / current_leverage * target_leverage
            
            # Project onto beta neutrality constraint
            if len(betas) == len(w_opt) and np.sum(betas**2) > 1e-10:
                w_opt = w_opt - (w_opt.T @ betas) / (betas.T @ betas) * betas
        else:
            w_opt = np.zeros(len(alpha))
        
        return w_opt, {
            'status': 'fallback',
            'success': False,
            'error': str(e),
            'expected_return': 0.0,
            'risk': 0.0,
            'transaction_cost': 0.0,
            'actual_leverage': float(np.sum(np.abs(w_opt))),
            'beta_exposure': float(w_opt.T @ betas if len(betas) == len(w_opt) else 0),
            'turnover': 0.0,
            'n_positions': int(np.sum(np.abs(w_opt) > 1e-6))
        }


def prepare_asset_universe(
    current_assets: list,
    previous_weights: Optional[Dict[str, float]] = None,
    new_assets: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare universe alignment: identify current assets and construct w_prev.
    
    Args:
        current_assets: List of current active assets
        previous_weights: Optional dict mapping asset names to previous weights
        new_assets: Optional list of new assets to initialize with zero weight
    
    Returns:
        Tuple of (asset_names_array, previous_weights_array)
    """
    if previous_weights is None:
        previous_weights = {}
    
    # Create the array of asset names
    asset_names = np.array(current_assets)
    
    # Create array of previous weights, filling 0.0 for new entrants
    w_prev = np.array([previous_weights.get(asset, 0.0) for asset in current_assets])
    
    return asset_names, w_prev


def optimize_spectrum_portfolio(
    composite_alpha: pd.DataFrame,
    residuals: pd.DataFrame,
    betas: pd.DataFrame,
    previous_weights: Optional[Dict[str, float]] = None,
    target_leverage: float = 1.0,
    max_single_pos: float = 0.1,
    base_cost: float = 0.0002,
    risk_aversion: float = 1.0,
    transaction_cost_weight: float = 1.0
) -> Tuple[pd.Series, Dict]:
    """
    Main function to optimize Spectrum portfolio.
    
    Args:
        composite_alpha: DataFrame with datetime index and asset columns containing composite alpha
        residuals: DataFrame with datetime index and asset columns containing idiosyncratic returns
        betas: DataFrame with datetime index and asset columns containing beta coefficients
        previous_weights: Optional dict mapping asset names to previous weights
        target_leverage: Target idiosyncratic leverage (default 1.0)
        max_single_pos: Maximum single position size (default 0.1)
        base_cost: Base transaction cost (default 2 bps)
        risk_aversion: Risk aversion parameter (default 1.0)
        transaction_cost_weight: Transaction cost weight (default 1.0)
    
    Returns:
        Tuple of (target_idio_weights, optimization_info)
    """
    if composite_alpha.empty or residuals.empty or betas.empty:
        return pd.Series(dtype=float), {'error': 'Input data is empty'}
    
    # Get the latest available data for optimization
    latest_date = composite_alpha.index[-1]  # Get the most recent date
    
    # Extract current alphas, residuals, and betas
    current_alpha = composite_alpha.loc[latest_date].dropna()
    current_residuals = residuals.loc[:latest_date].dropna(how='all')  # Use all history for covariance
    current_betas = betas.loc[latest_date].dropna()
    
    # Align assets between alpha, residuals, and betas
    common_assets = set(current_alpha.index) \
                   .intersection(set(current_residuals.columns)) \
                   .intersection(set(current_betas.index))
    
    if not common_assets:
        return pd.Series(dtype=float), {'error': 'No common assets between inputs'}
    
    common_assets = list(common_assets)
    current_alpha = current_alpha.reindex(common_assets)
    current_residuals = current_residuals[common_assets]
    current_betas = current_betas.reindex(common_assets)
    
    # Prepare previous weights
    if previous_weights is None:
        w_prev_dict = {asset: 0.0 for asset in common_assets}
    else:
        w_prev_dict = {asset: previous_weights.get(asset, 0.0) for asset in common_assets}
    
    # Compute cost vector based on volatilities
    # Use recent volatility from residuals
    recent_vol = current_residuals.iloc[-20:].std().reindex(common_assets)  # Use last 20 days
    recent_vol = recent_vol.fillna(0.02)  # Default to 2% if no data
    cost_vector = compute_cost_vector(recent_vol, base_cost=base_cost)
    cost_vector = cost_vector.reindex(common_assets)
    
    # Compute Ledoit-Wolf covariance matrix
    cov_matrix = compute_ledoit_wolf_covariance(current_residuals)
    
    # Prepare arrays for optimization
    alpha_array = current_alpha.values
    betas_array = current_betas.values
    w_prev_array = np.array([w_prev_dict[asset] for asset in common_assets])
    cost_array = cost_vector.values
    
    # Perform robust optimization
    w_opt, opt_info = compute_robust_optimization(
        alpha=alpha_array,
        cov_matrix=cov_matrix,
        betas=betas_array,
        w_prev=w_prev_array,
        cost_vector=cost_array,
        target_leverage=target_leverage,
        max_single_pos=max_single_pos,
        gamma=risk_aversion,
        transaction_cost_weight=transaction_cost_weight
    )
    
    # Create result series
    if len(w_opt) == len(common_assets):
        result_weights = pd.Series(w_opt, index=common_assets)
        # Add all other assets with 0 weight
        all_assets_in_alpha = composite_alpha.columns
        for asset in all_assets_in_alpha:
            if asset not in result_weights.index:
                result_weights[asset] = 0.0
    else:
        # Fallback if optimization dimensions don't match
        result_weights = pd.Series(0.0, index=composite_alpha.columns)
    
    return result_weights, opt_info