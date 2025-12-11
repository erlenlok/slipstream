"""
Dynamic Weighting for Spectrum Strategy: Rolling Pooled Ridge Regression

This module implements the Ridge regression for determining weights of risk factors
as specified in the Spectrum strategy specification.

Reference: spectrum_spec.md - Module C: Dynamic Weighting
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import warnings


def prepare_pooled_data(
    signal_matrix: Dict[str, pd.DataFrame],
    target_returns: pd.DataFrame,
    lookback_period: int = 60
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare pooled data for Ridge regression by stacking time x asset matrices.
    
    Args:
        signal_matrix: Dict with keys 'idio_carry', 'idio_momentum', 'idio_meanrev' containing signal DataFrames
        target_returns: DataFrame with datetime index and asset columns containing target returns (residuals)
        lookback_period: Number of periods to look back for pooling (default 60)
    
    Returns:
        Tuple of (X_pooled, y_pooled, valid_mask)
            - X_pooled: 2D array of shape (n_samples, n_factors) with pooled features
            - y_pooled: 1D array of shape (n_samples,) with pooled targets
            - valid_mask: DataFrame showing which samples were valid (no NaN in signals or targets)
    """
    if not signal_matrix or len(signal_matrix) == 0:
        raise ValueError("Signal matrix cannot be empty")
    
    # Get the first signal to determine the structure
    first_signal_key = list(signal_matrix.keys())[0]
    first_signal = signal_matrix[first_signal_key]
    
    # Align all data on common index and columns
    common_index = first_signal.index
    for sig_key, sig_df in signal_matrix.items():
        common_index = common_index.intersection(sig_df.index)
    
    common_index = common_index.intersection(target_returns.index)
    
    # Get common columns (assets)
    common_cols = set(first_signal.columns)
    for sig_key, sig_df in signal_matrix.items():
        common_cols = common_cols.intersection(set(sig_df.columns))
    common_cols = common_cols.intersection(set(target_returns.columns))
    common_cols = list(common_cols)
    
    if not common_cols:
        raise ValueError("No common assets found between signals and targets")
    
    # Take the most recent lookback_period data points
    if len(common_index) > lookback_period:
        common_index = common_index[-lookback_period:]
    
    # Prepare pooled data
    all_X_rows = []
    all_y_values = []
    all_valid_mask = pd.DataFrame(False, index=first_signal.index, columns=first_signal.columns)
    
    # For each timestamp in the lookback window
    for date in common_index:
        # Get signals for all factors at this date
        date_signals = []
        for factor_name in ['idio_carry', 'idio_momentum', 'idio_meanrev']:
            if factor_name in signal_matrix:
                factor_signals = signal_matrix[factor_name].loc[date]
                if not factor_signals.empty:
                    date_signals.append(factor_signals.reindex(common_cols).values)
        
        if not date_signals:
            continue
        
        # Stack signals across factors for each asset
        date_signals_array = np.column_stack(date_signals)  # Shape: (n_assets, n_factors)
        date_targets = target_returns.loc[date].reindex(common_cols).values  # Shape: (n_assets,)
        
        # Create mask for valid (not NaN) entries
        valid_mask = (~np.isnan(date_signals_array).any(axis=1)) & (~np.isnan(date_targets))
        
        # Update the valid mask DataFrame
        valid_assets = common_cols
        for i, asset in enumerate(valid_assets):
            if i < len(valid_mask) and valid_mask[i]:
                all_valid_mask.loc[date, asset] = True
        
        # Only add valid rows to the pooled data
        valid_signals = date_signals_array[valid_mask]
        valid_targets = date_targets[valid_mask]
        
        if len(valid_signals) > 0:
            all_X_rows.append(valid_signals)
            all_y_values.extend(valid_targets)
    
    if not all_X_rows:
        raise ValueError("No valid data points found for pooling")
    
    # Combine all data
    X_pooled = np.vstack(all_X_rows)
    y_pooled = np.array(all_y_values)
    
    return X_pooled, y_pooled, all_valid_mask


def fit_ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Optional[list] = None,
    cv_folds: int = 5
) -> Tuple[np.ndarray, float, Dict]:
    """
    Fit Ridge regression with cross-validation to find optimal alpha.
    
    Args:
        X: 2D array of shape (n_samples, n_features) containing features
        y: 1D array of shape (n_samples,) containing targets
        alphas: List of alpha values to try (regularization strength)
        cv_folds: Number of cross-validation folds
    
    Returns:
        Tuple of (coefficients, best_alpha, cv_results)
            - coefficients: Array of fitted coefficients
            - best_alpha: Best alpha value found via CV
            - cv_results: Dictionary with CV results
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Use time series cross-validation to respect temporal order
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # Perform RidgeCV with cross-validation
    ridge_cv = RidgeCV(alphas=alphas, cv=tscv, scoring='r2')
    ridge_cv.fit(X, y)
    
    # Get the best alpha
    best_alpha = ridge_cv.alpha_
    
    # Fit final model with best alpha
    final_model = Ridge(alpha=best_alpha, fit_intercept=True)
    final_model.fit(X, y)
    
    # Calculate cross-validation results
    cv_results = {
        'best_alpha': best_alpha,
        'cv_scores': ridge_cv.cv_values_.mean(axis=0) if hasattr(ridge_cv, 'cv_values_') else [],
        'intercept': final_model.intercept_,
        'r2_score': final_model.score(X, y)
    }
    
    return final_model.coef_, best_alpha, cv_results


def predict_with_ridge(
    X: np.ndarray,
    coefficients: np.ndarray,
    intercept: float = 0.0
) -> np.ndarray:
    """
    Make predictions using fitted Ridge coefficients.
    
    Args:
        X: 2D array of shape (n_samples, n_features) containing features
        coefficients: 1D array of fitted coefficients
        intercept: Intercept value (default 0.0)
    
    Returns:
        1D array of shape (n_samples,) with predictions
    """
    return X @ coefficients + intercept


def compute_factor_weights_rolling(
    signal_matrix: Dict[str, pd.DataFrame],
    target_returns: pd.DataFrame,
    lookback_period: int = 60,
    alphas: Optional[list] = None,
    cv_folds: int = 5
) -> Tuple[Dict[str, float], Dict]:
    """
    Compute rolling factor weights using pooled Ridge regression.
    
    Args:
        signal_matrix: Dict with signal DataFrames for each factor
        target_returns: DataFrame with target returns (next period residuals)
        lookback_period: Lookback period for pooling data (default 60)
        alphas: List of alpha values for Ridge regression
        cv_folds: Number of CV folds
    
    Returns:
        Tuple of (factor_weights, training_results)
            - factor_weights: Dict mapping factor names to their weights
            - training_results: Dict with additional training info
    """
    # Prepare pooled data
    X_pooled, y_pooled, valid_mask = prepare_pooled_data(
        signal_matrix, target_returns, lookback_period
    )
    
    if X_pooled.shape[0] == 0 or y_pooled.shape[0] == 0:
        # Return default weights if no valid data
        return {
            'idio_carry': 0.0,
            'idio_momentum': 0.0, 
            'idio_meanrev': 0.0
        }, {'error': 'No valid data found for training'}
    
    # Fit Ridge regression
    coefficients, best_alpha, cv_results = fit_ridge_regression(
        X_pooled, y_pooled, alphas, cv_folds
    )
    
    # Map coefficients to factor names
    factor_names = ['idio_carry', 'idio_momentum', 'idio_meanrev']
    factor_weights = {}
    
    for i, factor_name in enumerate(factor_names):
        if i < len(coefficients):
            factor_weights[factor_name] = coefficients[i]
        else:
            factor_weights[factor_name] = 0.0
    
    training_results = {
        'best_alpha': best_alpha,
        'cv_results': cv_results,
        'n_samples': X_pooled.shape[0],
        'n_features': X_pooled.shape[1],
        'r2_score': cv_results.get('r2_score', 0.0)
    }
    
    return factor_weights, training_results


def apply_factor_weights(
    signal_matrix: Dict[str, pd.DataFrame],
    factor_weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Apply factor weights to signals to get composite alpha.
    
    Args:
        signal_matrix: Dict with signal DataFrames for each factor
        factor_weights: Dict mapping factor names to their weights
    
    Returns:
        DataFrame with datetime index and asset columns containing composite alpha
    """
    # Start with zeros
    if not signal_matrix:
        raise ValueError("Signal matrix cannot be empty")
    
    # Use the first signal to determine the shape
    first_factor = list(signal_matrix.keys())[0]
    result = pd.DataFrame(0.0, index=signal_matrix[first_factor].index, 
                         columns=signal_matrix[first_factor].columns)
    
    # Weight and sum each factor
    for factor_name, weight in factor_weights.items():
        if factor_name in signal_matrix:
            result += signal_matrix[factor_name] * weight
    
    return result