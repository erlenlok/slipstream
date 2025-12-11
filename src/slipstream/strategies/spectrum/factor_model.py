"""
Factor Engine for Spectrum Strategy: OLS-based Factor Decomposition

This module implements the OLS (Ordinary Least Squares) regression for decomposing
asset returns into Market Beta (BTC/ETH) and Idiosyncratic Residuals as specified
in the Spectrum strategy specification.

Reference: spectrum_spec.md - Module A: Factor Engine
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import timedelta
import warnings


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from daily prices.
    
    Args:
        prices: DataFrame with datetime index and asset columns containing daily closing prices
    
    Returns:
        DataFrame with datetime index and asset columns containing daily log returns
    """
    return np.log(prices / prices.shift(1))


def compute_ols_factor_decomposition(
    returns: pd.DataFrame,
    btc_returns: pd.Series,
    eth_returns: pd.Series,
    lookback_window: int = 30,
    min_periods: int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform rolling OLS regression to decompose returns into beta and idiosyncratic components.
    
    For each asset i and each time t:
    r_i = alpha + beta_btc * r_btc + beta_eth * r_eth + epsilon_i
    
    Args:
        returns: Wide DataFrame with datetime index and asset columns containing daily returns
        btc_returns: Series with datetime index containing daily BTC returns
        eth_returns: Series with datetime index containing daily ETH returns  
        lookback_window: Rolling window size for OLS estimation (default 30 days)
        min_periods: Minimum number of observations required (default 15 days)
    
    Returns:
        Tuple of (betas, residuals, idio_vol)
            - betas: DataFrame with datetime index and asset columns containing [beta_btc, beta_eth]
            - residuals: DataFrame with same shape as returns containing idiosyncratic returns
            - idio_vol: DataFrame with same shape as returns containing idiosyncratic volatility
    """
    # Align all data on common dates
    common_dates = returns.index.intersection(btc_returns.index).intersection(eth_returns.index)
    returns_aligned = returns.loc[common_dates].copy()
    btc_aligned = btc_returns.loc[common_dates]
    eth_aligned = eth_returns.loc[common_dates]
    
    # Initialize result containers
    betas_btc = pd.DataFrame(index=returns_aligned.index, columns=returns_aligned.columns, dtype=float)
    betas_eth = pd.DataFrame(index=returns_aligned.index, columns=returns_aligned.columns, dtype=float)
    residuals = pd.DataFrame(index=returns_aligned.index, columns=returns_aligned.columns, dtype=float)
    
    # Prepare factor matrix once: [constant, btc_returns, eth_returns]
    factor_matrix = np.column_stack([
        np.ones(len(btc_aligned)),  # For intercept
        btc_aligned.values,
        eth_aligned.values
    ])
    
    # For each date, perform rolling regression
    for i, date in enumerate(returns_aligned.index):
        # Only proceed if we have sufficient lookback data
        if i < min_periods - 1:
            continue
            
        # Get lookback window indices
        start_idx = max(0, i - lookback_window + 1)
        lookback_factor_matrix = factor_matrix[start_idx:i+1]
        
        # Skip if insufficient valid data
        if len(lookback_factor_matrix) < min_periods:
            continue
        
        # Process each asset
        for asset in returns_aligned.columns:
            y = returns_aligned[asset].iloc[start_idx:i+1].values
            
            # Check for valid data
            valid_mask = ~(np.isnan(y) | np.isnan(lookback_factor_matrix).any(axis=1))
            if valid_mask.sum() < min_periods:
                continue
            
            X_valid = lookback_factor_matrix[valid_mask]
            y_valid = y[valid_mask]
            
            # Perform OLS: (X'X)^(-1)X'y
            try:
                XtX_inv = np.linalg.inv(X_valid.T @ X_valid)
                coefficients = XtX_inv @ X_valid.T @ y_valid
                
                # Extract coefficients
                alpha = coefficients[0]
                beta_btc = coefficients[1]
                beta_eth = coefficients[2]
                
                # Store betas
                betas_btc.loc[date, asset] = beta_btc
                betas_eth.loc[date, asset] = beta_eth
                
                # Calculate residual for current date if possible
                if date in returns_aligned.index:
                    current_date_idx = returns_aligned.index.get_loc(date)
                    if current_date_idx < len(returns_aligned):
                        current_return = returns_aligned.loc[date, asset]
                        if not np.isnan(current_return):
                            # Calculate predicted return
                            current_factor_row = factor_matrix[i]  # Current date factors
                            predicted_return = (alpha + 
                                              beta_btc * current_factor_row[1] + 
                                              beta_eth * current_factor_row[2])
                            residual = current_return - predicted_return
                            residuals.loc[date, asset] = residual
            except np.linalg.LinAlgError:
                # Singular matrix, skip this date
                continue
    
    # Combine BTC and ETH betas into a single betas DataFrame
    # For each asset, we'll have both BTC and ETH betas
    betas_combined = pd.concat([betas_btc.add_suffix('_btc'), betas_eth.add_suffix('_eth')], axis=1)
    
    # Calculate idiosyncratic volatility (rolling standard deviation of residuals)
    idio_vol = residuals.rolling(window=lookback_window, min_periods=min_periods).std()
    
    return betas_combined, residuals, idio_vol


def apply_universe_mask(
    returns: pd.DataFrame,
    min_history_days: int = 30,
    min_avg_volume: float = 10_000_000,  # $10M threshold
    volume_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Apply universe masking based on liquidity and minimum history requirements.
    
    Args:
        returns: Wide DataFrame with datetime index and asset columns
        min_history_days: Minimum days of history required (default 30)
        min_avg_volume: Minimum average volume threshold (default $10M)
        volume_data: Optional volume DataFrame for liquidity filtering
    
    Returns:
        Masked returns DataFrame with inactive assets set to NaN
    """
    # Create mask based on minimum history
    history_mask = pd.DataFrame(True, index=returns.index, columns=returns.columns)
    
    for asset in returns.columns:
        # For each date, check if asset has sufficient history up to that point
        asset_notna = returns[asset].notna()
        cumulative_history = asset_notna.rolling(window=min_history_days, min_periods=1).sum()
        history_mask[asset] = cumulative_history >= min_history_days
    
    # If volume data is provided, apply liquidity filter
    if volume_data is not None and not volume_data.empty:
        # Align volume data with returns index
        volume_aligned = volume_data.reindex(returns.index)
        
        # Calculate rolling average volume
        avg_volume = volume_aligned.rolling(window=min_history_days, min_periods=min_history_days).mean()
        
        # Apply volume filter
        volume_mask = avg_volume >= min_avg_volume
        history_mask = history_mask & volume_mask.fillna(True)  # Fill NaN with True if no volume data
    
    # Apply mask to returns
    masked_returns = returns.where(history_mask, np.nan)
    
    return masked_returns


def calculate_daily_funding_yield(funding_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily funding yield - for daily data, this is just the daily funding rate.
    
    Args:
        funding_rates: DataFrame with datetime index and asset columns containing daily funding rates
    
    Returns:
        DataFrame with datetime index and asset columns containing daily funding yields
    """
    # For daily data, the funding rate is already the daily yield
    return funding_rates


def compute_spectrum_factors(
    returns: pd.DataFrame,
    btc_returns: pd.Series,
    eth_returns: pd.Series,
    funding_rates: Optional[pd.DataFrame] = None,
    lookback_window: int = 30,
    min_history_days: int = 30,
    min_avg_volume: float = 10_000_000,
    volume_data: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to compute all Spectrum factor model components.
    
    Args:
        returns: Wide DataFrame with datetime index and asset columns for all assets
        btc_returns: Series with datetime index for BTC daily returns
        eth_returns: Series with datetime index for ETH daily returns
        funding_rates: Optional DataFrame for daily funding rate data
        lookback_window: OLS lookback window (default 30 days)
        min_history_days: Minimum history for universe inclusion (default 30)
        min_avg_volume: Minimum average volume threshold (default $10M)
        volume_data: Optional DataFrame with volume data for filtering
    
    Returns:
        Tuple of (masked_betas, residuals, idio_vol, daily_funding)
            - masked_betas: Beta coefficients with universe masking applied
            - residuals: Idiosyncratic returns
            - idio_vol: Idiosyncratic volatility
            - daily_funding: Daily funding yields (or empty if not provided)
    """
    # Apply universe masking to returns
    masked_returns = apply_universe_mask(
        returns, min_history_days, min_avg_volume, volume_data
    )
    
    # Compute OLS factor decomposition using masked returns
    betas, residuals, idio_vol = compute_ols_factor_decomposition(
        masked_returns, btc_returns, eth_returns, lookback_window
    )
    
    # Calculate daily funding yield if provided
    daily_funding = calculate_daily_funding_yield(funding_rates) if funding_rates is not None else pd.DataFrame()
    
    return betas, residuals, idio_vol, daily_funding