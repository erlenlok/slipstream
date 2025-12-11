"""
Signal Factory for Spectrum Strategy: Risk Factor Signal Generation

This module implements the signal generation for standardized risk factor scores
from residuals as specified in the Spectrum strategy specification.

Reference: spectrum_spec.md - Module B: Signal Factory
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta


def compute_idio_carry(
    daily_funding_yield: pd.DataFrame,
    idio_vol: pd.DataFrame,
    annualization_factor: float = 365.0
) -> pd.DataFrame:
    """
    Calculate hedged funding yield (annualized), normalized by idio_vol (annualized).
    
    Args:
        daily_funding_yield: DataFrame with datetime index and asset columns containing daily funding yields
        idio_vol: DataFrame with datetime index and asset columns containing idiosyncratic volatility
        annualization_factor: Factor to annualize returns (default 365.0 for daily data)
    
    Returns:
        DataFrame with datetime index and asset columns containing idio-carry signals
    """
    # Annualize the funding yield
    annualized_funding = daily_funding_yield * annualization_factor
    
    # Annualize idio volatility 
    annualized_vol = idio_vol * np.sqrt(annualization_factor)
    
    # Normalize funding by annualized volatility
    idio_carry = annualized_funding / annualized_vol
    
    # Replace division by zero or NaN with 0
    idio_carry = idio_carry.replace([np.inf, -np.inf], np.nan)
    
    return idio_carry


def compute_idio_momentum(
    residuals: pd.DataFrame,
    idio_vol: pd.DataFrame,
    fast_span: int = 3,
    slow_span: int = 10
) -> pd.DataFrame:
    """
    Calculate (EMA_fast - EMA_slow) of residuals, normalized by idio_vol.
    
    Args:
        residuals: DataFrame with datetime index and asset columns containing idiosyncratic returns
        idio_vol: DataFrame with datetime index and asset columns containing idiosyncratic volatility
        fast_span: Fast EMA span (default 3 days)
        slow_span: Slow EMA span (default 10 days)
    
    Returns:
        DataFrame with datetime index and asset columns containing idio-momentum signals
    """
    # Compute fast and slow EMAs of residuals
    fast_ema = residuals.ewm(span=fast_span, min_periods=fast_span).mean()
    slow_ema = residuals.ewm(span=slow_span, min_periods=slow_span).mean()
    
    # Calculate momentum as difference
    momentum_raw = fast_ema - slow_ema
    
    # Normalize by idio volatility
    idio_momentum = momentum_raw / (idio_vol.replace(0, np.nan))
    
    # Replace division by zero or NaN with 0
    idio_momentum = idio_momentum.replace([np.inf, -np.inf], np.nan)
    
    return idio_momentum


def compute_idio_meanrev(
    residuals: pd.DataFrame,
    idio_vol: pd.DataFrame,
    sma_period: int = 5
) -> pd.DataFrame:
    """
    Calculate negative deviation from short-term mean (SMA), normalized by idio_vol.
    
    Formula: -1 * (epsilon_t - SMA(epsilon, 5d)) / sigma_epsilon
    
    Args:
        residuals: DataFrame with datetime index and asset columns containing idiosyncratic returns
        idio_vol: DataFrame with datetime index and asset columns containing idiosyncratic volatility
        sma_period: SMA period for mean calculation (default 5 days)
    
    Returns:
        DataFrame with datetime index and asset columns containing idio-meanrev signals
    """
    # Calculate simple moving average of residuals
    sma_residuals = residuals.rolling(window=sma_period, min_periods=sma_period).mean()
    
    # Calculate deviation from mean and negate (for mean reversion)
    deviation = -(residuals - sma_residuals)  # Negative deviation
    
    # Normalize by idio volatility
    idio_meanrev = deviation / (idio_vol.replace(0, np.nan))
    
    # Replace division by zero or NaN with 0
    idio_meanrev = idio_meanrev.replace([np.inf, -np.inf], np.nan)
    
    return idio_meanrev


def apply_cross_sectional_zscore(
    signals: pd.DataFrame,
    winsorize_at: float = 3.0
) -> pd.DataFrame:
    """
    Apply Cross-Sectional Z-Score (winsorized at ±3) to signals.
    
    Args:
        signals: DataFrame with datetime index and asset columns containing raw signals
        winsorize_at: Winsorization threshold (default ±3)
    
    Returns:
        DataFrame with datetime index and asset columns containing z-scored and winsorized signals
    """
    # Calculate cross-sectional mean and std for each date
    means = signals.mean(axis=1)  # Mean across assets for each date
    stds = signals.std(axis=1, ddof=0)  # Std across assets for each date (population std)
    
    # Initialize result dataframe
    zscored_signals = pd.DataFrame(index=signals.index, columns=signals.columns, dtype=float)
    
    # For each date, compute z-score relative to cross-section
    for date in signals.index:
        date_signals = signals.loc[date]
        mean_val = means.loc[date]
        std_val = stds.loc[date]
        
        # Compute z-score: (x - mean) / std
        if std_val != 0 and not pd.isna(std_val):
            z_scores = (date_signals - mean_val) / std_val
        else:
            # If std is 0, z-score is 0
            z_scores = pd.Series(0.0, index=date_signals.index)
        
        # Winsorize at ±winsorize_at
        z_scores = z_scores.clip(lower=-winsorize_at, upper=winsorize_at)
        
        zscored_signals.loc[date] = z_scores
    
    return zscored_signals


def generate_spectrum_signals(
    residuals: pd.DataFrame,
    daily_funding_yield: pd.DataFrame,
    idio_vol: pd.DataFrame,
    warmup_periods: int = 10,
    momentum_fast_span: int = 3,
    momentum_slow_span: int = 10,
    meanrev_sma_period: int = 5,
    zscore_winsorize_at: float = 3.0
) -> Dict[str, pd.DataFrame]:
    """
    Generate all Spectrum risk factor signals: Idio-Carry, Idio-Momentum, Idio-MeanRev.
    
    Args:
        residuals: DataFrame with datetime index and asset columns containing idiosyncratic returns
        daily_funding_yield: DataFrame with datetime index and asset columns containing daily funding yields
        idio_vol: DataFrame with datetime index and asset columns containing idiosyncratic volatility
        warmup_periods: Number of periods to ignore for warmup (default 10)
        momentum_fast_span: Fast EMA span for momentum (default 3)
        momentum_slow_span: Slow EMA span for momentum (default 10)
        meanrev_sma_period: SMA period for mean reversion (default 5)
        zscore_winsorize_at: Winsorization threshold for z-scoring (default 3.0)
    
    Returns:
        Dictionary with keys 'idio_carry', 'idio_momentum', 'idio_meanrev' containing the signals
    """
    # Initialize all signal matrices with same shape as residuals
    signals_matrix = {
        'idio_carry': pd.DataFrame(index=residuals.index, columns=residuals.columns, dtype=float),
        'idio_momentum': pd.DataFrame(index=residuals.index, columns=residuals.columns, dtype=float),
        'idio_meanrev': pd.DataFrame(index=residuals.index, columns=residuals.columns, dtype=float)
    }
    
    # Only compute signals after warmup period to avoid look-ahead bias
    if len(residuals) > warmup_periods:
        # Compute raw signals
        raw_carry = compute_idio_carry(
            daily_funding_yield.loc[residuals.index], 
            idio_vol.loc[residuals.index]
        )
        raw_momentum = compute_idio_momentum(
            residuals, idio_vol, momentum_fast_span, momentum_slow_span
        )
        raw_meanrev = compute_idio_meanrev(
            residuals, idio_vol, meanrev_sma_period
        )
        
        # Apply cross-sectional z-score and winsorization to each signal
        signals_matrix['idio_carry'] = apply_cross_sectional_zscore(raw_carry, zscore_winsorize_at)
        signals_matrix['idio_momentum'] = apply_cross_sectional_zscore(raw_momentum, zscore_winsorize_at)
        signals_matrix['idio_meanrev'] = apply_cross_sectional_zscore(raw_meanrev, zscore_winsorize_at)
    
    return signals_matrix