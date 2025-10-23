"""
Utility functions for signal processing and analysis.

This module provides helper functions for signal normalization, alignment,
and diagnostic analysis (e.g., autocorrelation for half-life estimation).
"""

import pandas as pd
import numpy as np
from typing import Optional

from slipstream.signals.base import validate_signal_dataframe


def align_signals_to_universe(
    signals: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Align signals to a given universe of tradeable assets.

    Filters signals to only include assets present in the universe at each timestamp.
    Useful for ensuring signals only cover liquid, tradeable assets.

    Args:
        signals: Long DataFrame with (timestamp, asset) index and 'signal' column
        universe: Wide DataFrame with timestamp index and boolean columns for each asset
                  indicating whether the asset is in the tradeable universe

    Returns:
        Filtered signals DataFrame with only in-universe assets
    """
    validate_signal_dataframe(signals, "signals")

    # Convert universe to long format with (timestamp, asset) index
    universe_long = universe.stack()
    universe_long = universe_long[universe_long == True].index  # Only True entries

    # Filter signals to universe
    common_index = signals.index.intersection(universe_long)
    filtered_signals = signals.loc[common_index]

    return filtered_signals


def normalize_signal_cross_sectional(
    signals: pd.DataFrame,
    method: str = 'zscore',
) -> pd.DataFrame:
    """Apply cross-sectional normalization to signals at each timestamp.

    This ensures signals are comparable across assets and timestamps by
    standardizing the distribution at each time point.

    Args:
        signals: Long DataFrame with (timestamp, asset) index and 'signal' column
        method: Normalization method
            - 'zscore': Subtract mean and divide by std (default)
            - 'rank': Convert to percentile ranks [0, 1]
            - 'demean': Subtract mean only (preserve scale)

    Returns:
        Normalized signals DataFrame with same structure
    """
    validate_signal_dataframe(signals, "signals")

    # Convert to wide format for easier cross-sectional operations
    signals_wide = signals['signal'].unstack(level='asset')

    if method == 'zscore':
        normalized = signals_wide.sub(signals_wide.mean(axis=1), axis=0)
        normalized = normalized.div(signals_wide.std(axis=1), axis=0)
    elif method == 'rank':
        normalized = signals_wide.rank(axis=1, pct=True)
    elif method == 'demean':
        normalized = signals_wide.sub(signals_wide.mean(axis=1), axis=0)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Convert back to long format
    normalized_long = normalized.stack().rename('signal').to_frame()
    normalized_long.index.names = ['timestamp', 'asset']

    return normalized_long.dropna()


def compute_signal_autocorrelation(
    signals: pd.DataFrame,
    max_lag_hours: int = 168,  # 1 week default
    asset: Optional[str] = None,
) -> pd.Series:
    """Compute autocorrelation of signal time series.

    This is used for signal half-life analysis (strategy_spec.md Section 4.4).
    The decay rate of autocorrelation indicates how quickly signal information
    degrades, which informs optimal holding period H*.

    Args:
        signals: Long DataFrame with (timestamp, asset) index and 'signal' column
        max_lag_hours: Maximum lag in hours to compute autocorrelation for
        asset: Optional specific asset to analyze. If None, computes average across all assets

    Returns:
        Series with lag (in hours) as index and autocorrelation as values

    Example:
        >>> signal = compute_pca_momentum_signal(...)
        >>> acf = compute_signal_autocorrelation(signal, max_lag_hours=168)
        >>> # Fit exponential decay to find half-life
        >>> from scipy.optimize import curve_fit
        >>> half_life = fit_exponential_decay(acf)
    """
    validate_signal_dataframe(signals, "signals")

    if asset is not None:
        # Analyze single asset
        signal_ts = signals.xs(asset, level='asset')['signal']
        autocorr = pd.Series(
            [signal_ts.autocorr(lag=i) for i in range(max_lag_hours + 1)],
            index=range(max_lag_hours + 1),
            name='autocorrelation'
        )
    else:
        # Average across all assets
        signals_wide = signals['signal'].unstack(level='asset')
        autocorrs = []

        for lag in range(max_lag_hours + 1):
            lag_corrs = [
                signals_wide[col].autocorr(lag=lag)
                for col in signals_wide.columns
            ]
            avg_corr = np.nanmean(lag_corrs)
            autocorrs.append(avg_corr)

        autocorr = pd.Series(
            autocorrs,
            index=range(max_lag_hours + 1),
            name='autocorrelation'
        )

    return autocorr


def fit_signal_decay(
    autocorrelation: pd.Series,
) -> dict:
    """Fit exponential decay to signal autocorrelation and estimate half-life.

    Fits the model: ACF(h) = exp(-h / tau)
    where tau is the decay constant. Half-life = tau * ln(2)

    Args:
        autocorrelation: Series from compute_signal_autocorrelation()

    Returns:
        Dictionary with keys:
            - 'half_life': Estimated signal half-life in hours
            - 'decay_constant': Decay constant tau
            - 'r_squared': Goodness of fit
    """
    from scipy.optimize import curve_fit

    # Filter out NaN values
    acf_clean = autocorrelation.dropna()
    if len(acf_clean) < 3:
        raise ValueError("Not enough valid autocorrelation values for fitting")

    lags = acf_clean.index.values
    acf_values = acf_clean.values

    # Exponential decay function
    def exp_decay(h, tau):
        return np.exp(-h / tau)

    # Fit the curve (initial guess: tau = max_lag / 2)
    initial_tau = lags.max() / 2
    try:
        popt, pcov = curve_fit(
            exp_decay,
            lags,
            acf_values,
            p0=[initial_tau],
            bounds=(0, np.inf)
        )
        tau = popt[0]

        # Compute R-squared
        ss_res = np.sum((acf_values - exp_decay(lags, tau)) ** 2)
        ss_tot = np.sum((acf_values - np.mean(acf_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            'half_life': tau * np.log(2),
            'decay_constant': tau,
            'r_squared': r_squared,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to fit exponential decay: {e}")
