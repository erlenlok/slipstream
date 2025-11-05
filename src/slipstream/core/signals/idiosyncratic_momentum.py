"""
PCA-based momentum signals for return prediction.

This module implements the core alpha signal generation for the Slipstream strategy.
The primary signal is based on volatility-normalized idiosyncratic momentum computed from
residuals after removing the market factor (PC1 from PCA).

References:
    - docs/strategy_spec.md Section 3.1: Alpha Model
    - docs/volume_weighted_pca_research.md: Volume weighting methodology (for PCA, not signals)
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal

from .base import validate_returns_dataframe

BASE_INTERVAL_HOURS = 4


def compute_idiosyncratic_returns(
    returns: pd.DataFrame,
    pca_loadings: pd.Series,
    market_factor: pd.Series,
) -> pd.DataFrame:
    """Compute idiosyncratic returns by removing market factor exposure.

    This implements the residual calculation from the single-factor model:
        R^price = alpha^price + beta * R_m + epsilon

    where epsilon is the idiosyncratic component we want to extract.

    Args:
        returns: Wide DataFrame with timestamp index and asset columns containing returns.
        pca_loadings: Long Series with (timestamp, asset) MultiIndex and 'loading' values
                      (beta estimates).
        market_factor: Series with timestamp index containing PC1 returns (R_m).

    Returns:
        Wide DataFrame with the same shape as returns, containing idiosyncratic components.
    """
    validate_returns_dataframe(returns, "returns")

    # Convert returns to long format for easier computation
    returns_long = returns.stack().rename('return')
    returns_long.index.names = ['timestamp', 'asset']

    # Align all data on common timestamps
    common_index = returns_long.index.intersection(pca_loadings.index)
    returns_aligned = returns_long.loc[common_index]
    loadings_aligned = pca_loadings.loc[common_index]

    # Broadcast market factor to all assets
    market_aligned = market_factor.reindex(
        returns_aligned.index.get_level_values('timestamp')
    ).values

    # Compute idiosyncratic returns: epsilon = R - beta * R_m
    idio_long = returns_aligned - (loadings_aligned * market_aligned)

    # Convert back to wide format
    idio_wide = idio_long.unstack(level='asset')

    return idio_wide


def compute_multifactor_residuals(
    returns: pd.DataFrame,
    loadings_pc1: pd.Series,
    loadings_pc2: pd.Series,
    loadings_pc3: pd.Series,
    factor_pc1: pd.Series,
    factor_pc2: pd.Series,
    factor_pc3: pd.Series,
) -> pd.DataFrame:
    """Compute idiosyncratic returns by removing PC1, PC2, PC3 exposures.

    This implements the residual calculation from the multi-factor model:
        R = alpha + beta1*F1 + beta2*F2 + beta3*F3 + epsilon

    where epsilon is the truly idiosyncratic component we want to extract.

    By pre-orthogonalizing returns against all three factors, the resulting
    signals will automatically be neutral to PC1, PC2, and PC3. This allows
    the portfolio optimizer to use a single constraint (beta1^T w = 0) as a
    safety check rather than requiring three separate constraints.

    Args:
        returns: Wide DataFrame with timestamp index and asset columns
        loadings_pc1: Long Series with (timestamp, asset) index (PC1 betas)
        loadings_pc2: Long Series with (timestamp, asset) index (PC2 betas)
        loadings_pc3: Long Series with (timestamp, asset) index (PC3 betas)
        factor_pc1: Series with timestamp index containing PC1 returns
        factor_pc2: Series with timestamp index containing PC2 returns
        factor_pc3: Series with timestamp index containing PC3 returns

    Returns:
        Wide DataFrame with same shape as returns, containing pure idiosyncratic
        components with all three principal components removed.

    Example:
        >>> # Load multi-component PCA file
        >>> pca = pd.read_csv('data/features/pca_factor_H24_K30_sqrt_3pc.csv')
        >>> pca['timestamp'] = pd.to_datetime(pca.index)
        >>>
        >>> # Extract loadings for each component
        >>> assets = [col.split('_pc')[0] for col in pca.columns if '_pc1' in col]
        >>> loadings_pc1 = pca[[f'{a}_pc1' for a in assets]].stack()
        >>> loadings_pc2 = pca[[f'{a}_pc2' for a in assets]].stack()
        >>> loadings_pc3 = pca[[f'{a}_pc3' for a in assets]].stack()
        >>>
        >>> # Compute factor returns (weighted sums)
        >>> factor_pc1 = compute_market_factor(loadings_pc1, returns)
        >>> factor_pc2 = compute_market_factor(loadings_pc2, returns)
        >>> factor_pc3 = compute_market_factor(loadings_pc3, returns)
        >>>
        >>> # Remove all three factors
        >>> idio_returns = compute_multifactor_residuals(
        ...     returns, loadings_pc1, loadings_pc2, loadings_pc3,
        ...     factor_pc1, factor_pc2, factor_pc3
        ... )
    """
    validate_returns_dataframe(returns, "returns")

    # Convert returns to long format
    returns_long = returns.stack().rename('return')
    returns_long.index.names = ['timestamp', 'asset']

    # Align all data on common timestamps and assets
    common_index = (
        returns_long.index
        .intersection(loadings_pc1.index)
        .intersection(loadings_pc2.index)
        .intersection(loadings_pc3.index)
    )

    returns_aligned = returns_long.loc[common_index]
    loadings_pc1_aligned = loadings_pc1.loc[common_index]
    loadings_pc2_aligned = loadings_pc2.loc[common_index]
    loadings_pc3_aligned = loadings_pc3.loc[common_index]

    # Broadcast factor returns to all assets at each timestamp
    timestamps = returns_aligned.index.get_level_values('timestamp')
    factor_pc1_aligned = factor_pc1.reindex(timestamps).values
    factor_pc2_aligned = factor_pc2.reindex(timestamps).values
    factor_pc3_aligned = factor_pc3.reindex(timestamps).values

    # Remove all three systematic components
    # epsilon = R - (beta1*F1 + beta2*F2 + beta3*F3)
    idio_long = returns_aligned - (
        (loadings_pc1_aligned * factor_pc1_aligned) +
        (loadings_pc2_aligned * factor_pc2_aligned) +
        (loadings_pc3_aligned * factor_pc3_aligned)
    )

    # Convert back to wide format
    idio_wide = idio_long.unstack(level='asset')

    return idio_wide


def idiosyncratic_momentum(
    returns: pd.DataFrame,
    pca_loadings: pd.Series,
    market_factor: pd.Series,
    spans: Optional[list[int]] = None,
    normalization: Literal['volatility', 'none'] = 'volatility',
    vol_span: Optional[int] = None,
    clip: float = 2.5,
) -> pd.DataFrame:
    """Compute EWMA-based idiosyncratic momentum indicators.

    This is the primary alpha signal for price return prediction. It computes
    exponentially-weighted moving averages of idiosyncratic returns (after
    removing market factor) at multiple timescales, creating a panel of
    momentum indicators for each asset.

    The momentum at each span measures the recent trend in idiosyncratic returns,
    with shorter spans responding faster to changes and longer spans being smoother.

    Args:
        returns: Wide DataFrame with timestamp index and asset columns.
        pca_loadings: Long Series with (timestamp, asset) MultiIndex and 'loading' values (PC1 betas).
        market_factor: Series with timestamp index containing PC1 returns.
        spans: List of EWMA span parameters (in hours) to compute momentum for.
               Default: [2, 4, 8, 16, 32, 64] for multi-timescale panel.
        normalization: How to normalize the idiosyncratic returns before momentum calc.
            - 'volatility': Normalize returns by EWMA volatility before computing momentum (default).
            - 'none': No normalization (raw EWMA on unnormalized returns).
        vol_span: Span for volatility EWMA. If None, uses max(spans) * 2.
        clip: Value to clip the momentum indicators at.

    Returns:
        Long DataFrame with (timestamp, asset, span) MultiIndex and 'momentum' column
        containing the momentum indicator values.
    """
    validate_returns_dataframe(returns, "returns")

    if spans is None:
        spans = [2, 4, 8, 16, 32, 64]

    # Step 1: Extract idiosyncratic returns
    idio_returns = compute_idiosyncratic_returns(returns, pca_loadings, market_factor)

    # Step 2: Volatility-normalize idiosyncratic returns FIRST (if requested)
    if normalization == 'volatility':
        if vol_span is None:
            vol_span = max(spans) * 2

        # EWMA volatility
        vol_span_bars = max(1, int(round(vol_span / BASE_INTERVAL_HOURS)))
        volatility = idio_returns.ewm(
            span=vol_span_bars,
            min_periods=vol_span_bars
        ).std()
        # Avoid division by zero
        volatility = volatility.replace(0, np.nan)

        # Normalize returns before computing momentum
        # This puts all assets on the same volatility scale
        idio_returns_normalized = idio_returns / volatility
    else:
        idio_returns_normalized = idio_returns

    # Step 3: Compute EWMA momentum for each span (on normalized returns)
    momentum_panels = {}

    for span in spans:
        # EWMA of volatility-normalized idiosyncratic returns
        span_bars = max(1, int(round(span / BASE_INTERVAL_HOURS)))
        momentum = idio_returns_normalized.ewm(
            span=span_bars,
            min_periods=span_bars
        ).mean()
        momentum = momentum.clip(lower=-clip, upper=clip)

        # Convert to long format and store
        momentum_long = momentum.stack()
        momentum_long.index.names = ['timestamp', 'asset']
        momentum_panels[span] = momentum_long

    # Step 4: Combine all spans into single DataFrame with 3-level index
    momentum_combined = pd.concat(momentum_panels, names=['span'])
    momentum_combined = momentum_combined.reorder_levels(['timestamp', 'asset', 'span'])
    momentum_combined = momentum_combined.to_frame(name='momentum')

    # Drop NaN values
    momentum_combined = momentum_combined.dropna()

    return momentum_combined
