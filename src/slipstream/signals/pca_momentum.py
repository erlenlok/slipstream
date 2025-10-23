"""
PCA-based momentum signals for return prediction.

This module implements the core alpha signal generation for the Slipstream strategy.
The primary signal is based on volume-normalized idiosyncratic momentum computed from
residuals after removing the market factor (PC1 from PCA).

References:
    - docs/strategy_spec.md Section 3.1: Alpha Model
    - docs/volume_weighted_pca_research.md: Volume weighting methodology
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal

from slipstream.signals.base import validate_returns_dataframe


def compute_idiosyncratic_returns(
    returns: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    market_factor: pd.Series,
) -> pd.DataFrame:
    """Compute idiosyncratic returns by removing market factor exposure.

    This implements the residual calculation from the single-factor model:
        R^price = alpha^price + beta * R_m + epsilon

    where epsilon is the idiosyncratic component we want to extract.

    Args:
        returns: Wide DataFrame with timestamp index and asset columns containing returns
        pca_loadings: Long DataFrame with (timestamp, asset) index and 'loading' column
                      containing PC1 loadings (beta estimates)
        market_factor: Series with timestamp index containing PC1 returns (R_m)

    Returns:
        Wide DataFrame with same shape as returns, containing idiosyncratic components

    Example:
        >>> returns = load_all_returns()  # Wide format
        >>> pca_data = pd.read_csv('data/features/pca_factor_H24_K30_sqrt.csv')
        >>> loadings = pca_data.set_index(['timestamp', 'asset'])['loading']
        >>> market = pca_data.groupby('timestamp')['market_return'].first()
        >>> idio_returns = compute_idiosyncratic_returns(returns, loadings, market)
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


def idiosyncratic_momentum(
    returns: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    market_factor: pd.Series,
    spans: Optional[list[int]] = None,
    normalization: Literal['volatility', 'none'] = 'volatility',
    vol_span: Optional[int] = None,
) -> pd.DataFrame:
    """Compute EWMA-based idiosyncratic momentum indicators.

    This is the primary alpha signal for price return prediction. It computes
    exponentially-weighted moving averages of idiosyncratic returns (after
    removing market factor) at multiple timescales, creating a panel of
    momentum indicators for each asset.

    The momentum at each span measures the recent trend in idiosyncratic returns,
    with shorter spans responding faster to changes and longer spans being smoother.

    Args:
        returns: Wide DataFrame with timestamp index and asset columns
        pca_loadings: Long DataFrame with (timestamp, asset, loading) for PC1 betas
        market_factor: Series with timestamp index containing PC1 returns
        spans: List of EWMA span parameters (in hours) to compute momentum for.
               Default: [2, 4, 8, 16, 32, 64] for multi-timescale panel
        normalization: How to normalize the momentum signal
            - 'volatility': Divide by EWMA volatility (default)
            - 'none': No normalization (raw EWMA)
        vol_span: Span for volatility EWMA. If None, uses max(spans) * 2

    Returns:
        Long DataFrame with (timestamp, asset, span) MultiIndex and 'momentum' column
        containing the momentum indicator values. For example:
            - idio_mom_2: Fast momentum (span=2)
            - idio_mom_4: Medium-fast momentum (span=4)
            - etc.

    Example:
        >>> returns = load_all_returns()
        >>> pca_data = pd.read_csv('data/features/pca_factor_H24_K30_sqrt.csv')
        >>> pca_data['timestamp'] = pd.to_datetime(pca_data['timestamp'])
        >>> loadings = pca_data.set_index(['timestamp', 'asset'])['loading']
        >>> market = pca_data.groupby('timestamp')['market_return'].first()
        >>>
        >>> # Compute panel of momentum indicators
        >>> momentum = idiosyncratic_momentum(
        ...     returns=returns,
        ...     pca_loadings=loadings,
        ...     market_factor=market,
        ...     spans=[2, 4, 8, 16, 32],
        ...     normalization='volatility'
        ... )
        >>>
        >>> # Access specific momentum: momentum.xs(8, level='span') for idio_mom_8
    """
    validate_returns_dataframe(returns, "returns")

    if spans is None:
        spans = [2, 4, 8, 16, 32, 64]

    # Step 1: Extract idiosyncratic returns
    idio_returns = compute_idiosyncratic_returns(returns, pca_loadings, market_factor)

    # Step 2: Compute volatility if needed for normalization
    if normalization == 'volatility':
        if vol_span is None:
            vol_span = max(spans) * 2

        # EWMA volatility
        volatility = idio_returns.ewm(span=vol_span, min_periods=vol_span // 2).std()
        # Avoid division by zero
        volatility = volatility.replace(0, np.nan)

    # Step 3: Compute EWMA momentum for each span
    momentum_panels = {}

    for span in spans:
        # EWMA of idiosyncratic returns
        momentum = idio_returns.ewm(span=span, min_periods=span // 2).mean()

        # Normalize by volatility if requested
        if normalization == 'volatility':
            momentum = momentum / volatility

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
