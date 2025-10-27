"""
Run full end-to-end backtest with trained models.

This script:
1. Loads trained alpha and funding models
2. Generates predictions on historical data
3. Constructs portfolios using beta-neutral optimization
4. Simulates trading with realistic costs
5. Produces performance reports and visualizations

Usage:
    python scripts/run_full_backtest.py --H 8 --start 2024-01-01 --end 2024-12-31
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from slipstream.alpha.data_prep import load_all_returns, load_all_funding, BASE_INTERVAL_HOURS
from slipstream.signals import idiosyncratic_momentum, compute_idiosyncratic_returns
from slipstream.portfolio import (
    run_backtest,
    BacktestConfig,
    TransactionCostModel,
    optimize_portfolio,
)
from slipstream.portfolio.risk import compute_total_covariance


def load_trained_models(H: int, models_dir: Path = Path("data/features")):
    """Load trained alpha and funding models for given H."""
    alpha_file = models_dir / f"alpha_models/alpha_model_H{H}.json"
    funding_file = models_dir / f"funding_models/funding_model_H{H}.json"

    if not alpha_file.exists():
        raise FileNotFoundError(f"Alpha model not found: {alpha_file}")
    if not funding_file.exists():
        raise FileNotFoundError(f"Funding model not found: {funding_file}")

    with open(alpha_file) as f:
        alpha_model = json.load(f)

    with open(funding_file) as f:
        funding_model = json.load(f)

    print(f"✓ Loaded models for H={H}")
    print(f"  Alpha R² (OOS): {alpha_model['r2_oos']:.6f}")
    print(f"  Funding R² (OOS): {funding_model['r2_oos']:.6f}")

    return alpha_model, funding_model


def load_pca_factors(H: int, method: str = "sqrt", features_dir: Path = Path("data/features")):
    """Load PCA factors for given H."""
    pca_file = features_dir / f"pca_factor_H{H}_K30_{method}.csv"

    if not pca_file.exists():
        raise FileNotFoundError(
            f"PCA factors not found: {pca_file}\n"
            f"Run: python scripts/find_optimal_horizon.py --H {H} --K 30 --weight-method {method}"
        )

    pca_data = pd.read_csv(pca_file, index_col=0, parse_dates=True)
    print(f"✓ Loaded PCA factors: {pca_data.shape}")

    # Extract loadings
    metadata_cols = ['_variance_explained', '_n_assets']
    asset_cols = [col for col in pca_data.columns if col not in metadata_cols]
    loadings_wide = pca_data[asset_cols]

    return loadings_wide


def compute_market_factor(loadings_wide: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Compute market factor from loadings and returns."""
    # Align loadings with returns
    loadings_aligned = loadings_wide.reindex(returns.index, method="ffill")

    # Common assets
    common_assets = loadings_aligned.columns.intersection(returns.columns)
    loadings_aligned = loadings_aligned[common_assets]
    returns_aligned = returns[common_assets]

    # Drop rows with no valid loadings
    valid_rows = loadings_aligned.notna().any(axis=1)
    loadings_aligned = loadings_aligned[valid_rows]
    returns_aligned = returns_aligned.loc[loadings_aligned.index]

    if loadings_aligned.empty:
        return pd.Series(dtype=float)

    # Compute weighted average
    joint_valid = (~loadings_aligned.isna()) & (~returns_aligned.isna())
    weighted_returns = returns_aligned.where(joint_valid, 0.0) * loadings_aligned.where(joint_valid, 0.0)
    numerator = weighted_returns.sum(axis=1)

    beta_norm = (loadings_aligned.where(joint_valid, 0.0) ** 2).sum(axis=1)
    beta_norm = beta_norm.replace(0, np.nan)

    market_factor = numerator / np.sqrt(beta_norm)
    return market_factor


def generate_alpha_predictions(
    alpha_model: dict,
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    loadings_wide: pd.DataFrame,
    market_factor: pd.Series,
    H: int,
) -> pd.DataFrame:
    """Generate alpha predictions using trained model."""
    print("\nGenerating alpha predictions...")

    # Align data
    loadings_expanded = loadings_wide.reindex(returns.index, method="ffill")
    valid_rows = loadings_expanded.notna().any(axis=1)
    loadings_expanded = loadings_expanded[valid_rows]

    common_assets = returns.columns.intersection(loadings_expanded.columns).intersection(funding.columns)
    returns_aligned = returns.loc[loadings_expanded.index, common_assets]
    funding_aligned = funding.reindex(loadings_expanded.index, method="ffill")[common_assets]
    loadings_expanded = loadings_expanded[common_assets]

    # Filter valid rows
    joint_valid_rows = (
        loadings_expanded.notna().any(axis=1)
        & returns_aligned.notna().any(axis=1)
    )
    loadings_expanded = loadings_expanded[joint_valid_rows]
    returns_aligned = returns_aligned.loc[loadings_expanded.index]
    funding_aligned = funding_aligned.loc[loadings_expanded.index]

    market_factor = market_factor.loc[returns_aligned.index]
    valid_factor_mask = market_factor.notna()
    returns_aligned = returns_aligned.loc[valid_factor_mask]
    funding_aligned = funding_aligned.loc[valid_factor_mask]
    loadings_expanded = loadings_expanded.loc[valid_factor_mask]
    market_factor = market_factor.loc[valid_factor_mask]

    loadings_long = loadings_expanded.stack().rename('loading')

    # Compute momentum features
    momentum_panel = idiosyncratic_momentum(
        returns=returns_aligned,
        pca_loadings=loadings_long,
        market_factor=market_factor,
        spans=[2, 4, 8, 16, 32, 64],
        normalization='volatility',
        base_interval_hours=BASE_INTERVAL_HOURS,
    )

    # Compute funding features
    from slipstream.alpha.data_prep import compute_funding_features
    funding_features = compute_funding_features(
        funding_aligned,
        spans=[2, 4, 8, 16, 32, 64],
        vol_span=128,
        base_interval_hours=BASE_INTERVAL_HOURS,
        clip=5.0,
    )

    # Merge features
    momentum_wide = momentum_panel.unstack(level='span')
    # Handle both MultiIndex and Index cases
    if isinstance(momentum_wide.columns, pd.MultiIndex):
        momentum_wide.columns = [f'mom_{int(s)}' for _, s in momentum_wide.columns]
    else:
        momentum_wide.columns = [f'mom_{int(s)}' for s in momentum_wide.columns]

    # Align with funding features
    common_idx = momentum_wide.index.intersection(funding_features.index)
    X = pd.concat([momentum_wide.loc[common_idx], funding_features.loc[common_idx]], axis=1)

    # Ensure feature order matches model
    X = X[alpha_model['feature_names']]

    # Generate predictions
    coefficients = np.array(alpha_model['coefficients'])
    predictions = X @ coefficients

    # Convert to wide format
    predictions_df = predictions.unstack(level='asset')

    print(f"✓ Generated alpha predictions: {predictions_df.shape}")
    return predictions_df


def generate_funding_predictions(
    funding_model: dict,
    funding: pd.DataFrame,
    H: int,
) -> pd.DataFrame:
    """Generate funding predictions using trained model."""
    print("\nGenerating funding predictions...")

    from slipstream.funding import prepare_funding_training_data

    try:
        X, y, vol = prepare_funding_training_data(
            funding_rates=funding,
            H=H,
            spans=[2, 4, 8, 16, 32, 64],
            vol_span=128,
            base_interval_hours=BASE_INTERVAL_HOURS,
        )
    except Exception as e:
        print(f"⚠ Error preparing funding data: {e}")
        # Return zeros as fallback
        return pd.DataFrame(0, index=funding.index, columns=funding.columns)

    # Generate predictions
    coefficients = np.array(funding_model['coefficients'])
    predictions = X @ coefficients

    # Convert to wide format
    predictions_df = predictions.unstack(level='asset')

    print(f"✓ Generated funding predictions: {predictions_df.shape}")
    return predictions_df


def main():
    parser = argparse.ArgumentParser(description="Run full backtest with trained models")
    parser.add_argument("--H", type=int, default=8, help="Holding period in hours")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--leverage", type=float, default=1.0, help="Portfolio leverage")
    parser.add_argument("--no-costs", action="store_true", help="Disable transaction costs")
    parser.add_argument("--pca-method", type=str, default="sqrt", help="PCA weighting method")
    parser.add_argument("--output", type=str, default="backtest_results", help="Output directory")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"FULL BACKTEST SIMULATION (H={args.H} hours)")
    print(f"{'='*70}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Leverage: {args.leverage}x")
    print(f"Costs: {'Disabled' if args.no_costs else 'Enabled'}")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading trained models...")
    alpha_model, funding_model = load_trained_models(args.H)

    # Load data
    print("\nLoading market data...")
    returns = load_all_returns()
    funding = load_all_funding()

    print(f"✓ Returns: {returns.shape}")
    print(f"✓ Funding: {funding.shape}")

    # Load PCA factors
    print("\nLoading PCA factors...")
    loadings_wide = load_pca_factors(args.H, args.pca_method)

    # Compute market factor
    print("\nComputing market factor...")
    market_factor = compute_market_factor(loadings_wide, returns)
    print(f"✓ Market factor: {market_factor.shape}")

    # Generate predictions
    alpha_pred = generate_alpha_predictions(
        alpha_model, returns, funding, loadings_wide, market_factor, args.H
    )

    funding_pred = generate_funding_predictions(
        funding_model, funding, args.H
    )

    # Align predictions to common time grid
    # Resample both to H-hour intervals starting from epoch (aligns to UTC midnight)
    alpha_pred = alpha_pred.resample(f'{args.H}H', origin='epoch', offset='0H').last()
    funding_pred = funding_pred.resample(f'{args.H}H', origin='epoch', offset='0H').last()

    # Drop NaNs after resampling
    alpha_pred = alpha_pred.dropna(how='all')
    funding_pred = funding_pred.dropna(how='all')

    # Intersect timestamps
    common_timestamps = alpha_pred.index.intersection(funding_pred.index)

    # Filter by date range
    start_date = pd.to_datetime(args.start, utc=True)
    end_date = pd.to_datetime(args.end, utc=True)

    common_timestamps = common_timestamps[
        (common_timestamps >= start_date) & (common_timestamps <= end_date)
    ]

    alpha_pred = alpha_pred.loc[common_timestamps]
    funding_pred = funding_pred.loc[common_timestamps]

    print(f"\n✓ Aligned predictions: {len(common_timestamps)} timestamps")

    # Load cost model
    print("\nLoading cost model...")
    liquidity_file = Path("data/features/liquidity_metrics.csv")
    if liquidity_file.exists():
        liquidity_df = pd.read_csv(liquidity_file, index_col=0)
        cost_model = TransactionCostModel.from_liquidity_metrics(
            assets=list(alpha_pred.columns),
            liquidity_df=liquidity_df,
        )
        print("✓ Loaded liquidity-adjusted cost model")
    else:
        cost_model = TransactionCostModel.create_default(len(alpha_pred.columns))
        print("⚠ Using default cost model (liquidity_metrics.csv not found)")

    # Prepare beta (use loadings as beta)
    beta = loadings_wide.reindex(alpha_pred.index, method="ffill")
    beta = beta[alpha_pred.columns].fillna(1.0)

    print(f"\nRunning backtest...")
    print(f"Timestamps: {len(common_timestamps)}")
    print(f"Assets: {len(alpha_pred.columns)}")

    config = BacktestConfig(
        H=args.H,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        leverage=args.leverage,
        use_costs=not args.no_costs,
    )

    # Align all data to the backtest period
    realized_returns = returns.loc[common_timestamps, alpha_pred.columns]
    realized_funding = funding.loc[common_timestamps, alpha_pred.columns]
    
    # The covariance matrix S needs to be a dictionary of numpy arrays,
    # where each key is a timestamp. For this script, we'll compute a single
    # expanding covariance matrix for simplicity.
    print("\nComputing covariance matrix...")
    S_dict = {}
    for t in common_timestamps:
        # In a real scenario, you would compute this on a rolling or expanding window
        # For this example, we'll just use a fixed window of returns
        window_end = t
        window_start = window_end - pd.Timedelta(days=30)
        returns_window = returns.loc[window_start:window_end, alpha_pred.columns]
        if not returns_window.empty:
            S_dict[t] = returns_window.cov().values
        else:
            # Fallback to identity matrix if no data
            S_dict[t] = np.eye(len(alpha_pred.columns))
    print(f"✓ Computed covariance matrices for {len(S_dict)} timestamps")


    result = run_backtest(
        config=config,
        alpha_price=alpha_pred,
        alpha_funding=funding_pred,
        beta=beta,
        S=S_dict,
        realized_returns=realized_returns,
        realized_funding=realized_funding,
        cost_model=cost_model,
    )

    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print(result.summary())
    print("="*70 + "\n")

    # Save results
    summary_file = output_dir / f"backtest_summary_H{args.H}.json"
    with open(summary_file, 'w') as f:
        json.dump(result.summary(), f, indent=4)
    print(f"✓ Saved backtest summary to {summary_file}")

    equity_curve_file = output_dir / f"equity_curve_H{args.H}.csv"
    result.equity_curve.to_csv(equity_curve_file)
    print(f"✓ Saved equity curve to {equity_curve_file}")

    trades_file = output_dir / f"trades_H{args.H}.csv"
    result.trades.to_csv(trades_file)
    print(f"✓ Saved trades to {trades_file}")

    # Plot equity curve
    plt.figure(figsize=(12, 8))
    result.equity_curve.plot(title=f"Backtest Equity Curve (H={args.H})")
    plt.grid(True)
    plt.savefig(output_dir / f"equity_curve_H{args.H}.png")
    print(f"✓ Saved equity curve plot to {output_dir}/")




if __name__ == "__main__":
    main()
