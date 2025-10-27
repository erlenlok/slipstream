"""
Find optimal holding period H* via joint alpha + funding model optimization.

This script trains both price-alpha and funding models for each H, combines them
into α_total = α_price - F_hat, and evaluates the joint signal quality.

The optimal H* maximizes the predictive power of the combined signal.

Usage:
    python scripts/find_optimal_H_joint.py --H 4 8 12 24 48 --n-bootstrap 1000

Output:
    - Joint model results for each H
    - Combined quantile diagnostics showing alpha, funding, and total signal
    - Comparison table
    - Best H* recommendation
    - Saved models in data/features/joint_models/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from slipstream.alpha import prepare_alpha_training_data, train_alpha_model_complete
from slipstream.alpha.data_prep import load_all_returns, load_all_funding, BASE_INTERVAL_HOURS
from slipstream.funding import prepare_funding_training_data
from slipstream.signals import idiosyncratic_momentum, compute_idiosyncratic_returns


def compute_market_factor_from_loadings(
    loadings_wide: pd.DataFrame,
    returns: pd.DataFrame
) -> pd.Series:
    """
    Compute market factor as weighted average of returns using PCA loadings.

    Args:
        loadings_wide: DataFrame with timestamp index and asset columns
        returns: Wide DataFrame with timestamp index and asset columns

    Returns:
        Series with timestamp index containing market factor returns
    """
    # Align loadings with return index (forward fill to cover intra-period bars)
    loadings_aligned = loadings_wide.reindex(returns.index, method="ffill")

    # Restrict to common assets
    common_assets = loadings_aligned.columns.intersection(returns.columns)
    loadings_aligned = loadings_aligned[common_assets]
    returns_aligned = returns[common_assets]

    # Drop rows with no valid loadings
    valid_rows = loadings_aligned.notna().any(axis=1)
    loadings_aligned = loadings_aligned[valid_rows]
    returns_aligned = returns_aligned.loc[loadings_aligned.index]

    if loadings_aligned.empty:
        return pd.Series(dtype=float)

    # Only use entries where both return and loading exist
    joint_valid = (~loadings_aligned.isna()) & (~returns_aligned.isna())

    weighted_returns = returns_aligned.where(joint_valid, 0.0) * loadings_aligned.where(joint_valid, 0.0)
    numerator = weighted_returns.sum(axis=1)

    beta_norm = (loadings_aligned.where(joint_valid, 0.0) ** 2).sum(axis=1)
    beta_norm = beta_norm.replace(0, np.nan)

    market_factor = numerator / np.sqrt(beta_norm)
    return market_factor


def load_pca_data(H: int, method: str = "sqrt") -> tuple:
    """
    Load PCA factor data for given holding period H.

    Args:
        H: Holding period in hours
        method: Volume weighting method (sqrt, log, sqrt_dollar)

    Returns:
        Tuple of (loadings, metadata)
    """
    pca_file = Path(f"data/features/pca_factor_H{H}_K30_{method}.csv")

    if not pca_file.exists():
        raise FileNotFoundError(
            f"PCA factor file not found: {pca_file}\n"
            f"Run: python scripts/find_optimal_horizon.py --H {H} --K 30 --weight-method {method}"
        )

    print(f"Loading PCA data from {pca_file}...")
    pca_data = pd.read_csv(pca_file, index_col=0, parse_dates=True)

    # Extract asset columns (exclude metadata columns)
    metadata_cols = ['_variance_explained', '_n_assets']
    asset_cols = [col for col in pca_data.columns if col not in metadata_cols]

    loadings_wide = pca_data[asset_cols]

    print(f"  PCA loadings shape: {loadings_wide.shape}")
    print(f"  Date range: {loadings_wide.index.min()} to {loadings_wide.index.max()}")

    metadata = pca_data[metadata_cols] if metadata_cols[0] in pca_data else None

    return loadings_wide, metadata


def compute_combined_quantile_table(
    alpha_pred: pd.Series,
    funding_pred: pd.Series,
    alpha_actual: pd.Series,
    funding_actual: pd.Series,
    n_quantiles: int = 10
) -> pd.DataFrame:
    """
    Compute quantile diagnostics for alpha, funding, and combined signal.

    Shows that the combined signal (α_total = α_price - F_hat) has stronger
    predictive power than either component alone.

    Args:
        alpha_pred: Predicted price alpha
        funding_pred: Predicted funding
        alpha_actual: Actual forward returns
        funding_actual: Actual forward funding
        n_quantiles: Number of quantile bins

    Returns:
        DataFrame with quantile statistics for all three signals
    """
    # Align all series
    common_idx = alpha_pred.index.intersection(funding_pred.index) \
                    .intersection(alpha_actual.index) \
                    .intersection(funding_actual.index)

    alpha_pred = alpha_pred.loc[common_idx]
    funding_pred = funding_pred.loc[common_idx]
    alpha_actual = alpha_actual.loc[common_idx]
    funding_actual = funding_actual.loc[common_idx]

    # Compute combined signal: α_total = α_price - F_hat
    total_pred = alpha_pred - funding_pred
    total_actual = alpha_actual - funding_actual

    # Bin by combined signal quantiles
    quantile_bins = pd.qcut(total_pred, q=n_quantiles, labels=False, duplicates='drop')

    results = []
    for q in range(n_quantiles):
        mask = quantile_bins == q
        n = mask.sum()

        if n == 0:
            continue

        # Alpha component
        alpha_pred_mean = alpha_pred[mask].mean()
        alpha_actual_mean = alpha_actual[mask].mean()
        alpha_actual_std = alpha_actual[mask].std()
        alpha_t = alpha_actual_mean / (alpha_actual_std / np.sqrt(n)) if alpha_actual_std > 0 else 0

        # Funding component
        funding_pred_mean = funding_pred[mask].mean()
        funding_actual_mean = funding_actual[mask].mean()
        funding_actual_std = funding_actual[mask].std()
        funding_t = funding_actual_mean / (funding_actual_std / np.sqrt(n)) if funding_actual_std > 0 else 0

        # Combined total
        total_pred_mean = total_pred[mask].mean()
        total_actual_mean = total_actual[mask].mean()
        total_actual_std = total_actual[mask].std()
        total_t = total_actual_mean / (total_actual_std / np.sqrt(n)) if total_actual_std > 0 else 0

        results.append({
            'Quantile': q,
            'Count': n,
            'α_pred_µ': alpha_pred_mean,
            'α_actual_µ': alpha_actual_mean,
            'α_t': alpha_t,
            'F_pred_µ': funding_pred_mean,
            'F_actual_µ': funding_actual_mean,
            'F_t': funding_t,
            'Total_pred_µ': total_pred_mean,
            'Total_actual_µ': total_actual_mean,
            'Total_t': total_t,
        })

    df = pd.DataFrame(results)
    return df


def train_joint_model_for_H(
    H: int,
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    n_bootstrap: int = 1000,
    pca_method: str = "sqrt",
    spans: list = None,
    vol_span: int = 128
) -> dict:
    """
    Train both alpha and funding models for a specific holding period H.

    Returns combined results including joint quantile diagnostics.
    """
    if spans is None:
        spans = [2, 4, 8, 16, 32, 64]

    print(f"\n{'='*70}")
    print(f"JOINT MODEL TRAINING FOR H={H} HOURS")
    print(f"{'='*70}\n")

    try:
        if H % BASE_INTERVAL_HOURS != 0:
            raise ValueError(f"H={H} must be a multiple of {BASE_INTERVAL_HOURS} hours.")

        # =====================================================================
        # PART 1: TRAIN ALPHA MODEL
        # =====================================================================
        print(f"\n{'─'*70}")
        print("PART 1: PRICE ALPHA MODEL")
        print(f"{'─'*70}\n")

        # Load PCA data
        raw_loadings, _ = load_pca_data(H, pca_method)
        loadings_expanded = raw_loadings.reindex(returns.index, method="ffill")
        valid_rows = loadings_expanded.notna().any(axis=1)
        loadings_expanded = loadings_expanded[valid_rows]

        # Align assets
        common_assets = (
            returns.columns
            .intersection(loadings_expanded.columns)
            .intersection(funding.columns)
        )
        print(f"Common assets: {len(common_assets)}")

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

        # Compute market factor
        print("Computing market factor...")
        market_factor = compute_market_factor_from_loadings(loadings_expanded, returns_aligned)
        market_factor = market_factor.loc[returns_aligned.index]

        valid_factor_mask = market_factor.notna()
        returns_aligned = returns_aligned.loc[valid_factor_mask]
        funding_aligned = funding_aligned.loc[valid_factor_mask]
        loadings_expanded = loadings_expanded.loc[valid_factor_mask]
        market_factor = market_factor.loc[valid_factor_mask]

        loadings_long = loadings_expanded.stack().rename('loading')

        # Compute momentum signals
        print("Computing momentum signals...")
        momentum_panel = idiosyncratic_momentum(
            returns=returns_aligned,
            pca_loadings=loadings_long,
            market_factor=market_factor,
            spans=spans,
            normalization='volatility',
            base_interval_hours=BASE_INTERVAL_HOURS,
        )

        # Compute idiosyncratic returns
        print("Computing idiosyncratic returns...")
        idio_returns = compute_idiosyncratic_returns(returns_aligned, loadings_long, market_factor)

        # Prepare alpha training data
        print("Preparing alpha training data...")
        X_alpha, y_alpha, vol_alpha = prepare_alpha_training_data(
            idio_returns=idio_returns,
            funding_rates=funding_aligned,
            momentum_panel=momentum_panel,
            H=H,
            spans=spans,
            vol_span=vol_span,
            base_interval_hours=BASE_INTERVAL_HOURS,
        )

        # Train alpha model
        print("\nTraining alpha model...")
        alpha_results = train_alpha_model_complete(
            X=X_alpha,
            y=y_alpha,
            H=H,
            n_bootstrap=n_bootstrap,
            alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv_folds=10,
            n_cv_splits=10,
            label="ALPHA"
        )

        # =====================================================================
        # PART 2: TRAIN FUNDING MODEL
        # =====================================================================
        print(f"\n{'─'*70}")
        print("PART 2: FUNDING MODEL")
        print(f"{'─'*70}\n")

        X_funding, y_funding, vol_funding = prepare_funding_training_data(
            funding_rates=funding_aligned,
            H=H,
            spans=spans,
            vol_span=vol_span,
            base_interval_hours=BASE_INTERVAL_HOURS,
        )

        print("Training funding model...")
        funding_results = train_alpha_model_complete(
            X=X_funding,
            y=y_funding,
            H=H,
            n_bootstrap=n_bootstrap,
            alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv_folds=10,
            n_cv_splits=10,
            label="FUNDING"
        )

        # =====================================================================
        # PART 3: JOINT ANALYSIS
        # =====================================================================
        print(f"\n{'─'*70}")
        print("PART 3: JOINT SIGNAL ANALYSIS")
        print(f"{'─'*70}\n")

        # Get predictions on common sample
        # Note: Both models use walk-forward CV, so we extract OOS predictions
        # These are numpy arrays from CV, so we convert to Series for alignment
        alpha_pred = pd.Series(alpha_results['cv_results']['predictions_oos'])
        alpha_actual = pd.Series(alpha_results['cv_results']['actuals_oos'])

        funding_pred = pd.Series(funding_results['cv_results']['predictions_oos'])
        funding_actual = pd.Series(funding_results['cv_results']['actuals_oos'])

        # Compute combined quantile table
        print("Computing combined quantile diagnostics...\n")
        quantile_table = compute_combined_quantile_table(
            alpha_pred=alpha_pred,
            funding_pred=funding_pred,
            alpha_actual=alpha_actual,
            funding_actual=funding_actual,
            n_quantiles=10
        )

        print("COMBINED SIGNAL QUANTILE ANALYSIS")
        print("(Binned by α_total = α_price - F_hat)\n")
        print(quantile_table.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

        # Compute combined R² on aligned sample
        common_idx = alpha_pred.index.intersection(funding_pred.index) \
                        .intersection(alpha_actual.index) \
                        .intersection(funding_actual.index)

        alpha_pred_aligned = alpha_pred.loc[common_idx]
        funding_pred_aligned = funding_pred.loc[common_idx]
        alpha_actual_aligned = alpha_actual.loc[common_idx]
        funding_actual_aligned = funding_actual.loc[common_idx]

        total_pred = alpha_pred_aligned - funding_pred_aligned
        total_actual = alpha_actual_aligned - funding_actual_aligned

        # Combined R²
        ss_res = ((total_actual - total_pred) ** 2).sum()
        ss_tot = ((total_actual - total_actual.mean()) ** 2).sum()
        r2_combined = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"\n{'='*70}")
        print(f"JOINT MODEL SUMMARY (H={H})")
        print(f"{'='*70}")
        print(f"Alpha R² (OOS):        {alpha_results['r2_oos']:.6f} ({alpha_results['r2_oos_bp']:.2f} bp)")
        print(f"Funding R² (OOS):      {funding_results['r2_oos']:.6f} ({funding_results['r2_oos_bp']:.2f} bp)")
        print(f"Combined R² (OOS):     {r2_combined:.6f} ({r2_combined*10000:.2f} bp)")
        print(f"{'='*70}\n")

        # Return combined results
        return {
            'H': H,
            'alpha_results': alpha_results,
            'funding_results': funding_results,
            'quantile_table': quantile_table,
            'r2_alpha': alpha_results['r2_oos'],
            'r2_funding': funding_results['r2_oos'],
            'r2_combined': r2_combined,
            'r2_combined_bp': r2_combined * 10000,
        }

    except Exception as e:
        print(f"✗ ERROR training joint model for H={H}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_joint_models(results_dict: dict) -> pd.DataFrame:
    """
    Compare joint models across different H values.
    """
    comparison = []

    for H, results in results_dict.items():
        if results is None:
            continue

        alpha_res = results['alpha_results']
        funding_res = results['funding_results']

        comparison.append({
            'H': H,
            'R²_alpha': results['r2_alpha'],
            'R²_funding': results['r2_funding'],
            'R²_combined': results['r2_combined'],
            'R²_combined_bp': results['r2_combined_bp'],
            'Alpha_λ': alpha_res['lambda'],
            'Funding_λ': funding_res['lambda'],
            'Alpha_sig_coefs': alpha_res['n_significant'],
            'Funding_sig_coefs': funding_res['n_significant'],
        })

    df = pd.DataFrame(comparison)
    if not df.empty:
        df = df.sort_values('R²_combined', ascending=False)

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal holding period H* via joint alpha + funding optimization'
    )
    parser.add_argument('--H', nargs='+', type=int, default=[4, 8, 12, 24, 48],
                        help='Holding periods to test (hours)')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples')
    parser.add_argument('--pca-method', type=str, default='sqrt',
                        choices=['sqrt', 'log', 'sqrt_dollar'],
                        help='PCA volume weighting method')
    parser.add_argument('--output-dir', type=str, default='data/features/joint_models',
                        help='Output directory for results')
    parser.add_argument('--spans', nargs='+', type=int, default=[2, 4, 8, 16, 32, 64],
                        help='EWMA spans (hours) for features')
    parser.add_argument('--vol-span', type=int, default=128,
                        help='EWMA span (hours) for volatility normalization')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"JOINT H* OPTIMIZATION (ALPHA + FUNDING)")
    print(f"{'='*70}")
    print(f"Holding periods: {args.H}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"PCA method: {args.pca_method}")
    print(f"Feature spans: {args.spans}")
    print(f"Volatility span: {args.vol_span}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    # Validate horizons
    invalid_horizons = [h for h in args.H if h % BASE_INTERVAL_HOURS != 0]
    if invalid_horizons:
        raise ValueError(
            f"Horizons must be multiples of {BASE_INTERVAL_HOURS} hours. "
            f"Invalid values: {invalid_horizons}"
        )

    # Load data (common across all H)
    print("Loading market data...")
    returns = load_all_returns()
    funding = load_all_funding()

    # Ensure same assets
    common_assets = returns.columns.intersection(funding.columns)
    returns = returns[common_assets]
    funding = funding[common_assets]

    print(f"\nCommon assets: {len(common_assets)}")
    print(f"Returns shape: {returns.shape}")
    print(f"Funding shape: {funding.shape}")

    # Train joint model for each H
    results_dict = {}

    for H in args.H:
        results = train_joint_model_for_H(
            H=H,
            returns=returns,
            funding=funding,
            n_bootstrap=args.n_bootstrap,
            pca_method=args.pca_method,
            spans=args.spans,
            vol_span=args.vol_span
        )

        if results is not None:
            results_dict[H] = results

            # Save individual joint model
            model_file = output_dir / f"joint_model_H{H}.json"
            with open(model_file, 'w') as f:
                def convert(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.Series):
                        return obj.to_dict()
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient='records')
                    elif isinstance(obj, (np.integer,)):
                        return int(obj)
                    elif isinstance(obj, (np.floating,)):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert(item) for item in obj]
                    else:
                        return obj

                save_results = {
                    'H': results['H'],
                    'r2_alpha': float(results['r2_alpha']),
                    'r2_funding': float(results['r2_funding']),
                    'r2_combined': float(results['r2_combined']),
                    'r2_combined_bp': float(results['r2_combined_bp']),
                    'quantile_table': convert(results['quantile_table']),
                    'alpha_lambda': float(results['alpha_results']['lambda']),
                    'funding_lambda': float(results['funding_results']['lambda']),
                }
                json.dump(save_results, f, indent=2)
            print(f"✓ Saved joint model to {model_file}\n")

    # Compare all models
    print(f"\n{'='*70}")
    print(f"JOINT MODEL COMPARISON")
    print(f"{'='*70}\n")

    comparison = compare_joint_models(results_dict)
    if comparison.empty:
        print("✗ No models trained successfully")
        return

    print(comparison.to_string(index=False))

    # Identify H*
    best_H = comparison.iloc[0]['H']
    best_r2_combined = comparison.iloc[0]['R²_combined']
    best_r2_combined_bp = comparison.iloc[0]['R²_combined_bp']

    print(f"\n{'='*70}")
    print(f"RECOMMENDATION")
    print(f"{'='*70}")
    print(f"✓ Optimal holding period: H* = {best_H} hours")
    print(f"  Combined R² (OOS) = {best_r2_combined:.6f} ({best_r2_combined_bp:.2f} bp)")
    print(f"{'='*70}\n")

    # Save comparison table
    comparison_file = output_dir / "H_comparison.csv"
    comparison.to_csv(comparison_file, index=False)
    print(f"✓ Saved comparison table to {comparison_file}")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'H_optimal': int(best_H),
        'R2_combined_optimal': float(best_r2_combined),
        'R2_combined_bp': float(best_r2_combined_bp),
        'H_tested': [int(h) for h in args.H],
        'pca_method': args.pca_method,
        'n_bootstrap': args.n_bootstrap,
        'spans': args.spans,
        'vol_span': args.vol_span,
    }

    summary_file = output_dir / "optimization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
