"""
Find optimal holding period H* for alpha model.

This script trains separate alpha models for different holding periods and
identifies H* that maximizes out-of-sample R².

The alpha has term structure - different H values will have different predictive power.

Usage:
    python scripts/find_optimal_H_alpha.py --H 6 12 24 48 --n-bootstrap 1000

Output:
    - Model results for each H
    - Comparison table
    - Best H* recommendation
    - Saved models in data/features/alpha_models/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from slipstream.signals import idiosyncratic_momentum
from slipstream.alpha import prepare_alpha_training_data, train_alpha_model_complete
from slipstream.alpha.data_prep import load_all_returns, load_all_funding, BASE_INTERVAL_HOURS


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
        Tuple of (loadings, market_factor)
    """
    # Load timescale-matched PCA file
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
    print(f"  Unique timestamps: {len(loadings_wide.index)}")
    print(f"  Unique assets: {loadings_wide.columns.nunique()}")
    print(f"  Date range: {loadings_wide.index.min()} to {loadings_wide.index.max()}")

    metadata = pca_data[metadata_cols] if metadata_cols[0] in pca_data else None

    return loadings_wide, metadata


def train_model_for_H(
    H: int,
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    n_bootstrap: int = 1000,
    pca_method: str = "sqrt"
) -> dict:
    """
    Train alpha model for a specific holding period H.

    Args:
        H: Holding period in hours
        returns: Raw returns (wide format)
        funding: Funding rates (wide format)
        n_bootstrap: Number of bootstrap samples
        pca_method: PCA volume weighting method

    Returns:
        Model results dictionary
    """
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL FOR H={H} HOURS")
    print(f"{'='*70}\n")

    try:
        if H % BASE_INTERVAL_HOURS != 0:
            raise ValueError(f"H={H} must be a multiple of {BASE_INTERVAL_HOURS} hours.")

        # 1. Load PCA data for this H
        raw_loadings, _ = load_pca_data(H, pca_method)

        # 2. Align PCA loadings to the 4-hour grid
        loadings_expanded = raw_loadings.reindex(returns.index, method="ffill")
        valid_rows = loadings_expanded.notna().any(axis=1)
        loadings_expanded = loadings_expanded[valid_rows]

        if loadings_expanded.empty:
            raise ValueError("No PCA loadings available after alignment to returns index.")

        # 3. Align assets across all datasets
        common_assets = (
            returns.columns
            .intersection(loadings_expanded.columns)
            .intersection(funding.columns)
        )
        print(f"Common assets across PCA, returns, and funding: {len(common_assets)}")

        if len(common_assets) == 0:
            raise ValueError("No overlapping assets between PCA loadings, returns, and funding.")

        returns_aligned = returns.loc[loadings_expanded.index, common_assets]
        funding_aligned = funding.loc[loadings_expanded.index, common_assets]
        loadings_expanded = loadings_expanded[common_assets]

        # Drop rows that still have insufficient data
        joint_valid_rows = (
            loadings_expanded.notna().any(axis=1)
            & returns_aligned.notna().any(axis=1)
        )
        loadings_expanded = loadings_expanded[joint_valid_rows]
        returns_aligned = returns_aligned.loc[loadings_expanded.index]
        funding_aligned = funding_aligned.loc[loadings_expanded.index]

        if loadings_expanded.empty:
            raise ValueError("No overlapping timestamps between loadings and returns after filtering.")

        # 4. Compute market factor from loadings and returns
        print("\nComputing market factor...")
        market_factor = compute_market_factor_from_loadings(loadings_expanded, returns_aligned)
        market_factor = market_factor.loc[returns_aligned.index]
        print(f"Market factor shape: {market_factor.shape}")

        valid_factor_mask = market_factor.notna()
        if not valid_factor_mask.any():
            raise ValueError("Market factor computation produced all NaNs.")

        returns_aligned = returns_aligned.loc[valid_factor_mask]
        funding_aligned = funding_aligned.loc[valid_factor_mask]
        loadings_expanded = loadings_expanded.loc[valid_factor_mask]
        market_factor = market_factor.loc[valid_factor_mask]

        loadings_long = loadings_expanded.stack().rename('loading')

        # 5. Compute idiosyncratic momentum signals
        print("\nComputing momentum signals...")
        momentum_panel = idiosyncratic_momentum(
            returns=returns_aligned,
            pca_loadings=loadings_long,
            market_factor=market_factor,
            spans=[2, 4, 8, 16, 32, 64],
            normalization='volatility',
            base_interval_hours=BASE_INTERVAL_HOURS,
        )

        print(f"Momentum panel shape: {momentum_panel.shape}")

        if momentum_panel.empty:
            raise ValueError("Momentum panel is empty after alignment; check data coverage.")

        # 6. Compute idiosyncratic returns (for forward return target)
        print("\nComputing idiosyncratic returns...")
        from slipstream.signals import compute_idiosyncratic_returns
        idio_returns = compute_idiosyncratic_returns(returns_aligned, loadings_long, market_factor)

        print(f"Idiosyncratic returns shape: {idio_returns.shape}")

        # 7. Prepare training data
        print("\nPreparing training data...")
        X, y, vol = prepare_alpha_training_data(
            idio_returns=idio_returns,
            funding_rates=funding_aligned,
            momentum_panel=momentum_panel,
            H=H,
            spans=[2, 4, 8, 16, 32, 64],
            vol_span=128,
            base_interval_hours=BASE_INTERVAL_HOURS,
        )

        # 8. Train model
        results = train_alpha_model_complete(
            X=X,
            y=y,
            H=H,
            n_bootstrap=n_bootstrap,
            alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv_folds=10,
            n_cv_splits=10
        )

        return results

    except Exception as e:
        print(f"✗ ERROR training model for H={H}: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_models(results_dict: dict) -> pd.DataFrame:
    """
    Compare models across different H values.

    Args:
        results_dict: Dictionary mapping H -> model results

    Returns:
        Comparison DataFrame
    """
    comparison = []

    for H, results in results_dict.items():
        if results is None:
            continue

        comparison.append({
            'H': H,
            'R²_oos': results['r2_oos'],
            'R²_oos_bp': results['r2_oos_bp'],
            'R²_in': results['r2_insample'],
            'Correction_%': results['correction_pct'],
            'Lambda': results['lambda'],
            'N_sig_coefs': results['n_significant'],
            'Mean_fold_R²': results['cv_results']['mean_fold_r2'],
            'Std_fold_R²': results['cv_results']['std_fold_r2']
        })

    df = pd.DataFrame(comparison)
    df = df.sort_values('R²_oos', ascending=False)

    return df


def main():
    parser = argparse.ArgumentParser(description='Find optimal holding period H* for alpha model')
    parser.add_argument('--H', nargs='+', type=int, default=[4, 8, 12, 24, 48],
                        help='Holding periods to test (hours)')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples')
    parser.add_argument('--pca-method', type=str, default='sqrt',
                        choices=['sqrt', 'log', 'sqrt_dollar'],
                        help='PCA volume weighting method')
    parser.add_argument('--output-dir', type=str, default='data/features/alpha_models',
                        help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ALPHA MODEL H* OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Holding periods: {args.H}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"PCA method: {args.pca_method}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

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

    # Train model for each H
    results_dict = {}

    for H in args.H:
        results = train_model_for_H(
            H=H,
            returns=returns,
            funding=funding,
            n_bootstrap=args.n_bootstrap,
            pca_method=args.pca_method
        )

        if results is not None:
            results_dict[H] = results

            # Save individual model
            model_file = output_dir / f"alpha_model_H{H}.json"
            with open(model_file, 'w') as f:
                # Convert numpy arrays and scalars to Python types for JSON serialization
                def convert_to_python(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                        return int(obj)
                    elif isinstance(obj, (np.float64, np.float32, np.float16)):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_to_python(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_python(item) for item in obj]
                    else:
                        return obj

                save_results = {
                    k: convert_to_python(v)
                    for k, v in results.items()
                    if k not in ['distribution', 'cv_results', 'bootstrap_results']
                }
                json.dump(save_results, f, indent=2)
            print(f"✓ Saved model to {model_file}")

    # Compare all models
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}\n")

    comparison = compare_models(results_dict)
    print(comparison.to_string(index=False))

    # Identify H*
    if len(comparison) > 0:
        best_H = comparison.iloc[0]['H']
        best_r2_oos = comparison.iloc[0]['R²_oos']
        best_r2_bp = comparison.iloc[0]['R²_oos_bp']

        print(f"\n{'='*70}")
        print(f"RECOMMENDATION")
        print(f"{'='*70}")
        print(f"✓ Optimal holding period: H* = {best_H} hours")
        print(f"  Out-of-sample R² = {best_r2_oos:.6f} ({best_r2_bp:.2f} bp)")
        print(f"{'='*70}\n")

        # Save comparison table
        comparison_file = output_dir / "H_comparison.csv"
        comparison.to_csv(comparison_file, index=False)
        print(f"✓ Saved comparison table to {comparison_file}")

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'H_optimal': int(best_H),
            'R2_oos_optimal': float(best_r2_oos),
            'R2_oos_bp': float(best_r2_bp),
            'H_tested': [int(h) for h in args.H],
            'pca_method': args.pca_method,
            'n_bootstrap': args.n_bootstrap
        }

        summary_file = output_dir / "optimization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary to {summary_file}")

    else:
        print("✗ No models trained successfully")


if __name__ == "__main__":
    main()
