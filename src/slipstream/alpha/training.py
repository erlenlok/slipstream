"""
Alpha model training with Ridge regression and bootstrap.

Implements the bootstrap methodology adapted from Schmidhuber (2021) with:
- L2 regularization (Ridge regression) instead of OLS
- Cross-validated λ selection
- Bootstrap for coefficient distribution estimation
- Walk-forward cross-validation for out-of-sample R²

Reference: docs/ALPHA_MODEL_TRAINING.md Section 4-6
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from typing import Dict, List, Optional, Tuple

import pandas as pd


def find_optimal_lambda(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    cv_folds: int = 10
) -> float:
    """
    Find optimal regularization strength λ via cross-validation.

    Args:
        X: Feature matrix (n_samples x n_features)
        y: Target vector (n_samples,)
        alphas: Grid of λ values to try
        cv_folds: Number of CV folds

    Returns:
        Optimal λ value
    """
    print(f"Finding optimal λ via {cv_folds}-fold CV...")
    print(f"Testing λ ∈ {alphas}")

    ridge_cv = RidgeCV(alphas=alphas, cv=cv_folds)
    ridge_cv.fit(X, y)

    lambda_opt = ridge_cv.alpha_
    print(f"✓ Optimal λ = {lambda_opt:.4f}")

    return lambda_opt


def bootstrap_train_alpha_model(
    X: pd.DataFrame,
    y: pd.Series,
    lambda_reg: float,
    n_bootstrap: int = 1000,
    random_seed: int = 42
) -> Dict:
    """
    Train alpha model using bootstrap sampling with Ridge regression.

    Process:
    1. Sample timestamps with replacement (preserving all assets at each time)
    2. Fit Ridge(alpha=lambda_reg) on bootstrap sample
    3. Store coefficients
    4. Repeat n_bootstrap times
    5. Compute mean, std, t-statistics from distribution

    Args:
        X: Feature matrix with MultiIndex (timestamp, asset)
        y: Target vector with MultiIndex (timestamp, asset)
        lambda_reg: Regularization strength (from CV)
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with coefficient statistics and distributions
    """
    print(f"\nBootstrap training with λ={lambda_reg:.4f}, n={n_bootstrap}...")

    np.random.seed(random_seed)

    # Get unique timestamps
    timestamps = X.index.get_level_values('timestamp').unique()
    n_times = len(timestamps)

    print(f"Unique timestamps: {n_times}")
    print(f"Total samples: {len(X)}")

    # Storage for bootstrap results
    n_features = X.shape[1]
    beta_samples = np.zeros((n_bootstrap, n_features))
    r2_samples = np.zeros(n_bootstrap)

    # Bootstrap loop
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}...")

        # Sample timestamps with replacement
        sampled_times = np.random.choice(timestamps, size=n_times, replace=True)

        # Get all assets at sampled timestamps
        mask = X.index.get_level_values('timestamp').isin(sampled_times)
        X_boot = X[mask]
        y_boot = y[mask]

        # Fit Ridge regression with fixed λ
        model = Ridge(alpha=lambda_reg)
        model.fit(X_boot, y_boot)

        # Store coefficients and R²
        beta_samples[i] = model.coef_
        r2_samples[i] = model.score(X_boot, y_boot)

    # Compute statistics
    beta_mean = beta_samples.mean(axis=0)
    beta_std = beta_samples.std(axis=0)

    # t-statistics (16th-84th percentile for robustness to non-normality)
    beta_p16 = np.percentile(beta_samples, 16, axis=0)
    beta_p84 = np.percentile(beta_samples, 84, axis=0)
    beta_se = (beta_p84 - beta_p16) / 2  # Robust standard error
    t_stats = beta_mean / beta_se

    print(f"✓ Bootstrap complete")
    print(f"  Mean R² (in-sample): {r2_samples.mean():.6f}")
    print(f"  Std R² (in-sample): {r2_samples.std():.6f}")

    return {
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_se': beta_se,
        'beta_distribution': beta_samples,
        'r2_insample': r2_samples.mean(),
        'r2_std': r2_samples.std(),
        't_statistics': t_stats,
        'lambda': lambda_reg,
        'n_bootstrap': n_bootstrap
    }


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    lambda_reg: float,
    n_splits: int = 10,
    min_train_hours: int = 180 * 24  # 180 days
) -> Dict:
    """
    Walk-forward cross-validation with Ridge regression.

    Uses expanding window: train on all data BEFORE validation window.

    Args:
        X: Feature matrix with MultiIndex (timestamp, asset)
        y: Target vector with MultiIndex (timestamp, asset)
        lambda_reg: Regularization strength
        n_splits: Number of CV folds
        min_train_hours: Minimum training window size (hours)

    Returns:
        Dictionary with out-of-sample R² and predictions
    """
    print(f"\nWalk-forward CV with {n_splits} folds...")

    timestamps = X.index.get_level_values('timestamp').unique().sort_values()
    n_times = len(timestamps)
    fold_size = n_times // n_splits

    print(f"Timestamps: {n_times}")
    print(f"Fold size: ~{fold_size} timestamps")

    predictions_oos = []
    actuals_oos = []
    fold_r2 = []
    oos_indices: List[pd.MultiIndex] = []

    for i in range(n_splits):
        # Validation window
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_splits - 1 else n_times

        # Training window (all data before validation)
        if val_start < len(timestamps) // n_splits:  # Skip first fold if insufficient training data
            continue

        train_times = timestamps[:val_start]
        val_times = timestamps[val_start:val_end]

        # Split data
        train_mask = X.index.get_level_values('timestamp').isin(train_times)
        val_mask = X.index.get_level_values('timestamp').isin(val_times)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        if len(X_train) < 100:  # Skip if training set too small
            continue

        # Fit Ridge with optimal λ
        model = Ridge(alpha=lambda_reg)
        model.fit(X_train, y_train)

        # Predict on validation data
        y_pred = model.predict(X_val)

        # Store results
        predictions_oos.append(y_pred)
        actuals_oos.append(y_val.values)
        oos_indices.append(y_val.index)

        # Calculate fold R²
        ss_res = np.sum((y_val.values - y_pred) ** 2)
        ss_tot = np.sum((y_val.values - y_val.mean()) ** 2)
        r2_fold = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        fold_r2.append(r2_fold)
        print(f"  Fold {i+1}: R² = {r2_fold:.6f} (train={len(X_train)}, val={len(X_val)})")

    # Concatenate all OOS predictions
    all_predictions = np.concatenate(predictions_oos)
    all_actuals = np.concatenate(actuals_oos)

    # Overall OOS R²
    ss_res = np.sum((all_actuals - all_predictions) ** 2)
    ss_tot = np.sum((all_actuals - all_actuals.mean()) ** 2)
    r2_oos = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    print(f"✓ Out-of-sample R² = {r2_oos:.6f}")

    return {
        'r2_oos': r2_oos,
        'predictions_oos': all_predictions,
        'actuals_oos': all_actuals,
        'fold_r2': np.array(fold_r2),
        'mean_fold_r2': np.mean(fold_r2),
        'std_fold_r2': np.std(fold_r2)
    }


def _compute_quantile_statistics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_quantiles: int = 10
) -> pd.DataFrame:
    """Return mean realized value and t-stats per prediction quantile."""
    if len(predictions) == 0:
        return pd.DataFrame(
            columns=["quantile", "count", "pred_mean", "actual_mean", "actual_tstat"]
        )

    df = pd.DataFrame({"prediction": predictions, "actual": actuals})

    # Handle duplicate edges by adding small noise fallback
    try:
        df["quantile"] = pd.qcut(df["prediction"], q=n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        df["quantile"] = pd.cut(
            df["prediction"],
            bins=np.linspace(df["prediction"].min(), df["prediction"].max(), n_quantiles + 1),
            labels=False,
            include_lowest=True,
        )

    summary = (
        df.groupby("quantile")
        .apply(
            lambda g: pd.Series({
                "count": len(g),
                "pred_mean": g["prediction"].mean(),
                "actual_mean": g["actual"].mean(),
                "actual_std": g["actual"].std(ddof=1),
            })
        )
        .reset_index()
    )

    summary["actual_tstat"] = summary.apply(
        lambda row: row["actual_mean"] / (row["actual_std"] / np.sqrt(row["count"]))
        if row["actual_std"] and row["count"] > 1 else np.nan,
        axis=1,
    )
    summary = summary.drop(columns=["actual_std"])
    summary["quantile"] = summary["quantile"].astype(int)
    return summary


def train_alpha_model_complete(
    X: pd.DataFrame,
    y: pd.Series,
    H: int,
    n_bootstrap: int = 1000,
    alphas: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    cv_folds: int = 10,
    n_cv_splits: int = 10
) -> Dict:
    """
    Complete alpha model training pipeline.

    Steps:
    1. Find optimal λ via cross-validation
    2. Bootstrap training with optimal λ
    3. Walk-forward validation for out-of-sample R²
    4. Print comprehensive results

    Args:
        X: Feature matrix (momentum + funding features)
        y: Target vector (vol-normalized H-period forward returns)
        H: Holding period (for labeling results)
        n_bootstrap: Number of bootstrap samples
        alphas: Grid of λ values to try
        cv_folds: Number of folds for λ selection
        n_cv_splits: Number of folds for walk-forward CV

    Returns:
        Complete model specification + diagnostics
    """
    print(f"\n{'='*70}")
    print(f"ALPHA MODEL TRAINING (H={H} hours)")
    print(f"{'='*70}\n")

    # Step 1: Find optimal λ
    lambda_opt = find_optimal_lambda(X, y, alphas, cv_folds)

    # Step 2: Walk-forward CV for out-of-sample R²
    cv_results = walk_forward_cv(X, y, lambda_opt, n_cv_splits)
    r2_oos = cv_results['r2_oos']

    # Step 3: Bootstrap training with optimal λ
    bootstrap_results = bootstrap_train_alpha_model(X, y, lambda_opt, n_bootstrap)
    r2_in = bootstrap_results['r2_insample']

    # Step 4: Compile and print results
    correction = r2_in - r2_oos

    print(f"\n{'='*70}")
    print(f"FINAL MODEL SUMMARY (H={H})")
    print(f"{'='*70}")
    print(f"Regularization (λ):    {lambda_opt:.4f}")
    print(f"R² (in-sample):        {r2_in:.6f} ({r2_in*10000:.2f} bp)")
    print(f"R² (out-of-sample):    {r2_oos:.6f} ({r2_oos*10000:.2f} bp)")
    print(f"Correction:            {correction:.6f} ({100*correction/r2_in if r2_in > 0 else 0:.1f}% of R²)")
    print(f"Bootstrap samples:     {n_bootstrap}")
    print(f"CV folds:              {len(cv_results['fold_r2'])}")
    print(f"\nCoefficients:")
    print(f"{'Feature':<15} {'Beta':>10} {'Std Err':>10} {'t-stat':>8} {'Sig':>5}")
    print(f"{'-'*55}")

    for i, col in enumerate(X.columns):
        beta = bootstrap_results['beta_mean'][i]
        se = bootstrap_results['beta_se'][i]
        t = bootstrap_results['t_statistics'][i]
        sig = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.64 else ''
        print(f"{col:<15} {beta:>10.6f} {se:>10.6f} {t:>8.2f} {sig:>5}")

    print(f"\n*** p<0.01, ** p<0.05, * p<0.10")
    print(f"{'='*70}\n")

    # Count significant coefficients
    n_sig = np.sum(np.abs(bootstrap_results['t_statistics']) > 1.96)
    print(f"Significant coefficients (|t|>1.96): {n_sig}/{len(X.columns)}")

    quantile_stats = _compute_quantile_statistics(
        cv_results['predictions_oos'],
        cv_results['actuals_oos'],
        n_quantiles=10
    )

    if not quantile_stats.empty:
        print("\nQuantile analysis (prediction deciles):")
        print(f"{'Quantile':>8} {'Count':>8} {'Pred µ':>10} {'Actual µ':>10} {'t-stat':>10} {'Sig':>5}")
        print("-" * 52)
        for _, row in quantile_stats.iterrows():
            t = row["actual_tstat"]
            sig = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.64 else ''
            print(
                f"{int(row['quantile']):>8} "
                f"{int(row['count']):>8} "
                f"{row['pred_mean']:>10.4f} "
                f"{row['actual_mean']:>10.4f} "
                f"{t:>10.2f} {sig:>5}"
            )
        print("-" * 52)

    if r2_oos <= 0:
        print("⚠ WARNING: Out-of-sample R² ≤ 0 (no predictive power)")
    if correction / r2_in > 0.90 if r2_in > 0 else False:
        print("⚠ WARNING: Correction > 90% of R² (severe overfitting)")

    return {
        'H': H,
        'coefficients': bootstrap_results['beta_mean'],
        'std_errors': bootstrap_results['beta_se'],
        't_statistics': bootstrap_results['t_statistics'],
        'distribution': bootstrap_results['beta_distribution'],
        'lambda': lambda_opt,
        'r2_insample': r2_in,
        'r2_oos': r2_oos,
        'r2_oos_bp': r2_oos * 10000,
        'correction': correction,
        'correction_pct': 100 * correction / r2_in if r2_in > 0 else 0,
        'feature_names': X.columns.tolist(),
        'cv_results': cv_results,
        'bootstrap_results': bootstrap_results,
        'n_significant': n_sig,
        'quantile_stats': quantile_stats
    }
