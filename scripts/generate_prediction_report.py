"""
Generate performance report from prediction files.

This script creates visualizations and statistics for the generated predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_predictions(H: int, output_dir: Path = Path("backtest_results")):
    """Load saved predictions."""
    alpha_file = output_dir / f"alpha_predictions_H{H}.csv"
    funding_file = output_dir / f"funding_predictions_H{H}.csv"
    combined_file = output_dir / f"combined_alpha_H{H}.csv"

    alpha_pred = pd.read_csv(alpha_file, index_col=0, parse_dates=True)
    funding_pred = pd.read_csv(funding_file, index_col=0, parse_dates=True)
    combined_alpha = pd.read_csv(combined_file, index_col=0, parse_dates=True)

    return alpha_pred, funding_pred, combined_alpha


def compute_cross_sectional_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional statistics at each timestamp."""
    stats = pd.DataFrame({
        'mean': df.mean(axis=1),
        'median': df.median(axis=1),
        'std': df.std(axis=1),
        'q10': df.quantile(0.1, axis=1),
        'q90': df.quantile(0.9, axis=1),
    })
    return stats


def plot_time_series(alpha_pred, funding_pred, combined_alpha, output_dir):
    """Create time series plots."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Alpha predictions
    alpha_stats = compute_cross_sectional_stats(alpha_pred)
    ax = axes[0]
    ax.plot(alpha_stats.index, alpha_stats['mean'], label='Mean', linewidth=2)
    ax.fill_between(alpha_stats.index, alpha_stats['q10'], alpha_stats['q90'],
                     alpha=0.3, label='10th-90th percentile')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_title('Alpha Predictions (Price Returns)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Alpha (bps)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Funding predictions
    funding_stats = compute_cross_sectional_stats(funding_pred)
    ax = axes[1]
    ax.plot(funding_stats.index, funding_stats['mean'], label='Mean', linewidth=2, color='orange')
    ax.fill_between(funding_stats.index, funding_stats['q10'], funding_stats['q90'],
                     alpha=0.3, label='10th-90th percentile', color='orange')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_title('Funding Predictions (Expected Cost)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Funding (bps)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Combined alpha
    combined_stats = compute_cross_sectional_stats(combined_alpha)
    ax = axes[2]
    ax.plot(combined_stats.index, combined_stats['mean'], label='Mean', linewidth=2, color='green')
    ax.fill_between(combined_stats.index, combined_stats['q10'], combined_stats['q90'],
                     alpha=0.3, label='10th-90th percentile', color='green')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_title('Combined Alpha (α_price - F_hat)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Combined Alpha (bps)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_timeseries.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved time series plot: {output_dir / 'prediction_timeseries.png'}")
    plt.close()


def plot_distributions(alpha_pred, funding_pred, combined_alpha, output_dir):
    """Create distribution plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Alpha distribution
    ax = axes[0]
    alpha_flat = alpha_pred.values.flatten()
    alpha_flat = alpha_flat[~np.isnan(alpha_flat)]
    ax.hist(alpha_flat, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(alpha_flat), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(alpha_flat):.3f}')
    ax.set_title('Alpha Predictions Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Alpha (bps)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Funding distribution
    ax = axes[1]
    funding_flat = funding_pred.values.flatten()
    funding_flat = funding_flat[~np.isnan(funding_flat)]
    ax.hist(funding_flat, bins=100, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(funding_flat), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(funding_flat):.3f}')
    ax.set_title('Funding Predictions Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Funding (bps)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Combined alpha distribution
    ax = axes[2]
    combined_flat = combined_alpha.values.flatten()
    combined_flat = combined_flat[~np.isnan(combined_flat)]
    ax.hist(combined_flat, bins=100, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(combined_flat), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(combined_flat):.3f}')
    ax.set_title('Combined Alpha Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Combined Alpha (bps)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved distribution plot: {output_dir / 'prediction_distributions.png'}")
    plt.close()


def generate_summary_stats(alpha_pred, funding_pred, combined_alpha, output_dir):
    """Generate summary statistics."""
    def get_stats(df, name):
        flat = df.values.flatten()
        flat = flat[~np.isnan(flat)]
        return {
            'Component': name,
            'Mean': np.mean(flat),
            'Median': np.median(flat),
            'Std': np.std(flat),
            'Min': np.min(flat),
            'Max': np.max(flat),
            'Q10': np.percentile(flat, 10),
            'Q90': np.percentile(flat, 90),
            'N': len(flat),
        }

    stats = pd.DataFrame([
        get_stats(alpha_pred, 'Alpha (Price)'),
        get_stats(funding_pred, 'Funding'),
        get_stats(combined_alpha, 'Combined Alpha'),
    ])

    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY STATISTICS")
    print("=" * 80)
    print(stats.to_string(index=False))
    print("=" * 80 + "\n")

    # Save to CSV
    stats.to_csv(output_dir / 'prediction_summary.csv', index=False)
    print(f"✓ Saved summary statistics: {output_dir / 'prediction_summary.csv'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate prediction report")
    parser.add_argument("--H", type=int, default=8, help="Holding period in hours")
    parser.add_argument("--output", type=str, default="backtest_results", help="Output directory")

    args = parser.parse_args()
    output_dir = Path(args.output)

    print(f"\n{'='*80}")
    print(f"GENERATING PREDICTION REPORT (H={args.H} hours)")
    print(f"{'='*80}\n")

    # Load predictions
    print("Loading predictions...")
    alpha_pred, funding_pred, combined_alpha = load_predictions(args.H, output_dir)

    print(f"✓ Loaded predictions:")
    print(f"  Alpha: {alpha_pred.shape}")
    print(f"  Funding: {funding_pred.shape}")
    print(f"  Combined: {combined_alpha.shape}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_time_series(alpha_pred, funding_pred, combined_alpha, output_dir)
    plot_distributions(alpha_pred, funding_pred, combined_alpha, output_dir)

    # Generate summary statistics
    generate_summary_stats(alpha_pred, funding_pred, combined_alpha, output_dir)

    print(f"\n✓ Report generation complete!")
    print(f"  Time series: {output_dir / 'prediction_timeseries.png'}")
    print(f"  Distributions: {output_dir / 'prediction_distributions.png'}")
    print(f"  Statistics: {output_dir / 'prediction_summary.csv'}\n")


if __name__ == "__main__":
    main()
