#!/usr/bin/env python3
"""
Gradient Concentration Sensitivity Visualization

Generate plots from sensitivity analysis results.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_n_pct_vs_return(
    results_df: pd.DataFrame,
    output_path: Path,
    rebal_freqs_to_plot: list = None
):
    """
    Plot n% vs expected annualized return with error bands.

    Args:
        results_df: Results DataFrame
        output_path: Path to save plot
        rebal_freqs_to_plot: List of rebalance frequencies to plot (None = all)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    weight_schemes = results_df["weight_scheme"].unique()

    for idx, scheme in enumerate(weight_schemes):
        ax = axes[idx]
        scheme_data = results_df[results_df["weight_scheme"] == scheme]

        rebal_freqs = sorted(scheme_data["rebalance_freq_h"].unique())
        if rebal_freqs_to_plot:
            rebal_freqs = [f for f in rebal_freqs if f in rebal_freqs_to_plot]

        colors = plt.cm.viridis(np.linspace(0, 1, len(rebal_freqs)))

        for rebal_freq, color in zip(rebal_freqs, colors):
            freq_data = scheme_data[scheme_data["rebalance_freq_h"] == rebal_freq]
            freq_data = freq_data.sort_values("n_pct")

            ax.plot(
                freq_data["n_pct"],
                freq_data["mean_return_pct"],
                label=f"{rebal_freq}h",
                color=color,
                linewidth=2,
            )

            # Error bands (±1 std)
            ax.fill_between(
                freq_data["n_pct"],
                freq_data["mean_return_pct"] - freq_data["std_return_pct"],
                freq_data["mean_return_pct"] + freq_data["std_return_pct"],
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel("Concentration (n%)", fontsize=12)
        ax.set_ylabel("10-Day Return (%)", fontsize=12)
        ax.set_title(f"{scheme.replace('_', ' ').title()} Weighting", fontsize=14, fontweight="bold")
        ax.legend(title="Rebalance Freq", fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved n% vs return plot to {output_path}")
    plt.close()


def plot_sharpe_heatmap(results_df: pd.DataFrame, output_path: Path):
    """
    Plot heatmap of Sharpe ratio across n% and rebalance frequency.

    Args:
        results_df: Results DataFrame
        output_path: Path to save plot
    """
    weight_schemes = results_df["weight_scheme"].unique()

    fig, axes = plt.subplots(1, len(weight_schemes), figsize=(8 * len(weight_schemes), 6))

    if len(weight_schemes) == 1:
        axes = [axes]

    for idx, scheme in enumerate(weight_schemes):
        ax = axes[idx]
        scheme_data = results_df[results_df["weight_scheme"] == scheme]

        # Pivot to create heatmap data
        pivot_data = scheme_data.pivot_table(
            index="rebalance_freq_h",
            columns="n_pct",
            values="sharpe",
            aggfunc="mean"
        )

        sns.heatmap(
            pivot_data,
            ax=ax,
            cmap="RdYlGn",
            center=0,
            annot=False,
            fmt=".2f",
            cbar_kws={"label": "Sharpe Ratio"},
        )

        ax.set_xlabel("Concentration (n%)", fontsize=12)
        ax.set_ylabel("Rebalance Frequency (hours)", fontsize=12)
        ax.set_title(f"{scheme.replace('_', ' ').title()} - Sharpe Ratio", fontsize=14, fontweight="bold")
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved Sharpe heatmap to {output_path}")
    plt.close()


def plot_return_distributions(
    results_df: pd.DataFrame,
    output_path: Path,
    n_pcts_to_plot: list = [5, 10, 20, 30, 40, 50]
):
    """
    Plot return distributions for selected n% values.

    Args:
        results_df: Results DataFrame
        output_path: Path to save plot
        n_pcts_to_plot: List of n% values to plot
    """
    weight_schemes = results_df["weight_scheme"].unique()

    fig, axes = plt.subplots(len(weight_schemes), 1, figsize=(14, 6 * len(weight_schemes)))

    if len(weight_schemes) == 1:
        axes = [axes]

    for idx, scheme in enumerate(weight_schemes):
        ax = axes[idx]
        scheme_data = results_df[results_df["weight_scheme"] == scheme]

        # For each n%, plot distribution of mean returns across all rebal frequencies
        data_to_plot = []
        labels = []

        for n_pct in n_pcts_to_plot:
            n_data = scheme_data[scheme_data["n_pct"] == n_pct]
            if len(n_data) > 0:
                data_to_plot.append(n_data["mean_return_pct"].values)
                labels.append(f"{n_pct}%")

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(data_to_plot)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Concentration (n%)", fontsize=12)
        ax.set_ylabel("10-Day Return (%)", fontsize=12)
        ax.set_title(f"{scheme.replace('_', ' ').title()} - Return Distributions", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved return distributions to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from gradient concentration sensitivity analysis"
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/gradient/sensitivity"),
        help="Directory containing results CSVs",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gradient/sensitivity/plots"),
        help="Output directory for plots",
    )

    parser.add_argument(
        "--rebal-freqs",
        type=int,
        nargs="+",
        default=None,
        help="Rebalance frequencies to plot (default: all)",
    )

    parser.add_argument(
        "--n-pcts",
        type=int,
        nargs="+",
        default=[5, 10, 20, 30, 40, 50],
        help="N% values to include in distribution plots",
    )

    args = parser.parse_args()

    # Load results
    print("Loading results...")
    results_files = list(args.results_dir.glob("results_*.csv"))

    if not results_files:
        print(f"Error: No results files found in {args.results_dir}")
        return

    results_list = []
    for file in results_files:
        df = pd.read_csv(file)
        results_list.append(df)

    results_df = pd.concat(results_list, ignore_index=True)
    print(f"Loaded {len(results_df)} result rows")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")

    print("1. N% vs Return plot...")
    plot_n_pct_vs_return(
        results_df,
        args.output_dir / "n_pct_vs_return.png",
        rebal_freqs_to_plot=args.rebal_freqs
    )

    print("2. Sharpe ratio heatmap...")
    plot_sharpe_heatmap(results_df, args.output_dir / "sharpe_heatmap.png")

    print("3. Return distributions...")
    plot_return_distributions(
        results_df,
        args.output_dir / "return_distributions.png",
        n_pcts_to_plot=args.n_pcts
    )

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print(f"Plots saved to {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
