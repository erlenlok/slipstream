#!/usr/bin/env python3
"""
Gradient Concentration Sensitivity Analysis

This script runs a comprehensive sensitivity analysis to determine the optimal
portfolio concentration (n%) for the Gradient strategy.
"""

import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slipstream.strategies.gradient.sensitivity import (
    SensitivityConfig,
    build_panel_data,
    generate_sample_periods,
    run_sensitivity_sweep,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run concentration sensitivity analysis for Gradient strategy"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/market_data"),
        help="Directory containing 4h candle CSVs",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gradient/sensitivity"),
        help="Output directory for results",
    )

    parser.add_argument(
        "--panel-only",
        action="store_true",
        help="Only build panel data, don't run sensitivity sweep",
    )

    parser.add_argument(
        "--use-existing-panel",
        action="store_true",
        help="Use existing panel data if available",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples (K) to use for each configuration",
    )

    parser.add_argument(
        "--sample-days",
        type=int,
        default=10,
        help="Length of each sample period in days",
    )

    parser.add_argument(
        "--n-pct-min",
        type=int,
        default=1,
        help="Minimum concentration percentage",
    )

    parser.add_argument(
        "--n-pct-max",
        type=int,
        default=50,
        help="Maximum concentration percentage",
    )

    parser.add_argument(
        "--n-pct-step",
        type=int,
        default=1,
        help="Step size for concentration percentage",
    )

    parser.add_argument(
        "--rebal-min",
        type=int,
        default=4,
        help="Minimum rebalance frequency in hours",
    )

    parser.add_argument(
        "--rebal-max",
        type=int,
        default=48,
        help="Maximum rebalance frequency in hours",
    )

    parser.add_argument(
        "--rebal-step",
        type=int,
        default=4,
        help="Step size for rebalance frequency in hours",
    )

    parser.add_argument(
        "--lookbacks",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        help="EWMA lookback spans in hours",
    )

    parser.add_argument(
        "--vol-span",
        type=int,
        default=24,
        help="EWMA span for volatility estimation in hours",
    )

    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0,
        help="Transaction fee rate as decimal (e.g., 0.000144 for 0.0144%%)",
    )

    args = parser.parse_args()

    # Create configuration
    config = SensitivityConfig(
        lookback_spans=args.lookbacks,
        vol_span=args.vol_span,
        fee_rate=args.fee_rate,
        n_pct_range=list(range(args.n_pct_min, args.n_pct_max + 1, args.n_pct_step)),
        rebalance_freqs_hours=list(range(args.rebal_min, args.rebal_max + 1, args.rebal_step)),
        n_samples=args.n_samples,
        sample_period_days=args.sample_days,
    )

    print("=" * 80)
    print("Gradient Concentration Sensitivity Analysis")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Lookback spans: {config.lookback_spans}")
    print(f"Vol span: {config.vol_span}h")
    print(f"Fee rate: {args.fee_rate:.6f} ({args.fee_rate * 100:.4f}%)")
    print(f"N% range: {args.n_pct_min}-{args.n_pct_max}% (step {args.n_pct_step})")
    print(f"Rebalance frequencies: {args.rebal_min}-{args.rebal_max}h (step {args.rebal_step})")
    print(f"Samples per config: {args.n_samples}")
    print(f"Sample period: {args.sample_days} days")
    print("=" * 80)

    # Build or load panel data
    panel_path = args.output_dir / "panel_data.csv"

    if args.use_existing_panel and panel_path.exists():
        print(f"\nLoading existing panel data from {panel_path}...")
        import pandas as pd
        panel = pd.read_csv(panel_path)
        panel["timestamp"] = pd.to_datetime(panel["timestamp"])
        print(f"Loaded {len(panel)} rows for {panel['asset'].nunique()} assets")
    else:
        print("\nBuilding panel data...")
        panel = build_panel_data(
            data_dir=args.data_dir,
            config=config,
            output_path=panel_path
        )

    if args.panel_only:
        print("\nPanel data built. Exiting (--panel-only flag set).")
        return

    # Generate sample periods
    print("\nGenerating sample periods...")
    sample_periods_path = args.output_dir / "sample_periods.csv"
    sample_periods = generate_sample_periods(
        panel=panel,
        n_samples=config.n_samples,
        period_days=config.sample_period_days
    )
    sample_periods.to_csv(sample_periods_path, index=False)
    print(f"Generated {len(sample_periods)} sample periods")
    print(f"Saved to {sample_periods_path}")

    # Run sensitivity sweep
    print("\nRunning sensitivity sweep...")
    print(f"Total configurations: {len(config.n_pct_range)} n% × "
          f"{len(config.rebalance_freqs_hours)} rebal_freq × "
          f"{len(config.weight_schemes)} weight_schemes = "
          f"{len(config.n_pct_range) * len(config.rebalance_freqs_hours) * len(config.weight_schemes)}")
    print(f"Total backtests: {len(sample_periods) * len(config.n_pct_range) * len(config.rebalance_freqs_hours) * len(config.weight_schemes)}")
    print()

    results_df = run_sensitivity_sweep(
        panel=panel,
        sample_periods=sample_periods,
        config=config,
        output_dir=args.output_dir
    )

    # Find and report optimal configuration
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    # Top 10 by Sharpe ratio
    top_configs = results_df.nlargest(10, "sharpe")
    print("\nTop 10 configurations by Sharpe ratio:")
    print(top_configs[["n_pct", "rebalance_freq_h", "weight_scheme", "mean_return_pct", "sharpe"]].to_string(index=False))

    # Save optimal config
    optimal = top_configs.iloc[0]
    import json
    optimal_config = {
        "n_pct": float(optimal["n_pct"]),
        "rebalance_freq_h": int(optimal["rebalance_freq_h"]),
        "weight_scheme": optimal["weight_scheme"],
        "mean_return_pct": float(optimal["mean_return_pct"]),
        "std_return_pct": float(optimal["std_return_pct"]),
        "sharpe": float(optimal["sharpe"]),
    }

    optimal_path = args.output_dir / "optimal_config.json"
    with open(optimal_path, "w") as f:
        json.dump(optimal_config, f, indent=2)

    print(f"\nOptimal configuration saved to {optimal_path}")
    print("\nNext steps:")
    print("1. Run visualization script to generate plots")
    print("2. Review results in notebook for detailed analysis")
    print("3. Update Gradient strategy with optimal parameters")


if __name__ == "__main__":
    main()
