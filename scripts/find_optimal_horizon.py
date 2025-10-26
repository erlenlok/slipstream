#!/usr/bin/env python3
"""
Find optimal holding period (H*) for the Slipstream strategy.

Uses timescale-matched PCA: for each candidate holding period H, generates
PCA factor estimates at frequency H with lookback window K*H.

This solves the circular dependency between optimal H and optimal PCA parameters
by testing matched configurations systematically.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_pca_for_horizon(
    H_hours: int,
    K_multiplier: int,
    data_dir: Path,
    weight_method: str = "sqrt",
    interval: str = "1h",
) -> Path:
    """
    Build PCA factor matched to holding period H.

    Args:
        H_hours: Holding period in hours (also used as PCA estimation frequency)
        K_multiplier: Lookback window = K * H hours
        data_dir: Data directory
        weight_method: Volume weighting method
        interval: Base candle interval ("1h" or "4h")

    Returns:
        Path to generated PCA factor file
    """
    # Convert H to frequency string
    if H_hours == 1:
        freq = "H"
    elif H_hours == 24:
        freq = "D"
    elif H_hours == 168:
        freq = "W"
    else:
        freq = f"{H_hours}H"

    window_hours = K_multiplier * H_hours

    # Output path with H and K in filename
    output_path = data_dir / f"features/pca_factor_H{H_hours}_K{K_multiplier}_{weight_method}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Building PCA for H={H_hours}h, K={K_multiplier}, method={weight_method}")
    print(f"  Base interval: {interval}")
    print(f"  Frequency: {freq}")
    print(f"  Window: {window_hours} hours ({window_hours // H_hours} periods)")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # Call build_pca_factor.py with matched parameters
    cmd = [
        "python",
        "scripts/build_pca_factor.py",
        "--data-dir", str(data_dir),
        "--interval", interval,
        "--freq", freq,
        "--window", str(window_hours),
        "--weight-method", weight_method,
        "--output", str(output_path),
    ]

    subprocess.run(cmd, check=True)

    return output_path


def generate_pca_grid(
    H_candidates: list[int],
    K_multipliers: list[int],
    data_dir: Path,
    weight_methods: list[str],
    interval: str = "1h",
) -> None:
    """
    Generate all PCA factors for grid of (H, K, weight_method) combinations.

    Args:
        H_candidates: List of holding periods in hours to test
        K_multipliers: List of lookback window multipliers
        data_dir: Data directory
        weight_methods: List of volume weighting methods
    """
    total = len(H_candidates) * len(K_multipliers) * len(weight_methods)
    completed = 0

    print(f"\n{'#'*60}")
    print(f"# PCA Grid Generation: Timescale-Matched Approach")
    print(f"#")
    print(f"# H candidates: {H_candidates}")
    print(f"# K multipliers: {K_multipliers}")
    print(f"# Weight methods: {weight_methods}")
    print(f"# Total configurations: {total}")
    print(f"{'#'*60}\n")

    results = []

    for H in H_candidates:
        for K in K_multipliers:
            for method in weight_methods:
                completed += 1
                print(f"\n[{completed}/{total}] Processing H={H}h, K={K}, method={method}")

                try:
                    output_path = build_pca_for_horizon(
                        H_hours=H,
                        K_multiplier=K,
                        data_dir=data_dir,
                        weight_method=method,
                        interval=interval,
                    )
                    results.append({
                        "H": H,
                        "K": K,
                        "method": method,
                        "path": output_path,
                        "status": "success",
                    })
                except Exception as exc:
                    print(f"ERROR: Failed to build PCA for H={H}, K={K}, method={method}: {exc}")
                    results.append({
                        "H": H,
                        "K": K,
                        "method": method,
                        "path": None,
                        "status": f"failed: {exc}",
                    })

    # Print summary
    print(f"\n{'#'*60}")
    print(f"# Generation Complete")
    print(f"{'#'*60}\n")

    successful = sum(1 for r in results if r["status"] == "success")
    failed = total - successful

    print(f"Successful: {successful}/{total}")
    print(f"Failed: {failed}/{total}")

    if successful > 0:
        print(f"\nGenerated PCA factors:")
        for r in results:
            if r["status"] == "success":
                print(f"  - H={r['H']:3d}h K={r['K']:2d} {r['method']:12s} → {r['path']}")

    print("\nNext steps:")
    print("  1. Implement backtesting framework")
    print("  2. For each PCA factor file, run backtest with matching H")
    print("  3. Plot Sharpe ratio vs H to find H*")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate timescale-matched PCA factors for H* optimization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Coarse grid search (recommended starting point)
  python scripts/find_optimal_horizon.py --H 6 12 24 48 --K 30

  # Fine-tune around H=24 with multiple K values
  python scripts/find_optimal_horizon.py --H 20 24 28 --K 20 30 40

  # Test single configuration
  python scripts/find_optimal_horizon.py --H 24 --K 30 --weight-method sqrt

  # Full exploration
  python scripts/find_optimal_horizon.py --H 4 6 12 24 48 72 --K 20 30 40 60 --weight-method sqrt log sqrt_dollar
        """,
    )
    p.add_argument(
        "--H",
        type=int,
        nargs="+",
        default=[6, 12, 24, 48],
        help="Holding periods (hours) to test (default: 6 12 24 48)",
    )
    p.add_argument(
        "--K",
        type=int,
        nargs="+",
        default=[30],
        help="Lookback window multipliers (window = K × H) (default: 30)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)",
    )
    p.add_argument(
        "--weight-method",
        type=str,
        nargs="+",
        default=["sqrt"],
        choices=["none", "sqrt", "log", "dollar", "sqrt_dollar"],
        help="Volume weighting methods to test (default: sqrt)",
    )
    p.add_argument(
        "--interval",
        type=str,
        default="1h",
        choices=["1h", "4h"],
        help="Base candle interval (default: 1h)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    generate_pca_grid(
        H_candidates=args.H,
        K_multipliers=args.K,
        data_dir=args.data_dir,
        weight_methods=args.weight_method,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
