"""
Search for optimal funding-forecast horizon H*.

Mirrors the alpha workflow but trains models that predict forward funding
payments, which plug directly into the optimizer's carry term.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from slipstream.alpha.training import train_alpha_model_complete
from slipstream.alpha.data_prep import load_all_funding, BASE_INTERVAL_HOURS
from slipstream.funding import prepare_funding_training_data


def train_model_for_H(
    H: int,
    funding: np.ndarray,
    n_bootstrap: int = 1000,
    spans: list[int] | None = None,
    vol_span: int = 128,
) -> dict | None:
    if spans is None:
        spans = [2, 4, 8, 16, 32, 64]

    print(f"\n{'='*70}")
    print(f"TRAINING FUNDING MODEL FOR H={H} HOURS")
    print(f"{'='*70}\n")

    try:
        X, y, vol = prepare_funding_training_data(
            funding_rates=funding,
            H=H,
            spans=spans,
            vol_span=vol_span,
            base_interval_hours=BASE_INTERVAL_HOURS,
        )

        results = train_alpha_model_complete(
            X=X,
            y=y,
            H=H,
            n_bootstrap=n_bootstrap,
            alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            cv_folds=10,
            n_cv_splits=10,
            label="Funding",
        )

        results["vol_scale"] = vol
        return results

    except Exception as exc:  # noqa: BLE001
        print(f"✗ ERROR training funding model for H={H}: {exc}")
        import traceback

        traceback.print_exc()
        return None


def compare_models(results_dict: dict[int, dict]) -> np.ndarray:
    import pandas as pd

    comparison = []
    for H, results in results_dict.items():
        if results is None:
            continue

        comparison.append({
            "H": H,
            "R²_oos": results["r2_oos"],
            "R²_oos_bp": results["r2_oos_bp"],
            "R²_in": results["r2_insample"],
            "Correction_%": results["correction_pct"],
            "Lambda": results["lambda"],
            "N_sig_coefs": results["n_significant"],
            "Mean_fold_R²": results["cv_results"]["mean_fold_r2"],
            "Std_fold_R²": results["cv_results"]["std_fold_r2"],
        })

    df = pd.DataFrame(comparison)
    if not df.empty:
        df = df.sort_values("R²_oos", ascending=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Find optimal holding period H* for funding model")
    parser.add_argument("--H", nargs="+", type=int, default=[4, 8, 12, 24, 48], help="Holding periods to test (hours)")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--output-dir", type=str, default="data/features/funding_models", help="Output directory for results")
    parser.add_argument("--spans", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64], help="EWMA spans (in hours) for funding features")
    parser.add_argument("--vol-span", type=int, default=128, help="EWMA span (hours) for funding normalisation")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    invalid = [h for h in args.H if h % BASE_INTERVAL_HOURS != 0]
    if invalid:
        raise ValueError(
            f"Funding model horizons must be multiples of {BASE_INTERVAL_HOURS} hours. Invalid values: {invalid}"
        )

    print(f"\n{'='*70}")
    print("FUNDING MODEL H* OPTIMISATION")
    print(f"{'='*70}")
    print(f"Holding periods: {args.H}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Feature spans (hours): {args.spans}")
    print(f"Volatility span (hours): {args.vol_span}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    print("Loading funding data...")
    funding = load_all_funding()

    results_dict: dict[int, dict] = {}

    for H in args.H:
        results = train_model_for_H(
            H=H,
            funding=funding,
            n_bootstrap=args.n_bootstrap,
            spans=args.spans,
            vol_span=args.vol_span,
        )

        if results is None:
            continue

        results_dict[H] = results

        # Serialise run summary
        model_file = output_dir / f"funding_model_H{H}.json"
        with open(model_file, "w") as f:
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                import pandas as pd  # local import to avoid top-level dependency

                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient="records")
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj

            serialisable = {
                k: convert(v)
                for k, v in results.items()
                if k not in ["distribution", "cv_results", "bootstrap_results", "vol_scale"]
            }
            json.dump(serialisable, f, indent=2)
        print(f"✓ Saved model summary to {model_file}")

    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}\n")

    comparison = compare_models(results_dict)
    if comparison.empty:
        print("No funding models trained successfully.")
        return

    print(comparison.to_string(index=False))

    best_row = comparison.iloc[0]
    best_H = best_row["H"]
    best_r2 = best_row["R²_oos"]
    best_r2_bp = best_row["R²_oos_bp"]

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print(f"✓ Optimal holding period: H* = {best_H} hours")
    print(f"  Out-of-sample R² = {best_r2:.6f} ({best_r2_bp:.2f} bp)")
    print(f"{'='*70}\n")

    comparison_file = output_dir / "H_comparison.csv"
    comparison.to_csv(comparison_file, index=False)
    print(f"✓ Saved comparison table to {comparison_file}")

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "H_optimal": int(best_H),
        "R2_oos_optimal": float(best_r2),
        "R2_oos_bp": float(best_r2_bp),
        "H_tested": [int(h) for h in args.H],
        "n_bootstrap": args.n_bootstrap,
        "spans": args.spans,
        "vol_span": args.vol_span,
    }
    summary_file = output_dir / "optimization_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
