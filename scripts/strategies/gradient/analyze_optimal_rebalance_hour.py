"""
Analyze which 4-hour candle close is optimal for 24h rebalancing.

This script examines the sensitivity backtest results and determines which
specific hour of day (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC) produces
the best returns when rebalancing once per day.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slipstream.strategies.gradient.sensitivity import run_concentration_backtest


def analyze_rebalance_hour_sensitivity(
    panel_path: Path,
    sample_periods_path: Path,
    n_pct: float = 30.0,
    weight_scheme: str = "inverse_vol",
    fee_rate: float = 0.000144
):
    """
    Test each possible 4-hour rebalance time within a 24-hour window.

    Args:
        panel_path: Path to panel_data.csv
        sample_periods_path: Path to sample_periods.csv
        n_pct: Concentration percentage (default 30%)
        weight_scheme: "equal" or "inverse_vol" (default "inverse_vol")
        fee_rate: Trading fee rate (default 0.0144% = 0.000144)
    """
    print(f"Loading panel data from {panel_path}...")
    panel = pd.read_csv(panel_path)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])

    print(f"Loading sample periods from {sample_periods_path}...")
    sample_periods = pd.read_csv(sample_periods_path)
    sample_periods["start_time"] = pd.to_datetime(sample_periods["start_time"])
    sample_periods["end_time"] = pd.to_datetime(sample_periods["end_time"])

    # All possible 4-hour candle close times in a day (UTC)
    rebalance_hours = [0, 4, 8, 12, 16, 20]

    print(f"\nTesting 30% concentration with 24h rebalance at different hours...")
    print(f"Weight scheme: {weight_scheme}, Fee rate: {fee_rate:.6f}")
    print("-" * 80)

    results_by_hour = {}

    for hour in rebalance_hours:
        print(f"\n=== Testing rebalance at {hour:02d}:00 UTC ===")

        sample_returns = []

        for idx, sample in sample_periods.iterrows():
            start_time = sample["start_time"]
            end_time = sample["end_time"]

            # Filter panel to this sample period
            mask = (panel["timestamp"] >= start_time) & (panel["timestamp"] <= end_time)
            period_data = panel[mask].copy()

            if len(period_data) == 0:
                continue

            # Get all timestamps in period
            timestamps = sorted(period_data["timestamp"].unique())

            # Find rebalance times: all timestamps that match the target hour
            rebalance_times = [
                ts for ts in timestamps
                if ts.hour == hour
            ]

            if len(rebalance_times) < 2:
                # Need at least 2 rebalances to measure returns
                continue

            # Run custom backtest with these specific rebalance times
            period_return = run_backtest_with_fixed_times(
                panel=period_data,
                rebalance_times=rebalance_times,
                n_pct=n_pct,
                weight_scheme=weight_scheme,
                fee_rate=fee_rate
            )

            if not np.isnan(period_return):
                sample_returns.append(period_return)

        if len(sample_returns) > 0:
            mean_ret = np.mean(sample_returns)
            std_ret = np.std(sample_returns)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0

            results_by_hour[hour] = {
                "hour": hour,
                "mean_return_pct": mean_ret,
                "std_return_pct": std_ret,
                "sharpe": sharpe,
                "min_return": np.min(sample_returns),
                "max_return": np.max(sample_returns),
                "n_samples": len(sample_returns)
            }

            print(f"Hour {hour:02d}:00 - Mean: {mean_ret:+.2f}%, "
                  f"Std: {std_ret:.2f}%, Sharpe: {sharpe:.3f}, "
                  f"Samples: {len(sample_returns)}")
        else:
            print(f"Hour {hour:02d}:00 - No valid samples")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Rebalance Hour Analysis (30% concentration, 24h frequency)")
    print("=" * 80)

    results_df = pd.DataFrame(list(results_by_hour.values()))
    results_df = results_df.sort_values("sharpe", ascending=False)

    print("\nRanked by Sharpe Ratio:")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n🎯 BEST REBALANCE TIME: {int(best['hour']):02d}:00 UTC")
    print(f"   Mean Return: {best['mean_return_pct']:+.2f}%")
    print(f"   Std Dev: {best['std_return_pct']:.2f}%")
    print(f"   Sharpe: {best['sharpe']:.3f}")
    print(f"   Based on {int(best['n_samples'])} samples")

    return results_df


def run_backtest_with_fixed_times(
    panel: pd.DataFrame,
    rebalance_times: list,
    n_pct: float,
    weight_scheme: str,
    fee_rate: float
):
    """
    Run backtest with pre-specified rebalance times.

    This is similar to run_concentration_backtest but uses explicit rebalance times.
    """
    # Track portfolio value
    portfolio_value = 1.0
    current_weights = {}

    for i, rebal_time in enumerate(rebalance_times):
        # Get data at rebalance time
        rebal_data = panel[
            (panel["timestamp"] == rebal_time) &
            (panel["include_in_universe"] == True)
        ].copy()

        if len(rebal_data) == 0:
            continue

        # Rank by momentum
        rebal_data = rebal_data.sort_values("momentum_score", ascending=False)
        n_assets = len(rebal_data)
        n_select = max(1, int(np.ceil(n_assets * n_pct / 100.0)))

        # Select top n% (long) and bottom n% (short)
        long_assets = rebal_data.head(n_select)
        short_assets = rebal_data.tail(n_select)

        # Compute weights
        def compute_weights(assets_df, side: str):
            if weight_scheme == "equal":
                weights = pd.Series(1.0 / len(assets_df), index=assets_df["asset"])
            elif weight_scheme == "inverse_vol":
                inv_vol = 1.0 / assets_df["vol_24h"]
                weights = inv_vol / inv_vol.sum()
                weights.index = assets_df["asset"]
            else:
                raise ValueError(f"Unknown weight scheme: {weight_scheme}")

            weights = weights / weights.sum()
            if side == "short":
                weights = -weights
            return weights

        new_weights = {}
        if len(long_assets) > 0:
            long_weights = compute_weights(long_assets, "long")
            new_weights.update(long_weights.to_dict())

        if len(short_assets) > 0:
            short_weights = compute_weights(short_assets, "short")
            new_weights.update(short_weights.to_dict())

        # Calculate turnover and apply transaction costs
        if fee_rate > 0:
            all_assets = set(new_weights.keys()) | set(current_weights.keys())
            turnover = sum(
                abs(new_weights.get(asset, 0.0) - current_weights.get(asset, 0.0))
                for asset in all_assets
            )
            cost = turnover * fee_rate
            portfolio_value *= (1 - cost)

        # Calculate returns until next rebalance (or end of period)
        if i + 1 < len(rebalance_times):
            next_rebal_time = rebalance_times[i + 1]
        else:
            # Hold until end
            timestamps = sorted(panel["timestamp"].unique())
            next_rebal_time = timestamps[-1]

        # Get returns between rebalances
        return_mask = (
            (panel["timestamp"] > rebal_time) &
            (panel["timestamp"] <= next_rebal_time)
        )
        returns_data = panel[return_mask]

        if len(returns_data) == 0:
            current_weights = new_weights
            continue

        # Calculate portfolio return for this rebalance period
        period_timestamps = sorted(returns_data["timestamp"].unique())

        for ts in period_timestamps:
            ts_data = returns_data[returns_data["timestamp"] == ts]
            portfolio_return = 0.0

            for asset, weight in new_weights.items():
                asset_return = ts_data[ts_data["asset"] == asset]["log_return"]
                if len(asset_return) > 0:
                    simple_return = np.exp(asset_return.iloc[0]) - 1
                    portfolio_return += weight * simple_return

            portfolio_value *= (1 + portfolio_return)

        current_weights = new_weights

    # Return total return as percentage
    total_return = (portfolio_value - 1) * 100
    return total_return


if __name__ == "__main__":
    # Use net returns (after transaction costs)
    panel_path = Path("/root/slipstream/data/gradient/sensitivity_net/panel_data.csv")
    sample_periods_path = Path("/root/slipstream/data/gradient/sensitivity_net/sample_periods.csv")

    # Test with 0.0144% commission (typical maker fee)
    results = analyze_rebalance_hour_sensitivity(
        panel_path=panel_path,
        sample_periods_path=sample_periods_path,
        n_pct=30.0,
        weight_scheme="inverse_vol",
        fee_rate=0.000144
    )

    # Save results
    output_path = Path("/root/slipstream/data/gradient/sensitivity_net/rebalance_hour_analysis_30pct.csv")
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
