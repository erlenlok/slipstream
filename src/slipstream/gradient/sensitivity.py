"""
Concentration sensitivity analysis for Gradient strategy.

This module provides tools to analyze how portfolio concentration (n% of universe)
affects expected returns across different rebalancing frequencies and weighting schemes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Tuple, List
from dataclasses import dataclass


@dataclass
class SensitivityConfig:
    """Configuration for concentration sensitivity analysis."""

    lookback_spans: List[int]  # EWMA lookback windows in hours
    vol_span: int = 24  # Volatility estimation span in hours
    adv_window: int = 6  # Average daily volume window in periods (6 * 4h = 24h)
    liquidity_threshold: float = 10_000.0  # USD trade size
    liquidity_impact_pct: float = 2.5  # Max % of ADV

    # Transaction costs
    fee_rate: float = 0.0  # Trading fee as decimal (e.g., 0.000144 for 0.0144%)

    # Sensitivity sweep parameters
    n_pct_range: List[float] = None  # e.g., [1, 2, 3, ..., 50]
    rebalance_freqs_hours: List[int] = None  # e.g., [4, 8, 12, ..., 48]
    weight_schemes: List[Literal["equal", "inverse_vol"]] = None

    # Sampling parameters
    n_samples: int = 100  # K samples per configuration
    sample_period_days: int = 10  # Length of each sample period

    def __post_init__(self):
        if self.n_pct_range is None:
            self.n_pct_range = list(range(1, 51))  # 1% to 50%
        if self.rebalance_freqs_hours is None:
            self.rebalance_freqs_hours = list(range(4, 52, 4))  # 4h to 48h in 4h steps
        if self.weight_schemes is None:
            self.weight_schemes = ["equal", "inverse_vol"]


def load_all_candles(data_dir: Path) -> pd.DataFrame:
    """
    Load all 4h candle data from market_data directory.

    Args:
        data_dir: Path to data/market_data/ directory

    Returns:
        DataFrame with columns: timestamp, asset, open, high, low, close, volume
    """
    candle_files = list(data_dir.glob("*_candles_4h.csv"))

    dfs = []
    for file in candle_files:
        asset = file.stem.replace("_candles_4h", "")
        try:
            df = pd.read_csv(file)
            # Handle both 'datetime' and 'timestamp' column names
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["asset"] = asset
            dfs.append(df[["timestamp", "asset", "open", "high", "low", "close", "volume"]])
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
            continue

    if not dfs:
        raise ValueError(f"No candle files found in {data_dir}")

    panel = pd.concat(dfs, ignore_index=True)
    panel = panel.sort_values(["timestamp", "asset"]).reset_index(drop=True)

    return panel


def compute_log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 4h log returns for each asset.

    Args:
        panel: DataFrame with timestamp, asset, close columns

    Returns:
        Panel with added 'log_return' column
    """
    panel = panel.sort_values(["asset", "timestamp"])
    panel["log_return"] = panel.groupby("asset")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    return panel


def compute_ewma_vol(returns: pd.Series, span: int) -> pd.Series:
    """
    Compute EWMA volatility from returns.

    Args:
        returns: Series of returns
        span: EWMA span in periods

    Returns:
        Series of volatility estimates
    """
    return returns.ewm(span=span, min_periods=span).std()


def compute_adv_usd(panel: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """
    Compute average daily volume in USD.

    Args:
        panel: DataFrame with timestamp, asset, volume, close columns
        window: Rolling window in 4h periods (6 = 24h)

    Returns:
        Panel with added 'adv_usd' column
    """
    panel = panel.sort_values(["asset", "timestamp"])

    # Volume in USD for each period
    panel["volume_usd"] = panel["volume"] * panel["close"]

    # Rolling sum over 24h window, then divide by 1 day to get daily average
    panel["adv_usd"] = panel.groupby("asset")["volume_usd"].transform(
        lambda x: x.rolling(window=window, min_periods=window).sum()
    )

    return panel


def filter_universe_by_liquidity(
    panel: pd.DataFrame,
    trade_size_usd: float = 10_000.0,
    max_impact_pct: float = 2.5
) -> pd.DataFrame:
    """
    Filter universe to include only liquid assets.

    Include if: trade_size_usd < (max_impact_pct / 100) * adv_usd

    Args:
        panel: DataFrame with adv_usd column
        trade_size_usd: Size of trade in USD
        max_impact_pct: Maximum allowed impact as % of ADV

    Returns:
        Panel with added 'include_in_universe' boolean column
    """
    threshold = (max_impact_pct / 100.0) * panel["adv_usd"]
    panel["include_in_universe"] = trade_size_usd < threshold
    return panel


def compute_vol_normalized_returns(
    panel: pd.DataFrame,
    vol_span: int = 24
) -> pd.DataFrame:
    """
    Compute volatility-normalized returns.

    Args:
        panel: DataFrame with log_return column
        vol_span: EWMA span for volatility estimation

    Returns:
        Panel with added 'vol_24h' and 'vol_norm_return' columns
    """
    panel = panel.sort_values(["asset", "timestamp"])

    # Compute EWMA volatility per asset
    panel["vol_24h"] = panel.groupby("asset")["log_return"].transform(
        lambda x: compute_ewma_vol(x, span=vol_span)
    )

    # Normalize returns
    panel["vol_norm_return"] = panel["log_return"] / panel["vol_24h"]

    # Replace inf/nan with 0
    panel["vol_norm_return"] = panel["vol_norm_return"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return panel


def compute_multispan_momentum(
    panel: pd.DataFrame,
    lookback_spans: List[int]
) -> pd.DataFrame:
    """
    Compute momentum score as sum of EWMA(vol_norm_return) across multiple lookbacks.

    Args:
        panel: DataFrame with vol_norm_return column
        lookback_spans: List of EWMA spans in hours (e.g., [2, 4, 8, 16, ...])

    Returns:
        Panel with added 'momentum_score' column
    """
    panel = panel.sort_values(["asset", "timestamp"])

    # Compute EWMA for each span
    momentum_components = []
    for span in lookback_spans:
        col_name = f"ewma_{span}"
        panel[col_name] = panel.groupby("asset")["vol_norm_return"].transform(
            lambda x: x.ewm(span=span, min_periods=1).mean()
        )
        momentum_components.append(col_name)

    # Sum all components to get final momentum score
    panel["momentum_score"] = panel[momentum_components].sum(axis=1)

    # Clean up intermediate columns
    panel = panel.drop(columns=momentum_components)

    return panel


def build_panel_data(
    data_dir: Path,
    config: SensitivityConfig,
    output_path: Path = None
) -> pd.DataFrame:
    """
    Build high-quality panel data for concentration sensitivity analysis.

    Pipeline:
    1. Load all 4h candles
    2. Compute log returns
    3. Compute ADV and filter universe
    4. Compute vol-normalized returns
    5. Compute multi-span momentum scores

    Args:
        data_dir: Path to data/market_data/ directory
        config: Sensitivity configuration
        output_path: Optional path to save panel CSV

    Returns:
        Panel DataFrame with columns:
            timestamp, asset, close, log_return, vol_24h, adv_usd,
            include_in_universe, momentum_score, vol_norm_return
    """
    print("Loading 4h candle data...")
    panel = load_all_candles(data_dir)

    print(f"Loaded {len(panel)} rows for {panel['asset'].nunique()} assets")

    print("Computing log returns...")
    panel = compute_log_returns(panel)

    print("Computing ADV and filtering universe...")
    panel = compute_adv_usd(panel, window=config.adv_window)
    panel = filter_universe_by_liquidity(
        panel,
        trade_size_usd=config.liquidity_threshold,
        max_impact_pct=config.liquidity_impact_pct
    )

    # Report universe statistics
    valid_counts = panel.groupby("asset")["include_in_universe"].sum()
    included_assets = (valid_counts > 0).sum()
    print(f"Universe: {included_assets}/{panel['asset'].nunique()} assets pass liquidity filter")

    print("Computing vol-normalized returns...")
    panel = compute_vol_normalized_returns(panel, vol_span=config.vol_span)

    print(f"Computing multi-span momentum (spans: {config.lookback_spans})...")
    panel = compute_multispan_momentum(panel, lookback_spans=config.lookback_spans)

    # Drop rows with NaN momentum (insufficient history)
    initial_len = len(panel)
    panel = panel.dropna(subset=["momentum_score", "vol_24h", "adv_usd"])
    print(f"Dropped {initial_len - len(panel)} rows with insufficient history")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_path, index=False)
        print(f"Saved panel data to {output_path}")

    return panel


def generate_sample_periods(
    panel: pd.DataFrame,
    n_samples: int,
    period_days: int,
    candle_freq_hours: int = 4
) -> pd.DataFrame:
    """
    Generate K random non-overlapping sample periods.

    Args:
        panel: Panel data with timestamp column
        n_samples: Number of samples (K)
        period_days: Length of each period in days
        candle_freq_hours: Frequency of candles in hours

    Returns:
        DataFrame with columns: sample_id, start_time, end_time
    """
    timestamps = sorted(panel["timestamp"].unique())
    period_length = int((period_days * 24) / candle_freq_hours)  # Number of periods

    # Ensure we have enough data
    if len(timestamps) < period_length:
        raise ValueError(f"Not enough data: need {period_length} periods, have {len(timestamps)}")

    max_start_idx = len(timestamps) - period_length

    # Generate random start indices ensuring no overlaps
    samples = []
    used_ranges = []

    attempts = 0
    max_attempts = n_samples * 100

    while len(samples) < n_samples and attempts < max_attempts:
        start_idx = np.random.randint(0, max_start_idx)
        end_idx = start_idx + period_length

        # Check for overlap with existing samples
        overlap = False
        for used_start, used_end in used_ranges:
            if not (end_idx <= used_start or start_idx >= used_end):
                overlap = True
                break

        if not overlap:
            samples.append({
                "sample_id": len(samples),
                "start_time": timestamps[start_idx],
                "end_time": timestamps[end_idx - 1]
            })
            used_ranges.append((start_idx, end_idx))

        attempts += 1

    if len(samples) < n_samples:
        print(f"Warning: Could only generate {len(samples)}/{n_samples} non-overlapping samples")

    return pd.DataFrame(samples)


def run_concentration_backtest(
    panel: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    n_pct: float,
    rebalance_freq_hours: int,
    weight_scheme: Literal["equal", "inverse_vol"],
    candle_freq_hours: int = 4,
    fee_rate: float = 0.0
) -> float:
    """
    Run a single concentration backtest over a period.

    Args:
        panel: Panel data with momentum_score, log_return, vol_24h, include_in_universe
        start_time: Start timestamp
        end_time: End timestamp (inclusive)
        n_pct: Concentration percentage (e.g., 5.0 for top/bottom 5%)
        rebalance_freq_hours: Rebalancing frequency in hours
        weight_scheme: "equal" or "inverse_vol"
        candle_freq_hours: Candle frequency in hours (default 4h)
        fee_rate: Trading fee as decimal (e.g., 0.000144 for 0.0144%)

    Returns:
        Annualized return over the period
    """
    # Filter panel to period
    mask = (panel["timestamp"] >= start_time) & (panel["timestamp"] <= end_time)
    period_data = panel[mask].copy()

    if len(period_data) == 0:
        return np.nan

    # Get rebalance timestamps
    timestamps = sorted(period_data["timestamp"].unique())
    rebalance_freq_periods = rebalance_freq_hours // candle_freq_hours
    rebalance_times = timestamps[::rebalance_freq_periods]

    # Track portfolio value
    portfolio_value = 1.0

    # Track current weights
    current_weights = {}

    for i, rebal_time in enumerate(rebalance_times):
        # Get data at rebalance time
        rebal_data = period_data[
            (period_data["timestamp"] == rebal_time) &
            (period_data["include_in_universe"] == True)
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
        def compute_weights(assets_df, side: Literal["long", "short"]):
            if weight_scheme == "equal":
                weights = pd.Series(1.0 / len(assets_df), index=assets_df["asset"])
            elif weight_scheme == "inverse_vol":
                inv_vol = 1.0 / assets_df["vol_24h"]
                weights = inv_vol / inv_vol.sum()
                weights.index = assets_df["asset"]
            else:
                raise ValueError(f"Unknown weight scheme: {weight_scheme}")

            # Normalize to 100% per side
            weights = weights / weights.sum()

            # Apply sign
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
            # Turnover = sum of absolute weight changes
            all_assets = set(new_weights.keys()) | set(current_weights.keys())
            turnover = sum(
                abs(new_weights.get(asset, 0.0) - current_weights.get(asset, 0.0))
                for asset in all_assets
            )
            # Deduct costs from portfolio value
            cost = turnover * fee_rate
            portfolio_value *= (1 - cost)

        # Calculate returns until next rebalance (or end of period)
        next_rebal_idx = i + 1
        if next_rebal_idx < len(rebalance_times):
            next_rebal_time = rebalance_times[next_rebal_idx]
        else:
            next_rebal_time = timestamps[-1]

        # Get returns between rebalances
        return_mask = (
            (period_data["timestamp"] > rebal_time) &
            (period_data["timestamp"] <= next_rebal_time)
        )
        returns_data = period_data[return_mask]

        if len(returns_data) == 0:
            current_weights = new_weights
            continue

        # Calculate portfolio return for this rebalance period
        # Get unique timestamps in the rebalance period
        period_timestamps = sorted(returns_data["timestamp"].unique())

        for ts in period_timestamps:
            # Calculate weighted portfolio return for this single period
            ts_data = returns_data[returns_data["timestamp"] == ts]
            portfolio_return = 0.0

            for asset, weight in new_weights.items():
                asset_return = ts_data[ts_data["asset"] == asset]["log_return"]
                if len(asset_return) > 0:
                    # Convert log return to simple return and weight it
                    simple_return = np.exp(asset_return.iloc[0]) - 1
                    portfolio_return += weight * simple_return

            # Compound portfolio return
            portfolio_value *= (1 + portfolio_return)

        current_weights = new_weights

    # Calculate returns
    total_return = portfolio_value - 1
    n_periods = len(timestamps)
    periods_per_year = (365 * 24) / candle_freq_hours
    n_years = n_periods / periods_per_year

    if n_years <= 0:
        return np.nan

    # Return the total return percentage (not annualized)
    # Annualization can amplify returns unrealistically for short periods
    return total_return * 100  # Return as percentage


@dataclass
class SensitivityResult:
    """Results from a sensitivity analysis run."""

    n_pct: float
    rebalance_freq_h: int
    weight_scheme: str
    mean_return_pct: float  # Mean return over sample period (not annualized)
    std_return_pct: float   # Std dev of returns over sample period
    sharpe: float
    min_return: float
    max_return: float
    n_samples: int

    def to_dict(self):
        return {
            "n_pct": self.n_pct,
            "rebalance_freq_h": self.rebalance_freq_h,
            "weight_scheme": self.weight_scheme,
            "mean_return_pct": self.mean_return_pct,
            "std_return_pct": self.std_return_pct,
            "sharpe": self.sharpe,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "n_samples": self.n_samples,
        }


def run_sensitivity_sweep(
    panel: pd.DataFrame,
    sample_periods: pd.DataFrame,
    config: SensitivityConfig,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Run full sensitivity sweep across n%, rebalance frequency, and weight schemes.

    Args:
        panel: Panel data
        sample_periods: DataFrame with sample_id, start_time, end_time
        config: Sensitivity configuration
        output_dir: Optional directory to save results

    Returns:
        DataFrame with sensitivity results
    """
    results = []

    total_configs = (
        len(config.n_pct_range) *
        len(config.rebalance_freqs_hours) *
        len(config.weight_schemes)
    )

    config_count = 0

    for weight_scheme in config.weight_schemes:
        for rebal_freq in config.rebalance_freqs_hours:
            for n_pct in config.n_pct_range:
                config_count += 1

                print(f"[{config_count}/{total_configs}] Running n={n_pct}%, "
                      f"rebal={rebal_freq}h, weight={weight_scheme}")

                sample_returns = []

                for _, sample in sample_periods.iterrows():
                    period_return = run_concentration_backtest(
                        panel=panel,
                        start_time=sample["start_time"],
                        end_time=sample["end_time"],
                        n_pct=n_pct,
                        rebalance_freq_hours=rebal_freq,
                        weight_scheme=weight_scheme,
                        fee_rate=config.fee_rate
                    )

                    if not np.isnan(period_return):
                        sample_returns.append(period_return)

                if len(sample_returns) > 0:
                    mean_ret = np.mean(sample_returns)
                    std_ret = np.std(sample_returns)
                    sharpe = mean_ret / std_ret if std_ret > 0 else 0

                    result = SensitivityResult(
                        n_pct=n_pct,
                        rebalance_freq_h=rebal_freq,
                        weight_scheme=weight_scheme,
                        mean_return_pct=mean_ret,
                        std_return_pct=std_ret,
                        sharpe=sharpe,
                        min_return=np.min(sample_returns),
                        max_return=np.max(sample_returns),
                        n_samples=len(sample_returns)
                    )

                    results.append(result.to_dict())

    results_df = pd.DataFrame(results)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for scheme in config.weight_schemes:
            scheme_results = results_df[results_df["weight_scheme"] == scheme]
            output_path = output_dir / f"results_{scheme}.csv"
            scheme_results.to_csv(output_path, index=False)
            print(f"Saved {scheme} results to {output_path}")

    return results_df
