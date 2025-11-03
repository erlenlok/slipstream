#!/usr/bin/env python3
"""
Research: Candlestick Momentum Persistence Analysis

Analyzes the conditional probability and expected returns of 4hr candles
based on consecutive same-direction candles (up to 5 lookback).

Also analyzes trend persistence by hour of day.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats

# Configuration
DATA_DIR = Path("data/market_data")
OUTPUT_DIR = Path("output/candle_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Liquidity filter (same as gradient)
MIN_LIQUIDITY_USD = 10_000  # Daily volume threshold
MAX_LOOKBACK = 5

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_all_candle_data() -> pd.DataFrame:
    """Load all 4hr candle data and filter for liquid coins."""
    print("Loading 4hr candle data...")

    all_data = []
    files = list(DATA_DIR.glob("*_candles_4h.csv"))

    for file in files:
        coin = file.stem.replace("_candles_4h", "")
        try:
            df = pd.read_csv(file)

            # Handle different timestamp column names
            if 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                print(f"  Warning: No timestamp column in {coin}")
                continue

            df['coin'] = coin

            # Only keep necessary columns
            df = df[['timestamp', 'coin', 'open', 'high', 'low', 'close', 'volume']].copy()

            # Compute dollar volume (close * volume)
            df['dollar_volume'] = df['close'] * df['volume']

            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Failed to load {coin}: {e}")

    if len(all_data) == 0:
        raise ValueError("No data files loaded successfully")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"  Loaded {len(all_data)} coins, {len(combined):,} total candles")

    return combined


def filter_liquid_coins(df: pd.DataFrame, min_volume_usd: float) -> pd.DataFrame:
    """
    Filter for liquid coins based on daily dollar volume.

    Uses rolling 24hr (6 candles) average dollar volume.
    """
    print(f"\nFiltering for coins with >${min_volume_usd:,.0f} daily volume...")

    # Compute 24hr rolling volume (6 x 4hr candles)
    df = df.sort_values(['coin', 'timestamp'])
    df['volume_24h'] = df.groupby('coin')['dollar_volume'].transform(
        lambda x: x.rolling(6, min_periods=1).sum()
    )

    # Filter: keep coins where median 24hr volume > threshold
    coin_volumes = df.groupby('coin')['volume_24h'].median()
    liquid_coins = coin_volumes[coin_volumes >= min_volume_usd].index.tolist()

    df_liquid = df[df['coin'].isin(liquid_coins)].copy()

    print(f"  Kept {len(liquid_coins)}/{df['coin'].nunique()} liquid coins")
    print(f"  Total candles: {len(df_liquid):,}")

    return df_liquid


def compute_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candle direction and consecutive streaks."""
    print("\nComputing candle features...")

    df = df.sort_values(['coin', 'timestamp']).copy()

    # Candle direction: 1 = bullish (close > open), -1 = bearish
    df['direction'] = np.sign(df['close'] - df['open'])
    df.loc[df['direction'] == 0, 'direction'] = 1  # Treat doji as bullish

    # Candle return (close-to-close)
    df['return'] = df.groupby('coin')['close'].pct_change()

    # Hour of day (UTC)
    df['hour'] = df['timestamp'].dt.hour

    # Compute consecutive streaks for each lookback (1-5)
    for lookback in range(1, MAX_LOOKBACK + 1):
        df[f'streak_{lookback}'] = compute_streak(df, lookback)

    # Drop first MAX_LOOKBACK rows per coin (need history)
    df = df.groupby('coin').apply(
        lambda x: x.iloc[MAX_LOOKBACK:]
    ).reset_index(drop=True)

    print(f"  Computed features for {len(df):,} candles")

    return df


def compute_streak(df: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Compute streak indicator for given lookback.

    IMPORTANT: This looks at the PREVIOUS N candles, not including current.
    This way we can predict the NEXT candle based on HISTORICAL streaks.

    Returns:
        +lookback if previous N candles were all bullish
        -lookback if previous N candles were all bearish
        0 if mixed
    """
    # Shift direction by 1 so we're looking at PREVIOUS candles only
    direction_prev = df.groupby('coin')['direction'].shift(1)

    # Now compute rolling streak on the shifted direction
    # This means streak[t] looks at candles [t-lookback, t-1], NOT including t
    streak = df.groupby('coin').apply(
        lambda group: direction_prev.loc[group.index].rolling(
            lookback, min_periods=lookback
        ).apply(
            lambda vals: vals[0] if len(vals) == lookback and (vals == vals[0]).all() else 0,
            raw=True
        ),
        include_groups=False
    ).reset_index(level=0, drop=True)

    # Multiply by lookback to get streak length
    return streak * lookback


def analyze_conditional_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze next-candle returns conditional on streak.

    Returns summary statistics for each (lookback, streak_direction).
    """
    print("\nAnalyzing conditional returns...")

    results = []

    for lookback in range(1, MAX_LOOKBACK + 1):
        streak_col = f'streak_{lookback}'

        # Bullish streaks
        bullish_mask = df[streak_col] == lookback
        if bullish_mask.sum() > 0:
            returns_after_bull = df.loc[bullish_mask, 'return'].dropna()
            results.append({
                'lookback': lookback,
                'direction': 'bullish',
                'n_samples': len(returns_after_bull),
                'prob_continuation': (returns_after_bull > 0).mean(),
                'mean_return': returns_after_bull.mean(),
                'median_return': returns_after_bull.median(),
                'std_return': returns_after_bull.std(),
                't_stat': stats.ttest_1samp(returns_after_bull, 0)[0] if len(returns_after_bull) > 1 else np.nan,
                'p_value': stats.ttest_1samp(returns_after_bull, 0)[1] if len(returns_after_bull) > 1 else np.nan,
            })

        # Bearish streaks
        bearish_mask = df[streak_col] == -lookback
        if bearish_mask.sum() > 0:
            returns_after_bear = df.loc[bearish_mask, 'return'].dropna()
            results.append({
                'lookback': lookback,
                'direction': 'bearish',
                'n_samples': len(returns_after_bear),
                'prob_continuation': (returns_after_bear < 0).mean(),
                'mean_return': returns_after_bear.mean(),
                'median_return': returns_after_bear.median(),
                'std_return': returns_after_bear.std(),
                't_stat': stats.ttest_1samp(returns_after_bear, 0)[0] if len(returns_after_bear) > 1 else np.nan,
                'p_value': stats.ttest_1samp(returns_after_bear, 0)[1] if len(returns_after_bear) > 1 else np.nan,
            })

    summary = pd.DataFrame(results)

    print("\nConditional Return Summary:")
    print("=" * 100)
    print(summary.to_string(index=False))

    return summary


def analyze_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze streak persistence by hour of day."""
    print("\nAnalyzing by hour of day...")

    results = []
    hours = [0, 4, 8, 12, 16, 20]  # 4hr candle close hours

    for hour in hours:
        hour_df = df[df['hour'] == hour]

        for lookback in range(1, MAX_LOOKBACK + 1):
            streak_col = f'streak_{lookback}'

            # Bullish streaks
            bullish_mask = hour_df[streak_col] == lookback
            if bullish_mask.sum() > 10:  # Need minimum samples
                returns_bull = hour_df.loc[bullish_mask, 'return'].dropna()
                results.append({
                    'hour': hour,
                    'lookback': lookback,
                    'direction': 'bullish',
                    'n_samples': len(returns_bull),
                    'prob_continuation': (returns_bull > 0).mean(),
                    'mean_return': returns_bull.mean(),
                })

            # Bearish streaks
            bearish_mask = hour_df[streak_col] == -lookback
            if bearish_mask.sum() > 10:
                returns_bear = hour_df.loc[bearish_mask, 'return'].dropna()
                results.append({
                    'hour': hour,
                    'lookback': lookback,
                    'direction': 'bearish',
                    'n_samples': len(returns_bear),
                    'prob_continuation': (returns_bear < 0).mean(),
                    'mean_return': returns_bear.mean(),
                })

    hourly = pd.DataFrame(results)

    print("\nHourly Analysis Summary:")
    print("=" * 100)
    print(hourly.head(20).to_string(index=False))

    return hourly


def plot_conditional_distributions(df: pd.DataFrame):
    """Plot return distributions conditional on streak length."""
    print("\nGenerating distribution plots...")

    fig, axes = plt.subplots(2, MAX_LOOKBACK, figsize=(20, 8))
    fig.suptitle('Next-Candle Return Distributions by Consecutive Streak Length',
                 fontsize=16, y=1.02)

    for lookback in range(1, MAX_LOOKBACK + 1):
        streak_col = f'streak_{lookback}'

        # Bullish streaks
        bullish_returns = df.loc[df[streak_col] == lookback, 'return'].dropna()
        ax_bull = axes[0, lookback - 1]
        if len(bullish_returns) > 0:
            bullish_returns_clean = bullish_returns[
                (bullish_returns > bullish_returns.quantile(0.01)) &
                (bullish_returns < bullish_returns.quantile(0.99))
            ]
            ax_bull.hist(bullish_returns_clean * 100, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax_bull.axvline(0, color='red', linestyle='--', linewidth=2)
            ax_bull.axvline(bullish_returns.mean() * 100, color='blue', linestyle='-', linewidth=2,
                           label=f'Mean: {bullish_returns.mean()*100:.2f}%')
            prob_cont = (bullish_returns > 0).mean()
            ax_bull.set_title(f'{lookback} Bull→Next\nP(continue)={prob_cont:.1%}', fontsize=10)
            ax_bull.set_xlabel('Next Candle Return (%)')
            ax_bull.legend(fontsize=8)
            ax_bull.grid(True, alpha=0.3)

        # Bearish streaks
        bearish_returns = df.loc[df[streak_col] == -lookback, 'return'].dropna()
        ax_bear = axes[1, lookback - 1]
        if len(bearish_returns) > 0:
            bearish_returns_clean = bearish_returns[
                (bearish_returns > bearish_returns.quantile(0.01)) &
                (bearish_returns < bearish_returns.quantile(0.99))
            ]
            ax_bear.hist(bearish_returns_clean * 100, bins=50, alpha=0.7, color='red', edgecolor='black')
            ax_bear.axvline(0, color='green', linestyle='--', linewidth=2)
            ax_bear.axvline(bearish_returns.mean() * 100, color='blue', linestyle='-', linewidth=2,
                           label=f'Mean: {bearish_returns.mean()*100:.2f}%')
            prob_cont = (bearish_returns < 0).mean()
            ax_bear.set_title(f'{lookback} Bear→Next\nP(continue)={prob_cont:.1%}', fontsize=10)
            ax_bear.set_xlabel('Next Candle Return (%)')
            ax_bear.legend(fontsize=8)
            ax_bear.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel('Frequency\n(After Bullish Streak)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency\n(After Bearish Streak)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'conditional_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'conditional_distributions.png'}")
    plt.close()


def plot_summary_statistics(summary: pd.DataFrame):
    """Plot summary statistics of continuation probabilities and mean returns."""
    print("\nGenerating summary plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Continuation probability
    ax1 = axes[0]
    bull_data = summary[summary['direction'] == 'bullish']
    bear_data = summary[summary['direction'] == 'bearish']

    x = np.arange(1, MAX_LOOKBACK + 1)
    width = 0.35

    bars1 = ax1.bar(x - width/2, bull_data['prob_continuation'], width,
                    label='Bullish Streak', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, bear_data['prob_continuation'], width,
                    label='Bearish Streak', color='red', alpha=0.7)

    ax1.axhline(0.5, color='black', linestyle='--', linewidth=1, label='Random (50%)')
    ax1.set_xlabel('Consecutive Candles', fontsize=12)
    ax1.set_ylabel('Probability of Continuation', fontsize=12)
    ax1.set_title('Trend Continuation Probability by Streak Length', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.4, 0.6])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Mean return
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, bull_data['mean_return'] * 100, width,
                    label='After Bullish', color='green', alpha=0.7)
    bars4 = ax2.bar(x + width/2, bear_data['mean_return'] * 100, width,
                    label='After Bearish', color='red', alpha=0.7)

    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Consecutive Candles', fontsize=12)
    ax2.set_ylabel('Mean Next-Candle Return (%)', fontsize=12)
    ax2.set_title('Expected Next-Candle Return by Streak Length', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'summary_statistics.png'}")
    plt.close()


def plot_hourly_analysis(hourly: pd.DataFrame):
    """Plot trend persistence by hour of day."""
    print("\nGenerating hourly analysis plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Trend Persistence by Hour of Day (UTC)', fontsize=16, y=1.00)

    hours = [0, 4, 8, 12, 16, 20]

    for idx, hour in enumerate(hours):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        hour_data = hourly[hourly['hour'] == hour]

        if len(hour_data) == 0:
            continue

        bull_data = hour_data[hour_data['direction'] == 'bullish'].sort_values('lookback')
        bear_data = hour_data[hour_data['direction'] == 'bearish'].sort_values('lookback')

        # Plot continuation probability
        if len(bull_data) > 0:
            ax.plot(bull_data['lookback'], bull_data['prob_continuation'],
                   'o-', color='green', linewidth=2, markersize=8, label='Bull Continuation', alpha=0.8)
        if len(bear_data) > 0:
            ax.plot(bear_data['lookback'], bear_data['prob_continuation'],
                   's-', color='red', linewidth=2, markersize=8, label='Bear Continuation', alpha=0.8)

        ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(f'{hour:02d}:00 UTC', fontsize=12, fontweight='bold')
        ax.set_xlabel('Streak Length', fontsize=10)
        ax.set_ylabel('P(Continuation)', fontsize=10)
        ax.set_ylim([0.35, 0.65])
        ax.set_xticks(range(1, MAX_LOOKBACK + 1))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add sample size annotation
        total_samples = hour_data['n_samples'].sum()
        ax.text(0.02, 0.98, f'n={total_samples:,}', transform=ax.transAxes,
               fontsize=8, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hourly_persistence.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'hourly_persistence.png'}")
    plt.close()


def plot_hourly_heatmap(hourly: pd.DataFrame):
    """Create heatmap of continuation probabilities by hour and lookback."""
    print("\nGenerating hourly heatmap...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    hours = [0, 4, 8, 12, 16, 20]
    lookbacks = range(1, MAX_LOOKBACK + 1)

    # Bullish heatmap
    bull_data = hourly[hourly['direction'] == 'bullish']
    bull_pivot = bull_data.pivot_table(
        index='hour', columns='lookback', values='prob_continuation'
    )
    bull_pivot = bull_pivot.reindex(hours).reindex(columns=lookbacks)

    sns.heatmap(bull_pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0.5,
               vmin=0.4, vmax=0.6, ax=axes[0], cbar_kws={'label': 'P(Continuation)'})
    axes[0].set_title('Bullish Streak Continuation Probability', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Consecutive Bullish Candles', fontsize=12)
    axes[0].set_ylabel('Hour (UTC)', fontsize=12)
    axes[0].set_yticklabels([f'{h:02d}:00' for h in hours], rotation=0)

    # Bearish heatmap
    bear_data = hourly[hourly['direction'] == 'bearish']
    bear_pivot = bear_data.pivot_table(
        index='hour', columns='lookback', values='prob_continuation'
    )
    bear_pivot = bear_pivot.reindex(hours).reindex(columns=lookbacks)

    sns.heatmap(bear_pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0.5,
               vmin=0.4, vmax=0.6, ax=axes[1], cbar_kws={'label': 'P(Continuation)'})
    axes[1].set_title('Bearish Streak Continuation Probability', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Consecutive Bearish Candles', fontsize=12)
    axes[1].set_ylabel('Hour (UTC)', fontsize=12)
    axes[1].set_yticklabels([f'{h:02d}:00' for h in hours], rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hourly_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'hourly_heatmap.png'}")
    plt.close()


def main():
    """Run full candle momentum analysis."""
    print("=" * 100)
    print("CANDLESTICK MOMENTUM PERSISTENCE ANALYSIS")
    print("=" * 100)

    # Load and filter data
    df = load_all_candle_data()
    df = filter_liquid_coins(df, MIN_LIQUIDITY_USD)

    # Compute features
    df = compute_candle_features(df)

    # Analyze conditional returns
    summary = analyze_conditional_returns(df)

    # Analyze by hour
    hourly = analyze_by_hour(df)

    # Generate plots
    plot_conditional_distributions(df)
    plot_summary_statistics(summary)
    plot_hourly_analysis(hourly)
    plot_hourly_heatmap(hourly)

    # Save results
    summary.to_csv(OUTPUT_DIR / 'conditional_summary.csv', index=False)
    hourly.to_csv(OUTPUT_DIR / 'hourly_summary.csv', index=False)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
