#!/usr/bin/env python3
"""
Build transaction cost model from L2 liquidity + candle features.

Fits simple parametric models mapping candle-based features to execution cost.
Target: one-way slippage cost in decimal format for optimizer.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import argparse
import json


def load_candles_for_date(data_dir: Path, date: str, interval: str = "4h") -> pd.DataFrame:
    """Load all candle files and filter to specific date."""
    market_data_dir = data_dir / "market_data"
    files = sorted(market_data_dir.glob(f"*_candles_{interval}.csv"))
    
    all_candles = []
    for fpath in files:
        # Extract coin name
        parts = fpath.stem.split("_")
        if len(parts) >= 3 and parts[-2] == "candles":
            coin = "_".join(parts[:-2])
        else:
            coin = fpath.stem
        
        # Load and filter to date
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df['coin'] = coin
        
        # Filter to specific date
        date_filter = df.index.date == pd.Timestamp(date).date()
        df_filtered = df.loc[date_filter].copy()
        
        if not df_filtered.empty:
            all_candles.append(df_filtered)
    
    combined = pd.concat(all_candles, ignore_index=False)
    combined = combined.reset_index().rename(columns={'datetime': 'timestamp'})
    
    return combined


def engineer_features(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer liquidity proxy features from OHLCV candles.
    
    Returns DataFrame with additional feature columns.
    """
    df = candles.copy()
    
    # Dollar volume
    df['dollar_volume'] = df['close'] * df['volume']
    
    # Spread proxy: (high - low) / close
    df['spread_proxy'] = (df['high'] - df['low']) / df['close']
    
    # Volatility proxy: candle range as % of close
    df['candle_range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Price change: |close - open| / open
    df['price_change_abs'] = np.abs(df['close'] - df['open']) / df['open']
    
    # Log returns
    df['log_return'] = np.log(df['close'] / df['open'])
    df['abs_log_return'] = np.abs(df['log_return'])
    
    return df


def merge_l2_with_candles(liquidity_df: pd.DataFrame, candles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge L2 liquidity metrics with candle features.

    Aggregate L2 metrics to hourly, then merge with 4H candles.
    """
    # Convert timestamps
    liquidity_df['timestamp'] = pd.to_datetime(liquidity_df['timestamp'], format='ISO8601', utc=True)
    candles_df['timestamp'] = pd.to_datetime(candles_df['timestamp'], utc=True)
    
    # Round L2 timestamps to nearest hour for aggregation
    liquidity_df['timestamp_hour'] = liquidity_df['timestamp'].dt.floor('H')
    
    # Aggregate L2 to hourly: median slippage/spread
    l2_hourly = liquidity_df.groupby(['coin', 'timestamp_hour']).agg({
        'spread_bps': 'median',
        'depth_10bps_total': 'median',
        'slippage_buy_1000usd': 'median',
        'slippage_buy_5000usd': 'median',
        'slippage_buy_10000usd': 'median',
        'slippage_buy_50000usd': 'median',
        'slippage_buy_100000usd': 'median',
    }).reset_index()
    
    l2_hourly = l2_hourly.rename(columns={'timestamp_hour': 'timestamp'})
    
    # Merge with candle data (4H candles will have multiple hourly L2 observations)
    # Strategy: for each 4H candle, take the median L2 metrics from that 4H period
    merged = []
    
    for _, candle in candles_df.iterrows():
        coin = candle['coin']
        candle_start = candle['timestamp']
        candle_end = candle_start + pd.Timedelta(hours=4)
        
        # Find L2 observations in this 4H window
        l2_window = l2_hourly[
            (l2_hourly['coin'] == coin) &
            (l2_hourly['timestamp'] >= candle_start) &
            (l2_hourly['timestamp'] < candle_end)
        ]
        
        if not l2_window.empty:
            # Take median L2 metrics across this 4H period
            l2_agg = l2_window.select_dtypes(include=np.number).median()
            
            # Combine candle + L2
            combined = candle.copy()
            for col in l2_agg.index:
                combined[f'l2_{col}'] = l2_agg[col]
            
            merged.append(combined)
    
    return pd.DataFrame(merged)


def prepare_training_data(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare training dataset with target variables and features.
    
    Creates multiple rows per candle for different trade sizes.
    """
    # Trade sizes to model (in USD)
    trade_sizes = [1000, 5000, 10000, 50000, 100000]
    
    training_rows = []
    
    for _, row in merged_df.iterrows():
        for size in trade_sizes:
            # Create a row for this (candle, trade_size) pair
            train_row = {
                'coin': row['coin'],
                'timestamp': row['timestamp'],
                'position_usd': size,
                
                # Features from candles
                'dollar_volume': row['dollar_volume'],
                'spread_proxy': row['spread_proxy'],
                'candle_range_pct': row['candle_range_pct'],
                'price_change_abs': row['price_change_abs'],
                'abs_log_return': row['abs_log_return'],
                
                # Derived features
                'relative_position': size / row['dollar_volume'] if row['dollar_volume'] > 0 else np.inf,
                'sqrt_relative_position': np.sqrt(size / row['dollar_volume']) if row['dollar_volume'] > 0 else np.inf,
                
                # Target: actual slippage from L2
                'slippage_bps': row[f'l2_slippage_buy_{size}usd'],
            }
            
            training_rows.append(train_row)
    
    df = pd.DataFrame(training_rows)
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Convert slippage from bps to decimal
    df['slippage_decimal'] = df['slippage_bps'] / 10000
    
    return df


def fit_models(train_df: pd.DataFrame) -> dict:
    """
    Fit battery of simple cost models.
    
    Returns dict of {model_name: {'model': fitted_model, 'features': list, 'metrics': dict}}
    """
    models = {}
    
    # Model 1: Linear
    features_linear = ['spread_proxy', 'candle_range_pct', 'relative_position']
    X = train_df[features_linear].values
    y = train_df['slippage_decimal'].values
    
    model_linear = Ridge(alpha=0.01)
    model_linear.fit(X, y)
    
    y_pred = model_linear.predict(X)
    models['linear'] = {
        'model': model_linear,
        'features': features_linear,
        'metrics': {
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'coefficients': dict(zip(features_linear, model_linear.coef_)),
            'intercept': model_linear.intercept_,
        }
    }
    
    # Model 2: Square-root market impact
    features_sqrt = ['spread_proxy', 'candle_range_pct', 'sqrt_relative_position']
    X = train_df[features_sqrt].values
    y = train_df['slippage_decimal'].values
    
    model_sqrt = Ridge(alpha=0.01)
    model_sqrt.fit(X, y)
    
    y_pred = model_sqrt.predict(X)
    models['sqrt_impact'] = {
        'model': model_sqrt,
        'features': features_sqrt,
        'metrics': {
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'coefficients': dict(zip(features_sqrt, model_sqrt.coef_)),
            'intercept': model_sqrt.intercept_,
        }
    }
    
    # Model 3: Volatility-weighted
    features_vol = ['spread_proxy', 'abs_log_return', 'relative_position']
    X = train_df[features_vol].values
    y = train_df['slippage_decimal'].values
    
    model_vol = Ridge(alpha=0.01)
    model_vol.fit(X, y)
    
    y_pred = model_vol.predict(X)
    models['vol_weighted'] = {
        'model': model_vol,
        'features': features_vol,
        'metrics': {
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'coefficients': dict(zip(features_vol, model_vol.coef_)),
            'intercept': model_vol.intercept_,
        }
    }
    
    # Model 4: Full features
    features_full = ['spread_proxy', 'candle_range_pct', 'abs_log_return', 'sqrt_relative_position']
    X = train_df[features_full].values
    y = train_df['slippage_decimal'].values
    
    model_full = Ridge(alpha=0.01)
    model_full.fit(X, y)
    
    y_pred = model_full.predict(X)
    models['full'] = {
        'model': model_full,
        'features': features_full,
        'metrics': {
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'coefficients': dict(zip(features_full, model_full.coef_)),
            'intercept': model_full.intercept_,
        }
    }
    
    return models


def main():
    parser = argparse.ArgumentParser(description="Build transaction cost model from L2 + candles")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--date", type=str, default="2025-09-20", help="Date for training data")
    parser.add_argument("--liquidity-file", type=Path, default=Path("data/features/liquidity_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/features"))
    args = parser.parse_args()
    
    print("="*80)
    print("BUILDING TRANSACTION COST MODEL")
    print("="*80)
    
    # Load data
    print("\n1. Loading liquidity metrics...")
    liquidity_df = pd.read_csv(args.liquidity_file)
    print(f"   Loaded {len(liquidity_df)} L2 observations")
    
    print("\n2. Loading candle data for {args.date}...")
    candles_df = load_candles_for_date(args.data_dir, args.date, interval="4h")
    print(f"   Loaded {len(candles_df)} candles across {candles_df['coin'].nunique()} coins")
    
    print("\n3. Engineering features from candles...")
    candles_df = engineer_features(candles_df)
    print(f"   Engineered {candles_df.shape[1]} features")
    
    print("\n4. Merging L2 with candles...")
    merged_df = merge_l2_with_candles(liquidity_df, candles_df)
    print(f"   Merged dataset: {len(merged_df)} candles with L2 data")
    
    print("\n5. Preparing training data...")
    train_df = prepare_training_data(merged_df)
    print(f"   Training set: {len(train_df)} observations")
    print(f"   Target range: {train_df['slippage_decimal'].min():.6f} - {train_df['slippage_decimal'].max():.6f}")
    
    print("\n6. Fitting models...")
    models = fit_models(train_df)
    
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    for name, info in models.items():
        print(f"\n{name.upper()}:")
        print(f"  Features: {info['features']}")
        print(f"  MAE: {info['metrics']['mae']:.6f} ({info['metrics']['mae']*10000:.2f} bps)")
        print(f"  R²: {info['metrics']['r2']:.4f}")
        print(f"  Coefficients:")
        for feat, coef in info['metrics']['coefficients'].items():
            print(f"    {feat:25s}: {coef:10.6f}")
        print(f"  Intercept: {info['metrics']['intercept']:.6f}")
    
    # Export best model (lowest MAE)
    best_model_name = min(models.keys(), key=lambda k: models[k]['metrics']['mae'])
    best_model = models[best_model_name]
    
    print("\n" + "="*80)
    print(f"BEST MODEL: {best_model_name.upper()}")
    print("="*80)
    
    # Save model parameters
    output_file = args.output_dir / "transaction_cost_model.json"
    model_export = {
        'model_type': best_model_name,
        'features': best_model['features'],
        'coefficients': best_model['metrics']['coefficients'],
        'intercept': best_model['metrics']['intercept'],
        'metrics': {
            'mae': best_model['metrics']['mae'],
            'mae_bps': best_model['metrics']['mae'] * 10000,
            'r2': best_model['metrics']['r2'],
        },
        'training_date': args.date,
        'training_samples': len(train_df),
    }
    
    with open(output_file, 'w') as f:
        json.dump(model_export, f, indent=2)
    
    print(f"\nModel saved to {output_file}")
    
    # Save training data for inspection
    train_output = args.output_dir / "cost_model_training_data.csv"
    train_df.to_csv(train_output, index=False)
    print(f"Training data saved to {train_output}")


if __name__ == "__main__":
    main()
