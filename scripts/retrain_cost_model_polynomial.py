#!/usr/bin/env python3
"""
Retrain transaction cost model with polynomial features to fix overprediction near origin.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import json
from pathlib import Path

# Load cleaned training data
print("Loading cleaned training data...")
train_df = pd.read_csv('data/features/cost_model_training_data_cleaned.csv')

print(f"Dataset: {len(train_df)} observations")

# Define feature sets to test
feature_sets = {
    'baseline': ['spread_proxy', 'candle_range_pct', 'abs_log_return', 'sqrt_relative_position'],

    'polynomial_v1': [
        'spread_proxy', 'spread_proxy_sq',
        'candle_range_pct',
        'abs_log_return',
        'sqrt_relative_position', 'sqrt_relative_position_sq'
    ],

    'polynomial_v2': [
        'spread_proxy', 'spread_proxy_sq',
        'candle_range_pct', 'candle_range_pct_sq',
        'abs_log_return',
        'sqrt_relative_position', 'sqrt_relative_position_sq'
    ],

    'polynomial_v3': [
        'spread_proxy', 'spread_proxy_sq',
        'abs_log_return', 'abs_log_return_sq',
        'sqrt_relative_position', 'sqrt_relative_position_sq'
    ],
}

# Compute polynomial features
print("\nComputing polynomial features...")
train_df['spread_proxy_sq'] = train_df['spread_proxy'] ** 2
train_df['candle_range_pct_sq'] = train_df['candle_range_pct'] ** 2
train_df['abs_log_return_sq'] = train_df['abs_log_return'] ** 2
train_df['sqrt_relative_position_sq'] = train_df['sqrt_relative_position'] ** 2

# Test each model
results = []
models = {}

print("\n" + "="*80)
print("TRAINING POLYNOMIAL MODELS")
print("="*80)

for model_name, features in feature_sets.items():
    print(f"\nModel: {model_name}")
    print(f"  Features: {features}")

    X = train_df[features].values
    y = train_df['slippage_decimal'].values

    # Train with Ridge regularization
    model = Ridge(alpha=0.01).fit(X, y)
    y_pred = model.predict(X)

    # Metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred) * 10000

    # Residual analysis by predicted value bins
    residuals = (y - y_pred) * 10000

    # Bin by predicted value
    pred_bps = y_pred * 10000
    low_pred_mask = pred_bps < 5
    mid_pred_mask = (pred_bps >= 5) & (pred_bps < 10)
    high_pred_mask = pred_bps >= 10

    mean_resid_low = np.mean(residuals[low_pred_mask]) if np.any(low_pred_mask) else 0
    mean_resid_mid = np.mean(residuals[mid_pred_mask]) if np.any(mid_pred_mask) else 0
    mean_resid_high = np.mean(residuals[high_pred_mask]) if np.any(high_pred_mask) else 0

    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.2f} bps")
    print(f"  Mean residual [0-5 bps]:   {mean_resid_low:+.2f} bps (n={np.sum(low_pred_mask)})")
    print(f"  Mean residual [5-10 bps]:  {mean_resid_mid:+.2f} bps (n={np.sum(mid_pred_mask)})")
    print(f"  Mean residual [10+ bps]:   {mean_resid_high:+.2f} bps (n={np.sum(high_pred_mask)})")

    results.append({
        'model': model_name,
        'r2': r2,
        'mae_bps': mae,
        'mean_resid_low': mean_resid_low,
        'mean_resid_mid': mean_resid_mid,
        'mean_resid_high': mean_resid_high,
        'n_features': len(features)
    })

    models[model_name] = {
        'model_obj': model,
        'features': features,
        'r2': r2,
        'mae_bps': mae
    }

# Compare models
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Select best model (balance R² and bias reduction near origin)
print("\n" + "="*80)
print("MODEL SELECTION")
print("="*80)

# Criteria: Highest R², lowest absolute bias near origin
results_df['bias_metric'] = np.abs(results_df['mean_resid_low'])
results_df['composite_score'] = results_df['r2'] - 0.1 * results_df['bias_metric'] / 10  # Penalize bias

best_idx = results_df['composite_score'].idxmax()
best_model_name = results_df.iloc[best_idx]['model']

print(f"\nBest model: {best_model_name}")
print(f"  R²: {results_df.iloc[best_idx]['r2']:.4f}")
print(f"  MAE: {results_df.iloc[best_idx]['mae_bps']:.2f} bps")
print(f"  Bias [0-5 bps]: {results_df.iloc[best_idx]['mean_resid_low']:+.2f} bps")

# Export best model
best_model_info = models[best_model_name]
best_model = best_model_info['model_obj']
best_features = best_model_info['features']

output_file = Path('data/features/transaction_cost_model.json')
model_export = {
    'model_type': f'{best_model_name}_cleaned',
    'features': best_features,
    'coefficients': dict(zip(best_features, best_model.coef_.tolist())),
    'intercept': float(best_model.intercept_),
    'metrics': {
        'mae': float(best_model_info['mae_bps'] / 10000),
        'mae_bps': float(best_model_info['mae_bps']),
        'r2': float(best_model_info['r2']),
    },
    'training_date': '2025-09-20',
    'training_samples': len(train_df),
    'outliers_removed': 4,
}

with open(output_file, 'w') as f:
    json.dump(model_export, f, indent=2)

print(f"\nModel saved to {output_file}")

# Print coefficients
print(f"\nBest model coefficients ({best_model_name}):")
for feat, coef in zip(best_features, best_model.coef_):
    print(f"  {feat:30s}: {coef:+.6f}")
print(f"  Intercept: {best_model.intercept_:+.6f}")

# Save enhanced training data with predictions
train_df['predicted_slippage_bps'] = best_model.predict(train_df[best_features]) * 10000
train_df['residual_bps'] = (train_df['slippage_decimal'] - train_df['predicted_slippage_bps'] / 10000) * 10000

train_output = Path('data/features/cost_model_training_data_polynomial.csv')
train_df.to_csv(train_output, index=False)
print(f"\nEnhanced training data saved to {train_output}")

print("\n" + "="*80)
print("POLYNOMIAL MODEL TRAINING COMPLETE")
print("="*80)
