#!/usr/bin/env python3
"""
Retrain transaction cost model with outliers removed.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import json
from pathlib import Path

# Load training data
print("Loading training data...")
train_df = pd.read_csv('data/features/cost_model_training_data.csv')

# Fit initial model to identify outliers
print("Identifying outliers...")
features = ['spread_proxy', 'candle_range_pct', 'abs_log_return', 'sqrt_relative_position']
X = train_df[features].values
y = train_df['slippage_decimal'].values

model_initial = Ridge(alpha=0.01).fit(X, y)
y_pred = model_initial.predict(X)

# Find outliers
residuals = np.abs(y * 10000 - y_pred * 10000)

# Get 4 worst outliers
worst_4_idx = np.argsort(residuals)[-4:]
print("\nRemoving 4 worst outliers:")
for i, idx in enumerate(worst_4_idx[::-1], 1):
    print(f"  {i}. Observation {idx}: {train_df.iloc[idx]['coin']}, "
          f"${train_df.iloc[idx]['position_usd']:,.0f}, "
          f"residual={residuals[idx]:.2f} bps")

# Filter out outliers
mask_keep = np.ones(len(train_df), dtype=bool)
mask_keep[worst_4_idx] = False

train_df_clean = train_df[mask_keep].copy()

print(f"\nOriginal dataset: {len(train_df)} observations")
print(f"Cleaned dataset: {len(train_df_clean)} observations")
print(f"Removed: {len(train_df) - len(train_df_clean)} ({100*(1-len(train_df_clean)/len(train_df)):.2f}%)")

# Retrain on cleaned data
print("\nRetraining model on cleaned data...")
X_clean = train_df_clean[features].values
y_clean = train_df_clean['slippage_decimal'].values

model_clean = Ridge(alpha=0.01).fit(X_clean, y_clean)
y_pred_clean = model_clean.predict(X_clean)

# Metrics
r2_clean = r2_score(y_clean, y_pred_clean)
mae_clean = mean_absolute_error(y_clean, y_pred_clean) * 10000

print(f"\nCleaned model performance:")
print(f"  R²: {r2_clean:.4f}")
print(f"  MAE: {mae_clean:.2f} bps")
print(f"\nModel coefficients:")
for feat, coef in zip(features, model_clean.coef_):
    print(f"  {feat:25s}: {coef:.6f}")
print(f"  Intercept: {model_clean.intercept_:.6f}")

# Export model
output_file = Path('data/features/transaction_cost_model.json')
model_export = {
    'model_type': 'full_cleaned',
    'features': features,
    'coefficients': dict(zip(features, model_clean.coef_.tolist())),
    'intercept': float(model_clean.intercept_),
    'metrics': {
        'mae': float(mae_clean / 10000),  # Store as decimal
        'mae_bps': float(mae_clean),
        'r2': float(r2_clean),
    },
    'training_date': '2025-09-20',
    'training_samples': len(train_df_clean),
    'outliers_removed': 4,
}

with open(output_file, 'w') as f:
    json.dump(model_export, f, indent=2)

print(f"\nModel saved to {output_file}")

# Save cleaned training data
train_output = Path('data/features/cost_model_training_data_cleaned.csv')
train_df_clean.to_csv(train_output, index=False)
print(f"Cleaned training data saved to {train_output}")

print("\n" + "="*80)
print("RETRAINING COMPLETE")
print("="*80)

