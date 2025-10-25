#!/usr/bin/env python3
"""
Generate scientific visualizations of transaction cost models.

Creates publication-quality plots for model inspection and validation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import json

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def load_data():
    """Load training data and model."""
    # Try to load cleaned data first, fall back to original
    cleaned_path = Path('data/features/cost_model_training_data_cleaned.csv')
    if cleaned_path.exists():
        train_df = pd.read_csv(cleaned_path)
        print(f"Loaded cleaned training data: {len(train_df)} observations")
    else:
        train_df = pd.read_csv('data/features/cost_model_training_data.csv')
        print(f"Loaded original training data: {len(train_df)} observations")

    with open('data/features/transaction_cost_model.json', 'r') as f:
        model_params = json.load(f)

    return train_df, model_params


def fit_all_models(train_df):
    """Refit all models to get predictions."""
    # Compute polynomial features if not present
    if 'spread_proxy_sq' not in train_df.columns:
        train_df['spread_proxy_sq'] = train_df['spread_proxy'] ** 2
        train_df['candle_range_pct_sq'] = train_df['candle_range_pct'] ** 2
        train_df['abs_log_return_sq'] = train_df['abs_log_return'] ** 2
        train_df['sqrt_relative_position_sq'] = train_df['sqrt_relative_position'] ** 2

    models = {}
    y = train_df['slippage_decimal'].values

    # Model 1: Linear
    features = ['spread_proxy', 'candle_range_pct', 'relative_position']
    X = train_df[features].values
    model = Ridge(alpha=0.01).fit(X, y)
    models['Linear'] = {'model': model, 'features': features, 'predictions': model.predict(X)}

    # Model 2: Square-root impact
    features = ['spread_proxy', 'candle_range_pct', 'sqrt_relative_position']
    X = train_df[features].values
    model = Ridge(alpha=0.01).fit(X, y)
    models['Sqrt Impact'] = {'model': model, 'features': features, 'predictions': model.predict(X)}

    # Model 3: Vol-weighted
    features = ['spread_proxy', 'abs_log_return', 'relative_position']
    X = train_df[features].values
    model = Ridge(alpha=0.01).fit(X, y)
    models['Vol Weighted'] = {'model': model, 'features': features, 'predictions': model.predict(X)}

    # Model 4: Full (baseline)
    features = ['spread_proxy', 'candle_range_pct', 'abs_log_return', 'sqrt_relative_position']
    X = train_df[features].values
    model = Ridge(alpha=0.01).fit(X, y)
    models['Full'] = {'model': model, 'features': features, 'predictions': model.predict(X)}

    # Model 5: Full Polynomial (with squared terms)
    features = ['spread_proxy', 'spread_proxy_sq', 'candle_range_pct', 'candle_range_pct_sq',
                'abs_log_return', 'sqrt_relative_position', 'sqrt_relative_position_sq']
    X = train_df[features].values
    model = Ridge(alpha=0.01).fit(X, y)
    models['Full Poly'] = {'model': model, 'features': features, 'predictions': model.predict(X)}

    return models


def plot_model_comparison(train_df, models, output_dir):
    """Plot 1: Model comparison - R² and MAE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    model_names = list(models.keys())
    r2_scores = [r2_score(train_df['slippage_decimal'], models[m]['predictions']) for m in model_names]
    mae_scores = [mean_absolute_error(train_df['slippage_decimal'], models[m]['predictions']) * 10000 for m in model_names]
    
    # R² comparison
    colors = sns.color_palette("husl", len(model_names))
    bars1 = ax1.bar(model_names, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('Model Goodness-of-Fit', fontweight='bold')
    ax1.set_ylim([0, max(r2_scores) * 1.2])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.4, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target: 0.4')
    
    # Add values on bars
    for bar, val in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MAE comparison
    bars2 = ax2.bar(model_names, mae_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Mean Absolute Error (bps)', fontweight='bold')
    ax2.set_title('Model Prediction Error', fontweight='bold')
    ax2.set_ylim([0, max(mae_scores) * 1.2])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add values on bars
    for bar, val in zip(bars2, mae_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_comparison.png'}")
    plt.close()


def plot_actual_vs_predicted(train_df, models, output_dir):
    """Plot 2: Actual vs predicted scatter for all models."""
    n_models = len(models)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols  # Ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    axes = axes.flatten()

    y_actual = train_df['slippage_decimal'] * 10000  # Convert to bps

    for idx, (name, model_data) in enumerate(models.items()):
        ax = axes[idx]
        y_pred = model_data['predictions'] * 10000

        # Scatter plot - SWAPPED: predicted on x-axis, actual on y-axis
        ax.scatter(y_pred, y_actual, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)

        # Perfect prediction line
        max_val = max(y_actual.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')

        # Metrics
        r2 = r2_score(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)

        ax.set_xlabel('Predicted Slippage (bps)', fontweight='bold')
        ax.set_ylabel('Actual Slippage (bps)', fontweight='bold')
        ax.set_title(f'{name}\nR²={r2:.3f}, MAE={mae:.2f} bps', fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3, linestyle='--')

        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'actual_vs_predicted.png'}")
    plt.close()


def plot_residuals(train_df, models, output_dir):
    """Plot 3: Residual analysis for best model."""
    best_model_name = 'Full'
    y_actual = train_df['slippage_decimal'] * 10000
    y_pred = models[best_model_name]['predictions'] * 10000
    residuals = y_actual - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Residual histogram
    axes[0].hist(residuals, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residual (bps)', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title(f'Residual Distribution\nMean: {residuals.mean():.3f} bps', fontweight='bold')
    axes[0].grid(alpha=0.3, linestyle='--')
    
    # Residuals vs predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Slippage (bps)', fontweight='bold')
    axes[1].set_ylabel('Residual (bps)', fontweight='bold')
    axes[1].set_title('Residual vs Predicted', fontweight='bold')
    axes[1].grid(alpha=0.3, linestyle='--')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    axes[2].grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'residual_analysis.png'}")
    plt.close()


def plot_feature_importance(models, output_dir):
    """Plot 4: Feature importance (coefficients) for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect coefficients
    all_features = set()
    for model_data in models.values():
        all_features.update(model_data['features'])
    
    all_features = sorted(all_features)
    
    # Create coefficient matrix
    coef_matrix = np.zeros((len(models), len(all_features)))
    for i, (name, model_data) in enumerate(models.items()):
        for j, feat in enumerate(all_features):
            if feat in model_data['features']:
                feat_idx = model_data['features'].index(feat)
                coef_matrix[i, j] = model_data['model'].coef_[feat_idx]
    
    # Heatmap
    im = ax.imshow(coef_matrix, aspect='auto', cmap='RdBu_r', vmin=-0.005, vmax=0.005)
    
    # Ticks
    ax.set_xticks(np.arange(len(all_features)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(all_features, rotation=45, ha='right')
    ax.set_yticklabels(models.keys())
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coefficient Value', fontweight='bold')
    
    # Annotate cells
    for i in range(len(models)):
        for j in range(len(all_features)):
            text = ax.text(j, i, f'{coef_matrix[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    ax.set_title('Feature Coefficients Across Models', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'feature_importance.png'}")
    plt.close()


def plot_cost_curves(train_df, models, output_dir):
    """Plot 5: Cost curves for different position sizes."""
    best_model = models['Full']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by position size
    grouped = train_df.groupby('position_usd')
    
    position_sizes = sorted(train_df['position_usd'].unique())
    actual_costs = [grouped.get_group(size)['slippage_decimal'].median() * 10000 for size in position_sizes]
    predicted_costs = [np.median(best_model['predictions'][train_df['position_usd'] == size]) * 10000
                       for size in position_sizes]
    
    # Linear scale
    ax1.plot(position_sizes, actual_costs, 'o-', linewidth=2, markersize=8, label='Actual', color='steelblue')
    ax1.plot(position_sizes, predicted_costs, 's--', linewidth=2, markersize=8, label='Predicted', color='orangered')
    ax1.set_xlabel('Position Size (USD)', fontweight='bold')
    ax1.set_ylabel('Median Slippage (bps)', fontweight='bold')
    ax1.set_title('Cost vs Position Size (Linear)', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Log scale
    ax2.loglog(position_sizes, actual_costs, 'o-', linewidth=2, markersize=8, label='Actual', color='steelblue')
    ax2.loglog(position_sizes, predicted_costs, 's--', linewidth=2, markersize=8, label='Predicted', color='orangered')
    
    # Add power law reference line
    x_ref = np.array(position_sizes)
    y_ref = 0.5 * (x_ref / 10000) ** 0.5  # sqrt market impact
    ax2.loglog(x_ref, y_ref, ':', linewidth=2, color='green', alpha=0.7, label='Sqrt reference')
    
    ax2.set_xlabel('Position Size (USD)', fontweight='bold')
    ax2.set_ylabel('Median Slippage (bps)', fontweight='bold')
    ax2.set_title('Cost vs Position Size (Log-Log)', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cost_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'cost_curves.png'}")
    plt.close()


def plot_liquidity_analysis(train_df, models, output_dir):
    """Plot 6: Cost vs liquidity metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    y_actual = train_df['slippage_decimal'] * 10000
    
    # Cost vs dollar volume
    ax = axes[0, 0]
    scatter = ax.scatter(train_df['dollar_volume'], y_actual, 
                        c=train_df['position_usd'], cmap='viridis', 
                        alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dollar Volume', fontweight='bold')
    ax.set_ylabel('Slippage (bps)', fontweight='bold')
    ax.set_title('Cost vs Liquidity', fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--', which='both')
    plt.colorbar(scatter, ax=ax, label='Position Size (USD)')
    
    # Cost vs spread proxy
    ax = axes[0, 1]
    scatter = ax.scatter(train_df['spread_proxy'] * 10000, y_actual,
                        c=train_df['position_usd'], cmap='viridis',
                        alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Spread Proxy (bps)', fontweight='bold')
    ax.set_ylabel('Slippage (bps)', fontweight='bold')
    ax.set_title('Cost vs Spread', fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    plt.colorbar(scatter, ax=ax, label='Position Size (USD)')
    
    # Cost vs relative position size
    ax = axes[1, 0]
    ax.scatter(train_df['relative_position'], y_actual, alpha=0.6, s=50,
              edgecolors='black', linewidths=0.5, color='steelblue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Relative Position (size / volume)', fontweight='bold')
    ax.set_ylabel('Slippage (bps)', fontweight='bold')
    ax.set_title('Cost vs Market Impact', fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--', which='both')
    
    # Cost vs volatility
    ax = axes[1, 1]
    ax.scatter(train_df['abs_log_return'] * 10000, y_actual, alpha=0.6, s=50,
              edgecolors='black', linewidths=0.5, color='coral')
    ax.set_xlabel('Volatility (bps)', fontweight='bold')
    ax.set_ylabel('Slippage (bps)', fontweight='bold')
    ax.set_title('Cost vs Volatility', fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'liquidity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'liquidity_analysis.png'}")
    plt.close()


def plot_coin_performance(train_df, models, output_dir):
    """Plot 7: Model performance by coin."""
    best_model = models['Full']
    y_actual = train_df['slippage_decimal'] * 10000
    y_pred = best_model['predictions'] * 10000
    
    # Compute MAE per coin
    train_df['residual_abs'] = np.abs(y_actual - y_pred)
    coin_mae = train_df.groupby('coin')['residual_abs'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['green' if mae < 5 else 'orange' if mae < 8 else 'red' for mae in coin_mae.values]
    bars = ax.barh(coin_mae.index, coin_mae.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Mean Absolute Error (bps)', fontweight='bold')
    ax.set_ylabel('Coin', fontweight='bold')
    ax.set_title('Model Performance by Coin', fontweight='bold')
    ax.axvline(x=coin_mae.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {coin_mae.median():.2f} bps')
    ax.legend()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'coin_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'coin_performance.png'}")
    plt.close()


def main():
    print("="*80)
    print("GENERATING TRANSACTION COST MODEL VISUALIZATIONS")
    print("="*80)
    
    output_dir = Path('data/features/plots')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nLoading data...")
    train_df, model_params = load_data()
    
    print("Fitting all models...")
    models = fit_all_models(train_df)
    
    print("\nGenerating plots...")
    plot_model_comparison(train_df, models, output_dir)
    plot_actual_vs_predicted(train_df, models, output_dir)
    plot_residuals(train_df, models, output_dir)
    plot_feature_importance(models, output_dir)
    plot_cost_curves(train_df, models, output_dir)
    plot_liquidity_analysis(train_df, models, output_dir)
    plot_coin_performance(train_df, models, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  1. model_comparison.png - R² and MAE comparison")
    print("  2. actual_vs_predicted.png - Scatter plots for all models")
    print("  3. residual_analysis.png - Residual diagnostics")
    print("  4. feature_importance.png - Coefficient heatmap")
    print("  5. cost_curves.png - Position size vs cost")
    print("  6. liquidity_analysis.png - Cost drivers analysis")
    print("  7. coin_performance.png - Per-coin model accuracy")


if __name__ == "__main__":
    main()
