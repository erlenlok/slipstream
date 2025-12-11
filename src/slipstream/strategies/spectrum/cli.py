"""
CLI module for Spectrum Strategy
"""
import argparse
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.slipstream.strategies.spectrum.factor_model import compute_spectrum_factors
from src.slipstream.strategies.spectrum.signals import generate_spectrum_signals
from src.slipstream.strategies.spectrum.ridge_weighting import compute_factor_weights_rolling, apply_factor_weights
from src.slipstream.strategies.spectrum.optimizer import optimize_spectrum_portfolio
from src.slipstream.strategies.spectrum.execution import SpectrumExecutionBridge


def run_spectrum_strategy(
    data_dir: str = "data/market_data/1d",
    lookback_days: int = 60,
    target_leverage: float = 1.0
) -> Dict[str, Any]:
    """
    Run the full Spectrum strategy pipeline.
    
    Args:
        data_dir: Directory containing daily market data
        lookback_days: Number of days for lookback calculations
        target_leverage: Target portfolio leverage
    
    Returns:
        Dictionary with strategy results
    """
    print("Running Spectrum Strategy Pipeline...")
    
    # This is a simplified version - in practice you would:
    # 1. Load daily OHLCV data for universe
    # 2. Load daily funding data
    # 3. Load BTC/ETH returns for factor model
    # 4. Run the full pipeline
    
    # For now, we'll create synthetic data to demonstrate the pipeline
    
    # Create sample data to simulate loaded data
    n_days = lookback_days + 10  # Extra days for recent data
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Generate synthetic market data
    np.random.seed(42)
    assets = ['BTC', 'ETH', 'SOL', 'ADA', 'XRP']
    
    # BTC and ETH returns (for factor model)
    btc_returns = pd.Series(
        np.random.normal(0.0005, 0.015, len(dates)), 
        index=dates
    )
    eth_returns = pd.Series(
        np.random.normal(0.0004, 0.012, len(dates)), 
        index=dates
    )
    
    # All asset returns
    all_returns = pd.DataFrame(index=dates)
    for asset in assets:
        # Each asset has some correlation with BTC/ETH + idiosyncratic component
        asset_returns = (
            0.3 * btc_returns + 
            0.2 * eth_returns + 
            np.random.normal(0, 0.01, len(dates))
        )
        all_returns[asset] = asset_returns
    
    # Remove BTC and ETH from universe for factor analysis (they're the factors)
    universe_returns = all_returns.drop(columns=['BTC', 'ETH']) if 'BTC' in all_returns.columns and 'ETH' in all_returns.columns else all_returns
    
    # Generate synthetic funding data
    funding_rates = pd.DataFrame(index=dates)
    for asset in universe_returns.columns:
        funding_rates[asset] = np.random.normal(0.0001, 0.0005, len(dates))
    
    # Generate synthetic volume data for universe filtering
    volume_data = pd.DataFrame(index=dates)
    for asset in universe_returns.columns:
        volume_data[asset] = np.random.lognormal(15, 1, len(dates))  # Daily volumes in reasonable range
    
    # Step 1: Compute spectrum factors (Module A)
    print("Step 1: Computing spectrum factors...")
    betas, residuals, idio_vol, daily_funding = compute_spectrum_factors(
        universe_returns,
        btc_returns,
        eth_returns,
        funding_rates,
        lookback_window=30,
        min_history_days=30,
        min_avg_volume=10_000_000,
        volume_data=volume_data
    )
    
    # Step 2: Generate signals (Module B)
    print("Step 2: Generating signals...")
    signal_matrix = generate_spectrum_signals(
        residuals.iloc[-lookback_days:],  # Use recent data
        daily_funding.iloc[-lookback_days:],
        idio_vol.iloc[-lookback_days:],
        warmup_periods=10,
        momentum_fast_span=3,
        momentum_slow_span=10,
        meanrev_sma_period=5
    )
    
    # Step 3: Compute factor weights (Module C)
    print("Step 3: Computing factor weights...")
    # Prepare target returns (next day residuals)
    target_returns = residuals.shift(-1).iloc[-lookback_days:]  # Next day returns as target
    
    factor_weights, weight_training_results = compute_factor_weights_rolling(
        signal_matrix,
        target_returns,
        lookback_period=lookback_days
    )
    
    # Step 4: Generate composite alpha
    print("Step 4: Generating composite alpha...")
    composite_alpha = apply_factor_weights(signal_matrix, factor_weights)
    
    # Step 5: Optimize portfolio (Module D)
    print("Step 5: Optimizing portfolio...")
    # Use the most recent composite alpha and residuals for optimization
    latest_composite_alpha = composite_alpha.iloc[[-1]].T  # Latest row as series
    latest_residuals = residuals.iloc[-30:]  # Recent residuals for covariance
    latest_betas = betas.iloc[[-1]].T  # Latest betas reshaped for the optimizer
    
    # Since the optimizer expects proper data format, we'll use the latest values
    latest_date = composite_alpha.index[-1]
    current_composite_alpha = pd.DataFrame({col: [composite_alpha[col].iloc[-1]] for col in composite_alpha.columns}).T.iloc[:, 0]  # As Series
    current_residuals = residuals.loc[:latest_date]
    current_betas = betas.loc[latest_date]
    
    target_weights, optimization_info = optimize_spectrum_portfolio(
        composite_alpha=pd.DataFrame({col: [composite_alpha[col].iloc[-1]] for col in composite_alpha.columns}, index=[latest_date]),
        residuals=current_residuals,
        betas=pd.DataFrame({col: [current_betas[col]] if col in current_betas.index else [0.0] for col in current_residuals.columns}, index=[latest_date]),
        target_leverage=target_leverage,
        max_single_pos=0.1
    )
    
    # Step 6: Execute (Module E) - simplified
    print("Step 6: Preparing execution...")
    execution_bridge = SpectrumExecutionBridge(account_equity=1000000.0)
    
    # The execution bridge would handle the two-stage timing and hedging
    # This is a simplified execution for demonstration
    results = {
        'factor_weights': factor_weights,
        'target_weights': target_weights.to_dict(),
        'optimization_info': optimization_info,
        'strategy_date': latest_date,
        'n_assets': len(target_weights),
        'gross_exposure': float(np.sum(np.abs(target_weights))),
        'factor_performance': weight_training_results
    }
    
    print(f"Strategy run completed. Date: {latest_date}")
    print(f"Target weights: {dict(list(target_weights.items())[:3])}...")  # Show first few
    print(f"Gross exposure: {results['gross_exposure']:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Spectrum Strategy CLI')
    parser.add_argument('--data-dir', type=str, default='data/market_data/1d',
                        help='Directory containing daily market data')
    parser.add_argument('--lookback-days', type=int, default=60,
                        help='Number of days for lookback calculations')
    parser.add_argument('--target-leverage', type=float, default=1.0,
                        help='Target portfolio leverage')
    
    args = parser.parse_args()
    
    results = run_spectrum_strategy(
        data_dir=args.data_dir,
        lookback_days=args.lookback_days,
        target_leverage=args.target_leverage
    )
    
    print("\nStrategy Results Summary:")
    print(f"- Number of assets: {results['n_assets']}")
    print(f"- Gross exposure: {results['gross_exposure']:.3f}")
    print(f"- Factor weights: {results['factor_weights']}")
    print(f"- Optimization success: {results['optimization_info'].get('success', 'N/A')}")


if __name__ == "__main__":
    main()