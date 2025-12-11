"""
Final integration test for Spectrum strategy
"""
import pandas as pd
import numpy as np
from datetime import datetime

def test_full_spectrum_integration():
    """Test that all Spectrum modules work together in the expected sequence."""
    print("Running full Spectrum integration test...")

    # Import all modules
    from slipstream.strategies.spectrum.factor_model import compute_spectrum_factors
    from slipstream.strategies.spectrum.signals import generate_spectrum_signals
    from slipstream.strategies.spectrum.ridge_weighting import compute_factor_weights_rolling, apply_factor_weights
    from slipstream.strategies.spectrum.optimizer import optimize_spectrum_portfolio
    from slipstream.strategies.spectrum.execution import SpectrumExecutionBridge
    
    print("✓ All modules imported successfully")
    
    # Generate test data
    dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
    np.random.seed(123)
    
    # Factor returns (BTC and ETH)
    btc_returns = pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)
    eth_returns = pd.Series(np.random.normal(0.0008, 0.012, len(dates)), index=dates)
    
    # Universe assets returns (with some correlation to BTC/ETH)
    assets = ['SOL', 'ADA', 'XRP', 'DOGE']
    universe_returns = pd.DataFrame(index=dates)
    for asset in assets:
        # Each asset has different sensitivity to BTC/ETH + idiosyncratic component
        sensitivity_btc = np.random.uniform(0.3, 0.8)
        sensitivity_eth = np.random.uniform(0.1, 0.5)
        universe_returns[asset] = (
            sensitivity_btc * btc_returns + 
            sensitivity_eth * eth_returns + 
            np.random.normal(0, 0.01, len(dates)) * np.random.uniform(0.5, 1.5)
        )
    
    # Funding rates (uncorrelated with returns for realistic behavior)
    funding_rates = pd.DataFrame(index=dates)
    for asset in assets:
        funding_rates[asset] = np.random.normal(0.0001, 0.0003, len(dates))
    
    # Volume data for universe filtering
    volume_data = pd.DataFrame(index=dates)
    for asset in assets:
        volume_data[asset] = np.random.lognormal(16, 0.8, len(dates))
    
    print("✓ Test data generated")
    
    # Module A: Factor Engine
    print("Testing Module A: Factor Engine...")
    betas, residuals, idio_vol, daily_funding = compute_spectrum_factors(
        universe_returns,
        btc_returns,
        eth_returns,
        funding_rates,
        lookback_window=30,
        min_history_days=20,
        min_avg_volume=5_000_000,
        volume_data=volume_data
    )
    assert not residuals.empty, "Residuals should not be empty"
    assert not betas.empty, "Betas should not be empty"
    assert not idio_vol.empty, "Idio vol should not be empty"
    print("✓ Module A completed successfully")
    
    # Module B: Signal Factory
    print("Testing Module B: Signal Factory...")
    # Use recent data for signal generation
    recent_residuals = residuals.iloc[-40:]  # Last 40 days
    recent_funding = daily_funding.iloc[-40:]
    recent_vol = idio_vol.iloc[-40:]
    
    signal_matrix = generate_spectrum_signals(
        recent_residuals,
        recent_funding,
        recent_vol,
        warmup_periods=10,
        momentum_fast_span=3,
        momentum_slow_span=10,
        meanrev_sma_period=5,
        zscore_winsorize_at=3.0
    )
    
    assert 'idio_carry' in signal_matrix, "Idio carry signal should exist"
    assert 'idio_momentum' in signal_matrix, "Idio momentum signal should exist"
    assert 'idio_meanrev' in signal_matrix, "Idio meanrev signal should exist"
    print("✓ Module B completed successfully")
    
    # Module C: Dynamic Weighting
    print("Testing Module C: Dynamic Weighting...")
    # Prepare target returns (next day residuals as prediction targets)
    target_returns = residuals.shift(-1).iloc[-35:]  # Next day returns, recent 35 days
    
    factor_weights, weight_results = compute_factor_weights_rolling(
        signal_matrix,
        target_returns,
        lookback_period=30,
        cv_folds=3
    )
    
    assert isinstance(factor_weights, dict), "Factor weights should be a dictionary"
    assert set(factor_weights.keys()) == {'idio_carry', 'idio_momentum', 'idio_meanrev'}, "Should have all three factor weights"
    print(f"✓ Module C completed successfully with factor weights: {factor_weights}")
    
    # Module D: Robust Optimizer
    print("Testing Module D: Robust Optimizer...")
    # Generate composite alpha using the computed factor weights
    composite_alpha = apply_factor_weights(signal_matrix, factor_weights)
    
    # Use latest data for optimization
    latest_date = composite_alpha.index[-1]
    current_composite_alpha = pd.DataFrame({
        col: [composite_alpha[col].iloc[-1]] for col in composite_alpha.columns
    }, index=[latest_date])
    
    current_residuals = residuals.loc[:latest_date]
    
    # Create beta DataFrame with same structure as alpha
    latest_betas = betas.loc[latest_date]
    current_betas = pd.DataFrame({
        col: [latest_betas[col + '_btc'] if col + '_btc' in latest_betas.index else 0.5] 
        for col in current_residuals.columns
    }, index=[latest_date])
    
    # Fill any missing beta values
    for col in current_residuals.columns:
        if col not in current_betas.columns:
            current_betas[col] = [0.5]
    
    target_weights, opt_info = optimize_spectrum_portfolio(
        composite_alpha=current_composite_alpha,
        residuals=current_residuals,
        betas=current_betas,
        target_leverage=1.0,
        max_single_pos=0.3
    )
    
    assert isinstance(target_weights, pd.Series), "Target weights should be a pandas Series"
    assert len(target_weights) >= len(assets), "Should have weights for all assets"
    print(f"✓ Module D completed successfully with {len(target_weights)} assets")
    
    # Module E: Execution Bridge
    print("Testing Module E: Execution Bridge...")
    bridge = SpectrumExecutionBridge(account_equity=1_000_000)
    
    # Test the basic functionality without running full async execution
    current_prices = {asset: 100.0 for asset in assets}  # Mock prices
    initial_positions = {}
    
    # Convert series to dict for position quantities (mock values)
    position_quantities = {k: 0.0 for k in target_weights.index}
    
    orders = bridge.convert_weights_to_orders(target_weights, current_prices, position_quantities)
    
    assert isinstance(orders, list), "Orders should be a list"
    print(f"✓ Module E completed successfully with {len(orders)} orders ready for execution")
    
    print("\n🎉 Full Spectrum integration test PASSED!")
    print(f"✓ All 5 modules working together:")
    print(f"  - Module A: Factor Engine (OLS decomposition)")
    print(f"  - Module B: Signal Factory (Idio-Carry, Momentum, MeanRev)")  
    print(f"  - Module C: Dynamic Weighting (Ridge regression)")
    print(f"  - Module D: Robust Optimizer (CVXPY portfolio optimization)")
    print(f"  - Module E: Execution Bridge (Two-stage timing + beta hedging)")
    
    # Summary
    active_assets = [k for k, v in target_weights.items() if abs(v) > 1e-6]
    gross_exposure = float(np.sum(np.abs(target_weights)))
    
    print(f"\nPortfolio Summary:")
    print(f"- Assets with positions: {len(active_assets)}")
    print(f"- Gross exposure: {gross_exposure:.3f}")
    print(f"- Leverage target: 1.0")
    print(f"- Max single position limit: 0.3")
    
    return True

if __name__ == "__main__":
    success = test_full_spectrum_integration()
    if success:
        print("\n🎊 ALL INTEGRATION TESTS PASSED! 🎊")
        print("Spectrum strategy is fully integrated with the Slipstream platform!")
    else:
        print("\n❌ Integration test failed")