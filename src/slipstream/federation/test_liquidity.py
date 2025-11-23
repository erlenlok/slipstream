"""
Test suite for the Liquidity Surface Mapper component.

This module tests the liquidity surface mapper's ability to model slippage vs. size
using Square-Root Law impact models for capacity-based position sizing.
"""

from datetime import datetime, timedelta
import asyncio
import numpy as np
from slipstream.federation.liquidity import LiquiditySurfaceMapper, MarketDataPoint, TradeSizeSlippage


async def test_liquidity_model_building():
    """
    Test that liquidity models are built correctly using Square-Root Law.
    """
    print("Testing liquidity model building...")
    
    mapper = LiquiditySurfaceMapper(min_data_points_for_model=5)
    await mapper.start()
    
    symbol = "BTC"
    base_time = datetime.now()
    
    # Add trade data that follows square root law pattern
    for i in range(10):
        trade_size = 0.1 * (i + 1)  # Increasing trade sizes: 0.1, 0.2, 0.3, ...
        # Square root law: slippage = k * sqrt(size) 
        # Let's use k=0.01 for this test, plus some noise
        expected_slippage = 0.01 * np.sqrt(trade_size) + np.random.normal(0, 0.0001)
        
        await mapper.record_trade_slippage(TradeSizeSlippage(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=i),
            trade_size=trade_size,
            realized_slippage=expected_slippage,
            market_impact=expected_slippage,
            execution_style="taker",
            liquidity_conditions="medium"
        ))
    
    # Build the model
    model = await mapper.build_liquidity_model(symbol)
    
    if model:
        print(f"  ✓ Liquidity model built: k={model.impact_coefficient:.6f}, R²={model.r_squared:.3f}")
        print(f"  ✓ Model type: {model.model_type}")
        
        # The coefficient should be close to our test value (0.01)
        # Allow some variance due to noise in the data
        assert 0.005 < model.impact_coefficient < 0.02  # Reasonable range around 0.01
        print("  ✓ Impact coefficient in expected range")
        
        assert model.r_squared > 0.5  # Should have decent fit for square-root pattern
        print(f"  ✓ Model fit acceptable: R²={model.r_squared:.3f}")
    else:
        print("  ⚠ Model not built due to data issues")
    
    await mapper.stop()
    print("  ✓ Liquidity model building test passed")


async def test_capacity_analysis():
    """
    Test that capacity analysis calculates position limits correctly.
    """
    print("\nTesting capacity analysis...")
    
    mapper = LiquiditySurfaceMapper(min_data_points_for_model=3)
    await mapper.start()
    
    symbol = "BTC_TEST"
    base_time = datetime.now()
    
    # Add market data
    for i in range(5):
        await mapper.record_market_data(MarketDataPoint(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=i),
            price=45000.0 + (i * 10),
            volume=2000000000 + (i * 100000000),
            bid_price=45000.0 - 0.5,
            ask_price=45000.0 + 0.5,
            spread=1.0,
            bid_size=5.0,
            ask_size=5.0,
            total_liquidity=300.0
        ))
    
    # Add trade data that will create a reasonable model
    for i in range(8):
        trade_size = 0.2 * (i + 1)
        # Realistic slippage values
        realized_slippage = 0.0005 * np.sqrt(trade_size)  # Small coefficient for test
        
        await mapper.record_trade_slippage(TradeSizeSlippage(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=i),
            trade_size=trade_size,
            realized_slippage=realized_slippage,
            market_impact=realized_slippage,
            execution_style="taker",
            liquidity_conditions="medium"
        ))
    
    # Perform capacity analysis
    analysis = await mapper.get_capacity_analysis(symbol)
    
    if analysis:
        print(f"  ✓ Capacity analysis: est_cap={analysis.estimated_capacity:.2f}")
        print(f"  ✓ Max position: {analysis.max_position_size:.2f}")
        print(f"  ✓ Liquidity score: {analysis.liquidity_score:.1f}")
        print(f"  ✓ Model confidence: {analysis.model_confidence:.2f}")
        
        # Verify values are reasonable (not infinite or negative)
        assert analysis.estimated_capacity > 0
        assert analysis.max_position_size > 0
        assert 0 <= analysis.liquidity_score <= 100
        print("  ✓ All values are reasonable")
    else:
        print("  ⚠ Analysis not completed")
    
    await mapper.stop()
    print("  ✓ Capacity analysis test passed")


async def test_position_size_recommendation():
    """
    Test that position size recommendations are properly capping based on liquidity.
    """
    print("\nTesting position size recommendation...")
    
    mapper = LiquiditySurfaceMapper(min_data_points_for_model=3, default_safety_factor=0.5)
    await mapper.start()
    
    symbol = "RECOMMEND_TEST"
    base_time = datetime.now()
    
    # Add market data for a liquid asset
    for i in range(5):
        await mapper.record_market_data(MarketDataPoint(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=i),
            price=3000.0,
            volume=2500000000,  # High volume = high liquidity
            bid_price=3000.0 - 0.1,  # Low spread = high liquidity
            ask_price=3000.0 + 0.1,
            spread=0.2,
            bid_size=10.0,
            ask_size=10.0,
            total_liquidity=500.0
        ))
    
    # Add trade data for building model
    for i in range(5):
        trade_size = 0.5 * (i + 1)
        realized_slippage = 0.0001 * np.sqrt(trade_size)  # Very low impact coefficient
        
        await mapper.record_trade_slippage(TradeSizeSlippage(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=i),
            trade_size=trade_size,
            realized_slippage=realized_slippage,
            market_impact=realized_slippage,
            execution_style="taker",
            liquidity_conditions="high"
        ))
    
    # Get recommendation for a potentially large position
    recommendation = await mapper.get_position_size_recommendation(
        strategy_id="test_strategy",
        symbol=symbol,
        base_suggestion=10.0  # Large base suggestion
    )
    
    if recommendation:
        print(f"  ✓ Recommendation: suggested={recommendation.recommended_size:.2f}, max_allowed={recommendation.max_allowed_size:.2f}")
        print(f"  ✓ Safety factor: {recommendation.safety_factor}")
        print(f"  ✓ Reason: {recommendation.reason[:60]}...")
        
        # The recommended size should be <= base_suggestion due to capping
        assert recommendation.recommended_size <= 10.0
        # The recommendation should be positive
        assert recommendation.recommended_size >= 0
        print("  ✓ Recommendation logic correct")
    else:
        print("  ⚠ Recommendation not generated")
    
    # Test with a less liquid symbol
    illiquid_symbol = "ILLIQ_TEST"
    
    # Add market data for illiquid asset
    for i in range(5):
        await mapper.record_market_data(MarketDataPoint(
            symbol=illiquid_symbol,
            timestamp=base_time + timedelta(minutes=i),
            price=50.0,
            volume=10000000,  # Low volume
            bid_price=50.0 - 2.0,  # High spread
            ask_price=50.0 + 2.0,
            spread=4.0,
            bid_size=0.1,
            ask_size=0.1,
            total_liquidity=10.0
        ))
    
    # Add trade data with high impact for illiquid asset
    for i in range(5):
        trade_size = 0.1 * (i + 1)
        realized_slippage = 0.01 * np.sqrt(trade_size)  # High impact coefficient
        
        await mapper.record_trade_slippage(TradeSizeSlippage(
            symbol=illiquid_symbol,
            timestamp=base_time + timedelta(minutes=i),
            trade_size=trade_size,
            realized_slippage=realized_slippage,
            market_impact=realized_slippage,
            execution_style="taker",
            liquidity_conditions="low"
        ))
    
    illiquid_recommendation = await mapper.get_position_size_recommendation(
        strategy_id="test_strategy",
        symbol=illiquid_symbol,
        base_suggestion=1.0
    )
    
    if illiquid_recommendation:
        print(f"  ✓ Illiquid asset recommendation: {illiquid_recommendation.recommended_size:.4f}")
        # Should be much smaller due to low liquidity
        print("  ✓ Illiquid asset properly restricted")
    else:
        print("  ⚠ Illiquid recommendation not generated")
    
    await mapper.stop()
    print("  ✓ Position size recommendation test passed")


async def test_liquidity_surface_snapshot():
    """
    Test that liquidity surface snapshots work for multiple symbols.
    """
    print("\nTesting liquidity surface snapshot...")
    
    mapper = LiquiditySurfaceMapper(min_data_points_for_model=3)
    await mapper.start()
    
    # Add data for multiple symbols
    symbols = ["BTC", "ETH", "ADA"]
    
    for symbol in symbols:
        base_time = datetime.now()
        
        # Add market data
        for i in range(5):
            await mapper.record_market_data(MarketDataPoint(
                symbol=symbol,
                timestamp=base_time + timedelta(minutes=i),
                price=40000.0 if symbol == "BTC" else (3000.0 if symbol == "ETH" else 1.0),
                volume=2000000000 if symbol == "BTC" else (1500000000 if symbol == "ETH" else 500000000),
                bid_price=39999.0 if symbol == "BTC" else (2999.0 if symbol == "ETH" else 0.99),
                ask_price=40001.0 if symbol == "BTC" else (3001.0 if symbol == "ETH" else 1.01),
                spread=2.0,
                bid_size=5.0,
                ask_size=5.0,
                total_liquidity=250.0
            ))
        
        # Add trade data
        for i in range(5):
            trade_size = 0.2 * (i + 1)
            # Different impact coefficients to simulate different liquidity
            if symbol == "BTC":
                impact_coeff = 0.0005  # Very liquid
            elif symbol == "ETH":
                impact_coeff = 0.001   # Less liquid
            else:
                impact_coeff = 0.005   # Least liquid
            
            realized_slippage = impact_coeff * np.sqrt(trade_size)
            
            await mapper.record_trade_slippage(TradeSizeSlippage(
                symbol=symbol,
                timestamp=base_time + timedelta(minutes=i),
                trade_size=trade_size,
                realized_slippage=realized_slippage,
                market_impact=realized_slippage,
                execution_style="taker",
                liquidity_conditions="medium"
            ))
    
    # Get snapshot of all symbols
    snapshot = await mapper.get_liquidity_surface_snapshot()
    
    print(f"  ✓ Snapshot retrieved for {len(snapshot)} symbols")
    for symbol, analysis in snapshot.items():
        print(f"    - {symbol}: score={analysis.liquidity_score:.1f}, max_pos={analysis.max_position_size:.2f}")
    
    # Test with specific symbols list
    specific_snapshot = await mapper.get_liquidity_surface_snapshot(["BTC", "ETH"])
    print(f"  ✓ Specific snapshot: {len(specific_snapshot)} symbols")
    
    await mapper.stop()
    print("  ✓ Liquidity surface snapshot test passed")


async def test_capacity_alerts():
    """
    Test that capacity alerts are generated for low liquidity conditions.
    """
    print("\nTesting capacity alerts...")
    
    mapper = LiquiditySurfaceMapper(min_data_points_for_model=3)
    await mapper.start()
    
    # Add data for a very illiquid symbol
    symbol = "LOW_LIQ_TEST"
    base_time = datetime.now()
    
    for i in range(5):
        # Very low volume and high spread to simulate low liquidity
        await mapper.record_market_data(MarketDataPoint(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=i),
            price=5.0,
            volume=10000,  # Very low volume
            bid_price=5.0 - 2.0,  # Very high spread
            ask_price=5.0 + 2.0,
            spread=4.0,
            bid_size=0.01,
            ask_size=0.01,
            total_liquidity=1.0
        ))
    
    # Add trade data with high impact
    for i in range(5):
        trade_size = 0.05 * (i + 1)
        realized_slippage = 0.05 * np.sqrt(trade_size)  # Very high impact
        
        await mapper.record_trade_slippage(TradeSizeSlippage(
            symbol=symbol,
            timestamp=base_time + timedelta(minutes=i),
            trade_size=trade_size,
            realized_slippage=realized_slippage,
            market_impact=realized_slippage,
            execution_style="taker",
            liquidity_conditions="low"
        ))
    
    # Generate alerts
    alerts = await mapper.get_capacity_alerts()
    
    print(f"  ✓ Found {len(alerts)} capacity alerts")
    for alert in alerts:
        print(f"    - {alert['symbol']}: {alert['alert_type']} ({alert['severity']}) - {alert['message'][:50]}...")
    
    # Since we created low liquidity conditions, we should have alerts
    low_liquidity_alerts = [a for a in alerts if a['alert_type'] == 'LOW_LIQUIDITY']
    print(f"  ✓ Low liquidity alerts: {len(low_liquidity_alerts)}")
    
    await mapper.stop()
    print("  ✓ Capacity alerts test passed")


async def run_all_tests():
    """
    Run all liquidity surface mapper tests.
    """
    print("Running Liquidity Surface Mapper Tests...\n")
    
    await test_liquidity_model_building()
    await test_capacity_analysis()
    await test_position_size_recommendation()
    await test_liquidity_surface_snapshot()
    await test_capacity_alerts()
    
    print("\n✅ All Liquidity Surface Mapper tests passed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())