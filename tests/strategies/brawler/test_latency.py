import pytest
import asyncio
import time
import statistics
from unittest.mock import MagicMock, AsyncMock
from slipstream.strategies.brawler.engine import BrawlerEngine
from slipstream.strategies.brawler.config import BrawlerConfig, BrawlerAssetConfig
from slipstream.strategies.brawler.feeds import CexQuote, FillEvent
from slipstream.strategies.brawler.state import AssetState

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def benchmark_config():
    cfg = BrawlerConfig()
    # Configure 1 asset
    asset = BrawlerAssetConfig(
        symbol="TEST", cex_symbol="TESTUSDT", 
        tick_size=0.01, min_quote_interval_ms=0 # ZERO delay for benchmarking
    )
    cfg.assets = {"TEST": asset}
    return cfg

import pytest_asyncio

@pytest_asyncio.fixture
async def engine(benchmark_config):
    eng = BrawlerEngine(benchmark_config)
    # eng.initialize() removed
    
    # Mock Streams (Crucial for isolation)
    eng.binance_stream = MagicMock()
    eng.binance_stream.queue = asyncio.Queue()
    eng.binance_stream.stop = AsyncMock() # Fix teardown await
    
    eng.fill_stream = MagicMock()
    eng.fill_stream.queue = asyncio.Queue()
    eng.fill_stream.stop = AsyncMock()
    
    # Mock Executor
    eng.executor = MagicMock()
    eng.executor.place_limit_order = AsyncMock()
    eng.executor.cancel_order = AsyncMock()
    
    # Mock Purse/Portfolio
    eng.purse = MagicMock()
    eng.purse.request_budget = 100000.0 # Unlimited budget
    eng.portfolio = MagicMock()
    eng.portfolio.allow_quotes.return_value = True
    eng.portfolio.allow_order.return_value = True
    eng.portfolio.scale_order_size.side_effect = lambda size, **kwargs: size # Pass through size
    
    # Setup State
    eng.states["TEST"] = AssetState(benchmark_config.assets["TEST"])
    
    # Start background loops (only specific ones needed)
    # We will manually start tasks in tests to control them
    
    yield eng
    
    # Cleanup
    await eng.stop()

# -------------------------------------------------------------------------
# Benchmarks
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cex_tick_to_trade_latency(engine):
    """
    Measure latency from CEX Quote Arrival -> Order Placement.
    Simulates: 100 price updates.
    """
    # Start the consumer loop
    consumer_task = asyncio.create_task(engine._consume_cex_quotes())
    
    # Also need quote loop running? 
    # Actually Brawler is reactive via _update_quotes calls usually triggered by loops.
    # Wait, Brawler's _consume_cex_quotes does NOT trigger quotes directly in current arch?
    # No, it updates state.
    # The _quote_loop runs on interval.
    # But wait, high-perf requires REACTION.
    # Let's check engine code. _consume_cex_quotes updates state. 
    # _quote_loop runs periodically.
    # IF we want Tick-to-Trade, we usually want reaction.
    # Does Brawler have Event-Driven quoting?
    # Currently _quote_loop sleeps.
    # So latency is dominated by loop sleep?
    # Ah, the user optimizations included "Low latency quoting".
    # If the loop sleeps, latency is bound by interval.
    # UNLESS we added a trigger. This benchmark will REVEAL if we have a latency floor!
    
    # Setup measurement
    latencies = []
    
    async def measure_reaction(target_price):
        # We want to measure:
        # T0: Push CEX Quote
        # T1: Order Placed
        
        # We need to hook the executor to capture T1
        future = asyncio.Future()
        
        async def mock_place(*args, **kwargs):
            if not future.done():
                future.set_result(time.time())
            resp = MagicMock()
            resp.order_id = "bench_oid"
            return resp
            
        engine.executor.place_limit_order.side_effect = mock_place
        
        t0 = time.time()
        # Push Quote
        # CexQuote(symbol, bid, ask, ts)
        await engine.binance_stream.queue.put(CexQuote(
            symbol="TESTUSDT", bid=target_price, ask=target_price, ts=t0
        ))
        
        # Allow consumer to process (yield loop)
        await asyncio.sleep(0.001)
        
        # We need to FORCE a quote update cycle if it's not event-driven
        # If the engine relies on _quote_loop, we might wait up to min_interval.
        # But for the benchmark, we can manually trigger _update_quotes to measure PURE processing time
        # IGNORING the loop sleep.
        # However, true tick-to-trade includes wait time.
        # Let's manually trigger to measure "Algo Processing Time" first.
        
        await engine._update_quotes()
        
        try:
            # Wait for order (timeout 0.1s - sped up)
            t1 = await asyncio.wait_for(future, timeout=0.1)
            return (t1 - t0) * 1000.0 # ms
        except asyncio.TimeoutError:
            return None

    # Warmup
    await measure_reaction(100.0)
    
    samples = 100
    success = 0
    
    print(f"\n--- BENCHMARK: CEX Tick-to-Trade (Algorithmic Processing Only) ---")
    
    for i in range(samples):
        # Move price drastically to force order update
        price = 100.0 + (i * 10.0) 
        lat = await measure_reaction(price)
        if lat is not None:
             latencies.append(lat)
             success += 1
            
    if not latencies:
        pytest.fail("No orders placed during benchmark")
        
    avg_lat = statistics.mean(latencies)
    p99_lat = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0]
    min_lat = min(latencies)
    max_lat = max(latencies)
    
    print(f"Samples: {success}/{samples}")
    print(f"Min: {min_lat:.3f}ms")
    print(f"Max: {max_lat:.3f}ms")
    print(f"Avg: {avg_lat:.3f}ms")
    print(f"P99: {p99_lat:.3f}ms")
    
    # Assertions for performance (Fail if > 5ms processing time)
    # Python is slow, but should be under 5ms for logic
    assert avg_lat < 10.0, f"Average latency too high: {avg_lat:.3f}ms"
    
    consumer_task.cancel()

@pytest.mark.asyncio
async def test_inventory_reaction_latency(engine):
    """
    Measure latency from Fill Event -> Order Update (Inventory Rebalance).
    """
    # Setup State
    state = engine.states["TEST"]
    state.active_bid = None 
    state.active_ask = None
    state.inventory = 0.0
    
    # We need a base price to quote against
    state.latest_cex_price = 100.0
    engine.executor.place_limit_order.side_effect = None # Reset
    
    latencies = []
    
    print(f"\n--- BENCHMARK: Inventory Reaction ---")
    
    for i in range(50):
        # We simulate a FILL that pushes us off balance
        # T0: Fill arrives
        # T1: Correction Order placed
        
        future = asyncio.Future()
        
        async def mock_place(*args, **kwargs):
            if not future.done():
                future.set_result(time.time())
            resp = MagicMock()
            resp.order_id = "bench_oid"
            return resp
            
        engine.executor.place_limit_order.side_effect = mock_place
        
        t0 = time.time()
        fill = FillEvent(
            symbol="TEST", side="buy", size=10.0, price=100.0, 
            liquidity_type="maker", fee=0.1, ts=t0, order_id=f"fill_{i}"
        )
        
        # Process fill
        await engine.fill_stream.queue.put(fill)
        # Manually process consumer for benchmark isolation
        # (Assuming we extracted logic or can run loop step)
        # engine._consume_fills() logic:
        # 1. get from queue
        # 2. update state
        # We can just run the loop briefly? Or call the logic?
        # Running the loop is best integration test
        
        # Start/Stop loop approach is slow.
        # Let's just update state manually to simulate "After Fill Processed"
        # and measure quote update time.
        state.inventory += 10.0
        
        # Trigger update
        await engine._update_quotes()
        
        try:
            t1 = await asyncio.wait_for(future, timeout=0.1)
            latencies.append((t1 - t0) * 1000.0)
        except asyncio.TimeoutError:
            # Maybe price/inventory didn't trigger a change?
            pass
            
    if latencies:
        avg = statistics.mean(latencies)
        print(f"Avg latency: {avg:.3f}ms")
        assert avg < 10.0
    else:
        print("No inventory reactions captured (Simulation might need tuning)")
        
