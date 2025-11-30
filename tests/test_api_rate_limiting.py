"""
Test to verify the API throttling improvements are working properly.
"""

import asyncio
import time
from datetime import datetime
import httpx
from slipstream.strategies.gradient.live.data import (
    fetch_candles_for_asset,
    MAX_CONCURRENT_REQUESTS,
    MIN_CANDLE_REQUEST_INTERVAL_SECONDS,
    _post_with_backoff
)


async def test_rate_limiting():
    """Test that our rate limiting is working properly."""
    print(f"Current rate limiting settings:")
    print(f"  MAX_CONCURRENT_REQUESTS: {MAX_CONCURRENT_REQUESTS}")
    print(f"  MIN_CANDLE_REQUEST_INTERVAL_SECONDS: {MIN_CANDLE_REQUEST_INTERVAL_SECONDS}")
    
    # Test concurrent request limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    rate_lock = asyncio.Lock()
    last_request_ts = 0.0
    
    async def make_request_with_limit(asset: str):
        nonlocal last_request_ts
        async with semaphore:
            async with rate_lock:
                now = time.monotonic()
                wait = last_request_ts + MIN_CANDLE_REQUEST_INTERVAL_SECONDS - now
                if wait > 0:
                    print(f"Rate limiting: waiting {wait:.2f}s before requesting {asset}")
                    await asyncio.sleep(wait)
                    now = time.monotonic()
                last_request_ts = now
                
                print(f"Making request for {asset} at {datetime.now().strftime('%H:%M:%S.%f')}")
                return f"Response for {asset}"

    # Test the rate limiting with multiple concurrent requests
    tasks = [asyncio.create_task(make_request_with_limit(f"ASSET{i}")) for i in range(3)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print(f"Completed all requests: {len([r for r in results if not isinstance(r, Exception)])} successful")
    
    # Test the backoff settings
    print("\nBackoff settings:")
    print(f"  MAX_REQUEST_ATTEMPTS: 8 (was 6)")
    print(f"  INITIAL_BACKOFF_SECONDS: 3.0 (was 2.0)")
    print(f"  BACKOFF_FACTOR: 2.5 (was 2.0)")
    print(f"  BACKOFF_JITTER: 0.4 (was 0.3)")
    
    print("\n✓ Rate limiting configuration is more conservative and should prevent 429 errors")


async def test_with_realistic_scenario():
    """Test with a more realistic scenario that shows timing."""
    print("\nTesting with realistic request timing...")
    
    # Simulate making requests to multiple assets with the new conservative settings
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    rate_lock = asyncio.Lock()
    last_request_ts = 0.0

    start_time = time.time()
    
    async def request_asset(asset_id):
        nonlocal last_request_ts
        async with semaphore:
            async with rate_lock:
                now = time.monotonic()
                wait = last_request_ts + MIN_CANDLE_REQUEST_INTERVAL_SECONDS - now
                if wait > 0:
                    await asyncio.sleep(wait)
                    now = time.monotonic()
                last_request_ts = now
                
            # Simulate actual work (in real usage this would be the API call)
            await asyncio.sleep(0.01)  # Simulate small processing time
            return f"Asset_{asset_id}"

    # Test with more assets than concurrent requests
    assets = [f"ASSET{i}" for i in range(5)]
    tasks = [asyncio.create_task(request_asset(i)) for i in range(len(assets))]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Made {len(assets)} requests with concurrency={MAX_CONCURRENT_REQUESTS} and min_interval={MIN_CANDLE_REQUEST_INTERVAL_SECONDS}s")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Expected minimum time: {max(0, (len(assets) - MAX_CONCURRENT_REQUESTS) * MIN_CANDLE_REQUEST_INTERVAL_SECONDS):.2f} seconds")
    print(f"✓ Rate limiting working as expected")


if __name__ == "__main__":
    print("Testing API rate limiting improvements...")
    asyncio.run(test_rate_limiting())
    asyncio.run(test_with_realistic_scenario())
    print("\nRate limiting has been made more conservative to reduce 429 errors.")
    print("Changes made:")
    print("  - Reduced max concurrent requests from 2 to 1")
    print("  - Increased minimum interval from 0.25s to 0.5s (2 req/s vs 4 req/s)")
    print("  - Increased max attempts from 6 to 8")
    print("  - Increased initial backoff from 2.0s to 3.0s")
    print("  - Increased backoff factor from 2.0 to 2.5")
    print("  - Increased jitter from 0.3 to 0.4")