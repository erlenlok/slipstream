import asyncio
import aiohttp
import yaml
import statistics
import math

# Criteria
# Criteria
MIN_VOL_24H = 500_000     # $500k Min (Rinkydink)
MAX_VOL_24H = 5_000_000   # $5M Cap (Shittier shitcoins)
MAX_ASSETS = 10
EXCLUDE_SYMBOLS = {"BTC", "ETH", "SOL", "HYPE", "PURR", "USDC"} # Exclude L1s
ORDER_SIZE_USD = 100.0
MAX_INV_USD = 200.0

async def main():
    print("Fetching Hyperliquid metadata...")
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.hyperliquid.xyz/info", json={"type": "metaAndAssetCtxs"}) as resp:
            data = await resp.json()
    
    universe = data[0]["universe"]
    ctxs = data[1]
    
    candidates = []
    
    print(f"Scanning {len(universe)} assets...")
    
    for i, asset in enumerate(universe):
        symbol = asset["name"]
        ctx = ctxs[i]
        
        # skip spot? universe is perps only in this response usually, but let's be sure
        # "universe" is perps.
        
        if symbol in EXCLUDE_SYMBOLS:
            continue
            
        vol_24h = float(ctx.get("dayNtlVlm", 0.0))
        price = float(ctx.get("markPx", 0.0))
        
        if vol_24h < MIN_VOL_24H: # Changed from MIN_VOL
            continue
            
        if vol_24h > MAX_VOL_24H: # Cap to avoid majors
            continue

        if price <= 0:
            continue

        candidates.append({
            "symbol": symbol,
            "volume": vol_24h,
            "price": price,
            "ctx": ctx,
            "szDecimals": asset.get("szDecimals", 5)
        })
        
    # Sort by Volume Descending
    candidates.sort(key=lambda x: x["volume"], reverse=True)
    
    selected = candidates[:MAX_ASSETS]
    
    print(f"\nSelected Top {len(selected)} Candidates:")
    print(f"{'Symbol':<10} {'Price':<10} {'Volume 24h (M)':<15}")
    print("-" * 40)
    
    assets_config = {}
    
    for c in selected:
        sym = c["symbol"]
        price = c["price"]
        vol_m = c["volume"] / 1_000_000
        # Determine decimals from metadata if available (it is in universe list)
        # We need to find the matching universe item from earlier loop
        # But we only stored 'ctx'. We need 'szDecimals' from 'universe' item.
        # Let's fix the loop above to store it.
        sz_decimals = c.get("szDecimals", 5) # Default to 5 if missing?
        
        print(f"{sym:<10} ${price:<9.4f} ${vol_m:<14.2f}M (Decimals: {sz_decimals})")
        
        # Calculate Sizes
        raw_size = ORDER_SIZE_USD / price
        raw_max = MAX_INV_USD / price
        
        # Round 
        if sz_decimals == 0:
             size_tokens = int(raw_size)
             max_inv_tokens = int(raw_max)
        else:
             size_tokens = float(f"{raw_size:.{sz_decimals}f}")
             max_inv_tokens = float(f"{raw_max:.{sz_decimals}f}")

        # Safety: Ensure > 0
        if size_tokens <= 0:
             if sz_decimals == 0:
                 size_tokens = 1
             else:
                 size_tokens = float(f"{10**(-sz_decimals):.{sz_decimals}f}")
        
        # Calculate dynamic tick size for 5 sig figs
        # tick_size = 10^(floor(log10(price)) - 4)
        if price > 0:
            exponent = math.floor(math.log10(price)) - 4
            tick_size = float(f"{10**exponent:.10f}") 
            if tick_size < 1e-6: tick_size = 1e-6 # Clamp to 1e-6 for safety
        else:
            tick_size = 1e-5
            
        assets_config[sym] = {
            "symbol": sym,
            "cex_symbol": f"{sym}USDT", 
            "order_size": size_tokens,
            "max_inventory": max_inv_tokens,
            
            # Standard "Shitcoin" params suited for high vol
            "base_spread": 0.01,       # 100bps to avoid Post-Only rejections on thin books
            "risk_aversion": 2.0,      
            "inventory_aversion": 0.5, 
            "volatility_lookback": 60,
            "max_volatility": 0.10,    
            "min_quote_interval_ms": 50, 
            "reduce_only_ratio": 0.95,
            "tick_size": tick_size, 
            "quote_reprice_tolerance_ticks": 2.0,
            "vol_spread_multiplier": 5.0  # Reduced multiplier as base spread is wider
        }

    # Generate Full Config
    config = {
        "assets": assets_config,
        "risk": {
            "tick_interval_ms": 50,
            "inventory_check_interval": 1.0,
            "metrics_flush_seconds": 60.0
        },
        "economics": {
            "reload_threshold_budget": 500.0,
            "reload_target_budget": 5000.0,
            "reload_symbol": "BTC", 
            "max_spread_bps": 500.0
        },
        "discovery": {
            "enabled": True, # Keep enabled to find more? User said "Trade ten shitcoins". Maybe disable to stick to 10?
            # User said "I want it to trade ten". Let's DISABLE discovery to respect the strict set.
            "enabled": False
        }
    }
    
    with open("prod_brawler.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)
        
    print("\nGenerated 'prod_brawler.yaml'")

if __name__ == "__main__":
    asyncio.run(main())
