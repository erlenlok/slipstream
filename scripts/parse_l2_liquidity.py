#!/usr/bin/env python3
"""
Parse L2 orderbook snapshots and compute liquidity metrics.

Extracts:
1. Spread (bid-ask spread as % of mid)
2. Orderbook depth at various levels
3. Slippage cost function for different trade sizes
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import argparse


def decompress_lz4(lz4_path: Path) -> List[dict]:
    """Decompress L2 file and return list of snapshots."""
    result = subprocess.run(
        ["lz4", "-d", "-c", str(lz4_path)],
        capture_output=True,
        check=True
    )
    
    # Parse JSONL (one JSON object per line)
    snapshots = []
    for line in result.stdout.decode('utf-8').strip().split('\n'):
        if line:
            snapshots.append(json.loads(line))
    
    return snapshots


def parse_orderbook(snapshot: dict) -> Tuple[List[dict], List[dict], float, str]:
    """
    Extract orderbook from snapshot.
    
    Returns:
        (bids, asks, timestamp_ms, coin)
        where bids/asks are lists of {px: float, sz: float}
    """
    raw_data = snapshot['raw']['data']
    coin = raw_data['coin']
    timestamp_ms = raw_data['time']
    
    levels = raw_data['levels']
    bids = [{'px': float(level['px']), 'sz': float(level['sz'])} for level in levels[0]]
    asks = [{'px': float(level['px']), 'sz': float(level['sz'])} for level in levels[1]]
    
    return bids, asks, timestamp_ms, coin


def compute_spread(bids: List[dict], asks: List[dict]) -> float:
    """Compute bid-ask spread as percentage of mid price."""
    if not bids or not asks:
        return np.nan
    
    best_bid = bids[0]['px']
    best_ask = asks[0]['px']
    mid = (best_bid + best_ask) / 2
    
    spread_bps = 10000 * (best_ask - best_bid) / mid
    return spread_bps


def compute_depth(bids: List[dict], asks: List[dict], bps_from_mid: int = 10) -> Dict[str, float]:
    """
    Compute orderbook depth within X bps of mid price.
    
    Returns dict with bid_depth, ask_depth, total_depth (in native units).
    """
    if not bids or not asks:
        return {'bid_depth': 0.0, 'ask_depth': 0.0, 'total_depth': 0.0}
    
    best_bid = bids[0]['px']
    best_ask = asks[0]['px']
    mid = (best_bid + best_ask) / 2
    
    # Define price range
    price_range = mid * bps_from_mid / 10000
    bid_threshold = mid - price_range
    ask_threshold = mid + price_range
    
    # Sum up liquidity within range
    bid_depth = sum(level['sz'] for level in bids if level['px'] >= bid_threshold)
    ask_depth = sum(level['sz'] for level in asks if level['px'] <= ask_threshold)
    
    return {
        'bid_depth': bid_depth,
        'ask_depth': ask_depth,
        'total_depth': bid_depth + ask_depth
    }


def compute_slippage(bids: List[dict], asks: List[dict], trade_size_usd: float, side: str = 'buy') -> float:
    """
    Compute slippage cost for a market order of given USD size.
    
    Args:
        bids/asks: Orderbook levels
        trade_size_usd: Trade size in USD
        side: 'buy' (execute against asks) or 'sell' (execute against bids)
    
    Returns:
        Slippage in basis points (10000 * slippage_cost / trade_size)
    """
    if not bids or not asks:
        return np.nan
    
    best_bid = bids[0]['px']
    best_ask = asks[0]['px']
    mid = (best_bid + best_ask) / 2
    
    # Choose side
    levels = asks if side == 'buy' else bids
    arrival_price = best_ask if side == 'buy' else best_bid
    
    # Walk through orderbook
    remaining_usd = trade_size_usd
    total_cost_usd = 0.0
    total_qty = 0.0
    
    for level in levels:
        px = level['px']
        sz = level['sz']
        
        # How much USD liquidity at this level?
        level_usd = px * sz
        
        if remaining_usd <= level_usd:
            # Partial fill at this level
            qty_filled = remaining_usd / px
            total_qty += qty_filled
            total_cost_usd += remaining_usd
            remaining_usd = 0
            break
        else:
            # Full fill at this level, move to next
            total_qty += sz
            total_cost_usd += level_usd
            remaining_usd -= level_usd
    
    if remaining_usd > 0:
        # Orderbook exhausted - return inf slippage
        return np.inf
    
    # Compute average execution price
    avg_price = total_cost_usd / total_qty
    
    # Slippage relative to mid price (in bps)
    slippage_bps = 10000 * abs(avg_price - mid) / mid
    
    return slippage_bps


def compute_liquidity_metrics(snapshot: dict) -> dict:
    """Extract all liquidity metrics from a single snapshot."""
    bids, asks, timestamp_ms, coin = parse_orderbook(snapshot)
    
    spread_bps = compute_spread(bids, asks)
    depth_10bps = compute_depth(bids, asks, bps_from_mid=10)
    depth_50bps = compute_depth(bids, asks, bps_from_mid=50)
    
    # Compute slippage for various trade sizes (in USD)
    trade_sizes_usd = [1000, 5000, 10000, 50000, 100000]
    slippage_buy = {f'slippage_buy_{size}usd': compute_slippage(bids, asks, size, 'buy') 
                    for size in trade_sizes_usd}
    slippage_sell = {f'slippage_sell_{size}usd': compute_slippage(bids, asks, size, 'sell') 
                     for size in trade_sizes_usd}
    
    return {
        'coin': coin,
        'timestamp_ms': timestamp_ms,
        'spread_bps': spread_bps,
        'depth_10bps_bid': depth_10bps['bid_depth'],
        'depth_10bps_ask': depth_10bps['ask_depth'],
        'depth_10bps_total': depth_10bps['total_depth'],
        'depth_50bps_bid': depth_50bps['bid_depth'],
        'depth_50bps_ask': depth_50bps['ask_depth'],
        'depth_50bps_total': depth_50bps['total_depth'],
        **slippage_buy,
        **slippage_sell,
    }


def process_all_snapshots(l2_dir: Path) -> pd.DataFrame:
    """Process all L2 snapshots and return liquidity metrics DataFrame."""
    all_metrics = []
    
    for coin_file in sorted(l2_dir.rglob("*.lz4")):
        try:
            snapshots = decompress_lz4(coin_file)
            print(f"Processing {coin_file.name}: {len(snapshots)} snapshots")
            
            for snapshot in snapshots:
                metrics = compute_liquidity_metrics(snapshot)
                all_metrics.append(metrics)
        
        except Exception as exc:
            print(f"  Warning: failed to process {coin_file.name}: {exc}")
    
    df = pd.DataFrame(all_metrics)
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
    df = df.sort_values(['coin', 'timestamp'])
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Parse L2 snapshots and compute liquidity metrics")
    parser.add_argument("--input", type=Path, default=Path("data/l2_snapshots"), help="L2 snapshots directory")
    parser.add_argument("--output", type=Path, default=Path("data/features/liquidity_metrics.csv"), help="Output CSV")
    args = parser.parse_args()
    
    print(f"Processing L2 snapshots from {args.input}")
    df = process_all_snapshots(args.input)
    
    print(f"\nComputed metrics for {len(df)} snapshots across {df['coin'].nunique()} coins")
    print(f"\nSample metrics:")
    print(df.head(10))
    
    print(f"\nSummary statistics:")
    print(df[['spread_bps', 'depth_10bps_total', 'slippage_buy_10000usd']].describe())
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
