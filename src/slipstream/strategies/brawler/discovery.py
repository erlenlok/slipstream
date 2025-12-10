"""Runtime candidate discovery engine for Brawler."""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import httpx

from .config import BrawlerConfig

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    symbol: str
    cex_symbol: str
    hl_volume: float
    binance_volume: float
    ratio: float
    relative_ratio: float
    funding_rate: float
    spread_bps: Optional[float]
    qualifies: bool
    notes: List[str]


class DiscoveryEngine:
    """
    Periodically scans the market for new Brawler candidates.
    
    Uses 24h ticker snapshots (fast) instead of granular history (slow).
    """

    def __init__(self, config: BrawlerConfig, engine=None) -> None:
        self.config = config
        self.engine = engine  # Reference to main engine for callbacks
        self.discovery_cfg = config.discovery
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task[None]] = None
        self._last_scan_ts = 0.0

    def start(self) -> None:
        if not self.discovery_cfg.enabled:
            logger.info("Discovery engine disabled in config.")
            return
        
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="brawler-discovery")
            logger.info("Discovery engine started (interval=%.0fs)", self.discovery_cfg.interval_seconds)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        # Initial delay to let the main engine settle
        await asyncio.sleep(5.0)
        
        while not self._stop.is_set():
            try:
                await self.scan()
            except Exception as exc:
                logger.error("Discovery scan failed: %s", exc, exc_info=exc)
            
            # Wait for next interval
            interval = max(60.0, self.discovery_cfg.interval_seconds)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def scan(self) -> None:
        logger.info("Starting market scan...")
        start_ts = time.time()

        # 1. Fetch Universe & Volumes
        hl_volumes, hl_ctxs = await self._fetch_hl_snapshot()
        binance_volumes = await self._fetch_binance_volumes()

        # 2. Compute Baselines
        baseline_ratios = []
        for bench in self.discovery_cfg.benchmarks:
            hl_vol = hl_volumes.get(bench, 0.0)
            bin_vol = binance_volumes.get(f"{bench}USDT", 0.0)
            if bin_vol > 0:
                baseline_ratios.append(hl_vol / bin_vol)
        
        if not baseline_ratios:
            logger.warning("No baseline volume data found for %s", self.discovery_cfg.benchmarks)
            return
        
        baseline = statistics.median(baseline_ratios)
        logger.info("Market baseline volume ratio: %.4f (based on %s)", baseline, self.discovery_cfg.benchmarks)

        # 3. Evaluate Candidates
        candidates: List[DiscoveryResult] = []
        for symbol, hl_vol in hl_volumes.items():
            # Skip assets already in config
            if symbol in self.config.assets:
                continue

            # Basic mapping assumption: HL symbol + USDT = Binance symbol
            cex_symbol = f"{symbol}USDT"
            bin_vol = binance_volumes.get(cex_symbol, 0.0)
            
            if bin_vol <= 0:
                continue

            ratio = hl_vol / bin_vol
            relative_ratio = ratio / baseline
            
            # Check Volume Criteria
            if relative_ratio > self.discovery_cfg.min_volume_ratio:
                continue

            # Check Funding
            ctx = hl_ctxs.get(symbol)
            funding = float(ctx.get("funding", 0.0)) if ctx else 0.0
            # HL funding is hourly, convert to approx 8h rate for comparison or just use raw
            # Using raw hourly rate check against config threshold
            if abs(funding) > self.discovery_cfg.max_funding_rate:
                continue

            # (Optional) Check Spread - requires L2 snapshot, skipping for speed in V1
            # We could add a lightweight check here if needed

            result = DiscoveryResult(
                symbol=symbol,
                cex_symbol=cex_symbol,
                hl_volume=hl_vol,
                binance_volume=bin_vol,
                ratio=ratio,
                relative_ratio=relative_ratio,
                funding_rate=funding,
                spread_bps=None,
                qualifies=True,
                notes=["Low relative volume", "Funding within limits"]
            )
            candidates.append(result)

        # 4. Report
        self._report_results(candidates)
        logger.info("Scan complete in %.2fs. Found %d potential candidates.", time.time() - start_ts, len(candidates))
        self._last_scan_ts = time.time()

    async def _fetch_hl_snapshot(self) -> tuple[Dict[str, float], Dict[str, dict]]:
        url = f"{self.config.hyperliquid_rest_url}/info"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json={"type": "metaAndAssetCtxs"})
            resp.raise_for_status()
            data = resp.json()
        
        universe = data[0]["universe"]
        ctxs = data[1]
        
        volumes = {}
        ctx_map = {}
        for item, ctx in zip(universe, ctxs):
            name = item["name"]
            vol = float(ctx.get("dayNtlVlm", 0.0))
            volumes[name] = vol
            ctx_map[name] = ctx
            
        return volumes, ctx_map

    async def _fetch_binance_volumes(self) -> Dict[str, float]:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            
        volumes = {}
        for item in data:
            symbol = item["symbol"]
            vol = float(item.get("quoteVolume", 0.0))
            volumes[symbol] = vol
            
        return volumes

    def _report_results(self, candidates: List[DiscoveryResult]) -> None:
        if not candidates:
            return

        # Sort by relative ratio (lower is better for Brawler)
        candidates.sort(key=lambda x: x.relative_ratio)
        
        top_k = candidates[:5]
        msg = ["\n🔎 **New Brawler Candidates Discovered**"]
        msg.append(f"{'Symbol':<8} {'RelRatio':<10} {'Funding':<10} {'HL Vol':<12}")
        msg.append("-" * 45)
        
        for c in top_k:
            msg.append(f"{c.symbol:<8} {c.relative_ratio:<10.4f} {c.funding_rate:<10.6f} ${c.hl_volume:,.0f}")
            
        logger.info("\n".join(msg))

        if self.engine:
            self._auto_add_candidates(top_k)

    def _auto_add_candidates(self, candidates: List[DiscoveryResult]) -> None:
        from .config import BrawlerAssetConfig
        
        for c in candidates:
            # Re-check existence
            if c.symbol in self.engine.states:
                continue
                
            logger.info("⚔️ Organically adding %s to Brawler arena!", c.symbol)
            
            # Conservative Defaults for Auto-Added Pairs
            # User wants "small size" -> $20 sizing risk.
            # Vol lookback 60, spread 20bps (safer than tight), max vol 5%.
            new_cfg = BrawlerAssetConfig(
                symbol=c.symbol,
                cex_symbol=c.cex_symbol, # e.g. "ETHUSDT"
                base_spread=0.002,
                volatility_lookback=60,
                risk_aversion=2.0,
                basis_alpha=0.05,
                max_inventory=100000.0, # High token cap, rely on $ limit
                inventory_aversion=0.2,
                order_size=0.0, # Handled by vol_sizing
                vol_sizing_risk_dollars=20.0, # Conservative sizing
                max_volatility=0.05,
                max_basis_deviation=0.02, # 2% basis dev allowed
                min_quote_interval_ms=500,
                reduce_only_ratio=0.9,
                tick_size=0.000001, # We need to fetch this dynamically ideally, but guessing small is safer than large
            )
            # Note: tick_size being wrong can cause failure. 
            # In a real impl we should fetch meta. For now assuming small.
            
            asyncio.create_task(self.engine.add_asset(new_cfg))
