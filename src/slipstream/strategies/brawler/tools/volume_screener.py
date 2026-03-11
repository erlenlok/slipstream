"""
Volume screening logic for comparing Hyperliquid vs Binance volumes.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import httpx

from slipstream.strategies.brawler.config import BrawlerConfig


@dataclass
class VolumeRow:
    symbol: str
    cex_symbol: str
    hl_volume: float
    binance_volume: float
    ratio: float
    relative_ratio: Optional[float]
    qualifies: bool
    notes: List[str]


async def fetch_hl_volumes(endpoint: str) -> Dict[str, float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(endpoint, json={"type": "metaAndAssetCtxs"})
        resp.raise_for_status()
        payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected metaAndAssetCtxs payload: {payload!r}")
    universe = payload[0].get("universe") or []
    ctxs = payload[1]
    volumes: Dict[str, float] = {}
    for entry, ctx in zip(universe, ctxs):
        if not isinstance(entry, Mapping) or not isinstance(ctx, Mapping):
            continue
        name = str(entry.get("name") or "").upper()
        if not name:
            continue
        raw = ctx.get("dayNtlVlm") or 0.0
        try:
            vol = float(raw)
        except (TypeError, ValueError):
            vol = 0.0
        volumes[name] = max(vol, 0.0)
    return volumes


async def fetch_binance_volumes(endpoint: str) -> Dict[str, float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(endpoint)
        resp.raise_for_status()
        data = resp.json()
    volumes: Dict[str, float] = {}
    if isinstance(data, list):
        source = data
    else:
        source = [data]
    for entry in source:
        if not isinstance(entry, Mapping):
            continue
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            continue
        raw = entry.get("quoteVolume") or entry.get("volume") or 0.0
        try:
            vol = float(raw)
        except (TypeError, ValueError):
            vol = 0.0
        volumes[symbol] = max(vol, 0.0)
    return volumes


def compute_rows(
    config: BrawlerConfig,
    symbols: List[str],
    hl_volumes: Mapping[str, float],
    binance_volumes: Mapping[str, float],
    baseline_symbols: Sequence[str],
    ratio_threshold: float,
) -> List[VolumeRow]:
    rows: List[VolumeRow] = []
    baseline_ratios: List[float] = []

    for sym in symbols:
        asset = config.assets[sym]
        hl_symbol = asset.symbol.upper()
        binance_symbol = asset.cex_symbol.upper()
        hl_vol = hl_volumes.get(hl_symbol, 0.0)
        bin_vol = binance_volumes.get(binance_symbol, 0.0)
        notes: List[str] = []
        if hl_vol <= 0:
            notes.append("No HL volume")
        if bin_vol <= 0:
            notes.append("No Binance volume")
        ratio = hl_vol / bin_vol if bin_vol > 0 else 0.0
        row = VolumeRow(
            symbol=hl_symbol,
            cex_symbol=binance_symbol,
            hl_volume=hl_vol,
            binance_volume=bin_vol,
            ratio=ratio,
            relative_ratio=None,
            qualifies=False,
            notes=notes,
        )
        rows.append(row)
        if hl_symbol in (name.upper() for name in baseline_symbols) and ratio > 0:
            baseline_ratios.append(ratio)

    baseline = statistics.median(baseline_ratios) if baseline_ratios else None

    for row in rows:
        if baseline and baseline > 0:
            row.relative_ratio = row.ratio / baseline
            row.qualifies = row.relative_ratio <= ratio_threshold and row.ratio > 0
            if row.relative_ratio is not None and row.relative_ratio > ratio_threshold:
                row.notes.append(f"Relative ratio {row.relative_ratio:.2f} above threshold")
        else:
            row.notes.append("No baseline ratio available")
            row.qualifies = False
    return rows


def render_table(rows: List[VolumeRow], limit: Optional[int] = None) -> str:
    header = f"{'Symbol':<8} {'CEX':<10} {'HL Vol ($)':>14} {'Bin Vol ($)':>14} {'Ratio':>8} {'RelRatio':>9} {'Good?':>6} Notes"
    lines = [header, "-" * len(header)]
    sorted_rows = sorted(rows, key=lambda r: (r.relative_ratio or float("inf")))
    for idx, row in enumerate(sorted_rows):
        if limit is not None and idx >= limit:
            break
        lines.append(
            f"{row.symbol:<8} {row.cex_symbol:<10} {row.hl_volume:14.2f} {row.binance_volume:14.2f} "
            f"{row.ratio:8.4f} {row.relative_ratio or 0:9.4f} {str(row.qualifies):>6} {'; '.join(row.notes)}"
        )
    return "\n".join(lines)


def save_csv(rows: List[VolumeRow], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["symbol", "cex_symbol", "hl_volume", "binance_volume", "ratio", "relative_ratio", "qualifies", "notes"]
        )
        for row in rows:
            writer.writerow(
                [
                    row.symbol,
                    row.cex_symbol,
                    f"{row.hl_volume:.6f}",
                    f"{row.binance_volume:.6f}",
                    f"{row.ratio:.6f}",
                    "" if row.relative_ratio is None else f"{row.relative_ratio:.6f}",
                    row.qualifies,
                    "; ".join(row.notes),
                ]
            )


def save_json(rows: List[VolumeRow], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "symbol": row.symbol,
            "cex_symbol": row.cex_symbol,
            "hl_volume": row.hl_volume,
            "binance_volume": row.binance_volume,
            "ratio": row.ratio,
            "relative_ratio": row.relative_ratio,
            "qualifies": row.qualifies,
            "notes": row.notes,
        }
        for row in rows
    ]
    path.write_text(json.dumps(payload, indent=2))
