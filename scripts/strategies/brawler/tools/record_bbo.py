#!/usr/bin/env python3
"""
Record Hyperliquid + Binance BBO (and optional HL depth) snapshots to disk for candidate scanning.

Example:
    uv run python scripts/strategies/brawler/record_bbo.py \
        --config config/brawler_single_asset.example.yml \
        --hl-pattern 'data/hl_bbo/{symbol}_{session}.csv' \
        --cex-pattern 'data/binance_bbo/{cex_symbol}_{session}.csv' \
        --depth-pattern 'data/hl_depth/{symbol}_{session}.csv' \
        --duration 3600
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import websockets

from slipstream.strategies.brawler.config import BrawlerConfig, load_brawler_config

logger = logging.getLogger("brawler.recorder")


def _timestamp_fields(ts: float) -> Tuple[float, str]:
    return ts, datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _ensure_writer(path: Path, headers: List[str]) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("w", newline="")
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    writer.file = file  # type: ignore[attr-defined]
    return writer


def _close_writer(writer: csv.DictWriter) -> None:
    writer.file.flush()  # type: ignore[attr-defined]
    writer.file.close()  # type: ignore[attr-defined]


def _compute_depth(
    bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], mid: float, depth_bps: float
) -> Tuple[float, float, float]:
    if mid <= 0 or depth_bps <= 0:
        return 0.0, 0.0, 0.0
    price_range = mid * depth_bps / 10_000.0
    bid_threshold = mid - price_range
    ask_threshold = mid + price_range
    bid_depth = sum(sz for px, sz in bids if px >= bid_threshold)
    ask_depth = sum(sz for px, sz in asks if px <= ask_threshold)
    return bid_depth, ask_depth, bid_depth + ask_depth


def _parse_orderbook_levels(levels) -> List[Tuple[float, float]]:
    parsed: List[Tuple[float, float]] = []
    for level in levels or []:
        if isinstance(level, dict):
            px = level.get("px")
            sz = level.get("sz")
        else:
            px = level[0] if len(level) > 0 else None
            sz = level[1] if len(level) > 1 else None
        try:
            px_f = float(px)
            sz_f = float(sz)
        except (TypeError, ValueError):
            continue
        if px_f <= 0 or sz_f <= 0:
            continue
        parsed.append((px_f, sz_f))
    return parsed


@dataclass
class RecorderConfig:
    symbols: List[str]
    cex_symbols: Dict[str, str]
    hl_ws_url: str
    binance_ws_url: str


def _resolve_symbols(brawler_config: BrawlerConfig, symbols: Optional[Iterable[str]]) -> RecorderConfig:
    if symbols:
        hl_symbols = []
        cex_map: Dict[str, str] = {}
        for sym in symbols:
            if sym not in brawler_config.assets:
                raise ValueError(f"Symbol '{sym}' not found in config assets.")
            asset = brawler_config.assets[sym]
            hl_symbols.append(asset.symbol)
            cex_map[asset.symbol] = asset.cex_symbol
    else:
        hl_symbols = list(brawler_config.assets.keys())
        cex_map = {cfg.symbol: cfg.cex_symbol for cfg in brawler_config.assets.values()}

    if not hl_symbols:
        raise ValueError("No symbols provided or found in config.")

    return RecorderConfig(
        symbols=hl_symbols,
        cex_symbols=cex_map,
        hl_ws_url=brawler_config.hyperliquid_ws_url,
        binance_ws_url=brawler_config.binance_ws_url,
    )


class HyperliquidRecorder:
    def __init__(
        self,
        recorder_cfg: RecorderConfig,
        *,
        hl_pattern: str,
        depth_pattern: Optional[str],
        depth_bps: float,
        session_id: str,
    ) -> None:
        self.cfg = recorder_cfg
        self.hl_pattern = hl_pattern
        self.depth_pattern = depth_pattern
        self.depth_bps = depth_bps
        self.session_id = session_id
        self._stop = asyncio.Event()
        self._writers: Dict[str, csv.DictWriter] = {}
        self._depth_writers: Dict[str, csv.DictWriter] = {}

    async def run(self) -> None:
        subscriptions = [{"type": "l2Book", "coin": symbol} for symbol in self.cfg.symbols]
        subscribe_msg = json.dumps({"type": "subscribe", "subscriptions": subscriptions})
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.cfg.hl_ws_url, ping_interval=20, ping_timeout=10) as ws:
                    await ws.send(subscribe_msg)
                    logger.info("Hyperliquid recorder subscribed (%d symbols)", len(self.cfg.symbols))
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        update = payload.get("data")
                        if not update:
                            continue
                        symbol = update.get("coin")
                        book = update.get("book")
                        ts = float(update.get("time") or time.time())
                        if not symbol or not book:
                            continue
                        bids = _parse_orderbook_levels(book.get("bids"))
                        asks = _parse_orderbook_levels(book.get("asks"))
                        if not bids or not asks:
                            continue
                        try:
                            bid_px, bid_sz = bids[0]
                            ask_px, ask_sz = asks[0]
                        except (ValueError, IndexError):
                            continue
                        mid = (bid_px + ask_px) / 2.0
                        spread = ask_px - bid_px
                        unix_ts, iso_ts = _timestamp_fields(ts / 1000 if ts > 1e12 else ts)
                        self._write_bbo(symbol, {
                            "timestamp": f"{unix_ts:.6f}",
                            "timestamp_iso": iso_ts,
                            "symbol": symbol,
                            "bid": f"{bid_px:.10f}",
                            "ask": f"{ask_px:.10f}",
                            "mid": f"{mid:.10f}",
                            "spread": f"{spread:.10f}",
                            "bid_size": f"{bid_sz:.10f}",
                            "ask_size": f"{ask_sz:.10f}",
                        })
                        if self.depth_pattern:
                            bid_depth, ask_depth, total_depth = _compute_depth(bids, asks, mid, self.depth_bps)
                            self._write_depth(symbol, {
                                "timestamp": f"{unix_ts:.6f}",
                                "timestamp_iso": iso_ts,
                                "symbol": symbol,
                                "bid_depth": f"{bid_depth:.10f}",
                                "ask_depth": f"{ask_depth:.10f}",
                                "total_depth": f"{total_depth:.10f}",
                            })
            except Exception as exc:
                logger.warning("Hyperliquid recorder error: %s", exc, exc_info=True)
                await asyncio.sleep(2.0)

    def stop(self) -> None:
        self._stop.set()
        for writer in self._writers.values():
            _close_writer(writer)
        for writer in self._depth_writers.values():
            _close_writer(writer)

    def _writer_path(self, pattern: str, symbol: str) -> Path:
        return Path(pattern.format(symbol=symbol, lower=symbol.lower(), session=self.session_id))

    def _write_bbo(self, symbol: str, row: Dict[str, str]) -> None:
        if symbol not in self._writers:
            path = self._writer_path(self.hl_pattern, symbol)
            self._writers[symbol] = _ensure_writer(
                path,
                ["timestamp", "timestamp_iso", "symbol", "bid", "ask", "mid", "spread", "bid_size", "ask_size"],
            )
        writer = self._writers[symbol]
        writer.writerow(row)
        writer.file.flush()  # type: ignore[attr-defined]

    def _write_depth(self, symbol: str, row: Dict[str, str]) -> None:
        if not self.depth_pattern:
            return
        if symbol not in self._depth_writers:
            path = self._writer_path(self.depth_pattern, symbol)
            self._depth_writers[symbol] = _ensure_writer(
                path,
                ["timestamp", "timestamp_iso", "symbol", "bid_depth", "ask_depth", "total_depth"],
            )
        writer = self._depth_writers[symbol]
        writer.writerow(row)
        writer.file.flush()  # type: ignore[attr-defined]


class BinanceRecorder:
    def __init__(
        self,
        recorder_cfg: RecorderConfig,
        *,
        cex_pattern: str,
        session_id: str,
    ) -> None:
        self.cfg = recorder_cfg
        self.cex_pattern = cex_pattern
        self.session_id = session_id
        self._stop = asyncio.Event()
        self._writers: Dict[str, csv.DictWriter] = {}

    async def run(self) -> None:
        streams = "/".join(f"{symbol.lower()}@bookTicker" for symbol in self.cfg.cex_symbols.values())
        target = f"{self.cfg.binance_ws_url.rstrip('/')}/stream?streams={streams}"
        while not self._stop.is_set():
            try:
                async with websockets.connect(target, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("Binance recorder subscribed (%d symbols)", len(self.cfg.cex_symbols))
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        data = payload.get("data") or {}
                        symbol = data.get("s")
                        if not symbol:
                            continue
                        try:
                            bid_px = float(data["b"])
                            ask_px = float(data["a"])
                            bid_sz = float(data.get("B") or 0.0)
                            ask_sz = float(data.get("A") or 0.0)
                        except (KeyError, TypeError, ValueError):
                            continue
                        mid = (bid_px + ask_px) / 2.0
                        spread = ask_px - bid_px
                        ts_ms = data.get("T") or data.get("E") or int(time.time() * 1000)
                        ts = float(ts_ms) / 1000.0
                        unix_ts, iso_ts = _timestamp_fields(ts)
                        self._write_row(symbol, {
                            "timestamp": f"{unix_ts:.6f}",
                            "timestamp_iso": iso_ts,
                            "symbol": symbol.upper(),
                            "bid": f"{bid_px:.10f}",
                            "ask": f"{ask_px:.10f}",
                            "mid": f"{mid:.10f}",
                            "spread": f"{spread:.10f}",
                            "bid_size": f"{bid_sz:.10f}",
                            "ask_size": f"{ask_sz:.10f}",
                        })
            except Exception as exc:
                logger.warning("Binance recorder error: %s", exc, exc_info=True)
                await asyncio.sleep(2.0)

    def stop(self) -> None:
        self._stop.set()
        for writer in self._writers.values():
            _close_writer(writer)

    def _writer_path(self, pattern: str, symbol: str) -> Path:
        return Path(pattern.format(symbol=symbol.upper(), cex_symbol=symbol.upper(), cex_lower=symbol.lower(), session=self.session_id))

    def _write_row(self, symbol: str, row: Dict[str, str]) -> None:
        upper_symbol = symbol.upper()
        if upper_symbol not in self._writers:
            path = self._writer_path(self.cex_pattern, symbol.lower())
            self._writers[upper_symbol] = _ensure_writer(
                path,
                ["timestamp", "timestamp_iso", "symbol", "bid", "ask", "mid", "spread", "bid_size", "ask_size"],
            )
        writer = self._writers[upper_symbol]
        writer.writerow(row)
        writer.file.flush()  # type: ignore[attr-defined]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record Hyperliquid + Binance BBO snapshots for candidate scanning.")
    parser.add_argument("--config", help="Brawler YAML/JSON config to pull symbols + URLs from.")
    parser.add_argument("--symbols", nargs="+", help="Subset of Hyperliquid symbols to record (defaults to every config asset).")
    parser.add_argument("--hl-pattern", default="data/hl_bbo/{symbol}_{session}.csv", help="Output pattern for Hyperliquid BBO CSVs.")
    parser.add_argument("--cex-pattern", default="data/binance_bbo/{cex_symbol}_{session}.csv", help="Output pattern for Binance BBO CSVs.")
    parser.add_argument("--depth-pattern", default="data/hl_depth/{symbol}_{session}.csv", help="Output pattern for Hyperliquid depth CSVs.")
    parser.add_argument("--depth-bps", type=float, default=10.0, help="Depth window in bps around mid when aggregating HL depth.")
    parser.add_argument("--duration", type=int, default=0, help="Optional number of seconds to record before stopping (0 = infinite).")
    parser.add_argument("--log-level", default=os.getenv("BRAWLER_LOG_LEVEL", "INFO"), help="Logging level (default: INFO).")
    return parser


async def _run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_brawler_config(args.config)
    recorder_cfg = _resolve_symbols(config, args.symbols)
    session_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    hl_recorder = HyperliquidRecorder(
        recorder_cfg,
        hl_pattern=args.hl_pattern,
        depth_pattern=args.depth_pattern,
        depth_bps=args.depth_bps,
        session_id=session_id,
    )
    binance_recorder = BinanceRecorder(
        recorder_cfg,
        cex_pattern=args.cex_pattern,
        session_id=session_id,
    )

    stop_event = asyncio.Event()

    def _handle_signal(*_) -> None:
        stop_event.set()
        hl_recorder.stop()
        binance_recorder.stop()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    tasks = [
        asyncio.create_task(hl_recorder.run(), name="hl-recorder"),
        asyncio.create_task(binance_recorder.run(), name="binance-recorder"),
    ]

    if args.duration > 0:
        await asyncio.sleep(args.duration)
        _handle_signal()
    await stop_event.wait()
    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if isinstance(args.depth_pattern, str) and not args.depth_pattern.strip():
        args.depth_pattern = None
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":  # pragma: no cover
    main()
