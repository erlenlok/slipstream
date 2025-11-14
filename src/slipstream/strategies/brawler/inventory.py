"""Inventory bootstrap helpers for the Brawler engine."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Iterable, Protocol


class InventoryProvider(Protocol):
    """Protocol describing sources that can fetch initial per-asset inventory."""

    async def fetch(self, symbols: Iterable[str]) -> Dict[str, float]:
        ...


class FileInventoryProvider:
    """Reads initial inventory seeds from a JSON file: {\"BTC\": 0.1, ...}."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    async def fetch(self, symbols: Iterable[str]) -> Dict[str, float]:
        symbol_set = {symbol.upper() for symbol in symbols}
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(None, self._read_payload)
        results: Dict[str, float] = {}
        for symbol, value in payload.items():
            if symbol.upper() in symbol_set:
                try:
                    results[symbol] = float(value)
                except (TypeError, ValueError):
                    continue
        return results

    def _read_payload(self) -> Dict[str, float]:
        if not self.path.exists():
            return {}
        text = self.path.read_text().strip()
        if not text:
            return {}
        data = json.loads(text)
        if not isinstance(data, dict):
            return {}
        return data


class DictInventoryProvider:
    """In-memory provider useful for tests."""

    def __init__(self, mapping: Dict[str, float]) -> None:
        self.mapping = mapping

    async def fetch(self, symbols: Iterable[str]) -> Dict[str, float]:
        symbol_set = {symbol.upper() for symbol in symbols}
        return {
            symbol: qty
            for symbol, qty in self.mapping.items()
            if symbol.upper() in symbol_set
        }


__all__ = ["InventoryProvider", "FileInventoryProvider", "DictInventoryProvider"]
