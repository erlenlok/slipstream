"""State persistence helpers for Brawler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Protocol

from .state import AssetSnapshot


class StatePersistence(Protocol):
    """Protocol describing persistence backends for per-asset snapshots."""

    def load(self) -> Mapping[str, AssetSnapshot]:
        ...

    def save(self, snapshots: Mapping[str, AssetSnapshot]) -> None:
        ...


class FileStatePersistence:
    """JSON-based persistence for per-asset basis/inventory snapshots."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load(self) -> Mapping[str, AssetSnapshot]:
        if not self.path or not self.path.exists():
            return {}
        text = self.path.read_text().strip()
        if not text:
            return {}
        payload = json.loads(text)
        if not isinstance(payload, dict):
            return {}
        snapshots = {
            symbol: AssetSnapshot.from_mapping(symbol, data or {})
            for symbol, data in payload.items()
        }
        return snapshots

    def save(self, snapshots: Mapping[str, AssetSnapshot]) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {
            symbol: snapshot.as_dict()
            for symbol, snapshot in snapshots.items()
        }
        self.path.write_text(json.dumps(serialized, indent=2, sort_keys=True))


__all__ = ["StatePersistence", "FileStatePersistence"]
