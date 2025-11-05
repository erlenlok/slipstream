"""
Layered configuration loader for Slipstream strategies.

This module provides helper utilities to resolve configuration files in the
following order (later layers override earlier ones):

1. Global defaults (e.g. ``config/global.yaml``)
2. Strategy-specific configuration (e.g. ``config/gradient_live.json``)
3. Optional environment overlays (``config/environments/<env>/...``)
4. In-memory overrides supplied at call time

Both JSON and YAML formats are supported. YAML loading requires PyYAML to be
installed; a helpful error message is raised otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SUPPORTED_EXTENSIONS = (".json", ".yaml", ".yml")


class ConfigNotFoundError(FileNotFoundError):
    """Raised when expected configuration files are missing."""


def _load_config_file(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    if ext == ".json":
        return json.loads(path.read_text())
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"Unable to load YAML config {path}: PyYAML is not installed. "
                "Install with `uv add pyyaml` or use JSON instead."
            ) from exc
        data = yaml.safe_load(path.read_text())
        return data or {}
    raise ValueError(f"Unsupported config extension for {path}")


def _find_config_path(
    root: Path,
    name_or_file: str,
) -> Optional[Path]:
    candidate = root / name_or_file
    if candidate.is_file():
        return candidate

    stem = Path(name_or_file).stem
    for ext in SUPPORTED_EXTENSIONS:
        candidate = root / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_layered_config(
    strategy_name: str,
    *,
    config_dir: str | Path = "config",
    env: Optional[str] = None,
    filename: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    additional_files: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Load configuration dictionaries using layered precedence rules.

    Args:
        strategy_name: Identifier for the strategy/config being loaded.
        config_dir: Base directory containing configuration files.
        env: Optional environment overlay (e.g. ``"prod"`` or ``"staging"``).
        filename: Explicit filename to load for the strategy layer.
        overrides: Dict applied last, useful for tests.
        additional_files: Optional iterable of extra config filenames to load
            after the strategy layer (e.g. ``["risk_overrides.yaml"]``).

    Returns:
        Dictionary containing the merged configuration.
    """
    config_root = Path(config_dir)
    if not config_root.exists():
        raise ConfigNotFoundError(f"Config directory not found: {config_root}")

    layers: list[Dict[str, Any]] = []

    # Global defaults
    global_path = _find_config_path(config_root, "global")
    if global_path:
        layers.append(_load_config_file(global_path))

    # Strategy layer
    strategy_file = filename or f"{strategy_name}"
    strategy_path = _find_config_path(config_root, strategy_file)
    if strategy_path:
        layers.append(_load_config_file(strategy_path))
    else:
        raise ConfigNotFoundError(
            f"Strategy config not found for '{strategy_name}' in {config_root}"
        )

    # Additional optional files
    if additional_files:
        for extra in additional_files:
            extra_path = _find_config_path(config_root, extra)
            if extra_path:
                layers.append(_load_config_file(extra_path))

    # Environment overlays
    if env:
        env_root = config_root / "environments" / env
        if env_root.exists():
            env_global = _find_config_path(env_root, "global")
            if env_global:
                layers.append(_load_config_file(env_global))
            env_strategy = _find_config_path(env_root, strategy_file)
            if env_strategy:
                layers.append(_load_config_file(env_strategy))
        else:
            raise ConfigNotFoundError(
                f"Environment '{env}' not found under {config_root}/environments"
            )

    merged: Dict[str, Any] = {}
    for layer in layers:
        _merge_dicts(merged, layer)

    if overrides:
        _merge_dicts(merged, overrides)

    return merged


__all__ = [
    "ConfigNotFoundError",
    "load_layered_config",
]

