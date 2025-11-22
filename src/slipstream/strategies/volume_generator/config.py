"""Configuration helpers for the Volume Generator strategy."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import yaml


@dataclass
class VolumeBotConfig:
    """Configuration for volume generator bot."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    main_wallet: Optional[str] = None
    base_url: str = "https://api.hyperliquid.xyz"
    symbol: str = "BTC"
    trade_count: int = 42
    trade_size_usd: float = 20.0  # USD amount to trade each time
    delay_between_trades: float = 1.5  # seconds delay between trades
    slippage_tolerance_bps: float = 100  # 1% slippage tolerance
    dry_run: bool = False  # If True, don't place actual orders


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z0-9_]+)")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(_replace_env_token, value)
    return value


def _replace_env_token(match: re.Match[str]) -> str:
    key = match.group(1) or match.group(2)
    if not key:
        return match.group(0)
    env_val = os.getenv(key)
    if env_val is None:
        return match.group(0)
    return env_val


def load_volume_generator_config(path: Optional[str] = None) -> VolumeBotConfig:
    """
    Load volume generator configuration from disk or return defaults.

    Parameters
    ----------
    path: optional str
        Path to a YAML/JSON file. If omitted, returns default config.
    """
    if path is None:
        # Try to get API credentials from environment variables as defaults
        api_key = os.getenv("HYPERLIQUID_API_KEY", "")
        api_secret = os.getenv("HYPERLIQUID_API_SECRET", "")
        main_wallet = os.getenv("HYPERLIQUID_VOLUME_WALLET") or os.getenv("HYPERLIQUID_MAIN_WALLET", "")

        return VolumeBotConfig(
            api_key=api_key,
            api_secret=api_secret,
            main_wallet=main_wallet
        )

    import json
    from pathlib import Path

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Volume Generator config file '{path}' not found.")

    # Load the file based on extension
    text = cfg_path.read_text()
    suffix = cfg_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)

    # Expand environment variables
    data = _expand_env(data)

    # Extract volume generation config
    vg_config = data.get("volume_generation", {})

    # Extract hyperliquid config
    hl_config = data.get("hyperliquid", {})

    # Get API credentials from environment variables with fallback to config file
    api_key = (
        os.getenv("HYPERLIQUID_API_KEY") or
        hl_config.get("api_key") or
        ""
    )
    api_secret = (
        os.getenv("HYPERLIQUID_API_SECRET") or
        hl_config.get("api_secret") or
        ""
    )
    main_wallet = (
        os.getenv("HYPERLIQUID_VOLUME_WALLET") or os.getenv("HYPERLIQUID_MAIN_WALLET") or
        hl_config.get("main_wallet") or
        ""
    )

    # Create the config object with defaults overwritten by config values
    config = VolumeBotConfig(
        trade_count=vg_config.get("trade_count", 42),
        trade_size_usd=vg_config.get("trade_size_usd", 20.0),
        delay_between_trades=vg_config.get("delay_between_trades", 1.5),
        symbol=vg_config.get("symbol", "BTC"),
        slippage_tolerance_bps=vg_config.get("slippage_tolerance_bps", 100),
        dry_run=vg_config.get("dry_run", False),
        base_url=hl_config.get("base_url", "https://api.hyperliquid.xyz"),
        api_key=api_key,
        api_secret=api_secret,
        main_wallet=main_wallet,
    )

    return config


__all__ = [
    "VolumeBotConfig",
    "load_volume_generator_config",
]