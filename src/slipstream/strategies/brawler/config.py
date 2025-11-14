"""Configuration helpers for the Brawler market-making strategy."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


@dataclass
class BrawlerRiskConfig:
    """Global risk toggles that apply across every asset."""

    tick_interval_ms: int = 500
    inventory_check_interval: float = 1.0
    kill_switch_cooldown_seconds: float = 10.0
    resume_backoff_seconds: float = 5.0
    metrics_flush_seconds: float = 60.0
    cex_queue_maxsize: int = 2000
    local_queue_maxsize: int = 2000
    fill_queue_maxsize: int = 2000
    min_cancel_interval_ms: int = 250
    side_jitter_ms: int = 25


@dataclass
class BrawlerKillSwitchConfig:
    """Global kill-switch thresholds."""

    max_disconnection_seconds: float = 5.0
    max_feed_lag_seconds: float = 3.0


@dataclass
class BrawlerPortfolioConfig:
    """Portfolio-level exposure guardrails."""

    max_gross_inventory: float = 0.0
    halt_ratio: float = 1.0
    resume_ratio: float = 0.75
    reduce_only_ratio: float = 0.85
    taper_start_ratio: float = 0.5
    min_order_size_ratio: float = 0.25


@dataclass
class BrawlerAssetConfig:
    """Per-asset configuration parameters."""

    symbol: str
    cex_symbol: str
    base_spread: float = 0.001
    volatility_lookback: int = 60
    risk_aversion: float = 5.0
    basis_alpha: float = 0.05
    max_inventory: float = 1.5
    inventory_aversion: float = 0.25
    order_size: float = 0.1
    max_volatility: float = 0.02
    max_basis_deviation: float = 0.005
    min_quote_interval_ms: int = 500
    reduce_only_ratio: float = 0.92
    tick_size: float = 0.1
    quote_reprice_tolerance_ticks: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "cex_symbol": self.cex_symbol,
            "base_spread": self.base_spread,
            "volatility_lookback": self.volatility_lookback,
            "risk_aversion": self.risk_aversion,
            "basis_alpha": self.basis_alpha,
            "max_inventory": self.max_inventory,
            "inventory_aversion": self.inventory_aversion,
            "order_size": self.order_size,
            "max_volatility": self.max_volatility,
            "max_basis_deviation": self.max_basis_deviation,
            "min_quote_interval_ms": self.min_quote_interval_ms,
            "reduce_only_ratio": self.reduce_only_ratio,
            "tick_size": self.tick_size,
            "quote_reprice_tolerance_ticks": self.quote_reprice_tolerance_ticks,
        }


@dataclass
class BrawlerCandidateScreeningConfig:
    """Thresholds + weights for offline candidate discovery."""

    min_samples: int = 2000
    align_tolerance_ms: int = 750
    sigma_ratio_min: float = 0.8
    sigma_ratio_max: float = 1.2
    max_mean_basis_ticks: float = 2.0
    max_basis_std_ticks: float = 1.0
    min_spread_ratio: float = 2.5
    max_funding_std: float = 0.02
    min_depth_multiple: float = 2.0
    weight_spread_edge: float = 0.5
    weight_basis_penalty: float = 0.2
    weight_vol_penalty: float = 0.2
    weight_depth_penalty: float = 0.1
    weight_funding_penalty: float = 0.1
    score_min: float = 0.0


@dataclass
class BrawlerConfig:
    """Top-level configuration for the market-maker."""

    assets: Dict[str, BrawlerAssetConfig] = field(default_factory=dict)
    binance_ws_url: str = "wss://fstream.binance.com"
    hyperliquid_ws_url: str = "wss://api.hyperliquid.xyz/ws"
    hyperliquid_rest_url: str = "https://api.hyperliquid.xyz"
    hyperliquid_main_wallet: Optional[str] = None
    hyperliquid_api_key: Optional[str] = None
    hyperliquid_api_secret: Optional[str] = None
    state_snapshot_path: Optional[str] = None
    risk: BrawlerRiskConfig = field(default_factory=BrawlerRiskConfig)
    kill_switch: BrawlerKillSwitchConfig = field(default_factory=BrawlerKillSwitchConfig)
    portfolio: BrawlerPortfolioConfig = field(default_factory=BrawlerPortfolioConfig)
    candidate_screening: BrawlerCandidateScreeningConfig = field(
        default_factory=BrawlerCandidateScreeningConfig
    )

    def asset(self, symbol: str) -> BrawlerAssetConfig:
        try:
            return self.assets[symbol]
        except KeyError as exc:  # pragma: no cover - configuration bug
            raise KeyError(f"Asset '{symbol}' missing from Brawler config.") from exc


def _coerce_asset(key: str, payload: Mapping[str, Any]) -> BrawlerAssetConfig:
    params = dict(payload)
    params.setdefault("symbol", key)
    params.setdefault("cex_symbol", key)
    return BrawlerAssetConfig(**params)


def _validate_assets(assets: Dict[str, BrawlerAssetConfig]) -> None:
    seen_hl: set[str] = set()
    seen_cex: set[str] = set()
    for key, cfg in assets.items():
        if key != cfg.symbol:
            raise ValueError(
                f"Asset map key '{key}' does not match config symbol '{cfg.symbol}'."
            )

        symbol_key = cfg.symbol.upper()
        if symbol_key in seen_hl:
            raise ValueError(f"Duplicate Hyperliquid symbol '{cfg.symbol}'.")
        seen_hl.add(symbol_key)

        cex_key = cfg.cex_symbol.upper()
        if cex_key in seen_cex:
            raise ValueError(f"Duplicate CEX symbol '{cfg.cex_symbol}'.")
        seen_cex.add(cex_key)


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z0-9_]+)")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return _expand_env(data)


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


def load_brawler_config(path: Optional[str] = None) -> BrawlerConfig:
    """
    Load configuration from disk or return defaults for quick experiments.

    Parameters
    ----------
    path: optional str
        Path to a YAML/JSON file. If omitted, returns a single-asset template config.
    """

    if path is None:
        default_asset = BrawlerAssetConfig(symbol="BTC", cex_symbol="btcusdt")
        return BrawlerConfig(assets={default_asset.symbol: default_asset})

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Brawler config file '{path}' not found.")

    payload = _load_mapping(cfg_path)
    risk_payload = payload.get("risk") or {}
    kill_switch_payload = payload.get("kill_switch") or {}
    candidate_payload = payload.get("candidate_screening") or {}

    assets_payload = payload.get("assets")
    if not assets_payload:
        raise ValueError("Brawler config must define at least one asset under 'assets'.")

    assets = {
        symbol: _coerce_asset(symbol, params or {})
        for symbol, params in assets_payload.items()
    }
    _validate_assets(assets)

    portfolio_payload = payload.get("portfolio") or {}

    config = BrawlerConfig(
        assets=assets,
        binance_ws_url=payload.get("binance_ws_url") or "wss://fstream.binance.com",
        hyperliquid_ws_url=payload.get("hyperliquid_ws_url") or "wss://api.hyperliquid.xyz/ws",
        hyperliquid_rest_url=payload.get("hyperliquid_rest_url") or "https://api.hyperliquid.xyz",
        hyperliquid_main_wallet=payload.get("hyperliquid_main_wallet"),
        hyperliquid_api_key=payload.get("hyperliquid_api_key"),
        hyperliquid_api_secret=payload.get("hyperliquid_api_secret"),
        state_snapshot_path=payload.get("state_snapshot_path"),
        risk=BrawlerRiskConfig(**risk_payload),
        kill_switch=BrawlerKillSwitchConfig(**kill_switch_payload),
        portfolio=BrawlerPortfolioConfig(**portfolio_payload),
        candidate_screening=BrawlerCandidateScreeningConfig(**candidate_payload),
    )

    return config


__all__ = [
    "BrawlerAssetConfig",
    "BrawlerConfig",
    "BrawlerCandidateScreeningConfig",
    "BrawlerKillSwitchConfig",
    "BrawlerRiskConfig",
    "BrawlerPortfolioConfig",
    "load_brawler_config",
]
