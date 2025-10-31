"""Configuration management for live Gradient trading."""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class GradientConfig:
    """Live trading configuration."""

    # Strategy parameters
    capital_usd: float
    concentration_pct: float
    rebalance_freq_hours: int
    weight_scheme: str
    lookback_spans: List[int]
    vol_span: int

    # Risk limits
    max_position_pct: float
    max_total_leverage: float
    emergency_stop_drawdown_pct: float

    # Liquidity filters
    liquidity_threshold_usd: float
    liquidity_impact_pct: float

    # API configuration
    api_endpoint: str
    api_key: str
    api_secret: str
    mainnet: bool
    api: dict  # For dict-style access

    # Execution parameters
    execution: dict
    dry_run: bool

    # Logging
    log_dir: str
    log_level: str

    # Alerts
    alerts_enabled: bool
    telegram_token: str = ""
    telegram_chat_id: str = ""


def load_config(config_path: str = "config/gradient_live.json") -> GradientConfig:
    """
    Load live trading configuration from JSON file.

    Args:
        config_path: Path to configuration JSON

    Returns:
        GradientConfig instance

    Environment variables required:
        HYPERLIQUID_API_KEY: API key for Hyperliquid
        HYPERLIQUID_API_SECRET: API secret for Hyperliquid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file) as f:
        data = json.load(f)

    # Get API credentials from environment
    api_key = os.environ.get("HYPERLIQUID_API_KEY", "")
    api_secret = os.environ.get("HYPERLIQUID_API_SECRET", "")

    if not api_key or not api_secret:
        if not data.get("dry_run", True):
            raise ValueError(
                "API credentials not found. Set HYPERLIQUID_API_KEY and "
                "HYPERLIQUID_API_SECRET environment variables."
            )

    # Set default execution params if not in config
    execution_params = data.get("execution", {
        "passive_timeout_seconds": 3600,
        "limit_order_aggression": "join_best",
        "cancel_before_market_sweep": True,
        "min_order_size_usd": 10
    })

    return GradientConfig(
        capital_usd=data["capital_usd"],
        concentration_pct=data["concentration_pct"],
        rebalance_freq_hours=data["rebalance_freq_hours"],
        weight_scheme=data["weight_scheme"],
        lookback_spans=data["lookback_spans"],
        vol_span=data["vol_span"],
        max_position_pct=data["max_position_pct"],
        max_total_leverage=data["max_total_leverage"],
        emergency_stop_drawdown_pct=data["emergency_stop_drawdown_pct"],
        liquidity_threshold_usd=data["liquidity_threshold_usd"],
        liquidity_impact_pct=data["liquidity_impact_pct"],
        api_endpoint=data["api"]["endpoint"],
        api_key=api_key,
        api_secret=api_secret,
        mainnet=data["api"]["mainnet"],
        api=data["api"],  # Store full API config
        execution=execution_params,  # Store execution params
        dry_run=data.get("dry_run", True),
        log_dir=data["logging"]["dir"],
        log_level=data["logging"]["level"],
        alerts_enabled=data["alerts"]["enabled"],
        telegram_token=data["alerts"].get("telegram_token", ""),
        telegram_chat_id=data["alerts"].get("telegram_chat_id", ""),
    )


def validate_config(config: GradientConfig) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if config.capital_usd <= 0:
        raise ValueError(f"capital_usd must be positive, got {config.capital_usd}")

    if not (0 < config.concentration_pct <= 50):
        raise ValueError(
            f"concentration_pct must be in (0, 50], got {config.concentration_pct}"
        )

    if config.weight_scheme not in ["equal", "inverse_vol"]:
        raise ValueError(
            f"weight_scheme must be 'equal' or 'inverse_vol', got {config.weight_scheme}"
        )

    if config.max_position_pct <= 0 or config.max_position_pct > 100:
        raise ValueError(
            f"max_position_pct must be in (0, 100], got {config.max_position_pct}"
        )

    if config.max_total_leverage <= 0:
        raise ValueError(
            f"max_total_leverage must be positive, got {config.max_total_leverage}"
        )

    if len(config.lookback_spans) == 0:
        raise ValueError("lookback_spans cannot be empty")

    print(f"Configuration validated successfully")
    print(f"  Capital: ${config.capital_usd:,.2f}")
    print(f"  Concentration: {config.concentration_pct}%")
    print(f"  Rebalance: Every {config.rebalance_freq_hours}h")
    print(f"  Weighting: {config.weight_scheme}")
    print(f"  Dry-run: {config.dry_run}")
