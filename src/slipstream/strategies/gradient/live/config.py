"""Configuration management for live Gradient trading."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

from slipstream.core.config import load_layered_config


@dataclass
class GradientConfig:
    """Live trading configuration."""

    # Strategy parameters
    capital_usd: float
    concentration_pct: float
    rebalance_freq_hours: int
    weight_scheme: str  # Legacy: "equal" or "inverse_vol"
    lookback_spans: List[int]
    vol_span: int  # Legacy: for EWMA vol

    # VAR-based risk parameters (new)
    risk_method: str = "dollar_vol"  # "dollar_vol" (legacy) or "var" (new)
    target_side_var: float = 0.02  # Target one-day VAR per side
    var_lookback_days: int = 60  # Days for covariance estimation

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


def _resolve_strategy_config(
    config_path: str,
) -> Dict[str, Optional[str]]:
    """Infer config directory, strategy name, and filename from a user hint."""
    candidate = Path(config_path)
    if candidate.is_file():
        return {
            "config_dir": str(candidate.parent),
            "strategy_name": candidate.stem,
            "filename": candidate.name,
        }

    # Allow callers to pass relative names like "gradient_live"
    if candidate.suffix:
        # File with extension that does not exist yet – assume relative to CWD.
        return {
            "config_dir": str(candidate.parent) if str(candidate.parent) else "config",
            "strategy_name": candidate.stem,
            "filename": candidate.name,
        }

    # Pure strategy identifier – fall back to default config directory.
    return {
        "config_dir": "config",
        "strategy_name": candidate.name or config_path,
        "filename": None,
    }


def load_config(
    config_path: str = "config/gradient_live.json",
    *,
    env: Optional[str] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> GradientConfig:
    """
    Load live trading configuration with optional layered overrides.

    Args:
        config_path: Path or identifier for configuration
        env: Optional environment overlay name
        overrides: Dict applied last (primarily for tests)

    Returns:
        GradientConfig instance

    Environment variables required:
        HYPERLIQUID_API_KEY: API key for Hyperliquid
        HYPERLIQUID_API_SECRET: API secret for Hyperliquid
    """
    info = _resolve_strategy_config(config_path)
    raw_config = load_layered_config(
        strategy_name=info["strategy_name"] or "gradient_live",
        config_dir=info["config_dir"] or "config",
        filename=info["filename"],
        env=env,
        overrides=overrides,
    )

    # Get API credentials from environment
    api_key = os.environ.get("HYPERLIQUID_API_KEY", "")
    api_secret = os.environ.get("HYPERLIQUID_API_SECRET", "")

    if (not api_key or not api_secret) and not raw_config.get("dry_run", True):
        raise ValueError(
            "API credentials not found. Set HYPERLIQUID_API_KEY and "
            "HYPERLIQUID_API_SECRET environment variables."
        )

    # Set default execution params if not in config
    execution_params = raw_config.get("execution", {
        "passive_timeout_seconds": 3600,
        "limit_order_aggression": "join_best",
        "cancel_before_market_sweep": True,
        "min_order_size_usd": 10
    })

    return GradientConfig(
        capital_usd=raw_config["capital_usd"],
        concentration_pct=raw_config["concentration_pct"],
        rebalance_freq_hours=raw_config["rebalance_freq_hours"],
        weight_scheme=raw_config["weight_scheme"],
        lookback_spans=raw_config["lookback_spans"],
        vol_span=raw_config["vol_span"],
        # VAR parameters (with defaults for backward compatibility)
        risk_method=raw_config.get("risk_method", "dollar_vol"),
        target_side_var=raw_config.get("target_side_var", 0.02),
        var_lookback_days=raw_config.get("var_lookback_days", 60),
        max_position_pct=raw_config["max_position_pct"],
        max_total_leverage=raw_config["max_total_leverage"],
        emergency_stop_drawdown_pct=raw_config["emergency_stop_drawdown_pct"],
        liquidity_threshold_usd=raw_config["liquidity_threshold_usd"],
        liquidity_impact_pct=raw_config["liquidity_impact_pct"],
        api_endpoint=raw_config["api"]["endpoint"],
        api_key=api_key,
        api_secret=api_secret,
        mainnet=raw_config["api"]["mainnet"],
        api=raw_config["api"],  # Store full API config
        execution=execution_params,  # Store execution params
        dry_run=raw_config.get("dry_run", True),
        log_dir=raw_config["logging"]["dir"],
        log_level=raw_config["logging"]["level"],
        alerts_enabled=raw_config["alerts"]["enabled"],
        telegram_token=raw_config["alerts"].get("telegram_token", ""),
        telegram_chat_id=raw_config["alerts"].get("telegram_chat_id", ""),
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

    # Validate VAR parameters
    if config.risk_method not in ["dollar_vol", "var"]:
        raise ValueError(
            f"risk_method must be 'dollar_vol' or 'var', got {config.risk_method}"
        )

    if config.risk_method == "var":
        if config.target_side_var <= 0:
            raise ValueError(
                f"target_side_var must be positive, got {config.target_side_var}"
            )
        if config.var_lookback_days < 30:
            raise ValueError(
                f"var_lookback_days must be >= 30, got {config.var_lookback_days}"
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
    print(f"  Risk Method: {config.risk_method}")
    if config.risk_method == "var":
        print(f"    Target VAR: {config.target_side_var*100:.1f}% (per side, 1-day, 95% confidence)")
        print(f"    VAR Lookback: {config.var_lookback_days} days")
    else:
        print(f"    Weighting: {config.weight_scheme} (legacy)")
    print(f"  Dry-run: {config.dry_run}")
