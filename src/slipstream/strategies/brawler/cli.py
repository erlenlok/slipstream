"""Command-line interface for running the Brawler strategy."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Iterable, Optional

from . import BrawlerEngine, load_brawler_config
from .connectors import HyperliquidExecutionClient
from .inventory import FileInventoryProvider


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Brawler passive market-making loop.")
    parser.add_argument(
        "--config",
        required=False,
        help="Path to a YAML/JSON config file defining assets + risk parameters.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("HYPERLIQUID_API_KEY"),
        help="Hyperliquid API key (default: HYPERLIQUID_API_KEY env var).",
    )
    parser.add_argument(
        "--api-secret",
        default=os.getenv("HYPERLIQUID_API_SECRET"),
        help="Hyperliquid API secret (default: HYPERLIQUID_API_SECRET env var).",
    )
    parser.add_argument(
        "--wallet",
        default=os.getenv("HYPERLIQUID_BRAWLER_WALLET") or os.getenv("HYPERLIQUID_MAIN_WALLET"),
        help="Wallet address used for inventory tracking (default: HYPERLIQUID_BRAWLER_WALLET env var, with HYPERLIQUID_MAIN_WALLET fallback).",
    )
    parser.add_argument(
        "--inventory-file",
        help="Optional path to JSON file with initial per-symbol inventory seeds.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("BRAWLER_LOG_LEVEL") or "INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


async def _run_loop(
    config_path: Optional[str],
    api_key: Optional[str],
    api_secret: Optional[str],
    wallet: Optional[str],
    inventory_file: Optional[str] = None,
) -> None:
    config = load_brawler_config(config_path)
    resolved_api_key = _resolve_secret(api_key, config.hyperliquid_api_key, "HYPERLIQUID_API_KEY")
    resolved_api_secret = _resolve_secret(api_secret, config.hyperliquid_api_secret, "HYPERLIQUID_API_SECRET")
    resolved_wallet = wallet or config.hyperliquid_main_wallet or os.getenv("HYPERLIQUID_BRAWLER_WALLET") or os.getenv("HYPERLIQUID_MAIN_WALLET")
    if not resolved_api_key or not resolved_api_secret:
        raise SystemExit(
            "Hyperliquid API credentials are required via CLI args, config file, or environment."
        )
    config.hyperliquid_main_wallet = resolved_wallet
    config.hyperliquid_api_key = resolved_api_key
    config.hyperliquid_api_secret = resolved_api_secret
    inventory_provider = FileInventoryProvider(inventory_file) if inventory_file else None
    executor = HyperliquidExecutionClient(
        api_key=resolved_api_key,
        api_secret=resolved_api_secret,
        base_url=config.hyperliquid_rest_url,
        target_wallet=resolved_wallet,
    )
    engine = BrawlerEngine(
        config,
        executor=executor,
        inventory_provider=inventory_provider,
    )
    await engine.start()

    stop_event = asyncio.Event()

    def _handle_signal(*_) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    await stop_event.wait()
    await engine.stop()


def run_brawler_cli(argv: Optional[Iterable[str]] = None) -> None:
    _load_env_files()
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    asyncio.run(
        _run_loop(
            args.config,
            args.api_key,
            args.api_secret,
            args.wallet,
            args.inventory_file,
        )
    )


def _load_env_files() -> None:
    preferred = os.getenv("BRAWLER_ENV_FILE")
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([".env.brawler", ".env.gradient"])

    loaded = 0
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.exists():
            continue
        _apply_env_file(path)
        loaded += 1
    if loaded:
        os.environ.setdefault("BRAWLER_ENV_LOADED", str(loaded))


def _apply_env_file(path: Path) -> None:
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or key.startswith("#"):
            continue
        os.environ.setdefault(key, value)


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _resolve_secret(
    cli_value: Optional[str],
    config_value: Optional[str],
    env_key: str,
) -> Optional[str]:
    if cli_value:
        return cli_value
    if config_value:
        return config_value
    env_value = os.getenv(env_key)
    if env_value:
        return env_value
    return None


def run_backtest_cli(argv: Optional[Iterable[str]] = None) -> None:
    raise SystemExit("Brawler backtests are not yet implemented.")


__all__ = ["run_brawler_cli", "run_backtest_cli"]


if __name__ == "__main__":  # pragma: no cover
    run_brawler_cli()
