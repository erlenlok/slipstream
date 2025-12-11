"""Strategy packages bundled with Slipstream."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module as _import_module
from types import ModuleType
from typing import Callable, Dict, Iterable, List


@dataclass(frozen=True)
class StrategyInfo:
    """Metadata describing a registered strategy."""

    key: str
    title: str
    module: str
    description: str
    cli_entrypoints: Dict[str, str]

    def load_module(self) -> ModuleType:
        """Load and return the strategy's root module."""
        return _import_module(self.module)

    def load_cli_handler(self, command: str) -> Callable[..., object]:
        """Dynamically import a CLI handler (e.g., run_backtest_cli)."""
        dotted = self.cli_entrypoints.get(command)
        if dotted is None:
            available = ", ".join(sorted(self.cli_entrypoints))
            raise KeyError(
                f"Strategy '{self.key}' does not expose a '{command}' CLI. "
                f"Available commands: {available or 'none'}"
            )
        module_name, func_name = dotted.rsplit(".", 1)
        module = _import_module(module_name)
        handler = getattr(module, func_name, None)
        if handler is None:
            raise AttributeError(f"CLI handler '{dotted}' could not be resolved.")
        return handler


# Registry maps strategy identifiers to metadata + import hooks.
STRATEGY_REGISTRY: Dict[str, StrategyInfo] = {
    "gradient": StrategyInfo(
        key="gradient",
        title="Gradient Trend Overlay",
        module="slipstream.strategies.gradient",
        description="Balanced trend-following overlay built on shared Slipstream tooling.",
        cli_entrypoints={
            "run_backtest": "slipstream.strategies.gradient.cli.run_backtest_cli",
            "compute_signals": "slipstream.strategies.gradient.cli.compute_signals_cli",
        },
    ),
    "template": StrategyInfo(
        key="template",
        title="Template Mean-Reversion",
        module="slipstream.strategies.template",
        description="Reference implementation for onboarding new Slipstream strategies.",
        cli_entrypoints={
            "run_backtest": "slipstream.strategies.template.cli.run_backtest_cli",
            "compute_signals": "slipstream.strategies.template.cli.compute_signals_cli",
        },
    ),
    "brawler": StrategyInfo(
        key="brawler",
        title="Brawler Passive MM",
        module="slipstream.strategies.brawler",
        description="CEX-anchored passive market maker with latency-aware spreads.",
        cli_entrypoints={
            "run_brawler": "slipstream.strategies.brawler.cli.run_brawler_cli",
            "run_backtest": "slipstream.strategies.brawler.cli.run_backtest_cli",
        },
    ),
    "volume_generator": StrategyInfo(
        key="volume_generator",
        title="Volume Generator Bot",
        module="slipstream.strategies.volume_generator",
        description="Generates volume by making 42 in-and-out BTC trades.",
        cli_entrypoints={
            "run_volume_gen": "slipstream.strategies.volume_generator.cli.run_volume_generator_cli",
        },
    ),
    "spectrum": StrategyInfo(
        key="spectrum",
        title="Spectrum Idiosyncratic Arbitrage",
        module="slipstream.strategies.spectrum",
        description="Idiosyncratic statistical arbitrage system harvesting momentum, mean reversion, and funding carry premia.",
        cli_entrypoints={
            "run_spectrum": "slipstream.strategies.spectrum.cli.main",
        },
    ),
}


def load_strategy(name: str) -> ModuleType:
    """Import and return the top-level module for a registered strategy."""
    info = STRATEGY_REGISTRY.get(name)
    if info is None:
        raise KeyError(f"Unknown strategy '{name}'. Registered: {sorted(STRATEGY_REGISTRY)}")
    return info.load_module()


def get_strategy_info(name: str) -> StrategyInfo:
    """Return metadata for a registered strategy."""
    info = STRATEGY_REGISTRY.get(name)
    if info is None:
        raise KeyError(f"Unknown strategy '{name}'. Registered: {sorted(STRATEGY_REGISTRY)}")
    return info


def list_strategies() -> List[StrategyInfo]:
    """Return registered strategies sorted by key."""
    return [STRATEGY_REGISTRY[key] for key in sorted(STRATEGY_REGISTRY)]


def iter_strategy_cli(command: str) -> Iterable[Callable[..., object]]:
    """Yield CLI handlers for strategies that implement a given command."""
    for info in STRATEGY_REGISTRY.values():
        try:
            yield info.load_cli_handler(command)
        except KeyError:
            continue


__all__ = [
    "StrategyInfo",
    "STRATEGY_REGISTRY",
    "get_strategy_info",
    "iter_strategy_cli",
    "list_strategies",
    "load_strategy",
]
