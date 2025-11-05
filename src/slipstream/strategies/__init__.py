"""Strategy packages bundled with Slipstream."""

from importlib import import_module as _import_module
from typing import Dict

# Registry maps strategy identifiers to dotted module paths
STRATEGY_REGISTRY: Dict[str, str] = {
    "gradient": "slipstream.strategies.gradient",
    "template": "slipstream.strategies.template",
}


def load_strategy(name: str):
    """Import and return the top-level module for a registered strategy."""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Registered: {sorted(STRATEGY_REGISTRY)}")
    return _import_module(STRATEGY_REGISTRY[name])

__all__ = ["STRATEGY_REGISTRY", "load_strategy"]
