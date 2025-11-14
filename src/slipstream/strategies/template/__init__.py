"""Template module for onboarding new strategies."""

from .backtest import TemplateBacktestResult, run_template_backtest
from .config import TemplateConfig, load_template_config
from .signals import compute_template_signal


def describe() -> str:
    """Return a human-readable description used in scaffolding docs."""
    return (
        "Template strategy showcasing how to combine config, signals, "
        "and a lightweight backtest loop when onboarding to Slipstream."
    )


__all__ = [
    "TemplateBacktestResult",
    "TemplateConfig",
    "compute_template_signal",
    "describe",
    "load_template_config",
    "run_template_backtest",
]
