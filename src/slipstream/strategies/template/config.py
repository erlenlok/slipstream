"""Config helpers for the sample template strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional


@dataclass
class TemplateConfig:
    """Lightweight configuration used by the template strategy."""

    lookback: int = 24
    volatility_span: int = 64
    top_n: int = 3
    bottom_n: Optional[int] = None
    target_side_risk: float = 0.5  # Fraction of capital allocated to each side.
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


def load_template_config(config: Optional[Mapping[str, Any]] = None) -> TemplateConfig:
    """
    Build a TemplateConfig from a mapping (e.g., YAML/JSON) or return defaults.

    The helper is intentionally simple so new strategies can copy/paste and
    extend it during onboarding.
    """

    if config is None:
        return TemplateConfig()

    kwargs = {
        "lookback": int(config.get("lookback", TemplateConfig.lookback)),
        "volatility_span": int(config.get("volatility_span", TemplateConfig.volatility_span)),
        "top_n": int(config.get("top_n", TemplateConfig.top_n)),
        "bottom_n": config.get("bottom_n"),
        "target_side_risk": float(
            config.get("target_side_risk", TemplateConfig.target_side_risk)
        ),
        "metadata": dict(config.get("metadata", {})),
    }
    bottom_n = kwargs["bottom_n"]
    if bottom_n is not None:
        kwargs["bottom_n"] = int(bottom_n)
    return TemplateConfig(**kwargs)


__all__ = ["TemplateConfig", "load_template_config"]
