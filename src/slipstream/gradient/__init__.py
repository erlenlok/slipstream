"""Compatibility wrapper for the legacy ``slipstream.gradient`` namespace."""

from __future__ import annotations

from ._compat import alias_module as _alias_module

_alias_module(
    __name__,
    "slipstream.strategies.gradient",
    stacklevel=2,
    message=(
        "The 'slipstream.gradient' package is deprecated and will be removed in a future release. "
        "Import 'slipstream.strategies.gradient' instead."
    ),
)
