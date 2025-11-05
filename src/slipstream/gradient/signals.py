"""Deprecated compatibility shim for ``slipstream.gradient.signals``."""

from __future__ import annotations

from ._compat import alias_module as _alias_module

_alias_module(
    __name__,
    "slipstream.strategies.gradient.signals",
    stacklevel=2,
)

