"""
Compatibility wrapper for ``slipstream.funding`` redirecting to
``slipstream.core.funding``.
"""

from importlib import import_module as _import_module
import sys as _sys

_core_pkg = _import_module("slipstream.core.funding")

__all__ = getattr(_core_pkg, "__all__", [])
for _name in __all__:
    globals()[_name] = getattr(_core_pkg, _name)

_sys.modules[__name__] = _core_pkg
