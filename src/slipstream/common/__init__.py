"""
Compatibility wrapper for the legacy ``slipstream.common`` import path.

The underlying implementation now lives in ``slipstream.core.common``.
This shim preserves backwards compatibility for modules that have not
yet migrated to the new namespace.
"""

from importlib import import_module as _import_module
import sys as _sys

_core_pkg = _import_module("slipstream.core.common")

# Re-export public attributes.
__all__ = getattr(_core_pkg, "__all__", [])
for _name in __all__:
    globals()[_name] = getattr(_core_pkg, _name)

# Ensure runtime treats ``slipstream.common`` as the core package.
_sys.modules[__name__] = _core_pkg
