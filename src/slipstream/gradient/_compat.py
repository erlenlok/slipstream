"""
Shared helpers for Gradient compatibility shims.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from types import ModuleType


def alias_module(old_name: str, new_name: str, *, message: str | None = None, stacklevel: int = 3) -> ModuleType:
    """
    Import ``new_name`` and register it under ``old_name`` with a deprecation warning.

    Returns the imported module so callers can optionally re-export symbols.
    """
    warn_message = message or (
        f"Importing '{old_name}' is deprecated. Use '{new_name}' instead."
    )
    warnings.warn(warn_message, DeprecationWarning, stacklevel=stacklevel)
    module = importlib.import_module(new_name)
    sys.modules[old_name] = module
    return module

