#!/usr/bin/env python3
"""Validate that the Slipstream repository matches the monorepo layout."""

from __future__ import annotations

import sys
from pathlib import Path

EXPECTED_DIRS = {
    Path("src/slipstream/core"),
    Path("src/slipstream/strategies"),
}

EXPECTED_SHIMS = {
    Path("src/slipstream/common/__init__.py"),
    Path("src/slipstream/signals/__init__.py"),
    Path("src/slipstream/portfolio/__init__.py"),
    Path("src/slipstream/costs/__init__.py"),
    Path("src/slipstream/funding/__init__.py"),
    Path("src/slipstream/gradient/__init__.py"),
}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    missing_dirs = sorted(str(path) for path in EXPECTED_DIRS if not (repo_root / path).is_dir())
    missing_shims = sorted(str(path) for path in EXPECTED_SHIMS if not (repo_root / path).is_file())

    if missing_dirs or missing_shims:
        if missing_dirs:
            print("Missing directories:")
            for entry in missing_dirs:
                print(f"  - {entry}")
        if missing_shims:
            print("Missing compatibility shims:")
            for entry in missing_shims:
                print(f"  - {entry}")
        return 1

    print("Slipstream monorepo structure looks good ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
