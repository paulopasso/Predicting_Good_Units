#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    thesis_root = Path(__file__).resolve().parents[1]
    src_root = thesis_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from qc_thesis import validate_recipe_registry

    errors = validate_recipe_registry()
    if errors:
        for item in errors:
            print(item)
        return 1
    print("recipe registry OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
