#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
import argparse
from pathlib import Path

sys.dont_write_bytecode = True


THESIS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = THESIS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qc_thesis.modeling import validate_recipe_registry  # noqa: E402


REQUIRED_DIRS = [
    THESIS_ROOT / "src" / "qc_thesis",
    THESIS_ROOT / "src" / "qc_thesis" / "modeling",
    THESIS_ROOT / "notebooks",
    THESIS_ROOT / "data",
    THESIS_ROOT / "figures",
    THESIS_ROOT / "tables",
    THESIS_ROOT / "artifacts",
]


def audit_notebooks() -> list[str]:
    errors: list[str] = []
    for notebook_path in sorted((THESIS_ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(notebook_path.read_text())
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
        if "_find_thesis_root" not in joined:
            errors.append(f"{notebook_path}: missing _find_thesis_root bootstrap")
        if "ROOT = Path.cwd()" in joined:
            errors.append(f"{notebook_path}: stale Path.cwd bootstrap")
    return errors


def audit_dirs() -> list[str]:
    return [f"Missing required directory: {path}" for path in REQUIRED_DIRS if not path.exists()]


def find_cache_dirs() -> list[Path]:
    return sorted(
        path for path in THESIS_ROOT.rglob("*")
        if path.is_dir() and path.name in {"__pycache__", ".pytest_cache"}
    )


def audit_cache_dirs() -> list[str]:
    return [f"Remove cache directory: {path}" for path in find_cache_dirs()]


def clean_cache_dirs() -> list[Path]:
    removed: list[Path] = []
    for path in find_cache_dirs():
        shutil.rmtree(path)
        removed.append(path)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Remove cache directories before auditing.")
    parser.add_argument("--check-cache", action="store_true", help="Fail if cache directories are present after the audit starts.")
    args = parser.parse_args()

    if args.clean:
        clean_cache_dirs()

    errors: list[str] = []
    errors.extend(audit_dirs())
    errors.extend(audit_notebooks())
    if args.check_cache:
        errors.extend(audit_cache_dirs())
    errors.extend(validate_recipe_registry())

    if errors:
        print("thesis audit FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("thesis audit OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
