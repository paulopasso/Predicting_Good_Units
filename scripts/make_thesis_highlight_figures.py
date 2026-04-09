#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qc_thesis.highlight_figures import build_highlight_figures


def main() -> int:
    manifest = build_highlight_figures(ROOT)
    print(manifest[["figure_filename", "recommended_usage"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
