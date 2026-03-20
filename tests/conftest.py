from __future__ import annotations

from pathlib import Path
import sys


THESIS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = THESIS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
