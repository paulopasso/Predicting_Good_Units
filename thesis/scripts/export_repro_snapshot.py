#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), text=True).strip()
    except Exception:
        return ""


def main() -> int:
    thesis_root = Path(__file__).resolve().parents[1]
    project_root = thesis_root.parent
    out = thesis_root / "reproducibility/current_environment_snapshot.json"
    data = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        },
        "git": {
            "commit": _git(["git", "rev-parse", "HEAD"], project_root),
            "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"], project_root),
        },
        "environment": {
            "cwd": str(project_root),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        },
    }
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
