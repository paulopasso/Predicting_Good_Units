from __future__ import annotations

from pathlib import Path


def get_thesis_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "THESIS_BLUEPRINT.md").exists():
            return candidate
    raise FileNotFoundError("Could not find thesis root containing THESIS_BLUEPRINT.md")


THESIS_ROOT = Path(__file__).resolve().parents[2]

