from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import THESIS_ROOT


@dataclass(frozen=True)
class RunArtifactPaths:
    root: Path
    metrics_path: Path
    predictions_path: Path
    family_summary_path: Path
    recording_summary_path: Path
    manifest_path: Path


def make_run_artifact_paths(recipe_id: str, target: str, protocol: str, output_dir: Path | None = None) -> RunArtifactPaths:
    root = output_dir or (THESIS_ROOT / "artifacts" / "runs" / protocol / target / recipe_id)
    root.mkdir(parents=True, exist_ok=True)
    return RunArtifactPaths(
        root=root,
        metrics_path=root / "metrics.csv",
        predictions_path=root / "predictions.csv",
        family_summary_path=root / "per_family_summary.csv",
        recording_summary_path=root / "per_recording_summary.csv",
        manifest_path=root / "run_manifest.json",
    )


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
