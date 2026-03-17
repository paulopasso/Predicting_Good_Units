from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.spikeforest_audit.config import SpikeForestAuditConfig
from qc_framework.data import QCDataLoader


def _log(config: SpikeForestAuditConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SpikeForest paired extraction audit")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet", help="Path to the main QC parquet")
    parser.add_argument("--notebook", default="notebooks/Units_Spikeforest_Extraction_Colab.ipynb", help="Path to the extraction notebook")
    parser.add_argument("--output-root", default="crash_tests/spikeforest_audit/outputs", help="Root folder for outputs")
    parser.add_argument("--run-name", default=None, help="Optional explicit run name")
    parser.add_argument("--quiet", action="store_true", help="Reduce runner logging")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> SpikeForestAuditConfig:
    return SpikeForestAuditConfig(
        parquet_path=REPO_ROOT / args.parquet,
        extraction_notebook=REPO_ROOT / args.notebook,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )


def parse_requested_study_sets(notebook_path: Path) -> list[str]:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "SPIKEFOREST_STUDY_SETS" not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SPIKEFOREST_STUDY_SETS":
                    value = ast.literal_eval(node.value)
                    return [str(item) for item in value]
    raise RuntimeError(f"Could not find SPIKEFOREST_STUDY_SETS in {notebook_path}")


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns + [c for c in frame.columns if c not in columns]]


def _study_level_manifest(paired_df: pd.DataFrame, requested_studies: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    requested_set = {study for study in requested_studies if study.startswith("PAIRED_")}
    actual_studies = sorted(paired_df["study_set"].dropna().astype(str).unique().tolist())
    for study in sorted(requested_set | set(actual_studies)):
        study_df = paired_df[paired_df["study_set"].astype(str) == study].copy()
        sorter_union = sorted(study_df["sorter_name"].dropna().astype(str).unique().tolist())
        rows.append(
            {
                "study_set": study,
                "requested_in_notebook": bool(study in requested_set),
                "present_in_parquet": bool(not study_df.empty),
                "rows": int(len(study_df)),
                "matched_rows": int(study_df["matched_gt_unit_id"].notna().sum()) if not study_df.empty else 0,
                "unmatched_rows": int(study_df["matched_gt_unit_id"].isna().sum()) if not study_df.empty else 0,
                "recordings": int(study_df["recording_key"].nunique()) if not study_df.empty else 0,
                "studies": int(study_df["study_name"].nunique()) if not study_df.empty else 0,
                "sorters": int(study_df["sorter_name"].nunique()) if not study_df.empty else 0,
                "sorter_union": "|".join(sorter_union),
            }
        )
    columns = [
        "study_set",
        "requested_in_notebook",
        "present_in_parquet",
        "rows",
        "matched_rows",
        "unmatched_rows",
        "recordings",
        "studies",
        "sorters",
        "sorter_union",
    ]
    return _frame_with_columns(rows, columns).sort_values("study_set").reset_index(drop=True)


def _recording_level_manifest(paired_df: pd.DataFrame, common_sorter_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        paired_df.groupby(["study_set", "study_name", "recording_name", "recording_key"], dropna=False)
        .agg(
            rows=("row_uid", "size"),
            matched_rows=("matched_gt_unit_id", lambda s: int(s.notna().sum())),
            unmatched_rows=("matched_gt_unit_id", lambda s: int(s.isna().sum())),
            sorters=("sorter_name", "nunique"),
            sorter_names=("sorter_name", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
        )
        .reset_index()
    )

    coverage_rows: list[dict[str, Any]] = []
    for study_set, study_df in paired_df.groupby("study_set", dropna=False):
        recording_count = int(study_df["recording_key"].nunique())
        sorter_presence = (
            study_df[["recording_key", "sorter_name"]]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .groupby("sorter_name")
            .size()
            .rename("recording_support")
            .reset_index()
        )
        sorter_presence["support_fraction"] = sorter_presence["recording_support"] / max(recording_count, 1)
        common_sorters = sorter_presence[
            sorter_presence["support_fraction"] >= float(common_sorter_fraction)
        ]["sorter_name"].astype(str).tolist()
        union_sorters = sorted(sorter_presence["sorter_name"].astype(str).tolist())

        study_recordings = grouped[grouped["study_set"].astype(str) == str(study_set)].copy()
        for _, row in study_recordings.iterrows():
            present_sorters = set(str(row["sorter_names"]).split("|")) if str(row["sorter_names"]) else set()
            present_sorters.discard("")
            missing_common = sorted(set(common_sorters) - present_sorters)
            missing_union = sorted(set(union_sorters) - present_sorters)
            coverage_rows.append(
                {
                    "study_set": row["study_set"],
                    "recording_key": row["recording_key"],
                    "recording_name": row["recording_name"],
                    "common_sorter_count": len(common_sorters),
                    "common_sorters": "|".join(common_sorters),
                    "missing_common_sorters": "|".join(missing_common),
                    "missing_common_sorter_count": len(missing_common),
                    "union_sorter_count": len(union_sorters),
                    "union_sorters": "|".join(union_sorters),
                    "missing_union_sorters": "|".join(missing_union),
                    "missing_union_sorter_count": len(missing_union),
                }
            )

    coverage_df = _frame_with_columns(
        coverage_rows,
        [
            "study_set",
            "recording_key",
            "recording_name",
            "common_sorter_count",
            "common_sorters",
            "missing_common_sorters",
            "missing_common_sorter_count",
            "union_sorter_count",
            "union_sorters",
            "missing_union_sorters",
            "missing_union_sorter_count",
        ],
    )
    recording_manifest = grouped.merge(
        coverage_df,
        on=["study_set", "recording_key", "recording_name"],
        how="left",
    ).sort_values(["study_set", "recording_name"]).reset_index(drop=True)
    return recording_manifest, coverage_df


def _missing_items(
    *,
    study_manifest: pd.DataFrame,
    recording_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in study_manifest.iterrows():
        if bool(row["requested_in_notebook"]) and not bool(row["present_in_parquet"]):
            rows.append(
                {
                    "item_type": "missing_study",
                    "study_set": row["study_set"],
                    "recording_key": pd.NA,
                    "recording_name": pd.NA,
                    "sorter_name": pd.NA,
                    "severity": "critical",
                    "details": "Requested paired study set is absent from the parquet.",
                }
            )

    for _, row in recording_manifest.iterrows():
        if int(row["matched_rows"]) == 0:
            rows.append(
                {
                    "item_type": "no_matched_units",
                    "study_set": row["study_set"],
                    "recording_key": row["recording_key"],
                    "recording_name": row["recording_name"],
                    "sorter_name": pd.NA,
                    "severity": "warning",
                    "details": "Recording has paired rows but zero matched ground-truth units.",
                }
            )
        missing_common = [s for s in str(row.get("missing_common_sorters", "")).split("|") if s]
        for sorter_name in missing_common:
            rows.append(
                {
                    "item_type": "missing_sorter_output_common",
                    "study_set": row["study_set"],
                    "recording_key": row["recording_key"],
                    "recording_name": row["recording_name"],
                    "sorter_name": sorter_name,
                    "severity": "warning",
                    "details": "Sorter is missing from this recording but appears in at least the configured common-fraction of recordings for the study.",
                }
            )

    columns = ["item_type", "study_set", "recording_key", "recording_name", "sorter_name", "severity", "details"]
    return _frame_with_columns(rows, columns).sort_values(["item_type", "study_set", "recording_name", "sorter_name"]).reset_index(drop=True)


def _build_summary(
    *,
    requested_studies: list[str],
    study_manifest: pd.DataFrame,
    recording_manifest: pd.DataFrame,
    missing_items: pd.DataFrame,
) -> str:
    requested_paired = [study for study in requested_studies if study.startswith("PAIRED_")]
    present_requested = study_manifest[
        study_manifest["requested_in_notebook"].astype(bool) & study_manifest["present_in_parquet"].astype(bool)
    ]
    lines = [
        "# SpikeForest Audit Summary",
        "",
        "## Requested Paired Studies",
        f"- requested_paired_studies: `{', '.join(requested_paired)}`",
        f"- present_requested_studies: `{int(len(present_requested))}`",
        f"- requested_studies_missing: `{int((~study_manifest['present_in_parquet'].astype(bool) & study_manifest['requested_in_notebook'].astype(bool)).sum())}`",
        "",
        "## Recording Coverage",
        f"- paired_recordings_in_parquet: `{int(len(recording_manifest))}`",
        f"- paired_recordings_with_no_matched_units: `{int((recording_manifest['matched_rows'].fillna(0).astype(int) == 0).sum())}`",
        f"- recordings_with_missing_common_sorters: `{int((recording_manifest['missing_common_sorter_count'].fillna(0).astype(int) > 0).sum())}`",
        "",
        "## Missing Item Summary",
        f"- total_missing_items: `{int(len(missing_items))}`",
        "",
        "## Notes",
        "- `missing_sorter_output_common` is a heuristic based on study-level sorter coverage inside the current parquet.",
        "- Missing common sorters are likely the most actionable recovery candidates without re-querying SpikeForest manifests.",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_with_config(config: SpikeForestAuditConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(config, f"Loading parquet from {config.parquet_path}")
    loader = QCDataLoader()
    df = loader.load(config.parquet_path)
    requested_studies = parse_requested_study_sets(config.extraction_notebook)
    paired_df = df[df["study_set"].astype(str).str.startswith(config.paired_prefix)].copy()

    study_manifest = _study_level_manifest(paired_df, requested_studies)
    recording_manifest, _ = _recording_level_manifest(paired_df, config.common_sorter_fraction)
    missing_items = _missing_items(
        study_manifest=study_manifest,
        recording_manifest=recording_manifest,
    )
    summary_md = _build_summary(
        requested_studies=requested_studies,
        study_manifest=study_manifest,
        recording_manifest=recording_manifest,
        missing_items=missing_items,
    )

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)
    study_manifest.to_csv(output_dir / "study_manifest.csv", index=False)
    recording_manifest.to_csv(output_dir / "recording_manifest.csv", index=False)
    missing_items.to_csv(output_dir / "missing_items.csv", index=False)
    (output_dir / "audit_summary.md").write_text(summary_md, encoding="utf-8")

    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def main() -> int:
    args = _parse_args()
    config = _build_config(args)
    run_with_config(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
