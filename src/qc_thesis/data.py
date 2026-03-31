from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .features import add_context_features, infer_numeric_feature_columns
from .paths import THESIS_ROOT


RAW_PARQUET_PATH = THESIS_ROOT / "data" / "raw" / "33000_ROWS.parquet"


@dataclass(frozen=True)
class DatasetBundle:
    full_df: pd.DataFrame
    hybrid_df: pd.DataFrame
    paired_df: pd.DataFrame
    paired_labeled_df: pd.DataFrame
    paired_unlabeled_df: pd.DataFrame


def load_dataset_summary() -> dict:
    with open(THESIS_ROOT / "data/dataset_summary.json") as f:
        return json.load(f)


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or RAW_PARQUET_PATH).copy()
    df["dataset_kind"] = df["study_set"].astype(str).str.startswith("PAIRED").map({True: "paired", False: "hybrid"})
    df["recording_key"] = df["study_set"].astype(str) + "::" + df["recording_name"].astype(str)
    df["is_paired_labeled"] = df["dataset_kind"].eq("paired") & df["match_status"].isin(["matched_best", "well_detected"])
    df["is_paired_unlabeled"] = df["dataset_kind"].eq("paired") & ~df["is_paired_labeled"]
    df["is_hybrid_labeled"] = df["dataset_kind"].eq("hybrid")
    return df


def load_enriched_dataset(path: Path | None = None) -> pd.DataFrame:
    return add_context_features(load_raw_dataset(path=path))


def build_dataset_bundle(path: Path | None = None, *, with_context: bool = True) -> DatasetBundle:
    full_df = load_enriched_dataset(path=path) if with_context else load_raw_dataset(path=path)
    hybrid_df = full_df[full_df["dataset_kind"] == "hybrid"].copy()
    paired_df = full_df[full_df["dataset_kind"] == "paired"].copy()
    paired_labeled_df = full_df[full_df["is_paired_labeled"]].copy()
    paired_unlabeled_df = full_df[full_df["is_paired_unlabeled"]].copy()
    return DatasetBundle(
        full_df=full_df,
        hybrid_df=hybrid_df,
        paired_df=paired_df,
        paired_labeled_df=paired_labeled_df,
        paired_unlabeled_df=paired_unlabeled_df,
    )


def compute_dataset_summary(df: pd.DataFrame | None = None) -> dict[str, int | float]:
    df = load_raw_dataset() if df is None else df
    return {
        "total_rows": int(len(df)),
        "unique_recordings": int(df["recording_key"].nunique()),
        "unique_sorters": int(df["sorter_name"].nunique()),
        "feature_columns": int(len(infer_numeric_feature_columns(df))),
        "hybrid_labeled_rows": int(df["is_hybrid_labeled"].sum()),
        "paired_rows": int(df["dataset_kind"].eq("paired").sum()),
        "paired_labeled_rows": int(df["is_paired_labeled"].sum()),
        "paired_unlabeled_rows": int(df["is_paired_unlabeled"].sum()),
    }
