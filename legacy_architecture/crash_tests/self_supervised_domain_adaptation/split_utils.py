from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class MainExperimentSplit:
    hybrid_source: pd.DataFrame
    paired_finetune: pd.DataFrame
    paired_test: pd.DataFrame
    paired_non_test_all: pd.DataFrame
    ssl_pool: pd.DataFrame
    holdout_recordings: list[str]
    manifest: dict[str, Any]


@dataclass
class StressStudySplit:
    held_out_study: str
    hybrid_source: pd.DataFrame
    paired_finetune: pd.DataFrame
    paired_test: pd.DataFrame
    paired_non_test_all: pd.DataFrame
    ssl_pool: pd.DataFrame
    manifest: dict[str, Any]


def _grouped_holdout(
    df: pd.DataFrame,
    *,
    target_rows: int,
    random_state: int,
    n_candidates: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_rows = min(target_rows, max(len(df) - 1, 1))
    test_size = min(0.95, max(0.05, target_rows / max(len(df), 1)))
    splitter = GroupShuffleSplit(
        n_splits=n_candidates,
        test_size=test_size,
        random_state=random_state,
    )

    best_train: pd.DataFrame | None = None
    best_test: pd.DataFrame | None = None
    best_gap: int | None = None

    for train_idx, test_idx in splitter.split(df, groups=df["recording_key"].astype(str)):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        gap = abs(len(test) - target_rows)
        if best_gap is None or gap < best_gap:
            best_train = train
            best_test = test
            best_gap = gap
            if gap == 0:
                break

    assert best_train is not None and best_test is not None
    overlap = set(best_train["recording_key"].astype(str)) & set(best_test["recording_key"].astype(str))
    if overlap:
        raise AssertionError(f"Grouped holdout recording overlap: {sorted(overlap)[:5]}")
    return best_train, best_test


def make_main_split(
    df: pd.DataFrame,
    *,
    paired_final_holdout_rows: int,
    random_state: int,
) -> MainExperimentSplit:
    hybrid_source = df[df["dataset_type"] == "hybrid"].copy()
    paired = df[df["dataset_type"] == "paired"].copy()
    paired_matched = paired[paired["matched_gt_unit_id"].notna()].copy()

    paired_finetune, paired_test = _grouped_holdout(
        paired_matched,
        target_rows=paired_final_holdout_rows,
        random_state=random_state,
    )
    holdout_recordings = sorted(paired_test["recording_key"].astype(str).unique().tolist())
    paired_non_test_all = paired[~paired["recording_key"].astype(str).isin(holdout_recordings)].copy()
    ssl_pool = paired_non_test_all[paired_non_test_all["matched_gt_unit_id"].isna()].copy()

    overlap_ssl = set(ssl_pool["recording_key"].astype(str)) & set(holdout_recordings)
    overlap_finetune = set(paired_finetune["recording_key"].astype(str)) & set(holdout_recordings)
    if overlap_ssl:
        raise AssertionError(f"SSL pool overlaps main holdout recordings: {sorted(overlap_ssl)[:5]}")
    if overlap_finetune:
        raise AssertionError(f"Fine-tune pool overlaps main holdout recordings: {sorted(overlap_finetune)[:5]}")

    manifest = {
        "protocol": "recording_disjoint",
        "hybrid_source_rows": int(len(hybrid_source)),
        "paired_test_rows": int(len(paired_test)),
        "paired_finetune_rows": int(len(paired_finetune)),
        "paired_non_test_rows": int(len(paired_non_test_all)),
        "ssl_pool_rows": int(len(ssl_pool)),
        "paired_test_recordings": int(paired_test["recording_key"].nunique()),
        "paired_finetune_recordings": int(paired_finetune["recording_key"].nunique()),
        "ssl_pool_recordings": int(ssl_pool["recording_key"].nunique()),
        "holdout_recordings": holdout_recordings,
        "allow_test_time_unlabeled_context": True,
    }

    return MainExperimentSplit(
        hybrid_source=hybrid_source,
        paired_finetune=paired_finetune,
        paired_test=paired_test,
        paired_non_test_all=paired_non_test_all,
        ssl_pool=ssl_pool,
        holdout_recordings=holdout_recordings,
        manifest=manifest,
    )


def make_grouped_validation_splits(
    df: pd.DataFrame,
    *,
    n_splits: int,
    test_size: float,
    random_state: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if df.empty:
        return []
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for offset in range(n_splits):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state + offset,
        )
        train_idx, val_idx = next(splitter.split(df, groups=df["recording_key"].astype(str)))
        train = df.iloc[train_idx].copy()
        val = df.iloc[val_idx].copy()
        overlap = set(train["recording_key"].astype(str)) & set(val["recording_key"].astype(str))
        if overlap:
            raise AssertionError(f"Grouped validation overlap: {sorted(overlap)[:5]}")
        splits.append((train, val))
    return splits


def build_stress_splits(
    df: pd.DataFrame,
    *,
    max_stress_studies: int | None = None,
) -> list[StressStudySplit]:
    hybrid_source = df[df["dataset_type"] == "hybrid"].copy()
    paired = df[df["dataset_type"] == "paired"].copy()
    paired_matched = paired[paired["matched_gt_unit_id"].notna()].copy()
    studies = sorted(paired_matched["study_set"].dropna().astype(str).unique().tolist())
    if max_stress_studies is not None:
        studies = studies[:max_stress_studies]

    splits: list[StressStudySplit] = []
    for study in studies:
        paired_test = paired_matched[paired_matched["study_set"].astype(str) == study].copy()
        paired_non_test_all = paired[paired["study_set"].astype(str) != study].copy()
        paired_finetune = paired_non_test_all[paired_non_test_all["matched_gt_unit_id"].notna()].copy()
        ssl_pool = paired_non_test_all[paired_non_test_all["matched_gt_unit_id"].isna()].copy()

        if paired_test.empty or paired_finetune.empty or ssl_pool.empty:
            continue

        held_out_studies = set(paired_test["study_set"].astype(str))
        if held_out_studies != {study}:
            raise AssertionError("Stress-test paired test set leaked extra study_set values")
        if set(paired_finetune["study_set"].astype(str)) & {study}:
            raise AssertionError("Fine-tune pool contains held-out study_set rows")
        if set(ssl_pool["study_set"].astype(str)) & {study}:
            raise AssertionError("SSL pool contains held-out study_set rows")

        manifest = {
            "protocol": "leave_one_study_out",
            "held_out_study": study,
            "hybrid_source_rows": int(len(hybrid_source)),
            "paired_test_rows": int(len(paired_test)),
            "paired_finetune_rows": int(len(paired_finetune)),
            "paired_non_test_rows": int(len(paired_non_test_all)),
            "ssl_pool_rows": int(len(ssl_pool)),
            "paired_test_recordings": int(paired_test["recording_key"].nunique()),
            "paired_finetune_recordings": int(paired_finetune["recording_key"].nunique()),
            "ssl_pool_recordings": int(ssl_pool["recording_key"].nunique()),
            "allow_test_time_unlabeled_context": True,
        }
        splits.append(
            StressStudySplit(
                held_out_study=study,
                hybrid_source=hybrid_source,
                paired_finetune=paired_finetune,
                paired_test=paired_test,
                paired_non_test_all=paired_non_test_all,
                ssl_pool=ssl_pool,
                manifest=manifest,
            )
        )
    return splits
