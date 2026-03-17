from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class SplitResult:
    protocol: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    manifest: dict


def make_recording_key(df: pd.DataFrame) -> pd.Series:
    return df["study_set"].astype(str) + "::" + df["recording_name"].astype(str)


def recording_disjoint_split(df: pd.DataFrame, *, test_rows_target: int = 100, random_state: int = 42) -> SplitResult:
    groups = make_recording_key(df)
    unique_groups = groups.drop_duplicates().to_numpy()
    best_mask = None
    best_gap = float("inf")
    splitter = GroupShuffleSplit(n_splits=64, test_size=0.2, random_state=random_state)
    dummy = np.zeros(len(df))
    for train_idx, test_idx in splitter.split(dummy, groups=groups):
        gap = abs(len(test_idx) - test_rows_target)
        if gap < best_gap:
            best_gap = gap
            best_mask = (train_idx, test_idx)
    if best_mask is None:
        raise RuntimeError("Could not construct recording-disjoint split")
    train_idx, test_idx = best_mask
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    manifest = {
        "protocol": "recording_disjoint_main",
        "random_state": int(random_state),
        "test_rows_target": int(test_rows_target),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_recordings": sorted(make_recording_key(train_df).unique().tolist()),
        "test_recordings": sorted(make_recording_key(test_df).unique().tolist()),
    }
    return SplitResult("recording_disjoint_main", train_df, test_df, manifest)


def leave_one_family_out(df: pd.DataFrame, *, family: str) -> SplitResult:
    test_df = df[df["study_set"] == family].copy()
    train_df = df[df["study_set"] != family].copy()
    manifest = {
        "protocol": "family_held_out",
        "held_out_family": family,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_families": sorted(train_df["study_set"].dropna().unique().tolist()),
        "test_families": [family],
    }
    return SplitResult("family_held_out", train_df, test_df, manifest)


def build_label_budget_subset(df: pd.DataFrame, *, budget_rows: int, random_state: int = 42) -> pd.DataFrame:
    if budget_rows >= len(df):
        return df.copy()
    sampled = (
        df.sample(n=budget_rows, random_state=random_state)
        .sort_values(["study_set", "recording_name", "sorter_name", "unit_id"])
        .reset_index(drop=True)
    )
    return sampled


def validate_disjoint_recordings(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    return set(make_recording_key(train_df)).isdisjoint(set(make_recording_key(test_df)))
