from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import GroupShuffleSplit

from .specs import FeatureSpec


FeaturePayload = pd.DataFrame | Mapping[str, object] | None


def build_stack_frame(
    df: pd.DataFrame,
    *,
    context_cols: list[str],
    extra_features: Mapping[str, object],
) -> pd.DataFrame:
    frame = df.loc[:, context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    for name, values in extra_features.items():
        arr = np.asarray(values, dtype=float).reshape(len(df), -1)
        if arr.shape[1] == 1:
            frame[name] = arr[:, 0]
            continue
        for idx in range(arr.shape[1]):
            frame[f"{name}_{idx:02d}"] = arr[:, idx]
    return frame


def _coerce_extra_frame(extra_features: FeaturePayload, row_count: int) -> pd.DataFrame:
    if extra_features is None:
        return pd.DataFrame(index=range(row_count))
    if isinstance(extra_features, pd.DataFrame):
        return extra_features.reset_index(drop=True).copy()
    frame = pd.DataFrame(index=range(row_count))
    for name, values in extra_features.items():
        arr = np.asarray(values, dtype=float).reshape(row_count, -1)
        if arr.shape[1] == 1:
            frame[name] = arr[:, 0]
            continue
        for idx in range(arr.shape[1]):
            frame[f"{name}_{idx:02d}"] = arr[:, idx]
    return frame


def _append_extra_features(base: pd.DataFrame, extra_features: FeaturePayload) -> pd.DataFrame:
    extra = _coerce_extra_frame(extra_features, len(base))
    if extra.empty:
        return base.reset_index(drop=True).copy()
    return pd.concat([base.reset_index(drop=True), extra], axis=1)


@dataclass(frozen=True)
class StackSet:
    train: pd.DataFrame
    unlabeled: pd.DataFrame
    test: pd.DataFrame

    @classmethod
    def from_context_frames(
        cls,
        *,
        train_df: pd.DataFrame,
        unlabeled_df: pd.DataFrame,
        test_df: pd.DataFrame,
        context_cols: list[str],
        train_features: FeaturePayload = None,
        unlabeled_features: FeaturePayload = None,
        test_features: FeaturePayload = None,
    ) -> "StackSet":
        return cls(
            train=_append_extra_features(
                train_df.loc[:, context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                train_features,
            ),
            unlabeled=_append_extra_features(
                unlabeled_df.loc[:, context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                unlabeled_features,
            ),
            test=_append_extra_features(
                test_df.loc[:, context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                test_features,
            ),
        )

    def apply(self, augmenter: "StackAugmenter") -> "StackSet":
        return augmenter.apply(self)

    def with_frames(
        self,
        *,
        train: pd.DataFrame | None = None,
        unlabeled: pd.DataFrame | None = None,
        test: pd.DataFrame | None = None,
    ) -> "StackSet":
        return StackSet(
            train=train if train is not None else self.train.copy(),
            unlabeled=unlabeled if unlabeled is not None else self.unlabeled.copy(),
            test=test if test is not None else self.test.copy(),
        )


class StackAugmenter(ABC):
    @abstractmethod
    def apply(self, stack: StackSet) -> StackSet:
        raise NotImplementedError


@dataclass(frozen=True)
class FrameAugmenter(StackAugmenter):
    train_features: FeaturePayload = None
    unlabeled_features: FeaturePayload = None
    test_features: FeaturePayload = None

    def apply(self, stack: StackSet) -> StackSet:
        return StackSet(
            train=_append_extra_features(stack.train, self.train_features),
            unlabeled=_append_extra_features(stack.unlabeled, self.unlabeled_features),
            test=_append_extra_features(stack.test, self.test_features),
        )


def _infer_dataset_type(study_set: str) -> str:
    name = str(study_set)
    if name.startswith("HYBRID_"):
        return "hybrid"
    if name.startswith("PAIRED_"):
        return "paired"
    if name.startswith("SYNTH_"):
        return "synth"
    return "other"


class QCDataLoader:
    @staticmethod
    def load(path: str | "Path") -> pd.DataFrame:
        frame = pd.read_parquet(path).copy()
        if "study_set" in frame.columns:
            frame["dataset_type"] = frame["study_set"].map(_infer_dataset_type)
        else:
            frame["dataset_type"] = "unknown"

        if {"study_set", "study_name", "recording_name"}.issubset(frame.columns):
            frame["recording_key"] = (
                frame["study_set"].astype(str)
                + "::"
                + frame["study_name"].astype(str)
                + "::"
                + frame["recording_name"].astype(str)
            )
        else:
            frame["recording_key"] = pd.Series(np.arange(len(frame)), index=frame.index).astype(str)

        if "sorter_name" in frame.columns:
            frame["sorter_recording_key"] = frame["recording_key"] + "::" + frame["sorter_name"].astype(str)
        else:
            frame["sorter_recording_key"] = frame["recording_key"]

        if "matched_gt_unit_id" in frame.columns:
            frame["gt_key"] = np.where(
                frame["matched_gt_unit_id"].notna(),
                frame["recording_key"] + "::" + frame["matched_gt_unit_id"].astype(str),
                pd.NA,
            )
        else:
            frame["gt_key"] = pd.NA
        return frame


class QCDataCleaner:
    SOFT_CONTEXT_EXCLUSIONS: set[str] = {"rec_n_channels", "rec_sampling_rate", "rec_duration_sec"}

    def __init__(self, perfect_accuracy_threshold: float = 1.0 - 1e-12):
        self.perfect_accuracy_threshold = perfect_accuracy_threshold

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        cleaned, audit, drop_summary = self._drop_duplicate_perfect_matches(df)
        cleaned = self._add_fmiss_extended(cleaned)
        leakage_scan = self.dataset_type_leakage_scan(cleaned)
        return cleaned, {"drop_summary": drop_summary, "audit": audit, "leakage_scan": leakage_scan}

    def leakage_summary(self, df: pd.DataFrame) -> dict:
        summary: dict[str, Any] = {
            "rows": int(len(df)),
            "duplicate_row_uid": int(df["row_uid"].duplicated().sum()) if "row_uid" in df.columns else None,
            "duplicate_recording_sorter_unit": int(df[["recording_key", "sorter_name", "unit_id"]].duplicated().sum())
            if {"recording_key", "sorter_name", "unit_id"}.issubset(df.columns)
            else None,
        }
        matched = df[df["gt_key"].notna()].copy() if "gt_key" in df.columns else pd.DataFrame()
        if len(matched):
            gt_counts = matched.groupby("gt_key").size()
            sorter_counts = matched.groupby("gt_key")["sorter_name"].nunique()
            perfect_mask = pd.to_numeric(matched.get("accuracy"), errors="coerce").fillna(-1) >= self.perfect_accuracy_threshold
            perfect_multi = matched.loc[perfect_mask].groupby("gt_key")["sorter_name"].nunique()
            summary["matched_gt_keys_total"] = int(len(gt_counts))
            summary["matched_gt_keys_multi_row"] = int((gt_counts > 1).sum())
            summary["matched_gt_keys_multi_sorter"] = int((sorter_counts > 1).sum())
            summary["max_sorters_per_gt_key"] = int(sorter_counts.max())
            summary["perfect_accuracy_gt_multi_sorter"] = int((perfect_multi > 1).sum()) if len(perfect_multi) else 0
        return summary

    def dataset_type_leakage_scan(self, df: pd.DataFrame) -> pd.DataFrame:
        if "dataset_type" not in df.columns:
            return pd.DataFrame(columns=["feature", "stat", "p_value", "effect_size", "action"])
        rec_cols = [col for col in df.columns if col.startswith("rec_") and pd.api.types.is_numeric_dtype(df[col])]
        groups = df["dataset_type"].values
        rows = []
        for col in rec_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            tmp = pd.DataFrame({"v": vals, "g": groups}).dropna()
            if tmp.empty or tmp["g"].nunique() < 2:
                continue
            group_arrays = [group["v"].values for _, group in tmp.groupby("g")]
            try:
                result = stats.kruskal(*group_arrays)
                stat, p_value = float(result.statistic), float(result.pvalue)
            except Exception:
                continue
            k = len(group_arrays)
            n = len(tmp)
            eta2 = max(0.0, (stat - k + 1) / (n - k)) if n > k else 0.0
            rows.append(
                {
                    "feature": col,
                    "stat": round(stat, 3),
                    "p_value": float(p_value),
                    "effect_size": round(eta2, 4),
                    "action": "flag" if (p_value < 0.001 or eta2 > 0.1) else "ok",
                }
            )
        if not rows:
            return pd.DataFrame(columns=["feature", "stat", "p_value", "effect_size", "action"])
        return pd.DataFrame(rows).sort_values("effect_size", ascending=False)

    @staticmethod
    def _add_fmiss_extended(df: pd.DataFrame) -> pd.DataFrame:
        if "fmiss" not in df.columns:
            return df
        result = df.copy()
        fmiss_num = pd.to_numeric(df["fmiss"], errors="coerce")
        matched = df["gt_key"].notna() if "gt_key" in df.columns else pd.Series(False, index=df.index)
        has_spikes = pd.to_numeric(df.get("num_tested", pd.Series(np.nan, index=df.index)), errors="coerce").fillna(0) > 0
        unmatched_with_spikes = (~matched) & has_spikes
        result["fmiss_extended"] = fmiss_num.where(matched, pd.NA)
        result.loc[unmatched_with_spikes, "fmiss_extended"] = 1.0
        return result

    def _drop_duplicate_perfect_matches(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        required = {"recording_key", "matched_gt_unit_id", "sorter_name", "accuracy"}
        if not required.issubset(frame.columns):
            summary = pd.DataFrame([{"rows_before": len(frame), "rows_after": len(frame), "rows_dropped": 0, "affected_gt_groups": 0}])
            return frame.copy(), pd.DataFrame(), summary
        work = frame.copy()
        feature_cols = [col for col in work.columns if pd.api.types.is_numeric_dtype(work[col]) and col not in {"accuracy", "fpos", "fmiss"}]
        work["_feature_missing_fraction"] = work[feature_cols].isna().mean(axis=1) if feature_cols else 0.0
        work["_accuracy_num"] = pd.to_numeric(work["accuracy"], errors="coerce")
        work["_fpos_num"] = pd.to_numeric(work.get("fpos", 0), errors="coerce")
        work["_fmiss_num"] = pd.to_numeric(work.get("fmiss", 0), errors="coerce")
        work["_row_uid_safe"] = work["row_uid"].astype(str) if "row_uid" in work.columns else work.index.astype(str)

        perfect_mask = (work["_accuracy_num"].fillna(-1) >= self.perfect_accuracy_threshold) & work["matched_gt_unit_id"].notna()
        candidate_groups = work.loc[perfect_mask].groupby(["recording_key", "matched_gt_unit_id"], dropna=False)

        drop_index: list[Any] = []
        audit_rows: list[dict[str, Any]] = []
        for (recording_key, gt_unit), group in candidate_groups:
            if len(group) <= 1 or group["sorter_name"].nunique() <= 1:
                continue
            ranked = group.sort_values(
                by=["_fpos_num", "_fmiss_num", "_feature_missing_fraction", "sorter_name", "_row_uid_safe"],
                ascending=[True, True, True, True, True],
            )
            keep_idx = ranked.index[0]
            keep_row = ranked.loc[keep_idx]
            drop_rows = ranked.loc[ranked.index != keep_idx]
            drop_index.extend(drop_rows.index.tolist())
            audit_rows.append(
                {
                    "recording_key": recording_key,
                    "matched_gt_unit_id": str(gt_unit),
                    "n_rows_before": int(len(group)),
                    "n_sorters_before": int(group["sorter_name"].nunique()),
                    "kept_row_uid": keep_row["_row_uid_safe"],
                    "kept_sorter_name": keep_row.get("sorter_name"),
                    "dropped_row_uid": ", ".join(drop_rows["_row_uid_safe"].astype(str).tolist()),
                    "dropped_sorter_name": ", ".join(drop_rows["sorter_name"].astype(str).tolist()),
                }
            )

        cleaned = work.drop(index=drop_index).drop(
            columns=["_feature_missing_fraction", "_accuracy_num", "_fpos_num", "_fmiss_num", "_row_uid_safe"],
            errors="ignore",
        )
        summary = pd.DataFrame(
            [{
                "rows_before": int(len(frame)),
                "rows_after": int(len(cleaned)),
                "rows_dropped": int(len(drop_index)),
                "affected_gt_groups": int(len(audit_rows)),
            }]
        )
        return cleaned, pd.DataFrame(audit_rows), summary


class QCDataSplitter:
    def __init__(self, test_size: float = 0.20, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def grouped_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_idx, test_idx = next(splitter.split(df, groups=df["recording_key"]))
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        overlap = set(train["recording_key"]) & set(test["recording_key"])
        assert not overlap, f"Grouped split recording overlap: {sorted(overlap)[:5]}"
        return train, test

    def row_random_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=self.test_size, random_state=self.random_state)
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    def validate(self, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        overlap = set(train["recording_key"]) & set(test["recording_key"])
        result: dict[str, Any] = {
            "recording_overlap": sorted(overlap),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_recordings": int(train["recording_key"].nunique()),
            "test_recordings": int(test["recording_key"].nunique()),
        }
        if "dataset_type" in train.columns:
            result["train_dataset_types"] = train["dataset_type"].value_counts().to_dict()
            result["test_dataset_types"] = test["dataset_type"].value_counts().to_dict()
        return result


@dataclass(frozen=True)
class MainExperimentSplit:
    hybrid_source: pd.DataFrame
    paired_finetune: pd.DataFrame
    paired_test: pd.DataFrame
    paired_non_test_all: pd.DataFrame
    ssl_pool: pd.DataFrame
    holdout_recordings: list[str]
    manifest: dict[str, Any]


@dataclass(frozen=True)
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
    splitter = GroupShuffleSplit(n_splits=n_candidates, test_size=test_size, random_state=random_state)
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
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + offset)
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


def feature_group_for_column(column: str) -> str:
    name = str(column)
    if name.startswith("wf_bin_"):
        return "waveform_bins"
    if name.startswith("acg_"):
        return "acg_bins"
    if any(
        suffix in name
        for suffix in (
            "_recording_mean",
            "_recording_std",
            "_recording_median",
            "_recording_iqr",
            "_recording_delta",
        )
    ):
        return "recording_context"
    if name.endswith("_recording_pct"):
        return "recording_relative"
    if name.startswith("amp_"):
        return "amplitude"
    if name.startswith("si_"):
        return "spikeinterface"
    if "cosine" in name or "confusable" in name or "neighbor_dist" in name or "isolation" in name:
        return "confusability"
    if name.startswith("rec_"):
        return "recording_metadata"
    return "other"


def _numeric_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _safe_smd(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) == 0 or len(values_b) == 0:
        return float("nan")
    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    var_a = float(np.var(values_a))
    var_b = float(np.var(values_b))
    pooled = np.sqrt(max((var_a + var_b) / 2.0, 1e-12))
    return float((mean_b - mean_a) / pooled)


def compute_feature_shift_table(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    feature_cols: list[str],
    comparison: str,
    feature_set: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in feature_cols:
        a = _numeric_series(df_a, column)
        b = _numeric_series(df_b, column)
        missing_a = 1.0 - (len(a) / max(len(df_a), 1))
        missing_b = 1.0 - (len(b) / max(len(df_b), 1))
        ks_stat = float("nan")
        wasser = float("nan")
        if len(a) > 0 and len(b) > 0:
            ks_stat = float(ks_2samp(a, b).statistic)
            wasser = float(wasserstein_distance(a, b))
        smd = _safe_smd(a, b)
        rows.append(
            {
                "comparison": comparison,
                "feature_set": feature_set,
                "feature": column,
                "feature_group": feature_group_for_column(column),
                "rows_a": int(len(df_a)),
                "rows_b": int(len(df_b)),
                "mean_a": float(np.mean(a)) if len(a) else float("nan"),
                "mean_b": float(np.mean(b)) if len(b) else float("nan"),
                "median_a": float(np.median(a)) if len(a) else float("nan"),
                "median_b": float(np.median(b)) if len(b) else float("nan"),
                "std_a": float(np.std(a)) if len(a) else float("nan"),
                "std_b": float(np.std(b)) if len(b) else float("nan"),
                "missing_a": float(missing_a),
                "missing_b": float(missing_b),
                "missing_gap": float(missing_b - missing_a),
                "smd": smd,
                "abs_smd": abs(smd) if np.isfinite(smd) else float("nan"),
                "ks_stat": ks_stat,
                "wasserstein": wasser,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["comparison", "feature_set", "abs_smd", "ks_stat"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def summarize_feature_groups(feature_shift_df: pd.DataFrame) -> pd.DataFrame:
    if feature_shift_df.empty:
        return pd.DataFrame()
    grouped = (
        feature_shift_df.groupby(["comparison", "feature_set", "feature_group"], dropna=False)
        .agg(
            n_features=("feature", "size"),
            mean_abs_smd=("abs_smd", "mean"),
            median_abs_smd=("abs_smd", "median"),
            max_abs_smd=("abs_smd", "max"),
            mean_ks=("ks_stat", "mean"),
            mean_missing_gap=("missing_gap", "mean"),
        )
        .reset_index()
        .sort_values(["comparison", "feature_set", "mean_abs_smd"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return grouped


__all__ = [
    "MainExperimentSplit",
    "QCDataCleaner",
    "QCDataLoader",
    "QCDataSplitter",
    "FrameAugmenter",
    "StackAugmenter",
    "StackSet",
    "StressStudySplit",
    "build_stack_frame",
    "build_stress_splits",
    "compute_feature_shift_table",
    "feature_group_for_column",
    "make_grouped_validation_splits",
    "make_main_split",
    "summarize_feature_groups",
    "FeatureSetBuilder",
    "RecordingContextFeatureAugmenter",
    "RecordingRelativeFeatureAugmenter",
    "StudyIndicatorAugmenter",
]

# ── feature-engineering helpers absorbed from augmenters.py ──────────────────

_WF_PREFIX = "wf_bin_"
_ACG_PREFIX = "acg_"
_RELATIVE_SUFFIX = "_recording_pct"
_CONTEXT_MEAN_SUFFIX = "_recording_mean"
_CONTEXT_STD_SUFFIX = "_recording_std"
_CONTEXT_MEDIAN_SUFFIX = "_recording_median"
_CONTEXT_IQR_SUFFIX = "_recording_iqr"
_CONTEXT_CENTERED_SUFFIX = "_recording_delta"
_STUDY_INDICATOR_PREFIX = "study_indicator__"
_WF_RE = re.compile(r"^wf_bin_\d+$")
_ACG_RE = re.compile(r"^acg_\d+$")

_DEFAULT_RECORDING_RELATIVE_COLS = [
    "snr",
    "peak_to_trough_uv",
    "amp_mean",
    "amp_std",
    "amp_p50",
    "template_norm",
    "wf_energy",
    "si_snr",
    "si_amplitude_median",
    "si_amplitude_cutoff",
    "si_sd_ratio",
    "si_drift_ptp",
    "si_drift_std",
    "si_drift_mad",
    "max_cosine",
    "mean_cosine_top3",
    "mean_cosine_top5",
    "median_cosine",
    "std_cosine",
    "cosine_gap_top2",
    "n_confusable",
    "isolation_score",
    "max_waveform_cosine",
    "mean_waveform_cosine_top3",
    "median_waveform_cosine",
    "std_waveform_cosine",
    "n_waveform_confusable",
    "min_neighbor_dist_um",
    "mean_neighbor_dist_um",
]

_DEFAULT_RECORDING_CONTEXT_COLS = [
    "snr",
    "peak_to_trough_uv",
    "amp_mean",
    "amp_std",
    "amp_p50",
    "template_norm",
    "wf_energy",
    "si_snr",
    "si_amplitude_median",
    "si_amplitude_cutoff",
    "si_drift_ptp",
    "si_drift_std",
    "si_drift_mad",
    "firing_rate_hz",
    "isi_violation_rate",
    "max_cosine",
    "mean_cosine_top3",
    "median_cosine",
    "n_confusable",
    "isolation_score",
    "min_neighbor_dist_um",
    "mean_neighbor_dist_um",
]


class FeatureSetBuilder:
    ID_COLUMNS: frozenset[str] = frozenset(
        {
            "row_uid",
            "group_key",
            "recording_key",
            "sorter_recording_key",
            "recording_name",
            "unit_id",
            "matched_gt_unit_id",
            "hungarian_gt_unit_id",
            "gt_key",
        }
    )
    TARGET_DERIVED_COLUMNS: frozenset[str] = frozenset(
        {
            "accuracy",
            "agreement_score",
            "fpos",
            "fmiss",
            "fmiss_extended",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "num_gt",
            "num_tested",
            "match_status",
        }
    )
    FORBIDDEN_CONTEXT_COLUMNS: frozenset[str] = frozenset(
        {"study_set", "study_name", "recording_name", "sorter_name", "dataset_type"}
    )
    SOFT_CONTEXT_EXCLUSIONS: frozenset[str] = frozenset(
        {"rec_n_channels", "rec_sampling_rate", "rec_duration_sec"}
    )
    TRANSFER_VIEW_NAMES: frozenset[str] = frozenset(
        {
            "unit_raw",
            "norm_swap",
            "shape_only",
            "recording_relative",
            "recording_context",
            "study_identity",
        }
    )

    def build(
        self,
        df: pd.DataFrame,
        mode: str = "unit_only",
        exclude_soft: bool = False,
        extra_drop_cols: set[str] | None = None,
    ) -> FeatureSpec:
        hard_exclusions = self.ID_COLUMNS | self.TARGET_DERIVED_COLUMNS | self.FORBIDDEN_CONTEXT_COLUMNS
        extra_drop = extra_drop_cols or set()
        soft_excluded: set[str] = set()
        numeric: list[str] = []
        categorical: list[str] = []
        excluded: dict[str, str] = {}
        work = df.copy()

        for col in work.columns:
            if col in hard_exclusions:
                excluded[col] = "hard_exclusion"
                continue
            if col in extra_drop:
                excluded[col] = "exact_duplicate"
                continue
            if mode == "unit_only" and str(col).startswith("rec_"):
                excluded[col] = "recording_context_not_in_unit_only_mode"
                continue
            if exclude_soft and col in self.SOFT_CONTEXT_EXCLUSIONS:
                excluded[col] = "soft_context_exclusion"
                soft_excluded.add(col)
                continue
            if pd.api.types.is_numeric_dtype(work[col]):
                numeric.append(col)
                continue
            maybe_num = pd.to_numeric(work[col], errors="coerce")
            if maybe_num.notna().mean() > 0.95:
                work[col] = maybe_num
                numeric.append(col)
                continue
            if work[col].nunique(dropna=True) <= 20:
                categorical.append(col)
            else:
                excluded[col] = "high_cardinality_string"

        return FeatureSpec(
            mode=mode,
            numeric_cols=numeric,
            categorical_cols=categorical,
            excluded_cols=excluded,
            soft_excluded_cols=soft_excluded,
        )

    @staticmethod
    def find_exact_duplicate_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        pairs: list[tuple[str, str]] = []
        for idx, left in enumerate(numeric_cols):
            for right in numeric_cols[idx + 1 :]:
                if df[left].equals(df[right]):
                    pairs.append((left, right))
        return pairs

    @staticmethod
    def align_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
        missing = [c for c in spec.all_features if c not in df.columns]
        aligned = df.copy()
        for col in missing:
            aligned[col] = pd.NA
        return aligned[spec.all_features].copy()

    @staticmethod
    def waveform_bin_cols(columns: list[str]) -> list[str]:
        return [c for c in columns if _WF_RE.match(str(c))]

    @staticmethod
    def acg_cols(columns: list[str]) -> list[str]:
        return [c for c in columns if _ACG_RE.match(str(c))]

    @classmethod
    def shape_cols(cls, columns: list[str]) -> list[str]:
        return cls.waveform_bin_cols(columns) + cls.acg_cols(columns)

    @classmethod
    def transfer_feature_columns(
        cls,
        df: pd.DataFrame,
        spec: FeatureSpec,
        view_names: list[str] | tuple[str, ...],
        normalized_raw_cols: list[str] | tuple[str, ...] | None = None,
        normalizer_suffix: str = "_z",
        relative_suffix: str = _RELATIVE_SUFFIX,
        context_suffixes: tuple[str, ...] = (
            _CONTEXT_MEAN_SUFFIX,
            _CONTEXT_STD_SUFFIX,
            _CONTEXT_MEDIAN_SUFFIX,
            _CONTEXT_IQR_SUFFIX,
            _CONTEXT_CENTERED_SUFFIX,
        ),
        study_prefix: str = _STUDY_INDICATOR_PREFIX,
    ) -> list[str]:
        requested = list(view_names)
        unknown = sorted(set(requested) - set(cls.TRANSFER_VIEW_NAMES))
        if unknown:
            raise KeyError(f"Unknown transfer view(s): {unknown}. Available: {sorted(cls.TRANSFER_VIEW_NAMES)}")

        normalized_raw = set(normalized_raw_cols or [])
        base_cols = [c for c in spec.all_features if c in df.columns]
        selected: list[str] = []
        if "unit_raw" in requested:
            selected.extend(base_cols)
        if "norm_swap" in requested:
            selected.extend([c for c in base_cols if c not in normalized_raw])
            selected.extend([f"{c}{normalizer_suffix}" for c in normalized_raw if f"{c}{normalizer_suffix}" in df.columns])
        if "shape_only" in requested:
            selected.extend(cls.shape_cols(base_cols))
        if "recording_relative" in requested:
            selected.extend([c for c in df.columns if c.endswith(relative_suffix) and c[: -len(relative_suffix)] in df.columns])
        if "recording_context" in requested:
            selected.extend([c for c in df.columns if any(str(c).endswith(suffix) for suffix in context_suffixes)])
        if "study_identity" in requested:
            selected.extend([c for c in df.columns if str(c).startswith(study_prefix)])

        seen: set[str] = set()
        deduped: list[str] = []
        for col in selected:
            if col in df.columns and col not in seen:
                deduped.append(col)
                seen.add(col)
        return deduped


class RecordingRelativeFeatureAugmenter:
    def __init__(self, cols_to_rank: list[str] | None = None, suffix: str = _RELATIVE_SUFFIX) -> None:
        self.cols_to_rank = cols_to_rank
        self.suffix = suffix
        self.generated_cols_: list[str] = []

    def _effective_cols(self, df: pd.DataFrame) -> list[str]:
        base = self.cols_to_rank if self.cols_to_rank is not None else _DEFAULT_RECORDING_RELATIVE_COLS
        pattern_cols = [
            c
            for c in df.columns
            if str(c).startswith(("amp_", "si_amplitude_", "si_drift_"))
            or "cosine" in str(c)
            or "confusable" in str(c)
            or str(c).endswith("_neighbor_dist_um")
        ]
        merged = list(dict.fromkeys(list(base) + pattern_cols))
        return [c for c in merged if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and not str(c).endswith(self.suffix)]

    def fit(self, df: pd.DataFrame) -> "RecordingRelativeFeatureAugmenter":
        self.generated_cols_ = [f"{c}{self.suffix}" for c in self._effective_cols(df)]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "recording_key" not in df.columns:
            raise KeyError("RecordingRelativeFeatureAugmenter requires a recording_key column")
        cols = self._effective_cols(df)
        out = df.copy()
        groups = out["recording_key"].astype(str)
        for col in cols:
            vals = pd.to_numeric(out[col], errors="coerce")
            ranks = vals.groupby(groups).rank(method="average")
            counts = vals.notna().groupby(groups).transform("sum")
            pct = (ranks - 0.5) / counts.where(counts > 0)
            out[f"{col}{self.suffix}"] = pct.astype(float)
        self.generated_cols_ = [f"{c}{self.suffix}" for c in cols]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class RecordingContextFeatureAugmenter:
    def __init__(
        self,
        cols_to_summarize: list[str] | None = None,
        suffixes: tuple[str, str, str, str, str] = (
            _CONTEXT_MEAN_SUFFIX,
            _CONTEXT_STD_SUFFIX,
            _CONTEXT_MEDIAN_SUFFIX,
            _CONTEXT_IQR_SUFFIX,
            _CONTEXT_CENTERED_SUFFIX,
        ),
    ) -> None:
        self.cols_to_summarize = cols_to_summarize
        self.suffixes = suffixes
        self.generated_cols_: list[str] = []

    def _effective_cols(self, df: pd.DataFrame) -> list[str]:
        base = self.cols_to_summarize if self.cols_to_summarize is not None else _DEFAULT_RECORDING_CONTEXT_COLS
        pattern_cols = [
            c
            for c in df.columns
            if str(c).startswith(("amp_", "si_amplitude_", "si_drift_"))
            or "cosine" in str(c)
            or "confusable" in str(c)
            or str(c).endswith("_neighbor_dist_um")
        ]
        merged = list(dict.fromkeys(list(base) + pattern_cols))
        return [
            c
            for c in merged
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and not any(str(c).endswith(suffix) for suffix in self.suffixes)
        ]

    def fit(self, df: pd.DataFrame) -> "RecordingContextFeatureAugmenter":
        cols = self._effective_cols(df)
        generated: list[str] = []
        for col in cols:
            generated.extend([f"{col}{suffix}" for suffix in self.suffixes])
        self.generated_cols_ = generated
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "recording_key" not in df.columns:
            raise KeyError("RecordingContextFeatureAugmenter requires a recording_key column")
        cols = self._effective_cols(df)
        out = df.copy()
        groups = out["recording_key"].astype(str)
        generated: dict[str, pd.Series] = {}
        for col in cols:
            vals = pd.to_numeric(out[col], errors="coerce")
            mean = vals.groupby(groups).transform("mean")
            std = vals.groupby(groups).transform("std").fillna(0.0)
            median = vals.groupby(groups).transform("median")
            q25 = vals.groupby(groups).transform(lambda series: series.quantile(0.25))
            q75 = vals.groupby(groups).transform(lambda series: series.quantile(0.75))
            iqr = q75 - q25
            generated[f"{col}{self.suffixes[0]}"] = mean.astype(float)
            generated[f"{col}{self.suffixes[1]}"] = std.astype(float)
            generated[f"{col}{self.suffixes[2]}"] = median.astype(float)
            generated[f"{col}{self.suffixes[3]}"] = iqr.astype(float)
            generated[f"{col}{self.suffixes[4]}"] = (vals - median).astype(float)
        generated_df = pd.DataFrame(generated, index=out.index)
        self.generated_cols_ = list(generated_df.columns)
        return pd.concat([out, generated_df], axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class StudyIndicatorAugmenter:
    def __init__(self, source_col: str = "study_set", prefix: str = _STUDY_INDICATOR_PREFIX) -> None:
        self.source_col = source_col
        self.prefix = prefix
        self.levels_: list[str] = []
        self.generated_cols_: list[str] = []

    @staticmethod
    def _sanitize(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
        return cleaned.strip("_").lower() or "unknown"

    def fit(self, df: pd.DataFrame) -> "StudyIndicatorAugmenter":
        if self.source_col not in df.columns:
            self.levels_ = []
            self.generated_cols_ = []
            return self
        levels = df[self.source_col].fillna("unknown").astype(str).value_counts().index.tolist()
        self.levels_ = levels
        self.generated_cols_ = [f"{self.prefix}{self._sanitize(level)}" for level in levels]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.source_col not in out.columns:
            return out
        values = out[self.source_col].fillna("unknown").astype(str)
        for level, col in zip(self.levels_, self.generated_cols_):
            out[col] = (values == level).astype(float)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
