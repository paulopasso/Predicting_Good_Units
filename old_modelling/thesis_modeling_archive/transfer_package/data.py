from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit


def _infer_dataset_type(study_set: str) -> str:
    s = str(study_set)
    if s.startswith("HYBRID_"):
        return "hybrid"
    if s.startswith("PAIRED_"):
        return "paired"
    if s.startswith("SYNTH_"):
        return "synth"
    return "other"


class QCDataLoader:
    """Load a QC parquet and add derived key columns."""

    @staticmethod
    def load(path: str | Path) -> pd.DataFrame:
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
            frame["recording_key"] = pd.Series(
                np.arange(len(frame)), index=frame.index
            ).astype(str)

        if "sorter_name" in frame.columns:
            frame["sorter_recording_key"] = (
                frame["recording_key"] + "::" + frame["sorter_name"].astype(str)
            )
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
    """Clean a QC DataFrame: drop duplicate perfect matches, add fmiss_extended,
    and scan for dataset-type-encoding features."""

    SOFT_CONTEXT_EXCLUSIONS: set[str] = {
        "rec_n_channels",
        "rec_sampling_rate",
        "rec_duration_sec",
    }

    def __init__(self, perfect_accuracy_threshold: float = 1.0 - 1e-12):
        self.perfect_accuracy_threshold = perfect_accuracy_threshold

    # ------------------------------------------------------------------
    def clean(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Return (cleaned_df, report).

        Steps:
        1. Drop duplicate perfect matches (same GT key, multiple sorters).
        2. Add fmiss_extended.
        3. Run dataset-type leakage scan (stored in report).
        """
        cleaned, audit, drop_summary = self._drop_duplicate_perfect_matches(df)
        cleaned = self._add_fmiss_extended(cleaned)
        leakage_scan = self.dataset_type_leakage_scan(cleaned)
        report = {
            "drop_summary": drop_summary,
            "audit": audit,
            "leakage_scan": leakage_scan,
        }
        return cleaned, report

    # ------------------------------------------------------------------
    def leakage_summary(self, df: pd.DataFrame) -> dict:
        """Count duplicate row_uid, multi-sorter GT keys, perfect-match GT keys."""
        summary: dict = {
            "rows": int(len(df)),
            "duplicate_row_uid": (
                int(df["row_uid"].duplicated().sum())
                if "row_uid" in df.columns
                else None
            ),
            "duplicate_recording_sorter_unit": (
                int(df[["recording_key", "sorter_name", "unit_id"]].duplicated().sum())
                if {"recording_key", "sorter_name", "unit_id"}.issubset(df.columns)
                else None
            ),
        }
        matched = df[df["gt_key"].notna()].copy() if "gt_key" in df.columns else pd.DataFrame()
        if len(matched):
            gt_counts = matched.groupby("gt_key").size()
            sorter_counts = matched.groupby("gt_key")["sorter_name"].nunique()
            perfect_mask = (
                pd.to_numeric(matched.get("accuracy"), errors="coerce").fillna(-1)
                >= self.perfect_accuracy_threshold
            )
            perfect_multi = (
                matched.loc[perfect_mask]
                .groupby("gt_key")["sorter_name"]
                .nunique()
            )
            summary["matched_gt_keys_total"] = int(len(gt_counts))
            summary["matched_gt_keys_multi_row"] = int((gt_counts > 1).sum())
            summary["matched_gt_keys_multi_sorter"] = int((sorter_counts > 1).sum())
            summary["max_sorters_per_gt_key"] = int(sorter_counts.max())
            summary["perfect_accuracy_gt_multi_sorter"] = (
                int((perfect_multi > 1).sum()) if len(perfect_multi) else 0
            )
        return summary

    # ------------------------------------------------------------------
    def dataset_type_leakage_scan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kruskal-Wallis test of rec_* features vs dataset_type.

        Returns a DataFrame with columns:
          feature, stat, p_value, effect_size, action
        where action is 'flag' (p<0.001 or eta²>0.1) else 'ok'.
        """
        if "dataset_type" not in df.columns:
            return pd.DataFrame(
                columns=["feature", "stat", "p_value", "effect_size", "action"]
            )

        rec_cols = [
            c
            for c in df.columns
            if c.startswith("rec_") and pd.api.types.is_numeric_dtype(df[c])
        ]
        groups = df["dataset_type"].values
        rows = []
        for col in rec_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            tmp = pd.DataFrame({"v": vals, "g": groups}).dropna()
            if tmp.empty or tmp["g"].nunique() < 2:
                continue
            group_arrays = [g["v"].values for _, g in tmp.groupby("g")]
            try:
                res = stats.kruskal(*group_arrays)
                stat, p = float(res.statistic), float(res.pvalue)
            except Exception:
                continue
            # eta² = (H - k + 1) / (n - k) where H=kruskal stat, k=n_groups
            k = len(group_arrays)
            n = len(tmp)
            eta2 = max(0.0, (stat - k + 1) / (n - k)) if n > k else 0.0
            action = "flag" if (p < 0.001 or eta2 > 0.1) else "ok"
            rows.append(
                {
                    "feature": col,
                    "stat": round(stat, 3),
                    "p_value": float(p),
                    "effect_size": round(eta2, 4),
                    "action": action,
                }
            )
        if not rows:
            return pd.DataFrame(columns=["feature", "stat", "p_value", "effect_size", "action"])
        return pd.DataFrame(rows).sort_values("effect_size", ascending=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _add_fmiss_extended(df: pd.DataFrame) -> pd.DataFrame:
        """Add fmiss_extended: fmiss for matched, 1.0 for unmatched units with spikes."""
        if "fmiss" not in df.columns:
            return df
        result = df.copy()
        fmiss_num = pd.to_numeric(df["fmiss"], errors="coerce")
        matched = df["gt_key"].notna() if "gt_key" in df.columns else pd.Series(False, index=df.index)
        # Unmatched sorter units with spikes: num_tested > 0 and not matched
        has_spikes = (
            pd.to_numeric(df.get("num_tested", pd.Series(np.nan, index=df.index)), errors="coerce").fillna(0) > 0
        )
        unmatched_with_spikes = (~matched) & has_spikes
        result["fmiss_extended"] = fmiss_num.where(matched, pd.NA)
        result.loc[unmatched_with_spikes, "fmiss_extended"] = 1.0
        return result

    # ------------------------------------------------------------------
    def _drop_duplicate_perfect_matches(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        required = {"recording_key", "matched_gt_unit_id", "sorter_name", "accuracy"}
        if not required.issubset(frame.columns):
            summary = pd.DataFrame(
                [{"rows_before": len(frame), "rows_after": len(frame), "rows_dropped": 0, "affected_gt_groups": 0}]
            )
            return frame.copy(), pd.DataFrame(), summary

        work = frame.copy()
        feature_cols = [
            c
            for c in work.columns
            if pd.api.types.is_numeric_dtype(work[c]) and c not in {"accuracy", "fpos", "fmiss"}
        ]
        work["_feature_missing_fraction"] = (
            work[feature_cols].isna().mean(axis=1) if feature_cols else 0.0
        )
        work["_accuracy_num"] = pd.to_numeric(work["accuracy"], errors="coerce")
        work["_fpos_num"] = pd.to_numeric(work.get("fpos", 0), errors="coerce")
        work["_fmiss_num"] = pd.to_numeric(work.get("fmiss", 0), errors="coerce")
        work["_row_uid_safe"] = (
            work["row_uid"].astype(str) if "row_uid" in work.columns else work.index.astype(str)
        )

        perfect_mask = (
            (work["_accuracy_num"].fillna(-1) >= self.perfect_accuracy_threshold)
            & work["matched_gt_unit_id"].notna()
        )
        candidate_groups = work.loc[perfect_mask].groupby(
            ["recording_key", "matched_gt_unit_id"], dropna=False
        )

        drop_index: list = []
        audit_rows: list = []
        for (recording_key, gt_unit), grp in candidate_groups:
            if len(grp) <= 1 or grp["sorter_name"].nunique() <= 1:
                continue
            ranked = grp.sort_values(
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
                    "n_rows_before": int(len(grp)),
                    "n_sorters_before": int(grp["sorter_name"].nunique()),
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
            [
                {
                    "rows_before": int(len(frame)),
                    "rows_after": int(len(cleaned)),
                    "rows_dropped": int(len(drop_index)),
                    "affected_gt_groups": int(len(audit_rows)),
                }
            ]
        )
        return cleaned, pd.DataFrame(audit_rows), summary


class QCDataSplitter:
    """Grouped and row-random train/test splits."""

    def __init__(self, test_size: float = 0.20, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def grouped_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_idx, test_idx = next(
            splitter.split(df, groups=df["recording_key"])
        )
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        overlap = set(train["recording_key"]) & set(test["recording_key"])
        assert not overlap, f"Grouped split recording overlap: {sorted(overlap)[:5]}"
        return train, test

    def row_random_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    def validate(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> dict:
        overlap = set(train["recording_key"]) & set(test["recording_key"])
        result: dict = {
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
