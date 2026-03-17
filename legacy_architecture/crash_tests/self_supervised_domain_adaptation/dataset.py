from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qc_framework.data import QCDataCleaner, QCDataLoader
from qc_framework.features import (
    FeatureSetBuilder,
    FeatureSpec,
    RecordingContextFeatureAugmenter,
    RecordingRelativeFeatureAugmenter,
)


@dataclass
class PreparedFeatureTable:
    df: pd.DataFrame
    spec: FeatureSpec
    hybrid: pd.DataFrame
    paired: pd.DataFrame
    paired_matched: pd.DataFrame
    paired_unmatched: pd.DataFrame
    unit_feature_cols: list[str]
    context_feature_cols: list[str]
    context_feature_cols_by_target: dict[str, list[str]]
    lightgbm_context_cols: list[str]
    shape_feature_cols: list[str]
    wf_feature_cols: list[str]
    acg_feature_cols: list[str]
    scalar_feature_cols: list[str]


@dataclass
class NaNStandardScaler:
    columns: list[str]
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "NaNStandardScaler":
        values = df.loc[:, self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(values)
        counts = valid.sum(axis=0).astype(float)
        safe_counts = np.where(counts > 0.0, counts, 1.0)
        filled = np.where(valid, values, 0.0)
        mean = filled.sum(axis=0) / safe_counts
        centered = np.where(valid, values - mean, 0.0)
        scale = np.sqrt((centered**2).sum(axis=0) / safe_counts)
        mean = np.where(counts > 0.0, mean, 0.0)
        scale = np.where(counts > 0.0, scale, 1.0)
        scale[scale == 0.0] = 1.0
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("NaNStandardScaler must be fit before transform")
        values = df.loc[:, self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return (values - self.mean_) / self.scale_


@dataclass
class PreparedModelInput:
    row_uid: np.ndarray
    recording_key: np.ndarray
    study_set: np.ndarray
    wf: np.ndarray
    acg: np.ndarray
    scalars: np.ndarray
    wf_mask: np.ndarray
    acg_mask: np.ndarray
    scalar_mask: np.ndarray
    context: np.ndarray | None
    context_mask: np.ndarray | None

    def __len__(self) -> int:
        return int(len(self.row_uid))

    def subset(self, index: np.ndarray | list[int]) -> "PreparedModelInput":
        idx = np.asarray(index)
        return PreparedModelInput(
            row_uid=self.row_uid[idx],
            recording_key=self.recording_key[idx],
            study_set=self.study_set[idx],
            wf=self.wf[idx],
            acg=self.acg[idx],
            scalars=self.scalars[idx],
            wf_mask=self.wf_mask[idx],
            acg_mask=self.acg_mask[idx],
            scalar_mask=self.scalar_mask[idx],
            context=None if self.context is None else self.context[idx],
            context_mask=None if self.context_mask is None else self.context_mask[idx],
        )


_CURATED_CONTEXT_BASES: dict[str, tuple[str, ...]] = {
    "fpos": (
        "snr",
        "peak_to_trough_uv",
        "amp_mean",
        "amp_p50",
        "template_norm",
        "wf_energy",
        "si_snr",
        "max_cosine",
        "mean_cosine_top3",
        "isolation_score",
        "min_neighbor_dist_um",
        "mean_neighbor_dist_um",
    ),
    "fmiss": (
        "si_amplitude_cutoff",
        "si_drift_ptp",
        "si_drift_std",
        "si_drift_mad",
        "firing_rate_hz",
        "isi_violation_rate",
        "max_cosine",
        "mean_cosine_top3",
        "n_confusable",
        "isolation_score",
        "max_waveform_cosine",
        "n_waveform_confusable",
        "amp_cv",
        "amp_drift_slope",
        "amp_drift_r2",
        "amp_outlier_pct",
    ),
}

_CURATED_CONTEXT_SUFFIXES: tuple[str, ...] = (
    "recording_pct",
    "recording_mean",
    "recording_std",
    "recording_delta",
)


def curated_context_columns(all_context_cols: list[str], *, target: str) -> list[str]:
    bases = _CURATED_CONTEXT_BASES.get(target, ())
    selected: list[str] = []
    for base in bases:
        for suffix in _CURATED_CONTEXT_SUFFIXES:
            name = f"{base}_{suffix}"
            if name in all_context_cols:
                selected.append(name)
    if selected:
        return selected
    fallback = [
        col
        for col in all_context_cols
        if col.endswith("_recording_pct") or col.endswith("_recording_mean") or col.endswith("_recording_delta")
    ]
    if fallback:
        return fallback[: min(len(fallback), 24)]
    return selected


def prepare_feature_table_from_frame(frame: pd.DataFrame) -> PreparedFeatureTable:
    cleaner = QCDataCleaner()
    cleaned, _ = cleaner.clean(frame)

    hybrid_clean = cleaned[cleaned["dataset_type"] == "hybrid"].copy()
    dup_pairs = FeatureSetBuilder.find_exact_duplicate_pairs(hybrid_clean)
    extra_drop = {b for _, b in dup_pairs}
    spec = FeatureSetBuilder().build(
        hybrid_clean,
        mode="unit_only",
        exclude_soft=False,
        extra_drop_cols=extra_drop,
    )

    relative = RecordingRelativeFeatureAugmenter()
    df = relative.fit_transform(cleaned)
    context = RecordingContextFeatureAugmenter()
    df = context.fit_transform(df)

    unit_feature_cols = [c for c in spec.numeric_cols if c in df.columns]
    shape_feature_cols = FeatureSetBuilder.transfer_feature_columns(
        df,
        spec,
        view_names=["shape_only"],
    )
    context_feature_cols = FeatureSetBuilder.transfer_feature_columns(
        df,
        spec,
        view_names=["recording_relative", "recording_context"],
    )
    context_feature_cols_by_target = {
        target: curated_context_columns(context_feature_cols, target=target)
        for target in ("fpos", "fmiss")
    }
    lightgbm_context_cols = FeatureSetBuilder.transfer_feature_columns(
        df,
        spec,
        view_names=["unit_raw", "recording_relative", "recording_context"],
    )
    wf_feature_cols = FeatureSetBuilder.waveform_bin_cols(unit_feature_cols)
    acg_feature_cols = FeatureSetBuilder.acg_cols(unit_feature_cols)
    scalar_feature_cols = [c for c in unit_feature_cols if c not in set(wf_feature_cols + acg_feature_cols)]

    paired = df[df["dataset_type"] == "paired"].copy()

    return PreparedFeatureTable(
        df=df,
        spec=spec,
        hybrid=df[df["dataset_type"] == "hybrid"].copy(),
        paired=paired,
        paired_matched=paired[paired["matched_gt_unit_id"].notna()].copy(),
        paired_unmatched=paired[paired["matched_gt_unit_id"].isna()].copy(),
        unit_feature_cols=unit_feature_cols,
        context_feature_cols=context_feature_cols,
        context_feature_cols_by_target=context_feature_cols_by_target,
        lightgbm_context_cols=lightgbm_context_cols,
        shape_feature_cols=shape_feature_cols,
        wf_feature_cols=wf_feature_cols,
        acg_feature_cols=acg_feature_cols,
        scalar_feature_cols=scalar_feature_cols,
    )


def prepare_feature_table(parquet_path: str | Path) -> PreparedFeatureTable:
    loader = QCDataLoader()
    frame = loader.load(parquet_path)
    return prepare_feature_table_from_frame(frame)


def fit_training_side_scalers(
    train_side_df: pd.DataFrame,
    prepared: PreparedFeatureTable,
    *,
    context_columns: list[str] | None = None,
) -> tuple[NaNStandardScaler, NaNStandardScaler | None]:
    unit_scaler = NaNStandardScaler(prepared.unit_feature_cols).fit(train_side_df)
    context_scaler = None
    selected_context_cols = prepared.context_feature_cols if context_columns is None else context_columns
    if selected_context_cols:
        context_scaler = NaNStandardScaler(selected_context_cols).fit(train_side_df)
    return unit_scaler, context_scaler


def prepare_model_input(
    df: pd.DataFrame,
    prepared: PreparedFeatureTable,
    unit_scaler: NaNStandardScaler,
    context_scaler: NaNStandardScaler | None,
    *,
    use_context: bool,
    context_columns: list[str] | None = None,
) -> PreparedModelInput:
    unit_values = unit_scaler.transform(df)
    unit_mask = np.isfinite(unit_values)
    unit_filled = np.nan_to_num(unit_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    wf_idx = [prepared.unit_feature_cols.index(c) for c in prepared.wf_feature_cols]
    acg_idx = [prepared.unit_feature_cols.index(c) for c in prepared.acg_feature_cols]
    scalar_idx = [prepared.unit_feature_cols.index(c) for c in prepared.scalar_feature_cols]

    context_values: np.ndarray | None = None
    context_mask: np.ndarray | None = None
    if use_context:
        if context_scaler is None:
            raise RuntimeError("Context scaler is required when use_context=True")
        context_values = context_scaler.transform(df)
        context_mask = np.isfinite(context_values)
        context_values = np.nan_to_num(context_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return PreparedModelInput(
        row_uid=df["row_uid"].astype(str).to_numpy(),
        recording_key=df["recording_key"].astype(str).to_numpy(),
        study_set=df["study_set"].astype(str).to_numpy(),
        wf=unit_filled[:, wf_idx].astype(np.float32),
        acg=unit_filled[:, acg_idx].astype(np.float32),
        scalars=unit_filled[:, scalar_idx].astype(np.float32) if scalar_idx else np.zeros((len(df), 1), dtype=np.float32),
        wf_mask=unit_mask[:, wf_idx],
        acg_mask=unit_mask[:, acg_idx],
        scalar_mask=unit_mask[:, scalar_idx] if scalar_idx else np.ones((len(df), 1), dtype=bool),
        context=context_values,
        context_mask=context_mask,
    )


def numeric_target(df: pd.DataFrame, target: str) -> np.ndarray:
    return pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)


def target_name_for_source(eval_target: str, source_fmiss_target: str) -> str:
    return source_fmiss_target if eval_target == "fmiss" else eval_target
