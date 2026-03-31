from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLS = {"fpos", "fmiss", "accuracy"}
IDENTITY_COLS = {
    "row_uid",
    "study_set",
    "study_name",
    "recording_name",
    "sorter_name",
    "unit_id",
    "group_key",
    "matched_gt_unit_id",
    "hungarian_gt_unit_id",
    "match_status",
    "__filename",
}
META_NUMERIC_COLS = {"__fragment_index", "__batch_index", "__last_in_fragment"}

CONTEXT_BASE_COLS = [
    "amp_mean",
    "amp_std",
    "amp_cv",
    "si_amplitude_cv_median",
    "si_amplitude_cv_range",
    "snr",
    "wf_energy",
    "template_norm",
    "n_waveform_confusable",
    "isi_violation_rate",
]


def infer_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in TARGET_COLS or col in IDENTITY_COLS or col in META_NUMERIC_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def get_waveform_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("wf_bin_")]


def get_acg_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("acg_")]


def get_unit_feature_columns(df: pd.DataFrame) -> list[str]:
    return infer_numeric_feature_columns(df)


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_key = out["study_set"].astype(str) + "::" + out["recording_name"].astype(str)
    out["recording_key"] = group_key
    for col in CONTEXT_BASE_COLS:
        if col not in out.columns:
            continue
        grp = out.groupby("recording_key")[col]
        out[f"{col}_recording_mean"] = grp.transform("mean")
        out[f"{col}_recording_std"] = grp.transform("std").fillna(0.0)
        out[f"{col}_recording_delta"] = out[col] - out[f"{col}_recording_mean"]
        out[f"{col}_recording_pct"] = grp.rank(pct=True, method="average").fillna(0.5)
    return out


def get_feature_view(df: pd.DataFrame, feature_view: str) -> list[str]:
    waveform_cols = get_waveform_columns(df)
    acg_cols = get_acg_columns(df)
    unit_cols = get_unit_feature_columns(df)
    context_cols = [col for col in df.columns if "_recording_" in col]

    if feature_view == "none":
        return []
    if feature_view == "shape_only":
        return waveform_cols + acg_cols
    if feature_view == "unit_raw":
        return unit_cols
    if feature_view == "full_context":
        return unit_cols + context_cols
    if feature_view == "reduced_latent":
        selected = [
            "template_norm",
            "wf_energy",
            "snr",
            "isi_violation_rate",
            "amp_cv",
            "si_amplitude_cv_median",
            "si_amplitude_cv_range",
        ]
        selected += [col for col in waveform_cols[:8] + acg_cols[:10] if col in df.columns]
        selected += [col for col in context_cols if any(token in col for token in ("amp_", "wf_energy", "template_norm", "isi_violation_rate"))]
        return [col for col in selected if col in df.columns]
    if feature_view == "waveform_embedding":
        selected = waveform_cols + [col for col in unit_cols if col.startswith("wf_") or col.startswith("amp_")]
        selected += [col for col in context_cols if col.startswith("wf_") or col.startswith("amp_")]
        return list(dict.fromkeys([col for col in selected if col in df.columns]))
    if feature_view == "contextual":
        return context_cols
    raise KeyError(feature_view)
