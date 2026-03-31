from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_dataset_bundle
from .features import get_feature_view


@dataclass(frozen=True)
class DomainShiftBundle:
    domain_summary: pd.DataFrame
    feature_shift: pd.DataFrame
    feature_group_shift: pd.DataFrame
    pca_2d: pd.DataFrame
    pca_3d: pd.DataFrame


def _dataset_for_domain(with_context: bool = True) -> pd.DataFrame:
    bundle = build_dataset_bundle(with_context=with_context)
    df = bundle.full_df.copy()
    df["domain_label"] = (df["dataset_kind"] == "paired").astype(int)
    return df


def _cross_validated_auc(df: pd.DataFrame, feature_cols: list[str]) -> float:
    X = df[feature_cols]
    y = df["domain_label"].to_numpy()
    groups = df["recording_key"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    preds = np.zeros(len(df), dtype=float)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, n_jobs=None)),
        ])
        model.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return float(roc_auc_score(y, preds))


def _feature_shift_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    paired = df[df["dataset_kind"] == "paired"]
    hybrid = df[df["dataset_kind"] == "hybrid"]
    rows = []
    for col in feature_cols:
        p = paired[col].astype(float)
        h = hybrid[col].astype(float)
        pooled = np.nanstd(np.concatenate([p.fillna(np.nanmedian(p)), h.fillna(np.nanmedian(h))]))
        if not np.isfinite(pooled) or pooled == 0:
            pooled = 1.0
        rows.append({
            "feature": col,
            "paired_mean": float(np.nanmean(p)),
            "hybrid_mean": float(np.nanmean(h)),
            "abs_smd": float(abs(np.nanmean(p) - np.nanmean(h)) / pooled),
            "feature_group": infer_feature_group(col),
        })
    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)


def infer_feature_group(feature: str) -> str:
    if feature.startswith("wf_bin_") or feature.startswith("wf_"):
        return "waveform"
    if feature.startswith("acg_"):
        return "acg"
    if feature.startswith("amp_"):
        return "amplitude"
    if feature.startswith("si_") or feature.startswith("isi_"):
        return "spikeinterface_qc"
    if feature.endswith("_recording_mean") or feature.endswith("_recording_std") or feature.endswith("_recording_delta") or feature.endswith("_recording_pct"):
        return "recording_context"
    if feature in {"template_norm", "peak_channel_depth_um", "spread_um", "center_of_mass_um", "n_active_channels", "n_channels"}:
        return "spatial_template"
    return "other"


def _feature_group_summary(feature_shift: pd.DataFrame) -> pd.DataFrame:
    return (
        feature_shift.groupby("feature_group", as_index=False)
        .agg(feature_count=("feature", "size"), mean_abs_smd=("abs_smd", "mean"), max_abs_smd=("abs_smd", "max"))
        .sort_values("mean_abs_smd", ascending=False)
        .reset_index(drop=True)
    )


def _compute_pca(df: pd.DataFrame, feature_cols: list[str], n_components: int) -> pd.DataFrame:
    transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])
    scores = transformer.fit_transform(df[feature_cols])
    out = df[["study_set", "recording_name", "recording_key", "dataset_kind"]].copy()
    out["pc1"] = scores[:, 0]
    out["pc2"] = scores[:, 1]
    if n_components >= 3:
        out["pc3"] = scores[:, 2]
    out["dataset_type"] = out["dataset_kind"]
    return out


def get_domain_shift_bundle() -> DomainShiftBundle:
    df = _dataset_for_domain(with_context=True)
    unit_cols = get_feature_view(df, "unit_raw")
    context_cols = get_feature_view(df, "full_context")
    domain_summary = pd.DataFrame([
        {"comparison": "hybrid_vs_paired", "feature_set": "unit_only", "cv_auc_mean": _cross_validated_auc(df, unit_cols)},
        {"comparison": "hybrid_vs_paired", "feature_set": "full_context", "cv_auc_mean": _cross_validated_auc(df, context_cols)},
    ])
    feature_shift = _feature_shift_table(df, unit_cols)
    feature_group_shift = _feature_group_summary(feature_shift)
    pca_2d = _compute_pca(df, unit_cols, 2)
    pca_3d = _compute_pca(df, unit_cols, 3)
    return DomainShiftBundle(
        domain_summary=domain_summary,
        feature_shift=feature_shift,
        feature_group_shift=feature_group_shift,
        pca_2d=pca_2d,
        pca_3d=pca_3d,
    )
