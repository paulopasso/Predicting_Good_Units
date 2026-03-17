from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from qc_thesis.modeling.core.stack import StackSet, build_stack_frame as core_build_stack_frame
from qc_thesis.modeling.neural.mmd.config import MMDDomainAdaptationConfig
from qc_thesis.modeling.neural.mmd.dataset import (
    NaNStandardScaler,
    PreparedFeatureTable,
    PreparedModelInput,
    fit_training_side_scalers,
    prepare_model_input,
    target_name_for_source,
)
from qc_thesis.modeling.neural.mmd.model import (
    batch_to_device,
    make_unit_only_regressor,
    target_inverse,
    target_transform,
)
from qc_thesis.modeling.neural.mmd.train import (
    get_device,
    score_predictions,
    set_torch_runtime,
    train_source_with_mmd,
)

import torch


@dataclass
class FittedTreeRegressor:
    backend: str
    imputer: SimpleImputer
    model: Any
    target_transform_mode: str


@dataclass
class DomainWeightResult:
    weights: np.ndarray
    summary: dict[str, float]
    classifier: Pipeline
    source_target_prob: np.ndarray | None = None
    target_target_prob: np.ndarray | None = None


@dataclass
class MMDSourceArtifact:
    model_state: dict[str, Any]
    unit_scaler: NaNStandardScaler
    history: list[dict[str, Any]]
    source_target: str
    mmd_lambda: float


@dataclass
class ClusterLatentModel:
    n_clusters: int
    scaler: StandardScaler
    centers_: np.ndarray
    scale_: float
    target_cluster_counts_: np.ndarray


@dataclass
class StudyBiasCorrectionModel:
    global_offset: float
    study_offsets: dict[str, float]
    shrinkage: float


@dataclass
class QuantileTransportFeatureMap:
    feature: str
    source_grid: np.ndarray
    target_grid: np.ndarray
    target_std: float


@dataclass
class QuantileTransportModel:
    feature_maps: dict[str, QuantileTransportFeatureMap]
    selected_features: list[str]
    quantile_points: int
    blend: float
    noise_scale: float


@dataclass
class KMeansSubgroupModel:
    feature_cols: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    centers_: np.ndarray


@dataclass
class PCASupportProjector:
    feature_cols: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: PCA
    target_raw: np.ndarray
    target_scores: np.ndarray
    neighbors: NearestNeighbors


class CORALAligner:
    def __init__(self, *, regularization: float = 1e-3) -> None:
        self.regularization = float(regularization)
        self.imputer = SimpleImputer(strategy="median")
        self.source_mean_: np.ndarray | None = None
        self.target_mean_: np.ndarray | None = None
        self.transform_: np.ndarray | None = None

    @staticmethod
    def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.clip(eigvals, 1e-9, None)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    @staticmethod
    def _matrix_inv_sqrt(matrix: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.clip(eigvals, 1e-9, None)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    def fit(self, source: pd.DataFrame, target: pd.DataFrame) -> "CORALAligner":
        Xs = self.imputer.fit_transform(source)
        Xt = self.imputer.transform(target)
        self.source_mean_ = Xs.mean(axis=0)
        self.target_mean_ = Xt.mean(axis=0)
        centered_s = Xs - self.source_mean_
        centered_t = Xt - self.target_mean_
        feat_dim = centered_s.shape[1]
        cov_s = np.cov(centered_s, rowvar=False) + (self.regularization * np.eye(feat_dim))
        cov_t = np.cov(centered_t, rowvar=False) + (self.regularization * np.eye(feat_dim))
        self.transform_ = self._matrix_inv_sqrt(cov_s) @ self._matrix_sqrt(cov_t)
        return self

    def transform_source(self, source: pd.DataFrame) -> np.ndarray:
        if self.transform_ is None or self.source_mean_ is None or self.target_mean_ is None:
            raise RuntimeError("CORALAligner must be fit before transform_source")
        Xs = self.imputer.transform(source)
        return ((Xs - self.source_mean_) @ self.transform_) + self.target_mean_

    def transform_target(self, target: pd.DataFrame) -> np.ndarray:
        return self.imputer.transform(target)


def _logit_inverse(values: np.ndarray, mode: str) -> np.ndarray:
    return np.clip(target_inverse(values, mode=mode), 0.0, 1.0)


def fit_tree_regressor(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    *,
    backend: str,
    random_state: int,
    target_transform_mode: str,
    lightgbm_estimators: int,
    lightgbm_learning_rate: float,
    lightgbm_num_leaves: int,
    xgboost_estimators: int,
    xgboost_learning_rate: float,
    xgboost_max_depth: int,
    xgboost_min_child_weight: float,
    xgboost_subsample: float,
    xgboost_colsample_bytree: float,
    sample_weight: np.ndarray | None = None,
) -> FittedTreeRegressor:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    y_train_t = target_transform(y_train, mode=target_transform_mode)

    backend_norm = backend.lower()
    if backend_norm == "lightgbm":
        model = LGBMRegressor(
            n_estimators=lightgbm_estimators,
            learning_rate=lightgbm_learning_rate,
            num_leaves=lightgbm_num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    elif backend_norm == "xgboost":
        model = XGBRegressor(
            n_estimators=xgboost_estimators,
            learning_rate=xgboost_learning_rate,
            max_depth=xgboost_max_depth,
            min_child_weight=xgboost_min_child_weight,
            subsample=xgboost_subsample,
            colsample_bytree=xgboost_colsample_bytree,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        raise KeyError(f"Unknown tree backend: {backend}")

    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train_imp, y_train_t, **fit_kwargs)
    return FittedTreeRegressor(
        backend=backend_norm,
        imputer=imputer,
        model=model,
        target_transform_mode=target_transform_mode,
    )


def predict_tree_regressor(model: FittedTreeRegressor, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    X_imp = model.imputer.transform(X)
    pred_t = np.asarray(model.model.predict(X_imp), dtype=np.float32)
    return _logit_inverse(pred_t, model.target_transform_mode)


def filter_target_rows(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    return df.loc[mask].copy(), y.loc[mask].to_numpy(dtype=float)


def grouped_source_train_val(
    hybrid_df: pd.DataFrame,
    *,
    source_target: str,
    random_state: int,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_df, _ = filter_target_rows(hybrid_df, source_target)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=random_state,
    )
    train_idx, val_idx = next(splitter.split(target_df, groups=target_df["recording_key"].astype(str)))
    train_df = target_df.iloc[train_idx].copy()
    val_df = target_df.iloc[val_idx].copy()
    overlap = set(train_df["recording_key"].astype(str)) & set(val_df["recording_key"].astype(str))
    if overlap:
        raise AssertionError(f"Grouped source split overlap: {sorted(overlap)[:5]}")
    return train_df, val_df


def compute_importance_weights(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    random_state: int,
    clip_min: float,
    clip_max: float,
    c_value: float,
) -> DomainWeightResult:
    source_X = source_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    target_X = target_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    domain_X = pd.concat([source_X, target_X], ignore_index=True)
    domain_y = np.concatenate([np.zeros(len(source_X), dtype=int), np.ones(len(target_X), dtype=int)])
    clf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    clf.fit(domain_X, domain_y)
    p_target = np.clip(clf.predict_proba(source_X)[:, 1], 1e-4, 1.0 - 1e-4)
    odds = p_target / (1.0 - p_target)
    weights = odds / max(float(np.mean(odds)), 1e-6)
    weights = np.clip(weights, clip_min, clip_max)
    summary = {
        "n_source": float(len(source_df)),
        "n_target": float(len(target_df)),
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "min": float(np.min(weights)),
        "p10": float(np.quantile(weights, 0.10)),
        "p50": float(np.quantile(weights, 0.50)),
        "p90": float(np.quantile(weights, 0.90)),
        "max": float(np.max(weights)),
    }
    target_prob = np.clip(clf.predict_proba(target_X)[:, 1], 1e-4, 1.0 - 1e-4)
    return DomainWeightResult(
        weights=weights,
        summary=summary,
        classifier=clf,
        source_target_prob=p_target.astype(np.float32),
        target_target_prob=target_prob.astype(np.float32),
    )


def compute_overlap_importance_weights(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    random_state: int,
    clip_min: float,
    clip_max: float,
    c_value: float,
    support_quantiles: tuple[float, float],
    outside_support_scale: float,
) -> DomainWeightResult:
    base = compute_importance_weights(
        source_df,
        target_df,
        feature_cols=feature_cols,
        random_state=random_state,
        clip_min=clip_min,
        clip_max=clip_max,
        c_value=c_value,
    )
    if base.source_target_prob is None or base.target_target_prob is None:
        raise RuntimeError("Importance weights must expose source/target probabilities")

    q_lo, q_hi = support_quantiles
    target_probs = np.asarray(base.target_target_prob, dtype=float)
    source_probs = np.asarray(base.source_target_prob, dtype=float)
    lower = float(np.quantile(target_probs, q_lo))
    upper = float(np.quantile(target_probs, q_hi))
    in_support = (source_probs >= lower) & (source_probs <= upper)
    support_scale = np.where(in_support, 1.0, float(outside_support_scale))
    weights = np.asarray(base.weights, dtype=float) * support_scale
    weights = np.clip(weights, clip_min, clip_max)
    weights = weights / max(float(np.mean(weights)), 1e-6)
    weights = np.clip(weights, clip_min, clip_max)
    summary = dict(base.summary)
    summary.update(
        {
            "support_lower": lower,
            "support_upper": upper,
            "source_in_support_frac": float(np.mean(in_support)),
        }
    )
    return DomainWeightResult(
        weights=weights.astype(np.float32),
        summary=summary,
        classifier=base.classifier,
        source_target_prob=base.source_target_prob,
        target_target_prob=base.target_target_prob,
    )


def build_stack_frame(
    df: pd.DataFrame,
    *,
    context_cols: list[str],
    extra_features: dict[str, np.ndarray],
) -> pd.DataFrame:
    return core_build_stack_frame(df, context_cols=context_cols, extra_features=extra_features)


def build_stack_set(
    *,
    train_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    test_df: pd.DataFrame,
    context_cols: list[str],
    train_features: pd.DataFrame | dict[str, object] | None = None,
    unlabeled_features: pd.DataFrame | dict[str, object] | None = None,
    test_features: pd.DataFrame | dict[str, object] | None = None,
) -> StackSet:
    return StackSet.from_context_frames(
        train_df=train_df,
        unlabeled_df=unlabeled_df,
        test_df=test_df,
        context_cols=context_cols,
        train_features=train_features,
        unlabeled_features=unlabeled_features,
        test_features=test_features,
    )


def make_study_indicator_frame(
    df: pd.DataFrame,
    *,
    categories: list[str] | tuple[str, ...] | None = None,
    prefix: str = "study",
) -> pd.DataFrame:
    study = df["study_set"].astype(str)
    if categories is None:
        categories = sorted(study.unique().tolist())
    cat = pd.Categorical(study, categories=list(categories))
    frame = pd.get_dummies(cat, prefix=prefix, dtype=float)
    return frame.reset_index(drop=True)


def fit_study_bias_correction(
    study_labels: pd.Series | np.ndarray,
    residuals: np.ndarray,
    *,
    shrinkage: float,
) -> StudyBiasCorrectionModel:
    labels = pd.Series(study_labels).astype(str).reset_index(drop=True)
    resid = np.asarray(residuals, dtype=float).reshape(-1)
    if len(labels) != len(resid):
        raise ValueError("study_labels and residuals must have the same length")
    global_offset = float(np.mean(resid)) if len(resid) else 0.0
    grouped = pd.DataFrame({"study_set": labels, "residual": resid}).groupby("study_set", sort=True)
    offsets: dict[str, float] = {}
    for study, group in grouped:
        count = float(len(group))
        mean_resid = float(group["residual"].mean())
        shrunk = ((count * mean_resid) + (float(shrinkage) * global_offset)) / max(count + float(shrinkage), 1e-6)
        offsets[str(study)] = shrunk
    return StudyBiasCorrectionModel(
        global_offset=global_offset,
        study_offsets=offsets,
        shrinkage=float(shrinkage),
    )


def leave_one_out_study_bias_adjustment(
    study_labels: pd.Series | np.ndarray,
    residuals: np.ndarray,
    *,
    shrinkage: float,
) -> np.ndarray:
    labels = pd.Series(study_labels).astype(str).reset_index(drop=True)
    resid = np.asarray(residuals, dtype=float).reshape(-1)
    if len(labels) != len(resid):
        raise ValueError("study_labels and residuals must have the same length")
    if len(resid) <= 1:
        return np.zeros_like(resid, dtype=float)
    total_sum = float(np.sum(resid))
    total_count = int(len(resid))
    grouped = pd.DataFrame({"study_set": labels, "residual": resid}).groupby("study_set", sort=False)
    study_sum = grouped["residual"].sum().to_dict()
    study_count = grouped["residual"].count().to_dict()
    adjustments = np.zeros(len(resid), dtype=float)
    for idx, (study, value) in enumerate(zip(labels, resid, strict=True)):
        count = int(study_count[str(study)])
        if count <= 1:
            global_excl = (total_sum - float(value)) / max(total_count - 1, 1)
            adjustments[idx] = global_excl
            continue
        study_sum_excl = float(study_sum[str(study)]) - float(value)
        study_count_excl = count - 1
        study_mean_excl = study_sum_excl / max(study_count_excl, 1)
        global_excl = (total_sum - float(value)) / max(total_count - 1, 1)
        adjustments[idx] = (
            (study_count_excl * study_mean_excl) + (float(shrinkage) * global_excl)
        ) / max(study_count_excl + float(shrinkage), 1e-6)
    return adjustments


def apply_study_bias_correction(
    base_pred: np.ndarray,
    study_labels: pd.Series | np.ndarray,
    model: StudyBiasCorrectionModel,
) -> np.ndarray:
    pred = np.asarray(base_pred, dtype=float).reshape(-1)
    labels = pd.Series(study_labels).astype(str).reset_index(drop=True)
    if len(labels) != len(pred):
        raise ValueError("study_labels and base_pred must have the same length")
    offsets = np.array(
        [model.study_offsets.get(str(study), model.global_offset) for study in labels],
        dtype=float,
    )
    return np.clip(pred + offsets, 0.0, 1.0)


def fit_latent_cluster_model(
    latent: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
) -> ClusterLatentModel:
    arr = np.asarray(latent, dtype=float)
    if arr.ndim != 2:
        raise ValueError("latent must be a 2D array")
    if len(arr) == 0:
        raise ValueError("latent must contain at least one row")
    n_clusters_eff = min(max(2, int(n_clusters)), len(arr))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(arr)
    kmeans = KMeans(
        n_clusters=n_clusters_eff,
        n_init=20,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(scaled)
    centers = kmeans.cluster_centers_.astype(np.float32)
    distances_sq = ((scaled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    positive = distances_sq[distances_sq > 1e-9]
    scale = float(np.median(positive)) if positive.size else 1.0
    counts = np.bincount(labels, minlength=n_clusters_eff).astype(int)
    return ClusterLatentModel(
        n_clusters=n_clusters_eff,
        scaler=scaler,
        centers_=centers,
        scale_=max(scale, 1e-6),
        target_cluster_counts_=counts,
    )


def latent_cluster_responsibilities(
    latent: np.ndarray,
    cluster_model: ClusterLatentModel,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    arr = np.asarray(latent, dtype=float)
    if arr.ndim != 2:
        raise ValueError("latent must be a 2D array")
    scaled = cluster_model.scaler.transform(arr)
    distances_sq = ((scaled[:, None, :] - cluster_model.centers_[None, :, :]) ** 2).sum(axis=2)
    scale = max(float(cluster_model.scale_) * max(float(temperature), 1e-6), 1e-6)
    logits = -distances_sq / scale
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights_sum = np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    return (weights / weights_sum).astype(np.float32)


def is_shape_feature(column: str) -> bool:
    name = str(column)
    return name.startswith("wf_bin_") or name.startswith("acg_")


def mask_feature_columns(
    df: pd.DataFrame,
    drop_cols: list[str] | tuple[str, ...] | set[str],
) -> pd.DataFrame:
    cols = [col for col in drop_cols if col in df.columns]
    if not cols:
        return df.copy()
    out = df.copy()
    for column in cols:
        out[column] = np.nan
    return out


def select_shift_filtered_features(
    shift_df: pd.DataFrame,
    *,
    allowed_cols: list[str],
    top_n: int,
    min_abs_smd: float,
    preserve_shape_features: bool,
    feature_set: str,
) -> tuple[pd.DataFrame, list[str]]:
    allowed = set(str(col) for col in allowed_cols)
    working = shift_df[shift_df["feature"].astype(str).isin(allowed)].copy()
    if working.empty:
        manifest = pd.DataFrame(
            columns=[
                "feature_set",
                "feature",
                "feature_group",
                "abs_smd",
                "ks_stat",
                "missing_gap",
                "is_shape_feature",
                "protected_reason",
                "selected_for_drop",
            ]
        )
        return manifest, []

    working["feature_set"] = feature_set
    working["is_shape_feature"] = working["feature"].astype(str).map(is_shape_feature)
    working["protected_reason"] = np.where(
        working["is_shape_feature"] & preserve_shape_features,
        "preserve_shape_feature",
        "",
    )
    candidates = working[
        (working["abs_smd"].astype(float) >= float(min_abs_smd))
        & (~working["is_shape_feature"] | (not preserve_shape_features))
    ].copy()
    candidates = candidates.sort_values(["abs_smd", "ks_stat", "feature"], ascending=[False, False, True])
    selected = set(candidates.head(int(top_n))["feature"].astype(str).tolist())
    working["selected_for_drop"] = working["feature"].astype(str).isin(selected)
    manifest = working.loc[
        :,
        [
            "feature_set",
            "feature",
            "feature_group",
            "abs_smd",
            "ks_stat",
            "missing_gap",
            "is_shape_feature",
            "protected_reason",
            "selected_for_drop",
        ],
    ].sort_values(["selected_for_drop", "abs_smd", "feature"], ascending=[False, False, True]).reset_index(drop=True)
    return manifest, sorted(selected)


def select_shift_protected_features(
    shift_df: pd.DataFrame,
    *,
    train_df: pd.DataFrame,
    target: str,
    allowed_cols: list[str],
    top_n: int,
    min_abs_smd: float,
    preserve_shape_features: bool,
    protect_top_k: int,
    min_abs_corr_to_protect: float,
    feature_set: str,
) -> tuple[pd.DataFrame, list[str]]:
    manifest, _ = select_shift_filtered_features(
        shift_df,
        allowed_cols=allowed_cols,
        top_n=top_n,
        min_abs_smd=min_abs_smd,
        preserve_shape_features=preserve_shape_features,
        feature_set=feature_set,
    )
    if manifest.empty:
        manifest["target"] = target
        manifest["filter_strategy"] = "target_protected_shift_filter"
        manifest["paired_abs_corr"] = np.nan
        return manifest, []

    feature_df = train_df.loc[:, [col for col in allowed_cols if col in train_df.columns]].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(train_df[target], errors="coerce")
    corr_rows: list[dict[str, float | str]] = []
    for feature in feature_df.columns:
        mask = y.notna() & feature_df[feature].notna()
        if int(mask.sum()) < 8:
            corr = 0.0
        else:
            x_vals = feature_df.loc[mask, feature].to_numpy(dtype=float)
            y_vals = y.loc[mask].to_numpy(dtype=float)
            if float(np.nanstd(x_vals)) <= 1e-10 or float(np.nanstd(y_vals)) <= 1e-10:
                corr = 0.0
            else:
                corr = float(abs(np.corrcoef(x_vals, y_vals)[0, 1]))
                if not np.isfinite(corr):
                    corr = 0.0
        corr_rows.append({"feature": str(feature), "paired_abs_corr": corr})
    corr_df = pd.DataFrame(corr_rows)
    working = manifest.merge(corr_df, on="feature", how="left")
    working["paired_abs_corr"] = working["paired_abs_corr"].fillna(0.0)
    protectable = working[
        (~working["is_shape_feature"] | (not preserve_shape_features))
        & (working["paired_abs_corr"].astype(float) >= float(min_abs_corr_to_protect))
    ].copy()
    protectable = protectable.sort_values(
        ["paired_abs_corr", "abs_smd", "feature"],
        ascending=[False, False, True],
    )
    protected = set(protectable.head(int(protect_top_k))["feature"].astype(str).tolist())

    reasons = working["protected_reason"].astype(str).replace("nan", "")
    new_reasons: list[str] = []
    for feature, existing in zip(working["feature"].astype(str), reasons):
        reason = existing.strip()
        if feature in protected:
            reason = "preserve_target_predictive_feature" if not reason else f"{reason}|preserve_target_predictive_feature"
        new_reasons.append(reason)
    working["protected_reason"] = new_reasons

    candidates = working[
        (working["abs_smd"].astype(float) >= float(min_abs_smd))
        & (~working["feature"].astype(str).isin(protected))
        & (~working["is_shape_feature"] | (not preserve_shape_features))
    ].copy()
    candidates = candidates.sort_values(["abs_smd", "ks_stat", "feature"], ascending=[False, False, True])
    selected = set(candidates.head(int(top_n))["feature"].astype(str).tolist())
    working["selected_for_drop"] = working["feature"].astype(str).isin(selected)
    working["target"] = target
    working["filter_strategy"] = "target_protected_shift_filter"
    working = working.sort_values(["selected_for_drop", "abs_smd", "feature"], ascending=[False, False, True]).reset_index(drop=True)
    return working, sorted(selected)


def fit_quantile_transport_model(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    shift_df: pd.DataFrame | None = None,
    top_n: int | None = None,
    min_abs_smd: float = 0.0,
    quantile_points: int = 129,
    blend: float = 1.0,
    noise_scale: float = 0.0,
    min_feature_rows: int = 64,
) -> QuantileTransportModel:
    allowed = [str(col) for col in feature_cols if col in source_df.columns and col in target_df.columns]
    if shift_df is not None:
        working = shift_df[shift_df["feature"].astype(str).isin(allowed)].copy()
        working = working[working["abs_smd"].astype(float) >= float(min_abs_smd)].copy()
        working = working.sort_values(["abs_smd", "ks_stat", "feature"], ascending=[False, False, True])
        if top_n is not None:
            selected_features = working.head(int(top_n))["feature"].astype(str).tolist()
        else:
            selected_features = working["feature"].astype(str).tolist()
    else:
        selected_features = list(allowed if top_n is None else allowed[: int(top_n)])

    q = np.linspace(0.0, 1.0, int(max(9, quantile_points)))
    feature_maps: dict[str, QuantileTransportFeatureMap] = {}
    for feature in selected_features:
        source_vals = pd.to_numeric(source_df[feature], errors="coerce").to_numpy(dtype=float)
        target_vals = pd.to_numeric(target_df[feature], errors="coerce").to_numpy(dtype=float)
        source_vals = source_vals[np.isfinite(source_vals)]
        target_vals = target_vals[np.isfinite(target_vals)]
        if len(source_vals) < int(min_feature_rows) or len(target_vals) < int(min_feature_rows):
            continue
        source_grid = np.quantile(source_vals, q)
        target_grid = np.quantile(target_vals, q)
        source_grid = np.maximum.accumulate(source_grid.astype(float))
        target_grid = np.maximum.accumulate(target_grid.astype(float))
        feature_maps[str(feature)] = QuantileTransportFeatureMap(
            feature=str(feature),
            source_grid=source_grid.astype(np.float32),
            target_grid=target_grid.astype(np.float32),
            target_std=float(np.std(target_vals)),
        )
    return QuantileTransportModel(
        feature_maps=feature_maps,
        selected_features=sorted(feature_maps.keys()),
        quantile_points=int(len(q)),
        blend=float(blend),
        noise_scale=float(noise_scale),
    )


def apply_quantile_transport_model(
    df: pd.DataFrame,
    model: QuantileTransportModel,
    *,
    random_state: int,
) -> pd.DataFrame:
    if not model.feature_maps:
        return df.copy()
    rng = np.random.default_rng(int(random_state))
    out = df.copy()
    for feature, fmap in model.feature_maps.items():
        values = pd.to_numeric(out[feature], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values)
        if not np.any(mask):
            continue
        transported = np.interp(values[mask], fmap.source_grid, fmap.target_grid)
        if float(model.blend) < 1.0:
            transported = ((1.0 - float(model.blend)) * values[mask]) + (float(model.blend) * transported)
        if float(model.noise_scale) > 0.0 and fmap.target_std > 0.0:
            transported = transported + (rng.normal(0.0, float(model.noise_scale), size=transported.shape) * fmap.target_std)
        values[mask] = transported
        out[feature] = values
    return out


def fit_kmeans_subgroup_model(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    n_clusters: int,
    random_state: int,
) -> KMeansSubgroupModel:
    feature_cols = [str(col) for col in feature_cols if col in df.columns]
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one present column")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce"))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_clusters_eff = min(max(2, int(n_clusters)), len(X_scaled))
    kmeans = KMeans(n_clusters=n_clusters_eff, n_init=20, random_state=random_state)
    kmeans.fit(X_scaled)
    return KMeansSubgroupModel(
        feature_cols=feature_cols,
        imputer=imputer,
        scaler=scaler,
        centers_=kmeans.cluster_centers_.astype(np.float32),
    )


def assign_kmeans_subgroups(
    df: pd.DataFrame,
    subgroup_model: KMeansSubgroupModel,
) -> np.ndarray:
    X = subgroup_model.imputer.transform(df.loc[:, subgroup_model.feature_cols].apply(pd.to_numeric, errors="coerce"))
    X_scaled = subgroup_model.scaler.transform(X)
    distances_sq = ((X_scaled[:, None, :] - subgroup_model.centers_[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances_sq, axis=1).astype(int)


def select_target_transport_features(
    shift_df: pd.DataFrame,
    *,
    train_df: pd.DataFrame,
    target: str,
    allowed_cols: list[str],
    top_k: int,
    min_abs_smd: float,
    min_abs_corr: float,
) -> pd.DataFrame:
    allowed = set(str(col) for col in allowed_cols if col in train_df.columns)
    working = shift_df[shift_df["feature"].astype(str).isin(allowed)].copy()
    working = working[working["abs_smd"].astype(float) >= float(min_abs_smd)].copy()
    y = pd.to_numeric(train_df[target], errors="coerce")
    corr_rows: list[dict[str, Any]] = []
    for feature in working["feature"].astype(str).tolist():
        x = pd.to_numeric(train_df[feature], errors="coerce")
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 8:
            corr = 0.0
        else:
            x_vals = x.loc[mask].to_numpy(dtype=float)
            y_vals = y.loc[mask].to_numpy(dtype=float)
            if float(np.nanstd(x_vals)) <= 1e-10 or float(np.nanstd(y_vals)) <= 1e-10:
                corr = 0.0
            else:
                corr = float(abs(np.corrcoef(x_vals, y_vals)[0, 1]))
                if not np.isfinite(corr):
                    corr = 0.0
        corr_rows.append({"feature": feature, "paired_abs_corr": corr})
    corr_df = pd.DataFrame(corr_rows)
    scored = working.merge(corr_df, on="feature", how="left")
    scored["paired_abs_corr"] = scored["paired_abs_corr"].fillna(0.0)
    scored["corr_gate"] = scored["paired_abs_corr"].astype(float) >= float(min_abs_corr)
    scored["transport_score"] = scored["abs_smd"].astype(float) * np.maximum(scored["paired_abs_corr"].astype(float), float(min_abs_corr))
    scored = scored.sort_values(
        ["corr_gate", "transport_score", "abs_smd", "feature"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    scored["selected_for_transport"] = False
    if not scored.empty:
        selected_idx = scored.head(int(top_k)).index
        scored.loc[selected_idx, "selected_for_transport"] = True
    return scored


def fit_pca_support_projector(
    target_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    n_components: int,
    n_neighbors: int,
) -> PCASupportProjector:
    feature_cols = [str(col) for col in feature_cols if col in target_df.columns]
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one present column")
    imputer = SimpleImputer(strategy="median")
    raw = imputer.fit_transform(target_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce"))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    n_components_eff = min(max(2, int(n_components)), scaled.shape[1], max(2, len(scaled) - 1))
    pca = PCA(n_components=n_components_eff, random_state=0)
    scores = pca.fit_transform(scaled)
    neighbors = NearestNeighbors(n_neighbors=min(max(2, int(n_neighbors)), len(scores))).fit(scores)
    return PCASupportProjector(
        feature_cols=feature_cols,
        imputer=imputer,
        scaler=scaler,
        pca=pca,
        target_raw=raw.astype(np.float32),
        target_scores=scores.astype(np.float32),
        neighbors=neighbors,
    )


def apply_pca_support_projector(
    df: pd.DataFrame,
    projector: PCASupportProjector,
    *,
    blend: float,
    noise_scale: float,
    random_state: int,
) -> pd.DataFrame:
    out = df.copy()
    raw = projector.imputer.transform(df.loc[:, projector.feature_cols].apply(pd.to_numeric, errors="coerce"))
    scaled = projector.scaler.transform(raw)
    scores = projector.pca.transform(scaled)
    distances, neighbor_idx = projector.neighbors.kneighbors(scores, return_distance=True)
    inv_dist = 1.0 / np.clip(distances, 1e-6, None)
    weight = inv_dist / np.clip(inv_dist.sum(axis=1, keepdims=True), 1e-8, None)
    neighbor_mean = (weight[..., None] * projector.target_raw[neighbor_idx]).sum(axis=1)
    blended = ((1.0 - float(blend)) * raw) + (float(blend) * neighbor_mean)
    if float(noise_scale) > 0.0:
        rng = np.random.default_rng(int(random_state))
        local_std = projector.target_raw[neighbor_idx].std(axis=1)
        blended = blended + (rng.normal(0.0, float(noise_scale), size=blended.shape) * local_std)
    out.loc[:, projector.feature_cols] = blended
    return out


def compute_soft_group_balance_weights(
    responsibilities: np.ndarray,
    *,
    reference_mass: np.ndarray | None = None,
    power: float = 1.0,
) -> np.ndarray:
    resp = np.asarray(responsibilities, dtype=float)
    if resp.ndim != 2:
        raise ValueError("responsibilities must be a 2D array")
    if len(resp) == 0:
        return np.array([], dtype=np.float32)
    if reference_mass is None:
        mass = resp.sum(axis=0)
    else:
        mass = np.asarray(reference_mass, dtype=float).reshape(-1)
    mass = np.clip(mass, 1e-6, None)
    inverse_mass = 1.0 / np.power(mass, float(power))
    weights = resp @ inverse_mass
    weights = weights / max(float(np.mean(weights)), 1e-6)
    return weights.astype(np.float32)


def make_local_regression_mixup(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    *,
    subgroup_labels: pd.Series | np.ndarray | None = None,
    n_synthetic: int,
    k_neighbors: int,
    alpha_range: tuple[float, float],
    random_state: int,
    noise_scale: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = feature_frame.reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    y = np.asarray(target, dtype=float).reshape(-1)
    if len(frame) != len(y):
        raise ValueError("feature_frame and target must have the same length")
    if len(frame) < 2 or int(n_synthetic) <= 0:
        return frame.iloc[0:0].copy(), np.array([], dtype=float)

    labels = None
    if subgroup_labels is not None:
        labels = pd.Series(subgroup_labels).astype(str).reset_index(drop=True)
        if len(labels) != len(frame):
            raise ValueError("subgroup_labels must match feature_frame length")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(frame)
    col_scale = np.nanstd(X, axis=0, ddof=0)
    col_scale = np.where(np.isfinite(col_scale) & (col_scale > 1e-8), col_scale, 1.0)
    X_scaled = (X - np.nanmean(X, axis=0)) / col_scale

    rng = np.random.default_rng(int(random_state))
    alpha_lo, alpha_hi = alpha_range
    synth_rows: list[np.ndarray] = []
    synth_y: list[float] = []
    global_nbrs = NearestNeighbors(n_neighbors=min(max(2, int(k_neighbors) + 1), len(frame))).fit(X_scaled)

    subgroup_cache: dict[str, tuple[np.ndarray, NearestNeighbors]] = {}
    if labels is not None:
        for subgroup, idx in labels.groupby(labels).groups.items():
            indices = np.asarray(sorted(idx), dtype=int)
            if len(indices) < 2:
                continue
            n_neighbors_eff = min(max(2, int(k_neighbors) + 1), len(indices))
            subgroup_cache[str(subgroup)] = (
                indices,
                NearestNeighbors(n_neighbors=n_neighbors_eff).fit(X_scaled[indices]),
            )

    for _ in range(int(n_synthetic)):
        i = int(rng.integers(0, len(frame)))
        subgroup = str(labels.iloc[i]) if labels is not None else None
        if subgroup is not None and subgroup in subgroup_cache:
            subgroup_idx, subgroup_nbrs = subgroup_cache[subgroup]
            local_pos = int(np.where(subgroup_idx == i)[0][0]) if i in set(subgroup_idx.tolist()) else None
            distances, neighbors = subgroup_nbrs.kneighbors(X_scaled[i : i + 1], return_distance=True)
            candidates = subgroup_idx[neighbors[0]]
            candidates = candidates[candidates != i]
        else:
            distances, neighbors = global_nbrs.kneighbors(X_scaled[i : i + 1], return_distance=True)
            candidates = neighbors[0]
            candidates = candidates[candidates != i]
        if len(candidates) == 0:
            continue
        j = int(rng.choice(candidates))
        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        row = (alpha * X[i]) + ((1.0 - alpha) * X[j])
        if noise_scale > 0.0:
            row = row + rng.normal(0.0, noise_scale, size=row.shape) * col_scale
        y_row = float(np.clip((alpha * y[i]) + ((1.0 - alpha) * y[j]), 0.0, 1.0))
        synth_rows.append(row.astype(float))
        synth_y.append(y_row)

    if not synth_rows:
        return frame.iloc[0:0].copy(), np.array([], dtype=float)
    synth_df = pd.DataFrame(np.vstack(synth_rows), columns=frame.columns)
    return synth_df, np.asarray(synth_y, dtype=float)


def make_latent_summary_frame(
    latent: np.ndarray,
    *,
    prefix: str = "latent_summary",
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> pd.DataFrame:
    arr = np.asarray(latent, dtype=float)
    if arr.ndim != 2:
        raise ValueError("latent must be a 2D array")
    if arr.shape[1] == 0:
        arr = np.zeros((arr.shape[0], 1), dtype=float)
    summary = {
        f"{prefix}_mean": arr.mean(axis=1),
        f"{prefix}_std": arr.std(axis=1),
        f"{prefix}_min": arr.min(axis=1),
        f"{prefix}_max": arr.max(axis=1),
        f"{prefix}_l2": np.linalg.norm(arr, axis=1),
    }
    for quantile in quantiles:
        q_label = int(round(float(quantile) * 100.0))
        summary[f"{prefix}_q{q_label:02d}"] = np.quantile(arr, quantile, axis=1)
    return pd.DataFrame(summary)


def rank_latent_dims_by_correlation(
    latent: np.ndarray,
    target: np.ndarray,
    *,
    groups: np.ndarray,
    n_splits: int,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    arr = np.asarray(latent, dtype=float)
    y = np.asarray(target, dtype=float)
    if arr.ndim != 2:
        raise ValueError("latent must be a 2D array")
    if len(arr) != len(y):
        raise ValueError("latent and target must have the same number of rows")

    splitter = GroupShuffleSplit(
        n_splits=max(1, n_splits),
        test_size=test_size,
        random_state=random_state,
    )
    scores: list[np.ndarray] = []
    for train_idx, _ in splitter.split(arr, groups=groups):
        x_train = arr[train_idx]
        y_train = y[train_idx]
        fold_scores = np.zeros(arr.shape[1], dtype=float)
        y_center = y_train - np.nanmean(y_train)
        y_denom = float(np.sqrt(np.nansum(y_center ** 2)))
        for dim in range(arr.shape[1]):
            x = x_train[:, dim]
            x_center = x - np.nanmean(x)
            x_denom = float(np.sqrt(np.nansum(x_center ** 2)))
            if x_denom <= 1e-8 or y_denom <= 1e-8:
                fold_scores[dim] = 0.0
                continue
            corr = float(np.nansum(x_center * y_center) / (x_denom * y_denom))
            fold_scores[dim] = abs(corr)
        scores.append(fold_scores)

    score_matrix = np.vstack(scores) if scores else np.zeros((1, arr.shape[1]), dtype=float)
    mean_scores = score_matrix.mean(axis=0)
    rank_df = pd.DataFrame(
        {
            "latent_dim": np.arange(arr.shape[1], dtype=int),
            "mean_abs_corr": mean_scores,
        }
    )
    rank_df = rank_df.sort_values(["mean_abs_corr", "latent_dim"], ascending=[False, True]).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1, dtype=int)
    return rank_df


def pick_r2_first_recommendation(
    metrics_df: pd.DataFrame,
    *,
    target: str,
    reference_variant_id: str,
    selection_metric: str,
    mae_guardrail_pct: float,
    min_r2_improvement: float,
) -> dict[str, Any]:
    target_df = metrics_df[metrics_df["target"] == target].copy()
    if target_df.empty:
        raise ValueError(f"No metric rows available for target={target}")
    if reference_variant_id not in set(target_df["variant_id"].astype(str)):
        raise KeyError(f"Reference variant {reference_variant_id} missing for target={target}")

    reference = target_df.loc[target_df["variant_id"].astype(str) == reference_variant_id].iloc[0]
    reference_mae = float(reference["mae"])
    reference_r2 = float(reference["r2"])
    mae_limit = reference_mae * (1.0 + float(mae_guardrail_pct))

    eligible = target_df[target_df["mae"].astype(float) <= mae_limit].copy()
    metric = selection_metric.lower()
    if metric == "r2":
        eligible = eligible.sort_values(["r2", "mae", "variant_id"], ascending=[False, True, True])
        best = eligible.iloc[0]
        improves = float(best["r2"]) >= (reference_r2 + float(min_r2_improvement))
    else:
        eligible = eligible.sort_values(["mae", "r2", "variant_id"], ascending=[True, False, True])
        best = eligible.iloc[0]
        improves = float(best["mae"]) <= reference_mae

    recommended = best if improves else reference
    recommended_variant = str(recommended["variant_id"])
    recommended_mae = float(recommended["mae"])
    recommended_r2 = float(recommended["r2"])
    return {
        "target": target,
        "selection_metric": metric,
        "reference_variant_id": reference_variant_id,
        "reference_mae": reference_mae,
        "reference_r2": reference_r2,
        "mae_guardrail_limit": mae_limit,
        "recommended_variant_id": recommended_variant,
        "recommended_mae": recommended_mae,
        "recommended_r2": recommended_r2,
        "r2_improvement": recommended_r2 - reference_r2,
        "mae_regression_pct": ((recommended_mae / reference_mae) - 1.0) if reference_mae > 0 else 0.0,
        "passes_guardrail": bool(recommended_mae <= mae_limit),
        "beats_reference": bool(recommended_variant != reference_variant_id),
    }


def prepare_unit_input(
    df: pd.DataFrame,
    prepared: PreparedFeatureTable,
    unit_scaler: NaNStandardScaler,
) -> PreparedModelInput:
    return prepare_model_input(
        df,
        prepared,
        unit_scaler,
        None,
        use_context=False,
    )


def train_zero_shot_mmd_source(
    *,
    target: str,
    hybrid_source: pd.DataFrame,
    paired_non_test_all: pd.DataFrame,
    ssl_pool: pd.DataFrame,
    prepared: PreparedFeatureTable,
    random_state: int,
    source_fmiss_target: str,
    mmd_lambda: float,
    batch_size: int,
    hidden_dim: int,
    latent_dim: int,
    source_train_epochs: int,
    source_lr: float,
    weight_decay: float,
    patience: int,
    verbose: bool,
    checkpoint_path: Any = None,
) -> MMDSourceArtifact:
    source_target = target_name_for_source(target, source_fmiss_target)
    source_train_df, source_val_df = grouped_source_train_val(
        hybrid_source,
        source_target=source_target,
        random_state=random_state,
        val_fraction=0.15,
    )
    train_side_df = pd.concat([hybrid_source, paired_non_test_all], ignore_index=True)
    unit_scaler, _ = fit_training_side_scalers(train_side_df, prepared)
    source_train_input = prepare_unit_input(source_train_df, prepared, unit_scaler)
    source_val_input = prepare_unit_input(source_val_df, prepared, unit_scaler)
    target_unlabeled_input = prepare_unit_input(ssl_pool, prepared, unit_scaler)
    source_train_y = pd.to_numeric(source_train_df[source_target], errors="coerce").to_numpy(dtype=float)
    source_val_y = pd.to_numeric(source_val_df[source_target], errors="coerce").to_numpy(dtype=float)

    mmd_config = MMDDomainAdaptationConfig(
        parquet_path=Path("."),
        output_root=Path("."),
        random_state=random_state,
        source_fmiss_target=source_fmiss_target,
        lambda_grid=(float(mmd_lambda),),
        run_finetune_stage=False,
        source_train_epochs=source_train_epochs,
        finetune_epochs=1,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        source_lr=source_lr,
        finetune_lr=source_lr,
        weight_decay=weight_decay,
        patience=patience,
        verbose=verbose,
    )
    result = train_source_with_mmd(
        source_train_data=source_train_input,
        source_train_target=source_train_y,
        source_val_data=source_val_input,
        source_val_target=source_val_y,
        target_unlabeled_data=target_unlabeled_input,
        mmd_lambda=float(mmd_lambda),
        config=mmd_config,
        target_name=target,
        checkpoint_path=checkpoint_path,
        progress_prefix=f"workbench:{target}:lambda={mmd_lambda:.3f}:source",
    )
    return MMDSourceArtifact(
        model_state=result.model_state,
        unit_scaler=unit_scaler,
        history=result.history,
        source_target=source_target,
        mmd_lambda=float(mmd_lambda),
    )


def predict_embeddings_from_state(
    *,
    model_state: dict[str, Any],
    eval_data: PreparedModelInput,
    hidden_dim: int,
    latent_dim: int,
    random_state: int,
    target_transform_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    set_torch_runtime(random_state)
    device = get_device()
    n_wf = int(eval_data.wf.shape[1])
    n_acg = int(eval_data.acg.shape[1])
    n_scalar = int(eval_data.scalars.shape[1])
    model = make_unit_only_regressor(
        n_wf_features=n_wf,
        n_acg_features=n_acg,
        n_scalar_features=n_scalar,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        encoder_state=None,
        device=device,
    )
    model.load_state_dict(model_state)
    model.eval()
    batch = batch_to_device(
        wf=eval_data.wf,
        acg=eval_data.acg,
        scalars=eval_data.scalars,
        wf_mask=eval_data.wf_mask,
        acg_mask=eval_data.acg_mask,
        scalar_mask=eval_data.scalar_mask,
        context=None,
        context_mask=None,
        device=device,
    )
    with torch.no_grad():
        pred_t, latent = model(batch.wf, batch.acg, batch.scalars)
    pred = _logit_inverse(pred_t.detach().cpu().numpy(), target_transform_mode)
    return pred, latent.detach().cpu().numpy().astype(np.float32)


def make_metric_row(
    *,
    target: str,
    variant_id: str,
    family: str,
    backend: str,
    source_target: str | None,
    metrics: dict[str, float],
    notes: str,
) -> dict[str, Any]:
    return {
        "protocol": "main",
        "target": target,
        "variant_id": variant_id,
        "family": family,
        "backend": backend,
        "source_target": source_target,
        "notes": notes,
        **metrics,
    }


def make_prediction_frame(
    *,
    target: str,
    variant_id: str,
    family: str,
    backend: str,
    df: pd.DataFrame,
    y_true: np.ndarray,
    pred: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protocol": "main",
            "target": target,
            "variant_id": variant_id,
            "family": family,
            "backend": backend,
            "row_uid": df["row_uid"].astype(str).to_numpy(),
            "recording_key": df["recording_key"].astype(str).to_numpy(),
            "study_set": df["study_set"].astype(str).to_numpy(),
            "true": y_true,
            "pred": pred,
            "abs_error": np.abs(pred - y_true),
        }
    )
