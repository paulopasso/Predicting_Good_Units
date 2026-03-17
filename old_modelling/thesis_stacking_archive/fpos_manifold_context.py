from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qc_thesis.modeling.stacking.config import MLStatisticsWorkbenchConfig
from qc_thesis.modeling.stacking.methods import (
    build_stack_frame,
    fit_tree_regressor,
    make_metric_row,
    make_prediction_frame,
    mask_feature_columns,
    pick_r2_first_recommendation,
    predict_tree_regressor,
    score_predictions,
)
from qc_thesis.modeling.stacking.pseudo_label_manifold import (
    PseudoLabelManifoldConfig,
    _build_filtered_drop_cols,
    _build_support_artifacts,
    _fit_source_teacher_predictions,
    _fit_student_holdout,
    _manifold_support_table,
    _score_student_cv,
    _select_pseudo_labels,
    _teacher_predictions,
    _teacher_frames_for_target,
)
from qc_thesis.modeling.neural.mmd.dataset import prepare_feature_table
from qc_thesis.modeling.neural.ssl.split_utils import make_main_split


@dataclass
class FPosManifoldContextConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("artifacts/modeling/stacking")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    unit_pca_components: int = 8
    context_pca_components: int = 8
    local_neighbors: int = 12
    residual_neighbors: int = 12
    max_pseudo_multiplier: float = 1.5
    max_pseudo_per_recording: int = 48
    max_pseudo_per_cluster: int = 128
    max_pseudo_per_labeled_ratio: float = 1.5
    pseudo_weight_scale: float = 0.25
    pseudo_weight_floor: float = 0.08
    support_quantile: float = 0.60
    agreement_quantile: float = 0.55
    cluster_coverage_quantile: float = 0.35
    min_labeled_cluster_rows: int = 5
    manifold_clusters: int = 4
    curriculum_blend: float = 0.35
    selection_metric: str = "r2"
    mae_guardrail_pct: float = 0.03
    min_r2_improvement: float = 0.0
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_manifold_context_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fpos manifold+context pseudo benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosManifoldContextConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _grouped_oof_splits(df: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = df["recording_key"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    effective_splits = min(max(2, n_splits), len(unique_groups))
    splitter = GroupKFold(n_splits=effective_splits)
    return [
        (np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int))
        for train_idx, val_idx in splitter.split(df, groups=groups)
    ]


def _fit_pca_features(
    train_df: pd.DataFrame,
    transform_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    n_components: int,
    prefix: str,
) -> tuple[pd.DataFrame, tuple[SimpleImputer, StandardScaler, PCA]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = imputer.fit_transform(train_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce"))
    X_train = scaler.fit_transform(X_train)
    n_comp = min(max(2, int(n_components)), X_train.shape[0] - 1, X_train.shape[1])
    n_comp = max(2, n_comp)
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X_train)
    X_eval = imputer.transform(transform_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce"))
    X_eval = scaler.transform(X_eval)
    scores = pca.transform(X_eval)
    cols = [f"{prefix}_{idx:02d}" for idx in range(scores.shape[1])]
    return pd.DataFrame(scores, columns=cols), (imputer, scaler, pca)


def _transform_pca_features(
    transform_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    model: tuple[SimpleImputer, StandardScaler, PCA],
    prefix: str,
) -> pd.DataFrame:
    imputer, scaler, pca = model
    X_eval = imputer.transform(transform_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce"))
    X_eval = scaler.transform(X_eval)
    scores = pca.transform(X_eval)
    cols = [f"{prefix}_{idx:02d}" for idx in range(scores.shape[1])]
    return pd.DataFrame(scores, columns=cols)


def _distance_weighted_knn_predict(
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    eval_features: pd.DataFrame,
    *,
    n_neighbors: int,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = imputer.fit_transform(train_features)
    X_train = scaler.fit_transform(X_train)
    X_eval = imputer.transform(eval_features)
    X_eval = scaler.transform(X_eval)
    k = min(max(2, int(n_neighbors)), len(X_train))
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X_train)
    distances, indices = neighbors.kneighbors(X_eval, return_distance=True)
    weights = 1.0 / np.clip(distances, 1e-6, None)
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-8, None)
    pred = (weights * train_target[indices]).sum(axis=1)
    return np.clip(pred.astype(float), 0.0, 1.0)


def _oof_knn_predictions(
    meta_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    *,
    n_neighbors: int,
    n_splits: int,
) -> np.ndarray:
    splits = _grouped_oof_splits(meta_df, n_splits)
    pred = np.full(len(meta_df), np.nan, dtype=float)
    for train_idx, val_idx in splits:
        pred[val_idx] = _distance_weighted_knn_predict(
            feature_frame.iloc[train_idx].reset_index(drop=True),
            target[train_idx],
            feature_frame.iloc[val_idx].reset_index(drop=True),
            n_neighbors=n_neighbors,
        )
    return pred


def _fit_tree_oof_and_eval(
    *,
    meta_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    y: np.ndarray,
    eval_frame: pd.DataFrame,
    base: MLStatisticsWorkbenchConfig,
    random_state_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    splits = _grouped_oof_splits(meta_df, base.model_selection_splits)
    oof = np.full(len(meta_df), np.nan, dtype=float)
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        model = fit_tree_regressor(
            feature_frame.iloc[train_idx],
            y[train_idx],
            backend="lightgbm",
            random_state=base.random_state + random_state_offset + split_id,
            target_transform_mode=base.resolved_target_transform(),
            lightgbm_estimators=base.lightgbm_estimators,
            lightgbm_learning_rate=base.lightgbm_learning_rate,
            lightgbm_num_leaves=base.lightgbm_num_leaves,
            xgboost_estimators=base.xgboost_estimators,
            xgboost_learning_rate=base.xgboost_learning_rate,
            xgboost_max_depth=base.xgboost_max_depth,
            xgboost_min_child_weight=base.xgboost_min_child_weight,
            xgboost_subsample=base.xgboost_subsample,
            xgboost_colsample_bytree=base.xgboost_colsample_bytree,
        )
        oof[val_idx] = predict_tree_regressor(model, feature_frame.iloc[val_idx])
    full_pred = _fit_predict(
        feature_train=feature_frame,
        y_train=y,
        feature_eval=eval_frame,
        sample_weight=None,
        base_config=base,
        random_state_offset=random_state_offset + 100,
    )
    return oof, full_pred


def _fit_predict(
    *,
    feature_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_eval: pd.DataFrame,
    sample_weight: np.ndarray | None,
    base_config: MLStatisticsWorkbenchConfig,
    random_state_offset: int,
) -> np.ndarray:
    model = fit_tree_regressor(
        feature_train,
        y_train,
        backend="lightgbm",
        random_state=base_config.random_state + random_state_offset,
        target_transform_mode=base_config.resolved_target_transform(),
        lightgbm_estimators=base_config.lightgbm_estimators,
        lightgbm_learning_rate=base_config.lightgbm_learning_rate,
        lightgbm_num_leaves=base_config.lightgbm_num_leaves,
        xgboost_estimators=base_config.xgboost_estimators,
        xgboost_learning_rate=base_config.xgboost_learning_rate,
        xgboost_max_depth=base_config.xgboost_max_depth,
        xgboost_min_child_weight=base_config.xgboost_min_child_weight,
        xgboost_subsample=base_config.xgboost_subsample,
        xgboost_colsample_bytree=base_config.xgboost_colsample_bytree,
        sample_weight=sample_weight,
    )
    return predict_tree_regressor(model, feature_eval)


def _fit_error_model(
    *,
    meta_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    abs_error: np.ndarray,
    eval_frame: pd.DataFrame,
    base: MLStatisticsWorkbenchConfig,
    random_state_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof, eval_pred = _fit_tree_oof_and_eval(
        meta_df=meta_df,
        feature_frame=feature_frame,
        y=np.clip(abs_error.astype(float), 0.0, 1.0),
        eval_frame=eval_frame,
        base=base,
        random_state_offset=random_state_offset,
    )
    return np.clip(oof, 0.0, 1.0), np.clip(eval_pred, 0.0, 1.0)


def _pseudo_weight_from_error(
    predicted_error: np.ndarray,
    support_confidence: np.ndarray,
    *,
    scale: float,
    floor: float,
) -> np.ndarray:
    err = np.asarray(predicted_error, dtype=float)
    support = np.asarray(support_confidence, dtype=float)
    denom = max(float(np.quantile(err, 0.75)), 1e-4)
    err_conf = np.exp(-err / denom)
    weight = scale * (0.65 * err_conf + 0.35 * np.clip(support, 0.0, 1.0))
    return np.clip(weight, floor, scale).astype(np.float32)


def _curriculum_refine_pseudo(
    *,
    selected: pd.DataFrame,
    student_pred: np.ndarray,
    blend: float,
) -> pd.DataFrame:
    if selected.empty:
        return selected
    refined = selected.copy()
    refined["pseudo_label"] = np.clip(
        ((1.0 - float(blend)) * refined["pseudo_label"].astype(float))
        + (float(blend) * np.asarray(student_pred, dtype=float)),
        0.0,
        1.0,
    )
    return refined


def _assemble_training_data(
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    pseudo_frame: pd.DataFrame,
    pseudo_y: np.ndarray,
    pseudo_weight: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    full_frame = pd.concat([labeled_frame.reset_index(drop=True), pseudo_frame.reset_index(drop=True)], ignore_index=True)
    full_y = np.concatenate([np.asarray(labeled_y, dtype=float), np.asarray(pseudo_y, dtype=float)])
    full_weight = np.concatenate(
        [
            np.ones(len(labeled_frame), dtype=np.float32),
            np.asarray(pseudo_weight, dtype=np.float32),
        ]
    )
    return full_frame, full_y, full_weight


def _build_readme(
    *,
    config: FPosManifoldContextConfig,
    split_manifest: dict[str, Any],
    metrics_df: pd.DataFrame,
    pseudo_summary_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    holdout_best_df: pd.DataFrame,
) -> str:
    lines = [
        "# FPos Manifold + Context Benchmark",
        "",
        "This run specializes the pseudo-label pipeline for `fpos` by combining filtered context, source-derived signals, paired-domain manifold geometry, local kNN teachers, and an explicit pseudo-error model.",
        "",
        "## Config",
        f"- local_neighbors: `{config.local_neighbors}`",
        f"- residual_neighbors: `{config.residual_neighbors}`",
        f"- unit_pca_components: `{config.unit_pca_components}`",
        f"- context_pca_components: `{config.context_pca_components}`",
        f"- max_pseudo_multiplier: `{config.max_pseudo_multiplier}`",
        f"- max_pseudo_per_labeled_ratio: `{config.max_pseudo_per_labeled_ratio}`",
        f"- pseudo_weight_scale: `{config.pseudo_weight_scale}`",
        "",
        "## Split",
        f"- paired_test_rows: `{split_manifest['paired_test_rows']}`",
        f"- paired_finetune_rows: `{split_manifest['paired_finetune_rows']}`",
        f"- ssl_pool_rows: `{split_manifest['ssl_pool_rows']}`",
        "",
        "## Holdout Results",
        "",
    ]
    for _, row in metrics_df.sort_values(["r2", "mae"], ascending=[False, True]).iterrows():
        lines.append(f"- {row['variant_id']} | MAE `{row['mae']:.4f}` | R² `{row['r2']:.4f}`")
    lines.append("")
    if not pseudo_summary_df.empty:
        row = pseudo_summary_df.iloc[0]
        lines.extend(
            [
                "## Pseudo Selection",
                f"- candidate rows: `{int(row['candidate_rows'])}`",
                f"- selected pseudo rows: `{int(row['selected_pseudo_rows'])}`",
                f"- selected recordings: `{int(row['selected_recordings'])}`",
                f"- mean pseudo weight: `{row['mean_pseudo_weight']:.4f}`",
                "",
            ]
        )
    if not recommendation_df.empty:
        row = recommendation_df.iloc[0]
        lines.extend(
            [
                "## Recommendation",
                f"- reference CV variant: `{row['reference_variant_id']}`",
                f"- recommended CV variant: `{row['recommended_variant_id']}`",
                f"- recommended holdout MAE: `{row['recommended_holdout_mae']:.4f}`",
                f"- recommended holdout R²: `{row['recommended_holdout_r2']:.4f}`",
                "",
            ]
        )
    if not holdout_best_df.empty:
        row = holdout_best_df.iloc[0]
        lines.extend(
            [
                "## Diagnostic Best Holdout",
                f"- best holdout variant: `{row['best_holdout_variant_id']}`",
                f"- best holdout MAE: `{row['best_holdout_mae']:.4f}`",
                f"- best holdout R²: `{row['best_holdout_r2']:.4f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _run_with_config(config: FPosManifoldContextConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    base = MLStatisticsWorkbenchConfig(
        parquet_path=config.parquet_path,
        output_root=config.output_root,
        run_name=config.run_name,
        random_state=config.random_state,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        verbose=config.verbose,
    )
    pseudo_base = PseudoLabelManifoldConfig(
        parquet_path=config.parquet_path,
        output_root=config.output_root,
        run_name=config.run_name,
        random_state=config.random_state,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        support_quantile=config.support_quantile,
        agreement_quantile=config.agreement_quantile,
        cluster_coverage_quantile=config.cluster_coverage_quantile,
        max_pseudo_multiplier=config.max_pseudo_multiplier,
        max_pseudo_per_recording=config.max_pseudo_per_recording,
        max_pseudo_per_cluster=config.max_pseudo_per_cluster,
        max_pseudo_per_labeled_ratio=config.max_pseudo_per_labeled_ratio,
        pseudo_weight_scale=config.pseudo_weight_scale,
        pseudo_weight_floor=config.pseudo_weight_floor,
        manifold_clusters=config.manifold_clusters,
        min_labeled_cluster_rows=config.min_labeled_cluster_rows,
        verbose=config.verbose,
    )

    prepared = prepare_feature_table(config.parquet_path)
    split = make_main_split(
        prepared.df,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        random_state=config.random_state,
    )
    unit_drop_cols, context_drop_cols, filtered_manifest = _build_filtered_drop_cols(
        split=split,
        prepared=prepared,
        base_config=base,
    )

    (
        target_train_df,
        y_train,
        target_test_df,
        y_test,
        unlabeled_df,
        teachers,
        student_templates,
        transport_manifest,
    ) = _teacher_frames_for_target(
        target="fpos",
        split=split,
        prepared=prepared,
        base=base,
        config=pseudo_base,
        unit_drop_cols=unit_drop_cols,
        context_drop_cols=context_drop_cols,
    )

    teacher_train_df, teacher_unlabeled_df, teacher_test_df, teacher_perf_df = _teacher_predictions(
        teachers=teachers,
        labeled_df=target_train_df,
        y_train=y_train,
        base=base,
    )
    teacher_perf_df["target"] = "fpos"

    filtered_drop_cols = set(unit_drop_cols) | set(context_drop_cols)
    filtered_context_cols = [col for col in prepared.lightgbm_context_cols if col not in filtered_drop_cols]
    filtered_unit_cols = [col for col in prepared.unit_feature_cols if col not in set(unit_drop_cols)]
    target_train_masked = mask_feature_columns(target_train_df, filtered_drop_cols)
    target_test_masked = mask_feature_columns(target_test_df, filtered_drop_cols)
    unlabeled_masked = mask_feature_columns(unlabeled_df, filtered_drop_cols)
    paired_non_test_masked = mask_feature_columns(split.paired_non_test_all, filtered_drop_cols)

    raw_source_pred_train = teachers[1].train_frame["source_pred"].to_numpy(dtype=float)
    raw_source_pred_unlabeled = teachers[1].unlabeled_frame["source_pred"].to_numpy(dtype=float)
    raw_source_pred_test = teachers[1].test_frame["source_pred"].to_numpy(dtype=float)
    transported_source_pred_train = teachers[2].train_frame["source_pred"].to_numpy(dtype=float)
    transported_source_pred_unlabeled = teachers[2].unlabeled_frame["source_pred"].to_numpy(dtype=float)
    transported_source_pred_test = teachers[2].test_frame["source_pred"].to_numpy(dtype=float)

    context_pca_train, context_pca_model = _fit_pca_features(
        paired_non_test_masked,
        target_train_masked,
        feature_cols=filtered_context_cols,
        n_components=config.context_pca_components,
        prefix="ctxpc",
    )
    context_pca_test = _transform_pca_features(
        target_test_masked,
        feature_cols=filtered_context_cols,
        model=context_pca_model,
        prefix="ctxpc",
    )
    context_pca_unlabeled = _transform_pca_features(
        unlabeled_masked,
        feature_cols=filtered_context_cols,
        model=context_pca_model,
        prefix="ctxpc",
    )
    context_pca_all = _transform_pca_features(
        paired_non_test_masked,
        feature_cols=filtered_context_cols,
        model=context_pca_model,
        prefix="ctxpc",
    )

    unit_pca_train, unit_pca_model = _fit_pca_features(
        paired_non_test_masked,
        target_train_masked,
        feature_cols=filtered_unit_cols,
        n_components=config.unit_pca_components,
        prefix="unitpc",
    )
    unit_pca_test = _transform_pca_features(
        target_test_masked,
        feature_cols=filtered_unit_cols,
        model=unit_pca_model,
        prefix="unitpc",
    )
    unit_pca_unlabeled = _transform_pca_features(
        unlabeled_masked,
        feature_cols=filtered_unit_cols,
        model=unit_pca_model,
        prefix="unitpc",
    )
    unit_pca_all = _transform_pca_features(
        paired_non_test_masked,
        feature_cols=filtered_unit_cols,
        model=unit_pca_model,
        prefix="unitpc",
    )

    manifold_train = pd.concat(
        [
            context_pca_train.reset_index(drop=True),
            unit_pca_train.reset_index(drop=True),
            pd.DataFrame(
                {
                    "raw_source_pred": raw_source_pred_train,
                    "transported_source_pred": transported_source_pred_train,
                    "teacher_mean": teacher_train_df["teacher_mean"].to_numpy(dtype=float),
                    "teacher_std": teacher_train_df["teacher_std"].to_numpy(dtype=float),
                }
            ),
        ],
        axis=1,
    )
    manifold_test = pd.concat(
        [
            context_pca_test.reset_index(drop=True),
            unit_pca_test.reset_index(drop=True),
            pd.DataFrame(
                {
                    "raw_source_pred": raw_source_pred_test,
                    "transported_source_pred": transported_source_pred_test,
                    "teacher_mean": teacher_test_df["teacher_mean"].to_numpy(dtype=float),
                    "teacher_std": teacher_test_df["teacher_std"].to_numpy(dtype=float),
                }
            ),
        ],
        axis=1,
    )
    manifold_unlabeled = pd.concat(
        [
            context_pca_unlabeled.reset_index(drop=True),
            unit_pca_unlabeled.reset_index(drop=True),
            pd.DataFrame(
                {
                    "raw_source_pred": raw_source_pred_unlabeled,
                    "transported_source_pred": transported_source_pred_unlabeled,
                    "teacher_mean": teacher_unlabeled_df["teacher_mean"].to_numpy(dtype=float),
                    "teacher_std": teacher_unlabeled_df["teacher_std"].to_numpy(dtype=float),
                }
            ),
        ],
        axis=1,
    )
    manifold_all = pd.concat(
        [
            context_pca_all.reset_index(drop=True),
            unit_pca_all.reset_index(drop=True),
            pd.DataFrame(
                {
                    "recording_key": split.paired_non_test_all["recording_key"].astype(str).to_numpy(),
                }
            ),
        ],
        axis=1,
    )
    support_source = pd.concat(
        [
            split.paired_non_test_all.loc[:, ["row_uid", "recording_key", "study_set"]].reset_index(drop=True),
            manifold_all.drop(columns=["recording_key"]).reset_index(drop=True),
        ],
        axis=1,
    )
    support_train = pd.concat(
        [
            target_train_df.loc[:, ["row_uid", "recording_key", "study_set"]].reset_index(drop=True),
            manifold_train.reset_index(drop=True),
        ],
        axis=1,
    )
    support_unlabeled = pd.concat(
        [
            unlabeled_df.loc[:, ["row_uid", "recording_key", "study_set"]].reset_index(drop=True),
            manifold_unlabeled.reset_index(drop=True),
        ],
        axis=1,
    )

    support_artifacts = _build_support_artifacts(
        paired_non_test_all=support_source,
        labeled_train_df=support_train,
        unit_cols=manifold_train.columns.tolist(),
        config=pseudo_base,
    )
    support_df = _manifold_support_table(
        unlabeled_df=support_unlabeled,
        labeled_train_df=support_train,
        support_artifacts=support_artifacts,
    )

    local_knn_oof = _oof_knn_predictions(
        meta_df=target_train_df,
        feature_frame=manifold_train,
        target=y_train,
        n_neighbors=config.local_neighbors,
        n_splits=base.model_selection_splits,
    )
    local_knn_unlabeled = _distance_weighted_knn_predict(
        manifold_train,
        y_train,
        manifold_unlabeled,
        n_neighbors=config.local_neighbors,
    )
    local_knn_test = _distance_weighted_knn_predict(
        manifold_train,
        y_train,
        manifold_test,
        n_neighbors=config.local_neighbors,
    )

    residual_target = y_train - teacher_train_df["teacher_mean"].to_numpy(dtype=float)
    residual_knn_oof = _oof_knn_predictions(
        meta_df=target_train_df,
        feature_frame=manifold_train,
        target=residual_target,
        n_neighbors=config.residual_neighbors,
        n_splits=base.model_selection_splits,
    )
    residual_knn_unlabeled = _distance_weighted_knn_predict(
        manifold_train,
        residual_target,
        manifold_unlabeled,
        n_neighbors=config.residual_neighbors,
    )
    residual_knn_test = _distance_weighted_knn_predict(
        manifold_train,
        residual_target,
        manifold_test,
        n_neighbors=config.residual_neighbors,
    )
    residual_corrected_oof = np.clip(
        teacher_train_df["teacher_mean"].to_numpy(dtype=float) + residual_knn_oof,
        0.0,
        1.0,
    )
    residual_corrected_unlabeled = np.clip(
        teacher_unlabeled_df["teacher_mean"].to_numpy(dtype=float) + residual_knn_unlabeled,
        0.0,
        1.0,
    )
    residual_corrected_test = np.clip(
        teacher_test_df["teacher_mean"].to_numpy(dtype=float) + residual_knn_test,
        0.0,
        1.0,
    )

    support_train_eval = _manifold_support_table(
        unlabeled_df=support_train,
        labeled_train_df=support_train,
        support_artifacts=support_artifacts,
    )

    meta_train = pd.concat(
        [
            target_train_masked.loc[:, filtered_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
            pd.DataFrame(
                {
                    "teacher_contextual": teacher_train_df[teachers[0].variant_id].to_numpy(dtype=float),
                    "teacher_source": teacher_train_df[teachers[1].variant_id].to_numpy(dtype=float),
                    "teacher_transport": teacher_train_df[teachers[2].variant_id].to_numpy(dtype=float),
                    "teacher_mean": teacher_train_df["teacher_mean"].to_numpy(dtype=float),
                    "teacher_std": teacher_train_df["teacher_std"].to_numpy(dtype=float),
                    "teacher_range": teacher_train_df["teacher_range"].to_numpy(dtype=float),
                    "local_knn": local_knn_oof,
                    "residual_knn": residual_corrected_oof,
                    "support_distance": support_train_eval["support_distance"].to_numpy(dtype=float),
                    "support_confidence": support_train_eval["support_confidence"].to_numpy(dtype=float),
                    "cluster_labeled_count": support_train_eval["cluster_labeled_count"].to_numpy(dtype=float),
                    "cluster_labeled_share": support_train_eval["cluster_labeled_share"].to_numpy(dtype=float),
                    "raw_source_pred": raw_source_pred_train,
                    "transported_source_pred": transported_source_pred_train,
                }
            ),
        ],
        axis=1,
    )
    meta_unlabeled = pd.concat(
        [
            unlabeled_masked.loc[:, filtered_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
            pd.DataFrame(
                {
                    "teacher_contextual": teacher_unlabeled_df[teachers[0].variant_id].to_numpy(dtype=float),
                    "teacher_source": teacher_unlabeled_df[teachers[1].variant_id].to_numpy(dtype=float),
                    "teacher_transport": teacher_unlabeled_df[teachers[2].variant_id].to_numpy(dtype=float),
                    "teacher_mean": teacher_unlabeled_df["teacher_mean"].to_numpy(dtype=float),
                    "teacher_std": teacher_unlabeled_df["teacher_std"].to_numpy(dtype=float),
                    "teacher_range": teacher_unlabeled_df["teacher_range"].to_numpy(dtype=float),
                    "local_knn": local_knn_unlabeled,
                    "residual_knn": residual_corrected_unlabeled,
                    "support_distance": support_df["support_distance"].to_numpy(dtype=float),
                    "support_confidence": support_df["support_confidence"].to_numpy(dtype=float),
                    "cluster_labeled_count": support_df["cluster_labeled_count"].to_numpy(dtype=float),
                    "cluster_labeled_share": support_df["cluster_labeled_share"].to_numpy(dtype=float),
                    "raw_source_pred": raw_source_pred_unlabeled,
                    "transported_source_pred": transported_source_pred_unlabeled,
                }
            ),
        ],
        axis=1,
    )
    meta_test = pd.concat(
        [
            target_test_masked.loc[:, filtered_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
            pd.DataFrame(
                {
                    "teacher_contextual": teacher_test_df[teachers[0].variant_id].to_numpy(dtype=float),
                    "teacher_source": teacher_test_df[teachers[1].variant_id].to_numpy(dtype=float),
                    "teacher_transport": teacher_test_df[teachers[2].variant_id].to_numpy(dtype=float),
                    "teacher_mean": teacher_test_df["teacher_mean"].to_numpy(dtype=float),
                    "teacher_std": teacher_test_df["teacher_std"].to_numpy(dtype=float),
                    "teacher_range": teacher_test_df["teacher_range"].to_numpy(dtype=float),
                    "local_knn": local_knn_test,
                    "residual_knn": residual_corrected_test,
                    "support_distance": np.nan,
                    "support_confidence": np.nan,
                    "cluster_labeled_count": np.nan,
                    "cluster_labeled_share": np.nan,
                    "raw_source_pred": raw_source_pred_test,
                    "transported_source_pred": transported_source_pred_test,
                }
            ),
        ],
        axis=1,
    )
    support_test = _manifold_support_table(
        unlabeled_df=pd.concat(
            [
                target_test_df.loc[:, ["row_uid", "recording_key", "study_set"]].reset_index(drop=True),
                manifold_test.reset_index(drop=True),
            ],
            axis=1,
        ),
        labeled_train_df=support_train,
        support_artifacts=support_artifacts,
    )
    meta_test.loc[:, "support_distance"] = support_test["support_distance"].to_numpy(dtype=float)
    meta_test.loc[:, "support_confidence"] = support_test["support_confidence"].to_numpy(dtype=float)
    meta_test.loc[:, "cluster_labeled_count"] = support_test["cluster_labeled_count"].to_numpy(dtype=float)
    meta_test.loc[:, "cluster_labeled_share"] = support_test["cluster_labeled_share"].to_numpy(dtype=float)

    meta_oof, meta_unlabeled_pred = _fit_tree_oof_and_eval(
        meta_df=target_train_df,
        feature_frame=meta_train,
        y=y_train,
        eval_frame=meta_unlabeled,
        base=base,
        random_state_offset=7000,
    )
    meta_test_pred = _fit_predict(
        feature_train=meta_train,
        y_train=y_train,
        feature_eval=meta_test,
        sample_weight=None,
        base_config=base,
        random_state_offset=7100,
    )

    error_train = np.abs(meta_oof - y_train)
    error_feature_train = pd.concat(
        [
            meta_train.reset_index(drop=True),
            pd.DataFrame(
                {
                    "meta_teacher_pred": meta_oof,
                    "meta_delta_vs_mean": np.abs(meta_oof - teacher_train_df["teacher_mean"].to_numpy(dtype=float)),
                }
            ),
        ],
        axis=1,
    )
    error_feature_unlabeled = pd.concat(
        [
            meta_unlabeled.reset_index(drop=True),
            pd.DataFrame(
                {
                    "meta_teacher_pred": meta_unlabeled_pred,
                    "meta_delta_vs_mean": np.abs(meta_unlabeled_pred - teacher_unlabeled_df["teacher_mean"].to_numpy(dtype=float)),
                }
            ),
        ],
        axis=1,
    )
    error_oof, error_unlabeled_pred = _fit_error_model(
        meta_df=target_train_df,
        feature_frame=error_feature_train,
        abs_error=error_train,
        eval_frame=error_feature_unlabeled,
        base=base,
        random_state_offset=8000,
    )

    candidate_df, pseudo_selected = _select_pseudo_labels(
        unlabeled_df=unlabeled_df,
        unlabeled_teacher_df=pd.DataFrame(
            {
                "teacher_mean": meta_unlabeled_pred,
                "teacher_std": np.maximum(
                    teacher_unlabeled_df["teacher_std"].to_numpy(dtype=float),
                    np.abs(meta_unlabeled_pred - teacher_unlabeled_df["teacher_mean"].to_numpy(dtype=float)),
                ),
            }
        ),
        support_df=support_df,
        config=pseudo_base,
        max_rows=int(round(len(target_train_df) * config.max_pseudo_multiplier)),
    )
    candidate_df["predicted_error"] = error_unlabeled_pred
    if not pseudo_selected.empty:
        pseudo_selected = pseudo_selected.copy()
        pseudo_selected["predicted_error"] = error_unlabeled_pred[pseudo_selected["unlabeled_row_idx"].to_numpy(dtype=int)]
        pseudo_selected["pseudo_weight"] = _pseudo_weight_from_error(
            pseudo_selected["predicted_error"].to_numpy(dtype=float),
            pseudo_selected["support_confidence"].to_numpy(dtype=float),
            scale=config.pseudo_weight_scale,
            floor=config.pseudo_weight_floor,
        )

    pseudo_idx = pseudo_selected["unlabeled_row_idx"].to_numpy(dtype=int) if not pseudo_selected.empty else np.array([], dtype=int)
    pseudo_frame = meta_unlabeled.iloc[pseudo_idx].reset_index(drop=True)
    pseudo_y = pseudo_selected["pseudo_label"].to_numpy(dtype=float) if not pseudo_selected.empty else np.array([], dtype=float)
    pseudo_weight = pseudo_selected["pseudo_weight"].to_numpy(dtype=np.float32) if not pseudo_selected.empty else np.array([], dtype=np.float32)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []

    def _record_variant(variant_id: str, family: str, notes: str, pred: np.ndarray) -> None:
        metrics = score_predictions(y_test, pred)
        metric_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family=family,
                backend="lightgbm" if family != "teacher_ensemble" else "ensemble",
                source_target="fpos",
                metrics=metrics,
                notes=notes,
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family=family,
                backend="lightgbm" if family != "teacher_ensemble" else "ensemble",
                df=target_test_df,
                y_true=y_test,
                pred=pred,
            )
        )

    # teacher baselines
    _record_variant("teacher_contextual_paired_only_lightgbm_filtered", "paired_context_teacher", teachers[0].notes, teacher_test_df[teachers[0].variant_id].to_numpy(dtype=float))
    _record_variant("teacher_source_reweighted_fullctx_lightgbm_filtered", "source_teacher", teachers[1].notes, teacher_test_df[teachers[1].variant_id].to_numpy(dtype=float))
    _record_variant("teacher_paired_conditioned_global_fullctx_lightgbm_filtered", "transport_teacher", teachers[2].notes, teacher_test_df[teachers[2].variant_id].to_numpy(dtype=float))
    _record_variant("teacher_ensemble_weighted", "teacher_ensemble", "weighted base-teacher ensemble", teacher_test_df["teacher_mean"].to_numpy(dtype=float))
    _record_variant("local_knn_teacher", "manifold_local_teacher", "local manifold kNN teacher", local_knn_test)
    _record_variant("residual_knn_teacher", "manifold_local_teacher", "teacher mean + local residual kNN correction", residual_corrected_test)
    _record_variant("manifold_context_meta_teacher", "manifold_meta_teacher", "full filtered context + manifold local teachers + support metadata", meta_test_pred)

    base_reference = "teacher_paired_conditioned_global_fullctx_lightgbm_filtered"

    cv_rows.extend(
        [
            {"target": "fpos", "variant_id": base_reference, "family": "transport_teacher", "cv_mae": float(teacher_perf_df.loc[teacher_perf_df["variant_id"] == teachers[2].variant_id, "oof_mae"].iloc[0]), "cv_r2": float(teacher_perf_df.loc[teacher_perf_df["variant_id"] == teachers[2].variant_id, "oof_r2"].iloc[0]), "selected_pseudo_rows": 0, "mean_pseudo_weight": 0.0},
            {"target": "fpos", "variant_id": "manifold_context_meta_teacher", "family": "manifold_meta_teacher", "cv_mae": float(score_predictions(y_train, meta_oof)["mae"]), "cv_r2": float(score_predictions(y_train, meta_oof)["r2"]), "selected_pseudo_rows": 0, "mean_pseudo_weight": 0.0},
        ]
    )

    if not pseudo_selected.empty:
        full_train_meta, full_y_meta, full_w_meta = _assemble_training_data(
            meta_train,
            y_train,
            pseudo_frame,
            pseudo_y,
            pseudo_weight,
        )
        meta_student_pred = _fit_predict(
            feature_train=full_train_meta,
            y_train=full_y_meta,
            feature_eval=meta_test,
            sample_weight=full_w_meta,
            base_config=base,
            random_state_offset=9000,
        )
        _record_variant("manifold_context_meta_pseudo_student", "pseudo_label_student", "meta teacher feature stack retrained on real+x pseudo rows", meta_student_pred)
        meta_student_cv = _score_student_cv(
            labeled_meta_df=target_train_df,
            labeled_frame=meta_train,
            labeled_y=y_train,
            pseudo_selected=pseudo_selected,
            unlabeled_frame=meta_unlabeled,
            base=base,
            random_state_offset=9010,
        )

        stage1_unlabeled_student = _fit_predict(
            feature_train=full_train_meta,
            y_train=full_y_meta,
            feature_eval=meta_unlabeled.iloc[pseudo_idx].reset_index(drop=True),
            sample_weight=full_w_meta,
            base_config=base,
            random_state_offset=9050,
        )
        refined_selected = _curriculum_refine_pseudo(
            selected=pseudo_selected,
            student_pred=stage1_unlabeled_student,
            blend=config.curriculum_blend,
        )
        refined_frame = meta_unlabeled.iloc[refined_selected["unlabeled_row_idx"].to_numpy(dtype=int)].reset_index(drop=True)
        refined_train_meta, refined_y_meta, refined_w_meta = _assemble_training_data(
            meta_train,
            y_train,
            refined_frame,
            refined_selected["pseudo_label"].to_numpy(dtype=float),
            refined_selected["pseudo_weight"].to_numpy(dtype=np.float32),
        )
        curriculum_pred = _fit_predict(
            feature_train=refined_train_meta,
            y_train=refined_y_meta,
            feature_eval=meta_test,
            sample_weight=refined_w_meta,
            base_config=base,
            random_state_offset=9100,
        )
        _record_variant("manifold_context_meta_curriculum_student", "pseudo_label_student", "meta teacher pseudo student with one curriculum refresh", curriculum_pred)

        source_student_train = build_stack_frame(
            target_train_masked,
            context_cols=filtered_context_cols,
            extra_features={
                "source_pred": raw_source_pred_train,
                "meta_teacher_pred": meta_oof,
                "local_knn": local_knn_oof,
            },
        )
        source_student_unlabeled = build_stack_frame(
            unlabeled_masked,
            context_cols=filtered_context_cols,
            extra_features={
                "source_pred": raw_source_pred_unlabeled,
                "meta_teacher_pred": meta_unlabeled_pred,
                "local_knn": local_knn_unlabeled,
            },
        )
        source_student_test = build_stack_frame(
            target_test_masked,
            context_cols=filtered_context_cols,
            extra_features={
                "source_pred": raw_source_pred_test,
                "meta_teacher_pred": meta_test_pred,
                "local_knn": local_knn_test,
            },
        )
        pseudo_source_frame = source_student_unlabeled.iloc[pseudo_idx].reset_index(drop=True)
        source_train_full, source_y_full, source_w_full = _assemble_training_data(
            source_student_train,
            y_train,
            pseudo_source_frame,
            pseudo_y,
            pseudo_weight,
        )
        source_student_pred = _fit_predict(
            feature_train=source_train_full,
            y_train=source_y_full,
            feature_eval=source_student_test,
            sample_weight=source_w_full,
            base_config=base,
            random_state_offset=9200,
        )
        _record_variant("source_manifold_context_pseudo_lightgbm_filtered", "pseudo_label_student", "filtered context + raw source + manifold meta teacher signal + pseudo rows", source_student_pred)
        source_student_cv = _score_student_cv(
            labeled_meta_df=target_train_df,
            labeled_frame=source_student_train,
            labeled_y=y_train,
            pseudo_selected=pseudo_selected,
            unlabeled_frame=source_student_unlabeled,
            base=base,
            random_state_offset=9210,
        )

        cv_rows.extend(
            [
                {
                    "target": "fpos",
                    "variant_id": "manifold_context_meta_pseudo_student",
                    "family": "pseudo_label_student",
                    "cv_mae": float(meta_student_cv["mae"]),
                    "cv_r2": float(meta_student_cv["r2"]),
                    "selected_pseudo_rows": int(len(pseudo_selected)),
                    "mean_pseudo_weight": float(pseudo_selected["pseudo_weight"].mean()),
                },
                {
                    "target": "fpos",
                    "variant_id": "source_manifold_context_pseudo_lightgbm_filtered",
                    "family": "pseudo_label_student",
                    "cv_mae": float(source_student_cv["mae"]),
                    "cv_r2": float(source_student_cv["r2"]),
                    "selected_pseudo_rows": int(len(pseudo_selected)),
                    "mean_pseudo_weight": float(pseudo_selected["pseudo_weight"].mean()),
                },
            ]
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    teacher_perf_df = teacher_perf_df.copy()
    teacher_perf_df["target"] = "fpos"
    pseudo_summary_df = pd.DataFrame(
        [
            {
                "target": "fpos",
                "candidate_rows": int(candidate_df["selected_candidate"].sum()),
                "selected_pseudo_rows": int(len(pseudo_selected)),
                "selected_recordings": int(pseudo_selected["recording_key"].astype(str).nunique()) if not pseudo_selected.empty else 0,
                "mean_pseudo_weight": float(pseudo_selected["pseudo_weight"].mean()) if not pseudo_selected.empty else 0.0,
                "mean_predicted_error": float(pseudo_selected["predicted_error"].mean()) if not pseudo_selected.empty else float("nan"),
            }
        ]
    )
    holdout_best = metrics_df.iloc[0]
    holdout_best_df = pd.DataFrame(
        [
            {
                "target": "fpos",
                "best_holdout_variant_id": str(holdout_best["variant_id"]),
                "best_holdout_mae": float(holdout_best["mae"]),
                "best_holdout_r2": float(holdout_best["r2"]),
            }
        ]
    )
    cv_df = pd.DataFrame(cv_rows)
    if not cv_df.empty:
        cv_df = cv_df[np.isfinite(cv_df["cv_mae"].to_numpy(dtype=float)) & np.isfinite(cv_df["cv_r2"].to_numpy(dtype=float))].reset_index(drop=True)
    if not cv_df.empty:
        cv_for_picker = cv_df.rename(columns={"cv_mae": "mae", "cv_r2": "r2"})
        recommendation = pick_r2_first_recommendation(
            cv_for_picker,
            target="fpos",
            reference_variant_id=base_reference,
            selection_metric=config.selection_metric,
            mae_guardrail_pct=config.mae_guardrail_pct,
            min_r2_improvement=config.min_r2_improvement,
        )
        recommended_holdout = metrics_df.loc[
            metrics_df["variant_id"].astype(str) == str(recommendation["recommended_variant_id"])
        ].iloc[0]
        recommendation["recommended_holdout_mae"] = float(recommended_holdout["mae"])
        recommendation["recommended_holdout_r2"] = float(recommended_holdout["r2"])
        recommendation_df = pd.DataFrame([recommendation])
    else:
        recommendation_df = pd.DataFrame()
    cluster_summary_df = pd.DataFrame(
        support_df.groupby("cluster_id", dropna=False)
        .agg(
            unlabeled_rows=("row_uid", "size"),
            mean_support_distance=("support_distance", "mean"),
            labeled_share=("cluster_labeled_share", "mean"),
        )
        .reset_index()
    )
    if not pseudo_selected.empty:
        selected_cluster_counts = pseudo_selected["cluster_id"].value_counts().to_dict()
        cluster_summary_df["selected_pseudo_rows"] = cluster_summary_df["cluster_id"].map(lambda x: int(selected_cluster_counts.get(int(x), 0)))
    else:
        cluster_summary_df["selected_pseudo_rows"] = 0

    split_manifest = {
        **split.manifest,
        "max_pseudo_multiplier": float(config.max_pseudo_multiplier),
        "max_pseudo_per_recording": int(config.max_pseudo_per_recording),
        "max_pseudo_per_cluster": int(config.max_pseudo_per_cluster),
        "max_pseudo_per_labeled_ratio": float(config.max_pseudo_per_labeled_ratio),
    }
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    teacher_perf_df.to_csv(output_dir / "teacher_agreement_summary.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    pseudo_summary_df.to_csv(output_dir / "pseudo_label_summary.csv", index=False)
    candidate_df.to_csv(output_dir / "pseudo_candidate_table.csv", index=False)
    pseudo_selected.to_csv(output_dir / "pseudo_selected_rows.csv", index=False)
    cluster_summary_df.to_csv(output_dir / "pseudo_cluster_summary.csv", index=False)
    recommendation_df.to_csv(output_dir / "per_target_recommendation.csv", index=False)
    holdout_best_df.to_csv(output_dir / "holdout_best_by_target.csv", index=False)
    filtered_manifest.to_csv(output_dir / "filtered_feature_manifest.csv", index=False)
    transport_manifest.to_csv(output_dir / "transport_shift_summary.csv", index=False)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    (output_dir / "README_run.md").write_text(
        _build_readme(
            config=config,
            split_manifest=split_manifest,
            metrics_df=metrics_df,
            pseudo_summary_df=pseudo_summary_df,
            recommendation_df=recommendation_df,
            holdout_best_df=holdout_best_df,
        )
    )
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosManifoldContextConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosManifoldContextConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.unit_pca_components = 6
        cli_config.context_pca_components = 6
        cli_config.max_pseudo_multiplier = 1.0
        cli_config.max_pseudo_per_cluster = 48
        cli_config.max_pseudo_per_recording = 20
    _run_with_config(cli_config)
