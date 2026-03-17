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
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.domain_shift_audit.report import compute_feature_shift_table
from crash_tests.ml_statistics_workbench.config import MLStatisticsWorkbenchConfig
from crash_tests.ml_statistics_workbench.methods import (
    apply_quantile_transport_model,
    assign_kmeans_subgroups,
    build_stack_frame,
    compute_importance_weights,
    filter_target_rows,
    fit_kmeans_subgroup_model,
    fit_pca_support_projector,
    fit_quantile_transport_model,
    fit_tree_regressor,
    make_metric_row,
    make_prediction_frame,
    mask_feature_columns,
    pick_r2_first_recommendation,
    predict_tree_regressor,
    score_predictions,
    select_shift_filtered_features,
)
from crash_tests.mmd_domain_adaptation.dataset import prepare_feature_table
from crash_tests.self_supervised_domain_adaptation.split_utils import make_main_split


@dataclass
class PseudoLabelManifoldConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("crash_tests/ml_statistics_workbench/outputs")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    top_shift_features: int = 40
    min_abs_smd: float = 0.8
    quantile_points: int = 129
    manifold_pca_components: int = 10
    manifold_neighbors: int = 16
    manifold_clusters: int = 4
    min_labeled_cluster_rows: int = 5
    support_quantile: float = 0.60
    agreement_quantile: float = 0.55
    cluster_coverage_quantile: float = 0.35
    max_pseudo_multiplier: float = 1.5
    max_pseudo_per_recording: int = 48
    max_pseudo_per_cluster: int = 128
    max_pseudo_per_labeled_ratio: float = 1.5
    pseudo_weight_scale: float = 0.25
    pseudo_weight_floor: float = 0.08
    teacher_transport_blend: float = 1.0
    teacher_transport_noise: float = 0.0
    selection_metric: str = "r2"
    mae_guardrail_pct: float = 0.03
    min_r2_improvement: float = 0.0
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"pseudo_label_manifold_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


@dataclass
class TeacherBundle:
    variant_id: str
    family: str
    notes: str
    backend: str
    source_target: str | None
    train_frame: pd.DataFrame
    unlabeled_frame: pd.DataFrame
    test_frame: pd.DataFrame


@dataclass
class SupportArtifacts:
    cluster_model: Any
    projector: Any
    labeled_cluster_counts: dict[int, int]
    labeled_cluster_share: dict[int, float]
    labeled_support_radius: float
    support_distance_quantile: float
    agreement_quantile: float
    cluster_coverage_quantile: float
    pca_explained_variance: list[float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manifold-aware pseudo-label benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: PseudoLabelManifoldConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _grouped_oof_splits(df: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = df["recording_key"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    effective_splits = min(max(2, int(n_splits)), len(unique_groups))
    splitter = GroupKFold(n_splits=effective_splits)
    return [
        (np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int))
        for train_idx, val_idx in splitter.split(df, groups=groups)
    ]


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


def _build_filtered_drop_cols(
    *,
    split: Any,
    prepared: Any,
    base_config: MLStatisticsWorkbenchConfig,
) -> tuple[list[str], list[str], pd.DataFrame]:
    shift_unit = compute_feature_shift_table(
        split.hybrid_source,
        split.paired_non_test_all,
        feature_cols=prepared.unit_feature_cols,
        comparison="hybrid_vs_paired_non_test",
        feature_set="unit",
    )
    shift_full_context = compute_feature_shift_table(
        split.hybrid_source,
        split.paired_non_test_all,
        feature_cols=prepared.lightgbm_context_cols,
        comparison="hybrid_vs_paired_non_test",
        feature_set="full_context",
    )
    unit_manifest, unit_drop_cols = select_shift_filtered_features(
        shift_unit,
        allowed_cols=prepared.unit_feature_cols,
        top_n=base_config.filtered_shift_top_n_unit,
        min_abs_smd=base_config.filtered_shift_min_abs_smd,
        preserve_shape_features=base_config.filtered_preserve_shape_features,
        feature_set="unit",
    )
    context_manifest, context_drop_cols = select_shift_filtered_features(
        shift_full_context,
        allowed_cols=prepared.lightgbm_context_cols,
        top_n=base_config.filtered_shift_top_n_full_context,
        min_abs_smd=base_config.filtered_shift_min_abs_smd,
        preserve_shape_features=base_config.filtered_preserve_shape_features,
        feature_set="full_context",
    )
    manifest = pd.concat([unit_manifest, context_manifest], ignore_index=True)
    return unit_drop_cols, context_drop_cols, manifest


def _fit_global_transport(
    *,
    source_df: pd.DataFrame,
    paired_non_test_all: pd.DataFrame,
    unit_cols: list[str],
    config: PseudoLabelManifoldConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shift_unit = compute_feature_shift_table(
        source_df,
        paired_non_test_all,
        feature_cols=unit_cols,
        comparison="hybrid_vs_paired_non_test",
        feature_set="unit",
    )
    model = fit_quantile_transport_model(
        source_df,
        paired_non_test_all,
        feature_cols=unit_cols,
        shift_df=shift_unit,
        top_n=config.top_shift_features,
        min_abs_smd=config.min_abs_smd,
        quantile_points=config.quantile_points,
        blend=config.teacher_transport_blend,
        noise_scale=config.teacher_transport_noise,
    )
    translated = apply_quantile_transport_model(
        source_df.loc[:, unit_cols].apply(pd.to_numeric, errors="coerce").astype(float),
        model,
        random_state=config.random_state + 101,
    )
    post_shift = compute_feature_shift_table(
        translated,
        paired_non_test_all,
        feature_cols=unit_cols,
        comparison="teacher_transport",
        feature_set="unit",
    )
    return translated, post_shift


def _fit_source_teacher_predictions(
    *,
    source_features: pd.DataFrame,
    y_source: np.ndarray,
    train_features: pd.DataFrame,
    unlabeled_features: pd.DataFrame,
    test_features: pd.DataFrame,
    paired_target_df: pd.DataFrame,
    base: MLStatisticsWorkbenchConfig,
    random_state_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight_result = compute_importance_weights(
        source_features,
        paired_target_df,
        feature_cols=source_features.columns.tolist(),
        random_state=base.random_state + random_state_offset,
        clip_min=base.importance_weight_clip_min,
        clip_max=base.importance_weight_clip_max,
        c_value=base.domain_classifier_c,
    )
    model = fit_tree_regressor(
        source_features,
        y_source,
        backend="lightgbm",
        random_state=base.random_state + random_state_offset,
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
        sample_weight=weight_result.weights,
    )
    return (
        predict_tree_regressor(model, train_features),
        predict_tree_regressor(model, unlabeled_features),
        predict_tree_regressor(model, test_features),
    )


def _teacher_frames_for_target(
    *,
    target: str,
    split: Any,
    prepared: Any,
    base: MLStatisticsWorkbenchConfig,
    config: PseudoLabelManifoldConfig,
    unit_drop_cols: list[str],
    context_drop_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, list[TeacherBundle], pd.DataFrame, pd.DataFrame]:
    source_target = base.source_fmiss_target if target == "fmiss" else target
    source_df, y_source = filter_target_rows(split.hybrid_source, source_target)
    target_train_df, y_train = filter_target_rows(split.paired_finetune, target)
    target_test_df, y_test = filter_target_rows(split.paired_test, target)
    unlabeled_df = split.ssl_pool.copy()

    context_cols = prepared.lightgbm_context_cols
    source_unit = source_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    train_unit = target_train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    test_unit = target_test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    unlabeled_unit = unlabeled_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

    translated_source_unit, transport_shift = _fit_global_transport(
        source_df=source_df,
        paired_non_test_all=split.paired_non_test_all,
        unit_cols=prepared.unit_feature_cols,
        config=config,
    )
    translated_source_unit = translated_source_unit.reset_index(drop=True)

    raw_source_pred_train, raw_source_pred_unlabeled, raw_source_pred_test = _fit_source_teacher_predictions(
        source_features=source_unit,
        y_source=y_source,
        train_features=train_unit,
        unlabeled_features=unlabeled_unit,
        test_features=test_unit,
        paired_target_df=split.paired_non_test_all,
        base=base,
        random_state_offset=100 if target == "fpos" else 300,
    )
    translated_source_pred_train, translated_source_pred_unlabeled, translated_source_pred_test = _fit_source_teacher_predictions(
        source_features=translated_source_unit,
        y_source=y_source,
        train_features=train_unit,
        unlabeled_features=unlabeled_unit,
        test_features=test_unit,
        paired_target_df=split.paired_non_test_all,
        base=base,
        random_state_offset=200 if target == "fpos" else 400,
    )

    if target == "fpos":
        filtered_drop_cols = set(unit_drop_cols) | set(context_drop_cols)
        filtered_context_cols = [col for col in context_cols if col not in filtered_drop_cols]
        target_train_masked = mask_feature_columns(target_train_df, filtered_drop_cols)
        target_test_masked = mask_feature_columns(target_test_df, filtered_drop_cols)
        unlabeled_masked = mask_feature_columns(unlabeled_df, filtered_drop_cols)
        contextual_train = target_train_masked.loc[:, filtered_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        contextual_test = target_test_masked.loc[:, filtered_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        contextual_unlabeled = unlabeled_masked.loc[:, filtered_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        teachers = [
            TeacherBundle(
                variant_id="teacher_contextual_paired_only_lightgbm_filtered",
                family="paired_context_teacher",
                notes="paired-only filtered context teacher",
                backend="lightgbm",
                source_target=None,
                train_frame=contextual_train,
                unlabeled_frame=contextual_unlabeled,
                test_frame=contextual_test,
            ),
            TeacherBundle(
                variant_id="teacher_source_reweighted_fullctx_lightgbm_filtered",
                family="source_teacher",
                notes="filtered context + raw weighted hybrid source prediction",
                backend="lightgbm",
                source_target=source_target,
                train_frame=build_stack_frame(
                    target_train_masked,
                    context_cols=filtered_context_cols,
                    extra_features={"source_pred": raw_source_pred_train},
                ),
                unlabeled_frame=build_stack_frame(
                    unlabeled_masked,
                    context_cols=filtered_context_cols,
                    extra_features={"source_pred": raw_source_pred_unlabeled},
                ),
                test_frame=build_stack_frame(
                    target_test_masked,
                    context_cols=filtered_context_cols,
                    extra_features={"source_pred": raw_source_pred_test},
                ),
            ),
            TeacherBundle(
                variant_id="teacher_paired_conditioned_global_fullctx_lightgbm_filtered",
                family="transport_teacher",
                notes="filtered context + globally transported hybrid source prediction",
                backend="lightgbm",
                source_target=source_target,
                train_frame=build_stack_frame(
                    target_train_masked,
                    context_cols=filtered_context_cols,
                    extra_features={"source_pred": translated_source_pred_train},
                ),
                unlabeled_frame=build_stack_frame(
                    unlabeled_masked,
                    context_cols=filtered_context_cols,
                    extra_features={"source_pred": translated_source_pred_unlabeled},
                ),
                test_frame=build_stack_frame(
                    target_test_masked,
                    context_cols=filtered_context_cols,
                    extra_features={"source_pred": translated_source_pred_test},
                ),
            ),
        ]
        student_templates = pd.DataFrame(
            {
                "variant_id": [
                    "contextual_pseudo_lightgbm_filtered",
                    "source_reweighted_pseudo_lightgbm_filtered",
                    "transport_pseudo_lightgbm_filtered",
                ],
                "train_frame": [
                    contextual_train,
                    teachers[1].train_frame,
                    teachers[2].train_frame,
                ],
                "unlabeled_frame": [
                    contextual_unlabeled,
                    teachers[1].unlabeled_frame,
                    teachers[2].unlabeled_frame,
                ],
                "test_frame": [
                    contextual_test,
                    teachers[1].test_frame,
                    teachers[2].test_frame,
                ],
                "notes": [
                    "paired-only filtered context student with weighted pseudo rows",
                    "filtered context + raw weighted hybrid source prediction student with weighted pseudo rows",
                    "filtered context + globally transported hybrid source prediction student with weighted pseudo rows",
                ],
            }
        )
    else:
        contextual_train = target_train_df.loc[:, context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        contextual_test = target_test_df.loc[:, context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        contextual_unlabeled = unlabeled_df.loc[:, context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        teachers = [
            TeacherBundle(
                variant_id="teacher_contextual_paired_only_lightgbm",
                family="paired_context_teacher",
                notes="paired-only full context teacher",
                backend="lightgbm",
                source_target=None,
                train_frame=contextual_train,
                unlabeled_frame=contextual_unlabeled,
                test_frame=contextual_test,
            ),
            TeacherBundle(
                variant_id="teacher_source_reweighted_fullctx_lightgbm",
                family="source_teacher",
                notes="full context + raw weighted hybrid source prediction",
                backend="lightgbm",
                source_target=source_target,
                train_frame=build_stack_frame(
                    target_train_df,
                    context_cols=context_cols,
                    extra_features={"source_pred": raw_source_pred_train},
                ),
                unlabeled_frame=build_stack_frame(
                    unlabeled_df,
                    context_cols=context_cols,
                    extra_features={"source_pred": raw_source_pred_unlabeled},
                ),
                test_frame=build_stack_frame(
                    target_test_df,
                    context_cols=context_cols,
                    extra_features={"source_pred": raw_source_pred_test},
                ),
            ),
            TeacherBundle(
                variant_id="teacher_paired_conditioned_global_fullctx_lightgbm",
                family="transport_teacher",
                notes="full context + globally transported hybrid source prediction",
                backend="lightgbm",
                source_target=source_target,
                train_frame=build_stack_frame(
                    target_train_df,
                    context_cols=context_cols,
                    extra_features={"source_pred": translated_source_pred_train},
                ),
                unlabeled_frame=build_stack_frame(
                    unlabeled_df,
                    context_cols=context_cols,
                    extra_features={"source_pred": translated_source_pred_unlabeled},
                ),
                test_frame=build_stack_frame(
                    target_test_df,
                    context_cols=context_cols,
                    extra_features={"source_pred": translated_source_pred_test},
                ),
            ),
        ]
        student_templates = pd.DataFrame(
            {
                "variant_id": [
                    "contextual_pseudo_lightgbm",
                    "source_reweighted_pseudo_lightgbm",
                    "transport_pseudo_lightgbm",
                ],
                "train_frame": [
                    contextual_train,
                    teachers[1].train_frame,
                    teachers[2].train_frame,
                ],
                "unlabeled_frame": [
                    contextual_unlabeled,
                    teachers[1].unlabeled_frame,
                    teachers[2].unlabeled_frame,
                ],
                "test_frame": [
                    contextual_test,
                    teachers[1].test_frame,
                    teachers[2].test_frame,
                ],
                "notes": [
                    "paired-only full context student with weighted pseudo rows",
                    "full context + raw weighted hybrid source prediction student with weighted pseudo rows",
                    "full context + globally transported hybrid source prediction student with weighted pseudo rows",
                ],
            }
        )

    return (
        target_train_df,
        y_train,
        target_test_df,
        y_test,
        unlabeled_df,
        teachers,
        student_templates,
        transport_shift.assign(target=target),
    )


def _build_support_artifacts(
    *,
    paired_non_test_all: pd.DataFrame,
    labeled_train_df: pd.DataFrame,
    unit_cols: list[str],
    config: PseudoLabelManifoldConfig,
) -> SupportArtifacts:
    projector = fit_pca_support_projector(
        paired_non_test_all,
        feature_cols=unit_cols,
        n_components=config.manifold_pca_components,
        n_neighbors=config.manifold_neighbors,
    )
    cluster_model = fit_kmeans_subgroup_model(
        paired_non_test_all,
        feature_cols=unit_cols,
        n_clusters=config.manifold_clusters,
        random_state=config.random_state,
    )
    labeled_clusters = assign_kmeans_subgroups(labeled_train_df, cluster_model)
    cluster_counts = pd.Series(labeled_clusters).value_counts().to_dict()
    total_labeled = max(int(len(labeled_train_df)), 1)
    cluster_share = {int(k): float(v / total_labeled) for k, v in cluster_counts.items()}

    labeled_scores = projector.pca.transform(
        projector.scaler.transform(
            projector.imputer.transform(
                labeled_train_df.loc[:, projector.feature_cols].apply(pd.to_numeric, errors="coerce")
            )
        )
    )
    labeled_neighbors = NearestNeighbors(
        n_neighbors=min(max(2, config.manifold_neighbors), len(labeled_scores))
    ).fit(labeled_scores)
    distances, _ = labeled_neighbors.kneighbors(labeled_scores, return_distance=True)
    support_distance = distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else distances[:, 0]
    labeled_support_radius = float(np.quantile(support_distance, config.support_quantile))
    return SupportArtifacts(
        cluster_model=cluster_model,
        projector=projector,
        labeled_cluster_counts={int(k): int(v) for k, v in cluster_counts.items()},
        labeled_cluster_share=cluster_share,
        labeled_support_radius=labeled_support_radius,
        support_distance_quantile=float(config.support_quantile),
        agreement_quantile=float(config.agreement_quantile),
        cluster_coverage_quantile=float(config.cluster_coverage_quantile),
        pca_explained_variance=[float(x) for x in projector.pca.explained_variance_ratio_],
    )


def _manifold_support_table(
    unlabeled_df: pd.DataFrame,
    labeled_train_df: pd.DataFrame,
    support_artifacts: SupportArtifacts,
) -> pd.DataFrame:
    projector = support_artifacts.projector
    unlabeled_scores = projector.pca.transform(
        projector.scaler.transform(
            projector.imputer.transform(
                unlabeled_df.loc[:, projector.feature_cols].apply(pd.to_numeric, errors="coerce")
            )
        )
    )
    labeled_scores = projector.pca.transform(
        projector.scaler.transform(
            projector.imputer.transform(
                labeled_train_df.loc[:, projector.feature_cols].apply(pd.to_numeric, errors="coerce")
            )
        )
    )
    neighbors = NearestNeighbors(
        n_neighbors=min(max(2, projector.neighbors.n_neighbors), len(labeled_scores))
    ).fit(labeled_scores)
    distances, _ = neighbors.kneighbors(unlabeled_scores, return_distance=True)
    mean_distance = distances.mean(axis=1)
    clusters = assign_kmeans_subgroups(unlabeled_df, support_artifacts.cluster_model)
    cluster_count = np.array(
        [support_artifacts.labeled_cluster_counts.get(int(c), 0) for c in clusters],
        dtype=float,
    )
    cluster_share = np.array(
        [support_artifacts.labeled_cluster_share.get(int(c), 0.0) for c in clusters],
        dtype=float,
    )
    support_conf = np.exp(-mean_distance / max(support_artifacts.labeled_support_radius, 1e-6))
    return pd.DataFrame(
        {
            "unlabeled_row_idx": np.arange(len(unlabeled_df), dtype=int),
            "row_uid": unlabeled_df["row_uid"].astype(str).to_numpy(),
            "recording_key": unlabeled_df["recording_key"].astype(str).to_numpy(),
            "study_set": unlabeled_df["study_set"].astype(str).to_numpy(),
            "support_distance": mean_distance.astype(float),
            "support_confidence": support_conf.astype(float),
            "cluster_id": clusters.astype(int),
            "cluster_labeled_count": cluster_count.astype(int),
            "cluster_labeled_share": cluster_share.astype(float),
        }
    )


def _teacher_weight_table(teacher_perf_df: pd.DataFrame) -> pd.DataFrame:
    working = teacher_perf_df.copy()
    raw = np.clip(working["oof_r2"].astype(float).to_numpy(), 0.0, None)
    if float(raw.sum()) <= 1e-8:
        raw = 1.0 / np.clip(working["oof_mae"].astype(float).to_numpy(), 1e-4, None)
    weights = raw / max(float(raw.sum()), 1e-8)
    working["ensemble_weight"] = weights.astype(float)
    return working


def _aggregate_teacher_predictions(
    frame: pd.DataFrame,
    teacher_perf_df: pd.DataFrame,
) -> pd.DataFrame:
    weight_map = {
        str(row["variant_id"]): float(row["ensemble_weight"])
        for _, row in teacher_perf_df.iterrows()
    }
    teacher_cols = [str(col) for col in frame.columns if str(col) in weight_map]
    weights = np.array([weight_map[col] for col in teacher_cols], dtype=float)
    weight_sum = max(float(weights.sum()), 1e-8)
    norm_weights = weights / weight_sum
    pred_matrix = frame.loc[:, teacher_cols].to_numpy(dtype=float)
    weighted_mean = pred_matrix @ norm_weights
    weighted_var = (norm_weights[None, :] * ((pred_matrix - weighted_mean[:, None]) ** 2)).sum(axis=1)
    out = frame.copy()
    out["teacher_unweighted_mean"] = pred_matrix.mean(axis=1)
    out["teacher_unweighted_std"] = pred_matrix.std(axis=1, ddof=0)
    out["teacher_mean"] = weighted_mean
    out["teacher_std"] = np.sqrt(np.clip(weighted_var, 0.0, None))
    out["teacher_min"] = pred_matrix.min(axis=1)
    out["teacher_max"] = pred_matrix.max(axis=1)
    out["teacher_range"] = out["teacher_max"] - out["teacher_min"]
    return out


def _teacher_predictions(
    *,
    teachers: list[TeacherBundle],
    labeled_df: pd.DataFrame,
    y_train: np.ndarray,
    base: MLStatisticsWorkbenchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = _grouped_oof_splits(labeled_df, base.model_selection_splits)
    train_rows: list[dict[str, Any]] = []
    unlabeled_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    perf_rows: list[dict[str, Any]] = []

    for idx, teacher in enumerate(teachers, start=1):
        oof_pred = np.full(len(labeled_df), np.nan, dtype=float)
        for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
            fold_pred = _fit_predict(
                feature_train=teacher.train_frame.iloc[train_idx],
                y_train=y_train[train_idx],
                feature_eval=teacher.train_frame.iloc[val_idx],
                sample_weight=None,
                base_config=base,
                random_state_offset=(10000 * idx) + split_id,
            )
            oof_pred[val_idx] = fold_pred
        oof_metrics = score_predictions(y_train, oof_pred)
        unlabeled_pred = _fit_predict(
            feature_train=teacher.train_frame,
            y_train=y_train,
            feature_eval=teacher.unlabeled_frame,
            sample_weight=None,
            base_config=base,
            random_state_offset=(20000 * idx),
        )
        test_pred = _fit_predict(
            feature_train=teacher.train_frame,
            y_train=y_train,
            feature_eval=teacher.test_frame,
            sample_weight=None,
            base_config=base,
            random_state_offset=(30000 * idx),
        )
        train_rows.append({"variant_id": teacher.variant_id, "pred": oof_pred})
        unlabeled_rows.append({"variant_id": teacher.variant_id, "pred": unlabeled_pred})
        test_rows.append({"variant_id": teacher.variant_id, "pred": test_pred})
        perf_rows.append(
            {
                "variant_id": teacher.variant_id,
                "family": teacher.family,
                "backend": teacher.backend,
                "source_target": teacher.source_target,
                "notes": teacher.notes,
                "oof_mae": float(oof_metrics["mae"]),
                "oof_rmse": float(oof_metrics["rmse"]),
                "oof_r2": float(oof_metrics["r2"]),
                "oof_bias": float(oof_metrics["bias"]),
                "oof_calibration_slope": float(oof_metrics["calibration_slope"]),
                "oof_calibration_intercept": float(oof_metrics["calibration_intercept"]),
            }
        )

    def _to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
        payload: dict[str, np.ndarray] = {}
        for row in rows:
            payload[str(row["variant_id"])] = np.asarray(row["pred"], dtype=float)
        return pd.DataFrame(payload)

    teacher_perf_df = _teacher_weight_table(pd.DataFrame(perf_rows))
    train_frame = _aggregate_teacher_predictions(_to_frame(train_rows), teacher_perf_df)
    unlabeled_frame = _aggregate_teacher_predictions(_to_frame(unlabeled_rows), teacher_perf_df)
    test_frame = _aggregate_teacher_predictions(_to_frame(test_rows), teacher_perf_df)

    ensemble_metrics = score_predictions(y_train, train_frame["teacher_mean"].to_numpy(dtype=float))
    ensemble_row = {
        "variant_id": "teacher_ensemble_weighted",
        "family": "pseudo_teacher_ensemble",
        "backend": "ensemble",
        "source_target": None,
        "notes": "weighted teacher ensemble using grouped OOF teacher performance",
        "oof_mae": float(ensemble_metrics["mae"]),
        "oof_rmse": float(ensemble_metrics["rmse"]),
        "oof_r2": float(ensemble_metrics["r2"]),
        "oof_bias": float(ensemble_metrics["bias"]),
        "oof_calibration_slope": float(ensemble_metrics["calibration_slope"]),
        "oof_calibration_intercept": float(ensemble_metrics["calibration_intercept"]),
        "ensemble_weight": 1.0,
    }
    teacher_perf_df = pd.concat([teacher_perf_df, pd.DataFrame([ensemble_row])], ignore_index=True)
    teacher_perf_df["target"] = pd.NA
    teacher_perf_df["teacher_std_abs_error_corr"] = pd.NA
    teacher_perf_df.loc[
        teacher_perf_df["variant_id"] == "teacher_ensemble_weighted",
        "teacher_std_abs_error_corr",
    ] = float(
        np.corrcoef(
            train_frame["teacher_std"].to_numpy(dtype=float),
            np.abs(train_frame["teacher_mean"].to_numpy(dtype=float) - y_train),
        )[0, 1]
    ) if len(train_frame) > 2 else float("nan")
    return train_frame, unlabeled_frame, test_frame, teacher_perf_df


def _select_pseudo_labels(
    unlabeled_df: pd.DataFrame,
    *,
    unlabeled_teacher_df: pd.DataFrame,
    support_df: pd.DataFrame,
    config: PseudoLabelManifoldConfig,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.DataFrame(
        {
            "unlabeled_row_idx": np.arange(len(unlabeled_df), dtype=int),
            "row_uid": unlabeled_df["row_uid"].astype(str).to_numpy(),
            "recording_key": unlabeled_df["recording_key"].astype(str).to_numpy(),
            "study_set": unlabeled_df["study_set"].astype(str).to_numpy(),
        }
    )
    merged = merged.merge(unlabeled_teacher_df.reset_index().rename(columns={"index": "unlabeled_row_idx"}), on="unlabeled_row_idx", how="left")
    merged = merged.merge(
        support_df,
        on=["unlabeled_row_idx", "row_uid", "recording_key", "study_set"],
        how="left",
    )

    support_cut = float(merged["support_distance"].quantile(config.support_quantile))
    agreement_cut = float(merged["teacher_std"].quantile(config.agreement_quantile))
    coverage_cut = float(merged["cluster_labeled_share"].quantile(config.cluster_coverage_quantile))
    cluster_support_cut = (
        merged.groupby("cluster_id", dropna=False)["support_distance"]
        .transform(lambda s: float(s.quantile(config.support_quantile)))
    )
    cluster_agreement_cut = (
        merged.groupby("cluster_id", dropna=False)["teacher_std"]
        .transform(lambda s: float(s.quantile(config.agreement_quantile)))
    )
    candidate_mask = (
        (merged["support_distance"].astype(float) <= cluster_support_cut.astype(float))
        & (merged["teacher_std"].astype(float) <= cluster_agreement_cut.astype(float))
        & (merged["cluster_labeled_count"].astype(float) >= float(config.min_labeled_cluster_rows))
    )
    candidate_df = merged.copy()
    candidate_df["cluster_support_cutoff"] = cluster_support_cut.astype(float)
    candidate_df["cluster_agreement_cutoff"] = cluster_agreement_cut.astype(float)
    candidate_df["global_support_cutoff"] = support_cut
    candidate_df["global_agreement_cutoff"] = agreement_cut
    candidate_df["global_coverage_cutoff"] = coverage_cut
    candidate_df["selected_candidate"] = candidate_mask.astype(bool)
    if not candidate_mask.any():
        candidate_df["selected_final"] = False
        return candidate_df, candidate_df.iloc[0:0].copy()

    selected = candidate_df.loc[candidate_mask].copy()
    agreement_conf = 1.0 - np.clip(
        selected["teacher_std"].astype(float) / np.clip(selected["cluster_agreement_cutoff"].astype(float), 1e-6, None),
        0.0,
        1.0,
    )
    support_conf = np.clip(selected["support_confidence"].astype(float), 0.0, 1.0)
    coverage_conf = np.clip(selected["cluster_labeled_share"].astype(float) / max(coverage_cut, 1e-6), 0.0, 1.0)
    selected["pseudo_weight"] = np.clip(
        float(config.pseudo_weight_scale)
        * (
            (0.45 * support_conf)
            + (0.35 * agreement_conf)
            + (0.20 * np.minimum(coverage_conf, 1.0))
        ),
        float(config.pseudo_weight_floor),
        float(config.pseudo_weight_scale),
    )
    selected["pseudo_label"] = selected["teacher_mean"].astype(float)
    selected["support_cutoff"] = support_cut
    selected["agreement_cutoff"] = agreement_cut
    selected["coverage_cutoff"] = coverage_cut
    selected = selected.sort_values(
        ["pseudo_weight", "support_confidence", "teacher_std"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    cluster_candidate_counts = selected["cluster_id"].value_counts().to_dict()
    cluster_priority = {}
    for cluster_id, candidate_count in cluster_candidate_counts.items():
        labeled_count = float(selected.loc[selected["cluster_id"] == cluster_id, "cluster_labeled_count"].iloc[0])
        cluster_priority[int(cluster_id)] = np.sqrt(max(labeled_count, 1.0) * max(float(candidate_count), 1.0))
    total_priority = max(float(sum(cluster_priority.values())), 1e-8)
    cluster_quotas = {
        int(cluster_id): min(
            int(config.max_pseudo_per_cluster),
            max(
                1,
                int(
                    round(
                        float(
                            selected.loc[selected["cluster_id"] == cluster_id, "cluster_labeled_count"].iloc[0]
                        )
                        * float(config.max_pseudo_per_labeled_ratio)
                    )
                ),
            ),
            max(1, int(round((priority / total_priority) * float(max_rows)))),
        )
        for cluster_id, priority in cluster_priority.items()
    }

    recording_counts: dict[str, int] = {}
    cluster_counts: dict[int, int] = {}
    keep_rows: list[int] = []

    def _try_take(row_index: int, row_obj: Any) -> bool:
        recording_key = str(row_obj.recording_key)
        cluster_id = int(row_obj.cluster_id)
        if recording_counts.get(recording_key, 0) >= int(config.max_pseudo_per_recording):
            return False
        if cluster_counts.get(cluster_id, 0) >= int(config.max_pseudo_per_cluster):
            return False
        if cluster_counts.get(cluster_id, 0) >= int(cluster_quotas.get(cluster_id, 0)):
            return False
        keep_rows.append(int(row_index))
        recording_counts[recording_key] = recording_counts.get(recording_key, 0) + 1
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        return True

    for cluster_id in sorted(cluster_quotas):
        cluster_rows = selected[selected["cluster_id"] == cluster_id]
        for row in cluster_rows.itertuples(index=True):
            _try_take(int(row.Index), row)
            if len(keep_rows) >= int(max_rows):
                break
        if len(keep_rows) >= int(max_rows):
            break

    if len(keep_rows) < int(max_rows):
        for row in selected.itertuples(index=True):
            if int(row.Index) in keep_rows:
                continue
            recording_key = str(row.recording_key)
            cluster_id = int(row.cluster_id)
            if recording_counts.get(recording_key, 0) >= int(config.max_pseudo_per_recording):
                continue
            if cluster_counts.get(cluster_id, 0) >= int(config.max_pseudo_per_cluster):
                continue
            if cluster_counts.get(cluster_id, 0) >= int(cluster_quotas.get(cluster_id, 0)):
                continue
            keep_rows.append(int(row.Index))
            recording_counts[recording_key] = recording_counts.get(recording_key, 0) + 1
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            if len(keep_rows) >= int(max_rows):
                break

    selected = selected.iloc[sorted(set(keep_rows))].sort_values(
        ["pseudo_weight", "support_confidence", "teacher_std"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidate_df["selected_final"] = candidate_df["row_uid"].astype(str).isin(selected["row_uid"].astype(str))
    return candidate_df, selected


def _assemble_student_training_data(
    *,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    labeled_indices: np.ndarray | None,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    if labeled_indices is None:
        base_frame = labeled_frame.reset_index(drop=True)
        base_y = np.asarray(labeled_y, dtype=float)
    else:
        base_frame = labeled_frame.iloc[labeled_indices].reset_index(drop=True)
        base_y = np.asarray(labeled_y[labeled_indices], dtype=float)
    if pseudo_selected.empty:
        return base_frame, base_y, None

    pseudo_idx = pseudo_selected["unlabeled_row_idx"].to_numpy(dtype=int)
    pseudo_frame = unlabeled_frame.iloc[pseudo_idx].reset_index(drop=True)
    full_frame = pd.concat([base_frame, pseudo_frame], ignore_index=True)
    full_y = np.concatenate([base_y, pseudo_selected["pseudo_label"].to_numpy(dtype=float)])
    sample_weight = np.concatenate(
        [
            np.ones(len(base_frame), dtype=np.float32),
            pseudo_selected["pseudo_weight"].to_numpy(dtype=np.float32),
        ]
    )
    return full_frame, full_y, sample_weight


def _score_student_cv(
    *,
    labeled_meta_df: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    base: MLStatisticsWorkbenchConfig,
    random_state_offset: int,
) -> dict[str, float]:
    splits = _grouped_oof_splits(labeled_meta_df, base.model_selection_splits)
    oof_pred = np.full(len(labeled_meta_df), np.nan, dtype=float)
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        val_recordings = set(labeled_meta_df.iloc[val_idx]["recording_key"].astype(str))
        fold_pseudo = pseudo_selected[
            ~pseudo_selected["recording_key"].astype(str).isin(val_recordings)
        ].reset_index(drop=True)
        train_frame, train_y, train_weight = _assemble_student_training_data(
            labeled_frame=labeled_frame,
            labeled_y=labeled_y,
            labeled_indices=train_idx,
            pseudo_selected=fold_pseudo,
            unlabeled_frame=unlabeled_frame,
        )
        fold_pred = _fit_predict(
            feature_train=train_frame,
            y_train=train_y,
            feature_eval=labeled_frame.iloc[val_idx].reset_index(drop=True),
            sample_weight=train_weight,
            base_config=base,
            random_state_offset=random_state_offset + split_id,
        )
        oof_pred[val_idx] = fold_pred
    return score_predictions(labeled_y, oof_pred)


def _fit_student_holdout(
    *,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    base: MLStatisticsWorkbenchConfig,
    random_state_offset: int,
) -> np.ndarray:
    train_frame, train_y, train_weight = _assemble_student_training_data(
        labeled_frame=labeled_frame,
        labeled_y=labeled_y,
        labeled_indices=None,
        pseudo_selected=pseudo_selected,
        unlabeled_frame=unlabeled_frame,
    )
    return _fit_predict(
        feature_train=train_frame,
        y_train=train_y,
        feature_eval=test_frame.reset_index(drop=True),
        sample_weight=train_weight,
        base_config=base,
        random_state_offset=random_state_offset,
    )


def _summarize_clusters(
    *,
    target: str,
    paired_non_test_all: pd.DataFrame,
    labeled_train_df: pd.DataFrame,
    support_df: pd.DataFrame,
    pseudo_selected: pd.DataFrame,
    support_artifacts: SupportArtifacts,
) -> pd.DataFrame:
    all_clusters = assign_kmeans_subgroups(paired_non_test_all, support_artifacts.cluster_model)
    paired_all = paired_non_test_all.loc[:, ["study_set"]].copy().reset_index(drop=True)
    paired_all["cluster_id"] = all_clusters
    labeled_clusters = assign_kmeans_subgroups(labeled_train_df, support_artifacts.cluster_model)
    labeled_counts = pd.Series(labeled_clusters).value_counts().to_dict()
    unlabeled_counts = support_df["cluster_id"].value_counts().to_dict()
    selected_counts = pseudo_selected["cluster_id"].value_counts().to_dict() if not pseudo_selected.empty else {}
    rows: list[dict[str, Any]] = []
    n_clusters = int(np.max(all_clusters)) + 1 if len(all_clusters) else 0
    for cluster_id in range(n_clusters):
        subset = paired_all[paired_all["cluster_id"] == cluster_id]
        study_counts = subset["study_set"].astype(str).value_counts().head(3)
        rows.append(
            {
                "target": target,
                "cluster_id": int(cluster_id),
                "paired_non_test_rows": int(len(subset)),
                "labeled_rows": int(labeled_counts.get(cluster_id, 0)),
                "unlabeled_rows": int(unlabeled_counts.get(cluster_id, 0)),
                "selected_pseudo_rows": int(selected_counts.get(cluster_id, 0)),
                "selected_share_of_unlabeled": float(selected_counts.get(cluster_id, 0) / max(unlabeled_counts.get(cluster_id, 1), 1)),
                "top_studies": " | ".join(f"{study}:{count}" for study, count in study_counts.items()),
            }
        )
    return pd.DataFrame(rows)


def _build_readme(
    *,
    config: PseudoLabelManifoldConfig,
    split_manifest: dict[str, Any],
    metrics_df: pd.DataFrame,
    pseudo_summary_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    holdout_best_df: pd.DataFrame,
) -> str:
    lines = [
        "# Real-x / Pseudo-y Manifold Benchmark",
        "",
        "This run uses real unlabeled paired rows as target-domain `x`, discovers manifold/support structure on non-test paired rows, pseudo-labels only high-confidence support-consistent rows, and retrains context-first tree students with weighted pseudo labels.",
        "",
        "## Config",
        f"- support_quantile: `{config.support_quantile}`",
        f"- agreement_quantile: `{config.agreement_quantile}`",
        f"- cluster_coverage_quantile: `{config.cluster_coverage_quantile}`",
        f"- max_pseudo_multiplier: `{config.max_pseudo_multiplier}`",
        f"- max_pseudo_per_recording: `{config.max_pseudo_per_recording}`",
        f"- max_pseudo_per_cluster: `{config.max_pseudo_per_cluster}`",
        f"- max_pseudo_per_labeled_ratio: `{config.max_pseudo_per_labeled_ratio}`",
        f"- pseudo_weight_scale: `{config.pseudo_weight_scale}`",
        "",
        "## Split",
        f"- paired_test_rows: `{split_manifest['paired_test_rows']}`",
        f"- paired_finetune_rows: `{split_manifest['paired_finetune_rows']}`",
        f"- ssl_pool_rows: `{split_manifest['ssl_pool_rows']}`",
        "",
        "## Pseudo Selection",
        "",
    ]
    if not pseudo_summary_df.empty:
        for _, row in pseudo_summary_df.iterrows():
            lines.extend(
                [
                    f"### {row['target']}",
                    f"- candidate pseudo rows: `{int(row['candidate_rows'])}`",
                    f"- selected pseudo rows: `{int(row['selected_pseudo_rows'])}`",
                    f"- selected recordings: `{int(row['selected_recordings'])}`",
                    f"- mean pseudo weight: `{row['mean_pseudo_weight']:.4f}`",
                    f"- mean teacher std (selected): `{row['mean_teacher_std_selected']:.4f}`",
                    "",
                ]
            )
    if not recommendation_df.empty:
        lines.extend(["## Recommendation", ""])
        for _, row in recommendation_df.iterrows():
            lines.extend(
                [
                    f"### {row['target']}",
                    f"- reference CV variant: `{row['reference_variant_id']}`",
                    f"- recommended CV variant: `{row['recommended_variant_id']}`",
                    f"- holdout MAE: `{row['recommended_holdout_mae']:.4f}`",
                    f"- holdout R²: `{row['recommended_holdout_r2']:.4f}`",
                    f"- diagnostic best holdout variant: `{row['best_holdout_variant_id']}`",
                    f"- diagnostic best holdout R²: `{row['best_holdout_r2']:.4f}`",
                    "",
                ]
            )
    lines.extend(["## Holdout Results", ""])
    for target, group in metrics_df.groupby("target", dropna=False):
        lines.append(f"### {target}")
        for _, row in group.sort_values(["r2", "mae"], ascending=[False, True]).iterrows():
            lines.append(f"- {row['variant_id']} | MAE `{row['mae']:.4f}` | R² `{row['r2']:.4f}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def _run_with_config(config: PseudoLabelManifoldConfig) -> Path:
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

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    pseudo_summary_rows: list[dict[str, Any]] = []
    pseudo_candidate_frames: list[pd.DataFrame] = []
    pseudo_selected_frames: list[pd.DataFrame] = []
    manifold_rows: list[dict[str, Any]] = []
    teacher_perf_frames: list[pd.DataFrame] = []
    student_cv_rows: list[dict[str, Any]] = []
    cluster_summary_frames: list[pd.DataFrame] = []
    transport_rows: list[pd.DataFrame] = []
    recommendation_rows: list[dict[str, Any]] = []

    for target in ("fpos", "fmiss"):
        _log(config, f"[{target}] fitting teachers, manifold, and pseudo students")
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
            target=target,
            split=split,
            prepared=prepared,
            base=base,
            config=config,
            unit_drop_cols=unit_drop_cols,
            context_drop_cols=context_drop_cols,
        )
        transport_rows.append(transport_manifest)

        support_artifacts = _build_support_artifacts(
            paired_non_test_all=split.paired_non_test_all,
            labeled_train_df=target_train_df,
            unit_cols=prepared.unit_feature_cols,
            config=config,
        )
        support_df = _manifold_support_table(
            unlabeled_df=unlabeled_df,
            labeled_train_df=target_train_df,
            support_artifacts=support_artifacts,
        )
        manifold_rows.append(
            {
                "target": target,
                "labeled_support_radius": support_artifacts.labeled_support_radius,
                "n_clusters": int(config.manifold_clusters),
                "labeled_rows": int(len(target_train_df)),
                "unlabeled_rows": int(len(unlabeled_df)),
                "pca_explained_variance_1": float(support_artifacts.pca_explained_variance[0]) if support_artifacts.pca_explained_variance else float("nan"),
                "pca_explained_variance_2": float(support_artifacts.pca_explained_variance[1]) if len(support_artifacts.pca_explained_variance) > 1 else float("nan"),
                "pca_explained_variance_sum": float(np.sum(support_artifacts.pca_explained_variance)),
            }
        )

        teacher_train_df, teacher_unlabeled_df, teacher_test_df, teacher_perf_df = _teacher_predictions(
            teachers=teachers,
            labeled_df=target_train_df,
            y_train=y_train,
            base=base,
        )
        teacher_perf_df["target"] = target
        teacher_perf_frames.append(teacher_perf_df)

        for teacher in teachers:
            pred = teacher_test_df[teacher.variant_id].to_numpy(dtype=float)
            metrics = score_predictions(y_test, pred)
            metric_rows.append(
                make_metric_row(
                    target=target,
                    variant_id=teacher.variant_id,
                    family=teacher.family,
                    backend=teacher.backend,
                    source_target=teacher.source_target,
                    metrics=metrics,
                    notes=teacher.notes,
                )
            )
            prediction_frames.append(
                make_prediction_frame(
                    target=target,
                    variant_id=teacher.variant_id,
                    family=teacher.family,
                    backend=teacher.backend,
                    df=target_test_df,
                    y_true=y_test,
                    pred=pred,
                )
            )
            teacher_cv = teacher_perf_df.loc[teacher_perf_df["variant_id"] == teacher.variant_id].iloc[0]
            student_cv_rows.append(
                {
                    "target": target,
                    "variant_id": teacher.variant_id,
                    "family": teacher.family,
                    "backend": teacher.backend,
                    "selected_pseudo_rows": 0,
                    "mean_pseudo_weight": 0.0,
                    "cv_mae": float(teacher_cv["oof_mae"]),
                    "cv_rmse": float(teacher_cv["oof_rmse"]),
                    "cv_r2": float(teacher_cv["oof_r2"]),
                    "cv_bias": float(teacher_cv["oof_bias"]),
                    "cv_calibration_slope": float(teacher_cv["oof_calibration_slope"]),
                    "cv_calibration_intercept": float(teacher_cv["oof_calibration_intercept"]),
                }
            )

        ensemble_pred = teacher_test_df["teacher_mean"].to_numpy(dtype=float)
        ensemble_metrics = score_predictions(y_test, ensemble_pred)
        metric_rows.append(
            make_metric_row(
                target=target,
                variant_id="teacher_ensemble_weighted",
                family="pseudo_teacher_ensemble",
                backend="ensemble",
                source_target=base.source_fmiss_target if target == "fmiss" else target,
                metrics=ensemble_metrics,
                notes="weighted teacher ensemble using grouped OOF teacher performance",
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target=target,
                variant_id="teacher_ensemble_weighted",
                family="pseudo_teacher_ensemble",
                backend="ensemble",
                df=target_test_df,
                y_true=y_test,
                pred=ensemble_pred,
            )
        )
        ensemble_cv = teacher_perf_df.loc[teacher_perf_df["variant_id"] == "teacher_ensemble_weighted"].iloc[0]
        student_cv_rows.append(
            {
                "target": target,
                "variant_id": "teacher_ensemble_weighted",
                "family": "pseudo_teacher_ensemble",
                "backend": "ensemble",
                "selected_pseudo_rows": 0,
                "mean_pseudo_weight": 0.0,
                "cv_mae": float(ensemble_cv["oof_mae"]),
                "cv_rmse": float(ensemble_cv["oof_rmse"]),
                "cv_r2": float(ensemble_cv["oof_r2"]),
                "cv_bias": float(ensemble_cv["oof_bias"]),
                "cv_calibration_slope": float(ensemble_cv["oof_calibration_slope"]),
                "cv_calibration_intercept": float(ensemble_cv["oof_calibration_intercept"]),
            }
        )

        max_pseudo_rows = int(round(len(target_train_df) * float(config.max_pseudo_multiplier)))
        candidate_df, pseudo_selected = _select_pseudo_labels(
            unlabeled_df=unlabeled_df,
            unlabeled_teacher_df=teacher_unlabeled_df,
            support_df=support_df,
            config=config,
            max_rows=max_pseudo_rows,
        )
        candidate_df["target"] = target
        pseudo_selected["target"] = target
        pseudo_candidate_frames.append(candidate_df)
        pseudo_selected_frames.append(pseudo_selected)

        pseudo_summary_rows.append(
            {
                "target": target,
                "candidate_rows": int(candidate_df["selected_candidate"].sum()),
                "selected_pseudo_rows": int(len(pseudo_selected)),
                "selected_recordings": int(pseudo_selected["recording_key"].astype(str).nunique()) if not pseudo_selected.empty else 0,
                "mean_pseudo_weight": float(pseudo_selected["pseudo_weight"].mean()) if not pseudo_selected.empty else 0.0,
                "mean_teacher_std_selected": float(pseudo_selected["teacher_std"].mean()) if not pseudo_selected.empty else float("nan"),
                "mean_support_distance_selected": float(pseudo_selected["support_distance"].mean()) if not pseudo_selected.empty else float("nan"),
                "max_pseudo_rows_allowed": int(max_pseudo_rows),
            }
        )

        cluster_summary_frames.append(
            _summarize_clusters(
                target=target,
                paired_non_test_all=split.paired_non_test_all,
                labeled_train_df=target_train_df,
                support_df=support_df,
                pseudo_selected=pseudo_selected,
                support_artifacts=support_artifacts,
            )
        )

        for idx, row in student_templates.iterrows():
            variant_id = str(row["variant_id"])
            cv_metrics = _score_student_cv(
                labeled_meta_df=target_train_df,
                labeled_frame=row["train_frame"],
                labeled_y=y_train,
                pseudo_selected=pseudo_selected,
                unlabeled_frame=row["unlabeled_frame"],
                base=base,
                random_state_offset=5000 + (100 * int(idx + 1)),
            )
            holdout_pred = _fit_student_holdout(
                labeled_frame=row["train_frame"],
                labeled_y=y_train,
                pseudo_selected=pseudo_selected,
                unlabeled_frame=row["unlabeled_frame"],
                test_frame=row["test_frame"],
                base=base,
                random_state_offset=9000 + (100 * int(idx + 1)),
            )
            holdout_metrics = score_predictions(y_test, holdout_pred)
            metric_rows.append(
                make_metric_row(
                    target=target,
                    variant_id=variant_id,
                    family="pseudo_label_student",
                    backend="lightgbm",
                    source_target=base.source_fmiss_target if target == "fmiss" else target,
                    metrics=holdout_metrics,
                    notes=str(row["notes"]),
                )
            )
            prediction_frames.append(
                make_prediction_frame(
                    target=target,
                    variant_id=variant_id,
                    family="pseudo_label_student",
                    backend="lightgbm",
                    df=target_test_df,
                    y_true=y_test,
                    pred=holdout_pred,
                )
            )
            student_cv_rows.append(
                {
                    "target": target,
                    "variant_id": variant_id,
                    "family": "pseudo_label_student",
                    "backend": "lightgbm",
                    "selected_pseudo_rows": int(len(pseudo_selected)),
                    "mean_pseudo_weight": float(pseudo_selected["pseudo_weight"].mean()) if not pseudo_selected.empty else 0.0,
                    "cv_mae": float(cv_metrics["mae"]),
                    "cv_rmse": float(cv_metrics["rmse"]),
                    "cv_r2": float(cv_metrics["r2"]),
                    "cv_bias": float(cv_metrics["bias"]),
                    "cv_calibration_slope": float(cv_metrics["calibration_slope"]),
                    "cv_calibration_intercept": float(cv_metrics["calibration_intercept"]),
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["target", "r2", "mae"], ascending=[True, False, True]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    pseudo_summary_df = pd.DataFrame(pseudo_summary_rows)
    pseudo_candidates_df = pd.concat(pseudo_candidate_frames, ignore_index=True) if pseudo_candidate_frames else pd.DataFrame()
    pseudo_selected_df = pd.concat(pseudo_selected_frames, ignore_index=True) if pseudo_selected_frames else pd.DataFrame()
    manifold_df = pd.DataFrame(manifold_rows)
    teacher_perf_df = pd.concat(teacher_perf_frames, ignore_index=True) if teacher_perf_frames else pd.DataFrame()
    student_cv_df = pd.DataFrame(student_cv_rows)
    cluster_summary_df = pd.concat(cluster_summary_frames, ignore_index=True) if cluster_summary_frames else pd.DataFrame()
    transport_df = pd.concat(transport_rows, ignore_index=True) if transport_rows else pd.DataFrame()

    if not student_cv_df.empty:
        for target in sorted(student_cv_df["target"].dropna().astype(str).unique().tolist()):
            target_cv = student_cv_df[student_cv_df["target"].astype(str) == target].copy()
            teacher_target_cv = target_cv[target_cv["family"] != "pseudo_label_student"].copy()
            teacher_target_cv = teacher_target_cv.sort_values(["cv_r2", "cv_mae"], ascending=[False, True])
            reference_variant_id = str(teacher_target_cv.iloc[0]["variant_id"])
            cv_for_picker = target_cv.rename(
                columns={
                    "cv_mae": "mae",
                    "cv_rmse": "rmse",
                    "cv_r2": "r2",
                    "cv_bias": "bias",
                    "cv_calibration_slope": "calibration_slope",
                    "cv_calibration_intercept": "calibration_intercept",
                }
            )
            rec = pick_r2_first_recommendation(
                cv_for_picker,
                target=target,
                reference_variant_id=reference_variant_id,
                selection_metric=config.selection_metric,
                mae_guardrail_pct=config.mae_guardrail_pct,
                min_r2_improvement=config.min_r2_improvement,
            )
            holdout_row = metrics_df[
                (metrics_df["target"].astype(str) == target)
                & (metrics_df["variant_id"].astype(str) == rec["recommended_variant_id"])
            ].iloc[0]
            rec["recommended_holdout_mae"] = float(holdout_row["mae"])
            rec["recommended_holdout_r2"] = float(holdout_row["r2"])
            rec["recommended_holdout_bias"] = float(holdout_row["bias"])
            recommendation_rows.append(rec)

    recommendation_df = pd.DataFrame(recommendation_rows)
    holdout_best_rows: list[dict[str, Any]] = []
    for target in sorted(metrics_df["target"].dropna().astype(str).unique().tolist()):
        best = metrics_df[metrics_df["target"].astype(str) == target].sort_values(
            ["r2", "mae"],
            ascending=[False, True],
        ).iloc[0]
        holdout_best_rows.append(
            {
                "target": target,
                "best_holdout_variant_id": str(best["variant_id"]),
                "best_holdout_family": str(best["family"]),
                "best_holdout_mae": float(best["mae"]),
                "best_holdout_r2": float(best["r2"]),
                "best_holdout_bias": float(best["bias"]),
            }
        )
    holdout_best_df = pd.DataFrame(holdout_best_rows)
    if not recommendation_df.empty and not holdout_best_df.empty:
        recommendation_df = recommendation_df.merge(holdout_best_df, on="target", how="left")
    split_manifest = {
        **split.manifest,
        "support_quantile": float(config.support_quantile),
        "agreement_quantile": float(config.agreement_quantile),
        "cluster_coverage_quantile": float(config.cluster_coverage_quantile),
        "max_pseudo_multiplier": float(config.max_pseudo_multiplier),
        "max_pseudo_per_recording": int(config.max_pseudo_per_recording),
        "max_pseudo_per_cluster": int(config.max_pseudo_per_cluster),
        "max_pseudo_per_labeled_ratio": float(config.max_pseudo_per_labeled_ratio),
    }

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    pseudo_summary_df.to_csv(output_dir / "pseudo_label_summary.csv", index=False)
    pseudo_candidates_df.to_csv(output_dir / "pseudo_candidate_table.csv", index=False)
    pseudo_selected_df.to_csv(output_dir / "pseudo_selected_rows.csv", index=False)
    manifold_df.to_csv(output_dir / "manifold_summary.csv", index=False)
    teacher_perf_df.to_csv(output_dir / "teacher_agreement_summary.csv", index=False)
    student_cv_df.to_csv(output_dir / "student_cv_summary.csv", index=False)
    cluster_summary_df.to_csv(output_dir / "pseudo_cluster_summary.csv", index=False)
    transport_df.to_csv(output_dir / "transport_shift_summary.csv", index=False)
    filtered_manifest.to_csv(output_dir / "filtered_feature_manifest.csv", index=False)
    recommendation_df.to_csv(output_dir / "per_target_recommendation.csv", index=False)
    holdout_best_df.to_csv(output_dir / "holdout_best_by_target.csv", index=False)
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


def run_with_config(config: PseudoLabelManifoldConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = PseudoLabelManifoldConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.max_pseudo_multiplier = 1.5
        cli_config.top_shift_features = 24
        cli_config.manifold_clusters = 3
        cli_config.max_pseudo_per_recording = 20
        cli_config.max_pseudo_per_cluster = 48
    _run_with_config(cli_config)
