from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from qc_thesis.modeling.shared import StackSet, compute_feature_shift_table
from qc_thesis.modeling.neural import prepare_feature_table
from qc_thesis.modeling.shared import make_main_split

from .backends import (
    _family_weights,
    _fit_anchor_scores,
    _fit_backend_predict,
    _grouped_oof_splits,
    apply_quantile_transport_model,
    assign_kmeans_subgroups,
    build_stack_frame,
    build_stack_set,
    compute_importance_weights,
    filter_target_rows,
    fit_kmeans_subgroup_model,
    fit_pca_support_projector,
    fit_quantile_transport_model,
    fit_tree_regressor,
    mask_feature_columns,
    predict_tree_regressor,
    score_predictions,
    select_shift_filtered_features,
)
from .specs import (
    FPosClusterTrustPseudoConfig,
    MLStatisticsWorkbenchConfig,
    PseudoLabelManifoldConfig,
    SupportArtifacts,
    TeacherBundle,
)


class BaseStackConfig(Protocol):
    parquet_path: Any
    output_root: Any
    run_name: str | None
    random_state: int
    paired_final_holdout_rows: int
    max_pseudo_multiplier: float
    max_pseudo_per_recording: int
    max_pseudo_per_cluster: int
    max_pseudo_per_labeled_ratio: float
    cluster_trust_power: float
    study_trust_weight: float
    min_trust: float
    verbose: bool


@dataclass(frozen=True)
class BaseStackState:
    base: MLStatisticsWorkbenchConfig
    prepared: Any
    split: Any
    target_train_df: pd.DataFrame
    target_test_df: pd.DataFrame
    unlabeled_df: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    trust_selected: pd.DataFrame
    base_stack: StackSet
    source_variant_id: str


def _base_configs(config: BaseStackConfig) -> tuple[MLStatisticsWorkbenchConfig, PseudoLabelManifoldConfig, FPosClusterTrustPseudoConfig]:
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
        max_pseudo_multiplier=config.max_pseudo_multiplier,
        max_pseudo_per_recording=config.max_pseudo_per_recording,
        max_pseudo_per_cluster=config.max_pseudo_per_cluster,
        max_pseudo_per_labeled_ratio=config.max_pseudo_per_labeled_ratio,
        verbose=config.verbose,
    )
    trust_base = FPosClusterTrustPseudoConfig(
        parquet_path=config.parquet_path,
        output_root=config.output_root,
        run_name=config.run_name,
        random_state=config.random_state,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        max_pseudo_multiplier=config.max_pseudo_multiplier,
        max_pseudo_per_recording=config.max_pseudo_per_recording,
        max_pseudo_per_cluster=config.max_pseudo_per_cluster,
        max_pseudo_per_labeled_ratio=config.max_pseudo_per_labeled_ratio,
        cluster_trust_power=config.cluster_trust_power,
        study_trust_weight=config.study_trust_weight,
        min_trust=config.min_trust,
        verbose=config.verbose,
    )
    return base, pseudo_base, trust_base


def _source_variant_id(teachers: list[Any]) -> str:
    if len(teachers) >= 2:
        return str(teachers[1].variant_id)
    if not teachers:
        raise ValueError("teacher list is empty")
    return str(teachers[0].variant_id)


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
                "train_frame": [contextual_train, teachers[1].train_frame, teachers[2].train_frame],
                "unlabeled_frame": [contextual_unlabeled, teachers[1].unlabeled_frame, teachers[2].unlabeled_frame],
                "test_frame": [contextual_test, teachers[1].test_frame, teachers[2].test_frame],
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
                "train_frame": [contextual_train, teachers[1].train_frame, teachers[2].train_frame],
                "unlabeled_frame": [contextual_unlabeled, teachers[1].unlabeled_frame, teachers[2].unlabeled_frame],
                "test_frame": [contextual_test, teachers[1].test_frame, teachers[2].test_frame],
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
    cluster_count = np.array([support_artifacts.labeled_cluster_counts.get(int(c), 0) for c in clusters], dtype=float)
    cluster_share = np.array([support_artifacts.labeled_cluster_share.get(int(c), 0.0) for c in clusters], dtype=float)
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
    weight_map = {str(row["variant_id"]): float(row["ensemble_weight"]) for _, row in teacher_perf_df.iterrows()}
    teacher_cols = [str(col) for col in frame.columns if str(col) in weight_map]
    weights = np.array([weight_map[col] for col in teacher_cols], dtype=float)
    norm_weights = weights / max(float(weights.sum()), 1e-8)
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
            oof_pred[val_idx] = _fit_predict(
                feature_train=teacher.train_frame.iloc[train_idx],
                y_train=y_train[train_idx],
                feature_eval=teacher.train_frame.iloc[val_idx],
                sample_weight=None,
                base_config=base,
                random_state_offset=(10000 * idx) + split_id,
            )
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
        payload = {str(row["variant_id"]): np.asarray(row["pred"], dtype=float) for row in rows}
        return pd.DataFrame(payload)

    teacher_perf_df = _teacher_weight_table(pd.DataFrame(perf_rows))
    train_frame = _aggregate_teacher_predictions(_to_frame(train_rows), teacher_perf_df)
    unlabeled_frame = _aggregate_teacher_predictions(_to_frame(unlabeled_rows), teacher_perf_df)
    test_frame = _aggregate_teacher_predictions(_to_frame(test_rows), teacher_perf_df)
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
    cluster_support_cut = merged.groupby("cluster_id", dropna=False)["support_distance"].transform(
        lambda s: float(s.quantile(config.support_quantile))
    )
    cluster_agreement_cut = merged.groupby("cluster_id", dropna=False)["teacher_std"].transform(
        lambda s: float(s.quantile(config.agreement_quantile))
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
        * ((0.45 * support_conf) + (0.35 * agreement_conf) + (0.20 * np.minimum(coverage_conf, 1.0))),
        float(config.pseudo_weight_floor),
        float(config.pseudo_weight_scale),
    )
    selected["pseudo_label"] = selected["teacher_mean"].astype(float)
    selected = selected.sort_values(
        ["pseudo_weight", "support_confidence", "teacher_std"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    cluster_candidate_counts = selected["cluster_id"].value_counts().to_dict()
    cluster_priority: dict[int, float] = {}
    for cluster_id, candidate_count in cluster_candidate_counts.items():
        labeled_count = float(selected.loc[selected["cluster_id"] == cluster_id, "cluster_labeled_count"].iloc[0])
        cluster_priority[int(cluster_id)] = np.sqrt(max(labeled_count, 1.0) * max(float(candidate_count), 1.0))
    total_priority = max(float(sum(cluster_priority.values())), 1e-8)
    cluster_quotas = {
        int(cluster_id): min(
            int(config.max_pseudo_per_cluster),
            max(
                1,
                int(round(float(selected.loc[selected["cluster_id"] == cluster_id, "cluster_labeled_count"].iloc[0]) * float(config.max_pseudo_per_labeled_ratio))),
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
            if _try_take(int(row.Index), row) and len(keep_rows) >= int(max_rows):
                break

    selected = selected.iloc[sorted(set(keep_rows))].sort_values(
        ["pseudo_weight", "support_confidence", "teacher_std"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    candidate_df["selected_final"] = candidate_df["row_uid"].astype(str).isin(selected["row_uid"].astype(str))
    return candidate_df, selected


def _trust_from_group_stats(
    group_df: pd.DataFrame,
    *,
    group_col: str,
) -> pd.DataFrame:
    overall_mae = float(group_df["teacher_abs_error"].mean())
    max_n = max(int(group_df[group_col].value_counts().max()), 1)
    stats = (
        group_df.groupby(group_col, dropna=False)
        .agg(
            labeled_rows=("row_uid", "size"),
            teacher_mae=("teacher_abs_error", "mean"),
            mean_support_confidence=("support_confidence", "mean"),
        )
        .reset_index()
    )
    err_score = np.exp(-stats["teacher_mae"].astype(float) / max(overall_mae, 1e-6))
    size_score = np.sqrt(stats["labeled_rows"].astype(float) / float(max_n))
    support_score = np.clip(stats["mean_support_confidence"].astype(float), 0.0, 1.0)
    trust = 0.60 * err_score + 0.25 * support_score + 0.15 * size_score
    trust = trust / max(float(np.nanmax(trust)), 1e-6)
    stats["trust_score"] = trust.astype(float)
    return stats


def _apply_trust_selection(
    *,
    candidate_df: pd.DataFrame,
    cluster_trust_df: pd.DataFrame,
    study_trust_df: pd.DataFrame,
    max_rows: int,
    config: FPosClusterTrustPseudoConfig,
) -> pd.DataFrame:
    pool = candidate_df[candidate_df["selected_candidate"].astype(bool)].copy()
    if pool.empty:
        return pool

    pool = pool.merge(
        cluster_trust_df.loc[:, ["cluster_id", "trust_score"]].rename(columns={"trust_score": "cluster_trust"}),
        on="cluster_id",
        how="left",
    )
    pool = pool.merge(
        study_trust_df.loc[:, ["study_set", "trust_score"]].rename(columns={"trust_score": "study_trust"}),
        on="study_set",
        how="left",
    )
    pool["cluster_trust"] = pool["cluster_trust"].fillna(float(config.min_trust))
    pool["study_trust"] = pool["study_trust"].fillna(float(config.min_trust))
    pool["combined_trust"] = (
        ((1.0 - float(config.study_trust_weight)) * pool["cluster_trust"].astype(float))
        + (float(config.study_trust_weight) * pool["study_trust"].astype(float))
    ).clip(lower=float(config.min_trust), upper=1.0)
    std_denom = max(float(np.nanquantile(pool["teacher_std"].astype(float), 0.75)), 1e-6)
    base_conf = (
        0.65 * np.exp(-pool["teacher_std"].astype(float) / std_denom)
        + 0.35 * np.clip(pool["support_confidence"].astype(float), 0.0, 1.0)
    )
    pool["base_pseudo_weight"] = np.clip(0.25 * base_conf, 0.08, 0.25)
    pool["trust_adjusted_weight"] = (
        pool["base_pseudo_weight"].astype(float)
        * np.power(pool["combined_trust"].astype(float), float(config.cluster_trust_power))
    ).clip(lower=0.08, upper=0.25)
    cluster_caps = (
        pool.groupby("cluster_id")["combined_trust"]
        .mean()
        .map(
            lambda trust: max(
                1,
                min(
                    int(config.max_pseudo_per_cluster),
                    int(
                        round(
                            min(
                                float(config.max_pseudo_per_cluster),
                                float(config.max_pseudo_per_cluster)
                                * (max(float(trust), float(config.min_trust)) ** float(config.cluster_trust_power)),
                            )
                        )
                    ),
                ),
            )
        )
        .to_dict()
    )
    cluster_ratio_caps = (
        pool.groupby("cluster_id")["cluster_labeled_count"]
        .max()
        .map(lambda labeled_count: max(1, int(round(float(labeled_count) * float(config.max_pseudo_per_labeled_ratio)))))
        .to_dict()
    )

    selected_rows: list[pd.Series] = []
    recording_counts: dict[str, int] = {}
    cluster_counts: dict[int, int] = {}
    pool = pool.sort_values(
        ["trust_adjusted_weight", "combined_trust", "support_confidence", "base_pseudo_weight", "teacher_std"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    for _, row in pool.iterrows():
        if len(selected_rows) >= int(max_rows):
            break
        recording_key = str(row["recording_key"])
        cluster_id = int(row["cluster_id"])
        if recording_counts.get(recording_key, 0) >= int(config.max_pseudo_per_recording):
            continue
        cluster_cap = min(
            cluster_caps.get(cluster_id, int(config.max_pseudo_per_cluster)),
            cluster_ratio_caps.get(cluster_id, int(config.max_pseudo_per_cluster)),
        )
        if cluster_counts.get(cluster_id, 0) >= int(cluster_cap):
            continue
        selected_rows.append(row)
        recording_counts[recording_key] = recording_counts.get(recording_key, 0) + 1
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    if not selected_rows:
        return pool.head(0).copy()

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    if "pseudo_label" not in selected_df.columns:
        selected_df["pseudo_label"] = selected_df["teacher_mean"].astype(float)
    selected_df["pseudo_weight"] = selected_df["trust_adjusted_weight"].astype(np.float32)
    return selected_df


def build_base_stack_state(*, target: str, config: BaseStackConfig) -> BaseStackState:
    base, pseudo_base, trust_base = _base_configs(config)
    prepared = prepare_feature_table(config.parquet_path)
    split = make_main_split(
        prepared.df,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        random_state=config.random_state,
    )
    unit_drop_cols, context_drop_cols, _ = _build_filtered_drop_cols(
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
        _student_templates,
        _transport_manifest,
    ) = _teacher_frames_for_target(
        target=target,
        split=split,
        prepared=prepared,
        base=base,
        config=pseudo_base,
        unit_drop_cols=unit_drop_cols,
        context_drop_cols=context_drop_cols,
    )
    teacher_train_df, teacher_unlabeled_df, teacher_test_df, _ = _teacher_predictions(
        teachers=teachers,
        labeled_df=target_train_df,
        y_train=y_train,
        base=base,
    )
    support_artifacts = _build_support_artifacts(
        paired_non_test_all=split.paired_non_test_all,
        labeled_train_df=target_train_df,
        unit_cols=prepared.unit_feature_cols,
        config=pseudo_base,
    )
    support_df = _manifold_support_table(
        unlabeled_df=unlabeled_df,
        labeled_train_df=target_train_df,
        support_artifacts=support_artifacts,
    )
    support_train_df = _manifold_support_table(
        unlabeled_df=target_train_df,
        labeled_train_df=target_train_df,
        support_artifacts=support_artifacts,
    )
    train_trust_df = pd.concat(
        [
            target_train_df.loc[:, ["row_uid", "recording_key", "study_set"]].reset_index(drop=True),
            support_train_df.loc[:, ["cluster_id", "support_confidence"]].reset_index(drop=True),
        ],
        axis=1,
    )
    train_trust_df["teacher_abs_error"] = np.abs(y_train - teacher_train_df["teacher_mean"].to_numpy(dtype=float))
    cluster_trust_df = _trust_from_group_stats(train_trust_df, group_col="cluster_id")
    study_trust_df = _trust_from_group_stats(train_trust_df, group_col="study_set")
    candidate_df, _ = _select_pseudo_labels(
        unlabeled_df=unlabeled_df,
        unlabeled_teacher_df=teacher_unlabeled_df,
        support_df=support_df,
        config=pseudo_base,
        max_rows=int(round(len(target_train_df) * float(config.max_pseudo_multiplier))),
    )
    trust_selected = _apply_trust_selection(
        candidate_df=candidate_df,
        cluster_trust_df=cluster_trust_df,
        study_trust_df=study_trust_df,
        max_rows=int(round(len(target_train_df) * float(config.max_pseudo_multiplier))),
        config=trust_base,
    )

    filtered_drop_cols = set(unit_drop_cols) | set(context_drop_cols)
    filtered_context_cols = [col for col in prepared.lightgbm_context_cols if col not in filtered_drop_cols]
    filtered_unit_cols = [col for col in prepared.unit_feature_cols if col not in set(unit_drop_cols)]
    target_train_masked = mask_feature_columns(target_train_df, filtered_drop_cols)
    target_test_masked = mask_feature_columns(target_test_df, filtered_drop_cols)
    unlabeled_masked = mask_feature_columns(unlabeled_df, filtered_drop_cols)
    anchor_feature_cols = sorted(set(filtered_context_cols + filtered_unit_cols))
    anchor_oof, anchor_unlabeled, anchor_test = _fit_anchor_scores(
        matched_train_df=target_train_masked,
        matched_test_df=target_test_masked,
        unmatched_df=unlabeled_masked,
        feature_cols=anchor_feature_cols,
        n_splits=base.model_selection_splits,
        config=type(
            "AnchorCfg",
            (),
            {
                "anchor_learning_rate": 0.06,
                "anchor_max_depth": 4,
                "anchor_max_iter": 250,
                "random_state": config.random_state,
            },
        )(),
    )
    matched_support_conf_test = _manifold_support_table(
        unlabeled_df=target_test_df,
        labeled_train_df=target_train_df,
        support_artifacts=support_artifacts,
    )["support_confidence"].to_numpy(dtype=float)
    source_variant_id = _source_variant_id(teachers)
    base_stack = build_stack_set(
        train_df=target_train_masked,
        unlabeled_df=unlabeled_masked,
        test_df=target_test_masked,
        context_cols=filtered_context_cols,
        train_features={
            "source_pred": teacher_train_df[source_variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_oof,
            "matched_support_conf": support_train_df["support_confidence"].to_numpy(dtype=float),
            "anchor_support_margin": anchor_oof - support_train_df["support_confidence"].to_numpy(dtype=float),
        },
        unlabeled_features={
            "source_pred": teacher_unlabeled_df[source_variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_unlabeled,
            "matched_support_conf": support_df["support_confidence"].to_numpy(dtype=float),
            "anchor_support_margin": anchor_unlabeled - support_df["support_confidence"].to_numpy(dtype=float),
        },
        test_features={
            "source_pred": teacher_test_df[source_variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_test,
            "matched_support_conf": matched_support_conf_test,
            "anchor_support_margin": anchor_test - matched_support_conf_test,
        },
    )
    return BaseStackState(
        base=base,
        prepared=prepared,
        split=split,
        target_train_df=target_train_df,
        target_test_df=target_test_df,
        unlabeled_df=unlabeled_df,
        y_train=y_train,
        y_test=y_test,
        trust_selected=trust_selected,
        base_stack=base_stack,
        source_variant_id=source_variant_id,
    )


def build_embedding_universe(
    *,
    split: Any,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    universe_df = (
        pd.concat(
            [
                split.hybrid_source.reset_index(drop=True),
                split.paired_non_test_all.reset_index(drop=True),
                split.ssl_pool.reset_index(drop=True),
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=["row_uid"])
        .reset_index(drop=True)
    )
    if max_rows is not None and len(universe_df) > int(max_rows):
        universe_df = universe_df.sample(n=int(max_rows), random_state=random_state).reset_index(drop=True)
    return universe_df


def assemble_training_data(
    *,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    labeled_meta_df: pd.DataFrame,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    family_alpha: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    labeled_weight = _family_weights(labeled_meta_df["study_set"], alpha=family_alpha)
    if pseudo_selected.empty:
        return labeled_frame.reset_index(drop=True), np.asarray(labeled_y, dtype=float), labeled_weight
    pseudo_idx = pseudo_selected["unlabeled_row_idx"].to_numpy(dtype=int)
    pseudo_frame = unlabeled_frame.iloc[pseudo_idx].reset_index(drop=True)
    full_frame = pd.concat([labeled_frame.reset_index(drop=True), pseudo_frame], ignore_index=True)
    full_y = np.concatenate([np.asarray(labeled_y, dtype=float), pseudo_selected["pseudo_label"].to_numpy(dtype=float)])
    full_weight = np.concatenate([labeled_weight, pseudo_selected["pseudo_weight"].to_numpy(dtype=np.float32)])
    return full_frame, full_y, full_weight


def evaluate_stack_variant(
    *,
    state: BaseStackState,
    stack: StackSet,
    backend_id: str,
    family_alpha: float,
    random_state_offset: int,
) -> tuple[dict[str, float], np.ndarray]:
    splits = _grouped_oof_splits(state.target_train_df, state.base.model_selection_splits)
    oof = np.full(len(state.target_train_df), np.nan, dtype=float)
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        val_recordings = set(state.target_train_df.iloc[val_idx]["recording_key"].astype(str))
        fold_pseudo = state.trust_selected[
            ~state.trust_selected["recording_key"].astype(str).isin(val_recordings)
        ].reset_index(drop=True)
        train_fold, train_y, train_weight = assemble_training_data(
            labeled_frame=stack.train.iloc[train_idx].reset_index(drop=True),
            labeled_y=state.y_train[train_idx],
            labeled_meta_df=state.target_train_df.iloc[train_idx].reset_index(drop=True),
            pseudo_selected=fold_pseudo,
            unlabeled_frame=stack.unlabeled,
            family_alpha=family_alpha,
        )
        oof[val_idx] = _fit_backend_predict(
            X_train=train_fold,
            y_train=train_y,
            X_eval=stack.train.iloc[val_idx].reset_index(drop=True),
            sample_weight=train_weight,
            target_transform_mode=state.base.resolved_target_transform(),
            backend_id=backend_id,
            random_state=state.base.random_state + random_state_offset + split_id,
        )

    full_train, full_y, full_weight = assemble_training_data(
        labeled_frame=stack.train,
        labeled_y=state.y_train,
        labeled_meta_df=state.target_train_df,
        pseudo_selected=state.trust_selected,
        unlabeled_frame=stack.unlabeled,
        family_alpha=family_alpha,
    )
    holdout_pred = _fit_backend_predict(
        X_train=full_train,
        y_train=full_y,
        X_eval=stack.test.reset_index(drop=True),
        sample_weight=full_weight,
        target_transform_mode=state.base.resolved_target_transform(),
        backend_id=backend_id,
        random_state=state.base.random_state + random_state_offset + 100,
    )
    return score_predictions(state.y_train, oof), holdout_pred


__all__ = [
    "BaseStackState",
    "assemble_training_data",
    "build_base_stack_state",
    "build_embedding_universe",
    "evaluate_stack_variant",
]
