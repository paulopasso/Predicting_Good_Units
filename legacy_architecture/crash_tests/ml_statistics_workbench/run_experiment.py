from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.ml_statistics_workbench.config import MLStatisticsWorkbenchConfig
from crash_tests.ml_statistics_workbench.methods import (
    apply_study_bias_correction,
    build_stack_frame,
    compute_overlap_importance_weights,
    compute_soft_group_balance_weights,
    compute_importance_weights,
    filter_target_rows,
    fit_tree_regressor,
    fit_latent_cluster_model,
    fit_study_bias_correction,
    latent_cluster_responsibilities,
    leave_one_out_study_bias_adjustment,
    mask_feature_columns,
    make_latent_summary_frame,
    make_metric_row,
    make_prediction_frame,
    make_study_indicator_frame,
    pick_r2_first_recommendation,
    predict_embeddings_from_state,
    predict_tree_regressor,
    prepare_unit_input,
    rank_latent_dims_by_correlation,
    score_predictions,
    select_shift_filtered_features,
    select_shift_protected_features,
    train_zero_shot_mmd_source,
)
from crash_tests.domain_shift_audit.report import compute_feature_shift_table
from crash_tests.mmd_domain_adaptation.dataset import prepare_feature_table
from crash_tests.self_supervised_domain_adaptation.split_utils import make_main_split


_METRIC_COLUMNS = [
    "protocol",
    "target",
    "variant_id",
    "family",
    "backend",
    "source_target",
    "notes",
    "mae",
    "rmse",
    "r2",
    "bias",
    "calibration_slope",
    "calibration_intercept",
]
_PREDICTION_COLUMNS = [
    "protocol",
    "target",
    "variant_id",
    "family",
    "backend",
    "row_uid",
    "recording_key",
    "study_set",
    "true",
    "pred",
    "abs_error",
]
_HISTORY_COLUMNS = [
    "protocol",
    "target",
    "variant_id",
    "epoch",
    "train_task_loss",
    "train_mmd_loss",
    "train_total_loss",
    "val_mae",
]
_WEIGHT_COLUMNS = [
    "strategy",
    "target",
    "n_source",
    "n_target",
    "mean",
    "std",
    "min",
    "p10",
    "p50",
    "p90",
    "max",
    "support_lower",
    "support_upper",
    "source_in_support_frac",
]
_ABLATION_COLUMNS = [
    "target",
    "variant_id",
    "family",
    "backend",
    "feature_components",
    "latent_feature_mode",
    "latent_topk",
    "selected_latent_dims",
    "cv_mae",
    "cv_rmse",
    "cv_r2",
    "cv_bias",
    "cv_calibration_slope",
    "cv_calibration_intercept",
    "gate_reference_variant_id",
    "gate_reference_cv_mae",
    "gate_reference_cv_r2",
    "passed_cv_gate",
    "holdout_scored",
    "holdout_mae",
    "holdout_r2",
]
_RECOMMEND_COLUMNS = [
    "target",
    "selection_metric",
    "reference_variant_id",
    "reference_mae",
    "reference_r2",
    "mae_guardrail_limit",
    "recommended_variant_id",
    "recommended_mae",
    "recommended_r2",
    "r2_improvement",
    "mae_regression_pct",
    "passes_guardrail",
    "beats_reference",
    "is_breakthrough",
]
_FILTERED_MANIFEST_COLUMNS = [
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
_CLUSTER_COLUMNS = [
    "target",
    "variant_id",
    "backend",
    "n_clusters",
    "cluster_id",
    "target_non_test_rows",
    "source_soft_mass",
    "paired_train_soft_mass",
    "paired_test_soft_mass",
    "top_studies",
]
_STUDY_HOLDOUT_COLUMNS = [
    "target",
    "variant_id",
    "study_set",
    "n_rows",
    "mae",
    "rmse",
    "r2",
    "bias",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the focused ML/statistics transfer workbench")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet", help="Path to the main QC parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs", help="Root folder for outputs")
    parser.add_argument("--run-name", default=None, help="Optional explicit run name")
    parser.add_argument("--smoke", action="store_true", help="Run a faster smoke configuration")
    parser.add_argument("--quiet", action="store_true", help="Reduce runner logging")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> MLStatisticsWorkbenchConfig:
    config = MLStatisticsWorkbenchConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        config.lightgbm_estimators = 100
        config.xgboost_estimators = 140
        config.mmd_source_train_epochs = 6
        config.mmd_patience = 2
        config.model_selection_splits = 2
        config.latent_candidate_topk = (4, 8)
    return config


def _log(config: MLStatisticsWorkbenchConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns + [c for c in frame.columns if c not in columns]]


def _grouped_oof_splits(df: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = df["recording_key"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    effective_splits = min(max(2, n_splits), len(unique_groups))
    splitter = GroupKFold(n_splits=effective_splits)
    return [(np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)) for train_idx, val_idx in splitter.split(df, groups=groups)]


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.loc[:, columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)


def _concat_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([frame.reset_index(drop=True) for frame in frames], axis=1)


def _build_filtered_feature_manifests(
    *,
    split: Any,
    prepared: Any,
    config: MLStatisticsWorkbenchConfig,
) -> tuple[pd.DataFrame, list[str], list[str]]:
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
        top_n=config.filtered_shift_top_n_unit,
        min_abs_smd=config.filtered_shift_min_abs_smd,
        preserve_shape_features=config.filtered_preserve_shape_features,
        feature_set="unit",
    )
    full_context_manifest, full_context_drop_cols = select_shift_filtered_features(
        shift_full_context,
        allowed_cols=prepared.lightgbm_context_cols,
        top_n=config.filtered_shift_top_n_full_context,
        min_abs_smd=config.filtered_shift_min_abs_smd,
        preserve_shape_features=config.filtered_preserve_shape_features,
        feature_set="full_context",
    )
    manifest_df = pd.concat([unit_manifest, full_context_manifest], ignore_index=True)
    return manifest_df, unit_drop_cols, full_context_drop_cols


def _score_candidate_cv(
    *,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    feature_train: pd.DataFrame,
    backend: str,
    config: MLStatisticsWorkbenchConfig,
    random_state_offset: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    splits = _grouped_oof_splits(train_df, config.model_selection_splits)
    oof_pred = np.full(len(train_df), np.nan, dtype=float)
    transform_mode = config.resolved_target_transform()
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        model = fit_tree_regressor(
            feature_train.iloc[train_idx],
            y_train[train_idx],
            backend=backend,
            random_state=config.random_state + random_state_offset + split_id,
            target_transform_mode=transform_mode,
            lightgbm_estimators=config.lightgbm_estimators,
            lightgbm_learning_rate=config.lightgbm_learning_rate,
            lightgbm_num_leaves=config.lightgbm_num_leaves,
            xgboost_estimators=config.xgboost_estimators,
            xgboost_learning_rate=config.xgboost_learning_rate,
            xgboost_max_depth=config.xgboost_max_depth,
            xgboost_min_child_weight=config.xgboost_min_child_weight,
            xgboost_subsample=config.xgboost_subsample,
            xgboost_colsample_bytree=config.xgboost_colsample_bytree,
            sample_weight=sample_weight[train_idx] if sample_weight is not None else None,
        )
        oof_pred[val_idx] = predict_tree_regressor(model, feature_train.iloc[val_idx])
    if np.isnan(oof_pred).any():
        raise AssertionError("Grouped OOF predictions contain NaNs")
    return score_predictions(y_train, oof_pred), oof_pred


def _fit_predict_holdout(
    *,
    feature_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_test: pd.DataFrame,
    backend: str,
    config: MLStatisticsWorkbenchConfig,
    random_state_offset: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    model = fit_tree_regressor(
        feature_train,
        y_train,
        backend=backend,
        random_state=config.random_state + random_state_offset,
        target_transform_mode=config.resolved_target_transform(),
        lightgbm_estimators=config.lightgbm_estimators,
        lightgbm_learning_rate=config.lightgbm_learning_rate,
        lightgbm_num_leaves=config.lightgbm_num_leaves,
        xgboost_estimators=config.xgboost_estimators,
        xgboost_learning_rate=config.xgboost_learning_rate,
        xgboost_max_depth=config.xgboost_max_depth,
        xgboost_min_child_weight=config.xgboost_min_child_weight,
        xgboost_subsample=config.xgboost_subsample,
        xgboost_colsample_bytree=config.xgboost_colsample_bytree,
        sample_weight=sample_weight,
    )
    return predict_tree_regressor(model, feature_test)


def _topk_choice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: (-float(row["cv_r2"]), float(row["cv_mae"]), int(row["latent_topk"])))
    return ranked[0]


def _build_target_readme(
    *,
    target: str,
    metrics_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
) -> list[str]:
    lines = [f"### {target}", ""]
    target_metrics = metrics_df[metrics_df["target"] == target].sort_values(["r2", "mae"], ascending=[False, True])
    if target_metrics.empty:
        lines.append("- No holdout metrics produced.")
        lines.append("")
        return lines
    for _, row in target_metrics.iterrows():
        lines.append(f"- {row['variant_id']} | MAE `{row['mae']:.4f}` | R² `{row['r2']:.4f}`")
    rec = recommendations_df[recommendations_df["target"] == target]
    if not rec.empty:
        row = rec.iloc[0]
        lines.extend(
            [
                "",
                f"- recommendation: `{row['recommended_variant_id']}`",
                f"- reference: `{row['reference_variant_id']}`",
                f"- R² improvement: `{row['r2_improvement']:.4f}`",
            ]
        )
    lines.append("")
    return lines


def _build_readme(
    *,
    config: MLStatisticsWorkbenchConfig,
    split_manifest: dict[str, Any],
    metrics_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
) -> str:
    lines = [
        "# ML / Statistics Workbench Run",
        "",
        "## Config",
        f"- parquet_path: `{config.parquet_path}`",
        f"- selection_metric: `{config.selection_metric}`",
        f"- mae_guardrail_pct: `{config.mae_guardrail_pct:.3f}`",
        f"- target_transform: `{config.resolved_target_transform()}`",
        f"- use_domain_reweighted_source: `{config.use_domain_reweighted_source}`",
        f"- use_overlap_weighted_source: `{config.use_overlap_weighted_source}`",
        f"- use_subgroup_balanced_variants: `{config.use_subgroup_balanced_variants}`",
        f"- use_target_protected_shift_variants: `{config.use_target_protected_shift_variants}`",
        f"- use_cluster_aware_variants: `{config.use_cluster_aware_variants}`",
        f"- use_study_aware_variants: `{config.use_study_aware_variants}`",
        f"- use_study_bias_correction_variants: `{config.use_study_bias_correction_variants}`",
        f"- latent_selection_mode: `{config.latent_selection_mode}`",
        "",
        "## Main Split",
        f"- paired_test_rows: `{split_manifest['paired_test_rows']}`",
        f"- paired_finetune_rows: `{split_manifest['paired_finetune_rows']}`",
        f"- ssl_pool_rows: `{split_manifest['ssl_pool_rows']}`",
        f"- paired_test_recordings: `{split_manifest['paired_test_recordings']}`",
        f"- filtered_unit_drop_count: `{split_manifest.get('filtered_unit_drop_count', 0)}`",
        f"- filtered_full_context_drop_count: `{split_manifest.get('filtered_full_context_drop_count', 0)}`",
        "",
        "## Holdout Results",
        "",
    ]
    for target in config.targets:
        lines.extend(_build_target_readme(target=target, metrics_df=metrics_df, recommendations_df=recommendations_df))
    return "\n".join(lines) + "\n"


def _build_study_holdout_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame(columns=_STUDY_HOLDOUT_COLUMNS)
    rows: list[dict[str, Any]] = []
    grouped = predictions_df.groupby(["target", "variant_id", "study_set"], dropna=False)
    for (target, variant_id, study_set), group in grouped:
        y_true = group["true"].to_numpy(dtype=float)
        pred = group["pred"].to_numpy(dtype=float)
        metrics = score_predictions(y_true, pred)
        rows.append(
            {
                "target": target,
                "variant_id": variant_id,
                "study_set": study_set,
                "n_rows": int(len(group)),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "bias": metrics["bias"],
            }
        )
    out = pd.DataFrame(rows)
    return out.loc[:, _STUDY_HOLDOUT_COLUMNS].sort_values(
        ["target", "variant_id", "study_set"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def _run_target(
    *,
    target: str,
    split: Any,
    prepared: Any,
    config: MLStatisticsWorkbenchConfig,
    output_dir: Path,
    unit_drop_cols: list[str],
    full_context_drop_cols: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[pd.DataFrame],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    source_target = config.source_fmiss_target if target == "fmiss" else target
    target_train_df, y_train = filter_target_rows(split.paired_finetune, target)
    target_test_df, y_test = filter_target_rows(split.paired_test, target)
    source_df, y_source = filter_target_rows(split.hybrid_source, source_target)
    if target_train_df.empty or target_test_df.empty or source_df.empty:
        raise RuntimeError(f"Target {target} does not have enough rows for the workbench")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    target_filter_manifest_rows: list[dict[str, Any]] = []

    contextual_cols = prepared.lightgbm_context_cols
    unit_cols = prepared.unit_feature_cols
    study_categories = sorted(split.paired_non_test_all["study_set"].astype(str).unique().tolist())
    unlabeled_target_df = split.ssl_pool if not split.ssl_pool.empty else split.paired_non_test_all
    filtered_drop_cols = set(unit_drop_cols) | set(full_context_drop_cols)
    filtered_contextual_cols = [col for col in contextual_cols if col not in filtered_drop_cols]
    filtered_unit_cols = [col for col in unit_cols if col not in set(unit_drop_cols)]

    filtered_target_train_df = mask_feature_columns(target_train_df, filtered_drop_cols)
    filtered_target_test_df = mask_feature_columns(target_test_df, filtered_drop_cols)
    filtered_source_df = mask_feature_columns(source_df, unit_drop_cols)
    filtered_hybrid_source = mask_feature_columns(split.hybrid_source, unit_drop_cols)
    filtered_paired_non_test_all = mask_feature_columns(split.paired_non_test_all, unit_drop_cols)
    filtered_unlabeled_target_df = mask_feature_columns(unlabeled_target_df, unit_drop_cols)

    target_protected_context_drop_cols: list[str] = []
    target_protected_contextual_cols = list(contextual_cols)
    target_protected_target_train_df = target_train_df.copy()
    target_protected_target_test_df = target_test_df.copy()
    if config.use_target_protected_shift_variants:
        shift_full_context = compute_feature_shift_table(
            split.hybrid_source,
            split.paired_non_test_all,
            feature_cols=prepared.lightgbm_context_cols,
            comparison=f"hybrid_vs_paired_non_test_{target}",
            feature_set="full_context_target_protected",
        )
        target_context_manifest, target_protected_context_drop_cols = select_shift_protected_features(
            shift_full_context,
            train_df=target_train_df,
            target=target,
            allowed_cols=prepared.lightgbm_context_cols,
            top_n=config.target_protected_top_n_full_context,
            min_abs_smd=config.target_protected_min_abs_smd,
            preserve_shape_features=config.filtered_preserve_shape_features,
            protect_top_k=config.target_protected_feature_top_k,
            min_abs_corr_to_protect=config.target_protected_min_abs_corr,
            feature_set="full_context_target_protected",
        )
        target_filter_manifest_rows.extend(target_context_manifest.to_dict(orient="records"))
        target_protected_contextual_cols = [
            col for col in contextual_cols if col not in set(target_protected_context_drop_cols)
        ]
        target_protected_target_train_df = mask_feature_columns(target_train_df, target_protected_context_drop_cols)
        target_protected_target_test_df = mask_feature_columns(target_test_df, target_protected_context_drop_cols)

    _log(config, f"[{target}] computing source-domain reweighting")
    weight_result = compute_importance_weights(
        source_df,
        unlabeled_target_df,
        feature_cols=unit_cols,
        random_state=config.random_state,
        clip_min=config.importance_weight_clip_min,
        clip_max=config.importance_weight_clip_max,
        c_value=config.domain_classifier_c,
    )
    weight_rows.append({"strategy": "logistic_odds", "target": target, **weight_result.summary})
    overlap_weight_result = None
    if config.use_overlap_weighted_source:
        overlap_weight_result = compute_overlap_importance_weights(
            source_df,
            unlabeled_target_df,
            feature_cols=unit_cols,
            random_state=config.random_state + 11,
            clip_min=config.importance_weight_clip_min,
            clip_max=config.importance_weight_clip_max,
            c_value=config.domain_classifier_c,
            support_quantiles=config.overlap_support_quantiles,
            outside_support_scale=config.overlap_outside_support_scale,
        )
        weight_rows.append({"strategy": "overlap_trimmed_odds", "target": target, **overlap_weight_result.summary})

    _log(config, f"[{target}] training source encoder artifact")
    embed_artifact = train_zero_shot_mmd_source(
        target=target,
        hybrid_source=split.hybrid_source,
        paired_non_test_all=split.paired_non_test_all,
        ssl_pool=unlabeled_target_df,
        prepared=prepared,
        random_state=config.random_state,
        source_fmiss_target=config.source_fmiss_target,
        mmd_lambda=0.0,
        batch_size=config.mmd_batch_size,
        hidden_dim=config.mmd_hidden_dim,
        latent_dim=config.mmd_latent_dim,
        source_train_epochs=config.mmd_source_train_epochs,
        source_lr=config.mmd_source_lr,
        weight_decay=config.mmd_weight_decay,
        patience=config.mmd_patience,
        verbose=config.verbose,
        checkpoint_path=(output_dir / "checkpoints" / f"embed_nommd_{target}_source.pt") if config.save_checkpoints else None,
    )
    history_rows.extend(
        {
            "protocol": "main",
            "target": target,
            "variant_id": "embed_nommd_source",
            **row,
        }
        for row in embed_artifact.history
    )
    filtered_embed_artifact = None
    if config.use_shift_filtered_variants:
        _log(config, f"[{target}] training filtered source encoder artifact")
        filtered_embed_artifact = train_zero_shot_mmd_source(
            target=target,
            hybrid_source=filtered_hybrid_source,
            paired_non_test_all=filtered_paired_non_test_all,
            ssl_pool=filtered_unlabeled_target_df,
            prepared=prepared,
            random_state=config.random_state + 17,
            source_fmiss_target=config.source_fmiss_target,
            mmd_lambda=0.0,
            batch_size=config.mmd_batch_size,
            hidden_dim=config.mmd_hidden_dim,
            latent_dim=config.mmd_latent_dim,
            source_train_epochs=config.mmd_source_train_epochs,
            source_lr=config.mmd_source_lr,
            weight_decay=config.mmd_weight_decay,
            patience=config.mmd_patience,
            verbose=config.verbose,
            checkpoint_path=(output_dir / "checkpoints" / f"embed_nommd_filtered_{target}_source.pt") if config.save_checkpoints else None,
        )
        history_rows.extend(
            {
                "protocol": "main",
                "target": target,
                "variant_id": "embed_nommd_filtered_source",
                **row,
            }
            for row in filtered_embed_artifact.history
        )

    source_input = prepare_unit_input(source_df, prepared, embed_artifact.unit_scaler)
    paired_non_test_input = prepare_unit_input(split.paired_non_test_all, prepared, embed_artifact.unit_scaler)
    paired_train_input = prepare_unit_input(target_train_df, prepared, embed_artifact.unit_scaler)
    paired_test_input = prepare_unit_input(target_test_df, prepared, embed_artifact.unit_scaler)
    _, source_latent = predict_embeddings_from_state(
        model_state=embed_artifact.model_state,
        eval_data=source_input,
        hidden_dim=config.mmd_hidden_dim,
        latent_dim=config.mmd_latent_dim,
        random_state=config.random_state + 98,
        target_transform_mode=config.resolved_target_transform(),
    )
    _, paired_non_test_latent = predict_embeddings_from_state(
        model_state=embed_artifact.model_state,
        eval_data=paired_non_test_input,
        hidden_dim=config.mmd_hidden_dim,
        latent_dim=config.mmd_latent_dim,
        random_state=config.random_state + 99,
        target_transform_mode=config.resolved_target_transform(),
    )
    embed_train_pred, embed_train_latent = predict_embeddings_from_state(
        model_state=embed_artifact.model_state,
        eval_data=paired_train_input,
        hidden_dim=config.mmd_hidden_dim,
        latent_dim=config.mmd_latent_dim,
        random_state=config.random_state + 100,
        target_transform_mode=config.resolved_target_transform(),
    )
    embed_test_pred, embed_test_latent = predict_embeddings_from_state(
        model_state=embed_artifact.model_state,
        eval_data=paired_test_input,
        hidden_dim=config.mmd_hidden_dim,
        latent_dim=config.mmd_latent_dim,
        random_state=config.random_state + 101,
        target_transform_mode=config.resolved_target_transform(),
    )
    filtered_embed_train_pred = None
    filtered_embed_test_pred = None
    filtered_embed_train_latent = None
    filtered_embed_test_latent = None
    filtered_latent_rank = None
    filtered_ranked_dims: list[int] = []
    if filtered_embed_artifact is not None:
        filtered_paired_train_input = prepare_unit_input(filtered_target_train_df, prepared, filtered_embed_artifact.unit_scaler)
        filtered_paired_test_input = prepare_unit_input(filtered_target_test_df, prepared, filtered_embed_artifact.unit_scaler)
        filtered_embed_train_pred, filtered_embed_train_latent = predict_embeddings_from_state(
            model_state=filtered_embed_artifact.model_state,
            eval_data=filtered_paired_train_input,
            hidden_dim=config.mmd_hidden_dim,
            latent_dim=config.mmd_latent_dim,
            random_state=config.random_state + 102,
            target_transform_mode=config.resolved_target_transform(),
        )
        filtered_embed_test_pred, filtered_embed_test_latent = predict_embeddings_from_state(
            model_state=filtered_embed_artifact.model_state,
            eval_data=filtered_paired_test_input,
            hidden_dim=config.mmd_hidden_dim,
            latent_dim=config.mmd_latent_dim,
            random_state=config.random_state + 103,
            target_transform_mode=config.resolved_target_transform(),
        )

    subgroup_cluster_model = None
    subgroup_train_weights = None
    subgroup_test_resp = None
    subgroup_train_resp = None
    if config.use_subgroup_balanced_variants:
        for n_clusters in config.cluster_candidate_k:
            probe_model = fit_latent_cluster_model(
                paired_non_test_latent,
                n_clusters=int(n_clusters),
                random_state=config.random_state + 880 + int(n_clusters),
            )
            if int(probe_model.target_cluster_counts_.min()) >= int(config.cluster_min_target_rows):
                subgroup_cluster_model = probe_model
                break
        if subgroup_cluster_model is not None:
            subgroup_train_resp = latent_cluster_responsibilities(
                embed_train_latent,
                subgroup_cluster_model,
                temperature=config.cluster_assignment_temperature,
            )
            subgroup_test_resp = latent_cluster_responsibilities(
                embed_test_latent,
                subgroup_cluster_model,
                temperature=config.cluster_assignment_temperature,
            )
            subgroup_train_weights = compute_soft_group_balance_weights(
                subgroup_train_resp,
                reference_mass=subgroup_cluster_model.target_cluster_counts_,
                power=config.subgroup_balance_power,
            )

    contextual_train = _numeric_frame(target_train_df, contextual_cols)
    contextual_test = _numeric_frame(target_test_df, contextual_cols)
    study_train = make_study_indicator_frame(target_train_df, categories=study_categories)
    study_test = make_study_indicator_frame(target_test_df, categories=study_categories)
    target_protected_contextual_train = _numeric_frame(target_protected_target_train_df, target_protected_contextual_cols)
    target_protected_contextual_test = _numeric_frame(target_protected_target_test_df, target_protected_contextual_cols)
    latent_summary_train = make_latent_summary_frame(
        embed_train_latent,
        quantiles=config.latent_summary_quantiles,
    )
    latent_summary_test = make_latent_summary_frame(
        embed_test_latent,
        quantiles=config.latent_summary_quantiles,
    )
    latent_rank = rank_latent_dims_by_correlation(
        embed_train_latent,
        y_train,
        groups=target_train_df["recording_key"].astype(str).to_numpy(),
        n_splits=config.model_selection_splits,
        test_size=config.model_selection_test_size,
        random_state=config.random_state,
    )
    ranked_dims = latent_rank["latent_dim"].astype(int).tolist()
    if filtered_embed_train_latent is not None:
        filtered_latent_rank = rank_latent_dims_by_correlation(
            filtered_embed_train_latent,
            y_train,
            groups=target_train_df["recording_key"].astype(str).to_numpy(),
            n_splits=config.model_selection_splits,
            test_size=config.model_selection_test_size,
            random_state=config.random_state + 19,
        )
        filtered_ranked_dims = filtered_latent_rank["latent_dim"].astype(int).tolist()

    candidate_specs: dict[str, dict[str, Any]] = {}
    reference_variant_id = config.reference_variant_by_target[target]
    always_score_variants: set[str] = {
        "contextual_paired_only_lightgbm",
        "embed_nommd_fullctx_lightgbm",
    }
    always_score_variants.update(config.reference_variant_by_target.values())
    if config.use_shift_filtered_variants:
        always_score_variants.add("contextual_paired_only_lightgbm_filtered")
        always_score_variants.add("embed_nommd_fullctx_lightgbm_filtered")

    candidate_specs["contextual_paired_only_lightgbm"] = {
        "family": "baseline",
        "backend": "lightgbm",
        "source_target": None,
        "notes": "paired-only full context baseline",
        "feature_components": "full_context",
        "latent_feature_mode": "none",
        "latent_topk": 0,
        "selected_latent_dims": "[]",
        "train_frame": contextual_train,
        "test_frame": contextual_test,
        "gate_candidate": False,
    }
    if config.use_shift_filtered_variants and filtered_contextual_cols:
        candidate_specs["contextual_paired_only_lightgbm_filtered"] = {
            "family": "baseline_filtered",
            "backend": "lightgbm",
            "source_target": None,
            "notes": "paired-only filtered full context baseline",
            "feature_components": "filtered_full_context",
            "latent_feature_mode": "none",
            "latent_topk": 0,
            "selected_latent_dims": "[]",
            "train_frame": _numeric_frame(filtered_target_train_df, filtered_contextual_cols),
            "test_frame": _numeric_frame(filtered_target_test_df, filtered_contextual_cols),
            "gate_candidate": False,
        }
    if config.use_study_aware_variants:
        candidate_specs["study_contextual_paired_only_lightgbm"] = {
            "family": "study_baseline",
            "backend": "lightgbm",
            "source_target": None,
            "notes": "paired-only full context baseline + study one-hot",
            "feature_components": "full_context+study_one_hot",
            "latent_feature_mode": "study_one_hot",
            "latent_topk": 0,
            "selected_latent_dims": "[]",
            "train_frame": _concat_frames(contextual_train, study_train),
            "test_frame": _concat_frames(contextual_test, study_test),
            "gate_candidate": True,
        }

    embed_full_train = build_stack_frame(
        target_train_df,
        context_cols=contextual_cols,
        extra_features={"source_pred": embed_train_pred, "latent": embed_train_latent},
    )
    embed_full_test = build_stack_frame(
        target_test_df,
        context_cols=contextual_cols,
        extra_features={"source_pred": embed_test_pred, "latent": embed_test_latent},
    )
    candidate_specs["embed_nommd_fullctx_lightgbm"] = {
        "family": "embedding_stack",
        "backend": "lightgbm",
        "source_target": source_target,
        "notes": "embed_nommd source prediction + full latent + full context",
        "feature_components": "full_context+source_pred+full_latent",
        "latent_feature_mode": "full_latent",
        "latent_topk": int(embed_train_latent.shape[1]),
        "selected_latent_dims": json.dumps(ranked_dims),
        "train_frame": embed_full_train,
        "test_frame": embed_full_test,
        "gate_candidate": False,
    }
    if (
        config.use_shift_filtered_variants
        and filtered_embed_train_pred is not None
        and filtered_embed_train_latent is not None
        and filtered_contextual_cols
    ):
        filtered_embed_full_train = build_stack_frame(
            filtered_target_train_df,
            context_cols=filtered_contextual_cols,
            extra_features={"source_pred": filtered_embed_train_pred, "latent": filtered_embed_train_latent},
        )
        filtered_embed_full_test = build_stack_frame(
            filtered_target_test_df,
            context_cols=filtered_contextual_cols,
            extra_features={"source_pred": filtered_embed_test_pred, "latent": filtered_embed_test_latent},
        )
        candidate_specs["embed_nommd_fullctx_lightgbm_filtered"] = {
            "family": "embedding_stack_filtered",
            "backend": "lightgbm",
            "source_target": source_target,
            "notes": "filtered embed_nommd source prediction + full latent + filtered full context",
            "feature_components": "filtered_full_context+source_pred+full_latent",
            "latent_feature_mode": "full_latent",
            "latent_topk": int(filtered_embed_train_latent.shape[1]),
            "selected_latent_dims": json.dumps(filtered_ranked_dims),
            "train_frame": filtered_embed_full_train,
            "test_frame": filtered_embed_full_test,
            "gate_candidate": False,
        }
    candidate_specs["embed_nommd_fullctx_xgboost"] = {
        **candidate_specs["embed_nommd_fullctx_lightgbm"],
        "backend": "xgboost",
        "notes": "embed_nommd source prediction + full latent + full context",
        "gate_candidate": False,
    }
    always_score_variants.add("embed_nommd_fullctx_xgboost")

    source_pred_train_by_backend: dict[str, np.ndarray] = {}
    source_pred_test_by_backend: dict[str, np.ndarray] = {}
    for backend_idx, backend in enumerate(config.tree_backends):
        _log(config, f"[{target}] fitting domain-reweighted source model backend={backend}")
        weighted_source_model = fit_tree_regressor(
            source_df.loc[:, unit_cols],
            y_source,
            backend=backend,
            random_state=config.random_state + 200 + backend_idx,
            target_transform_mode=config.resolved_target_transform(),
            lightgbm_estimators=config.lightgbm_estimators,
            lightgbm_learning_rate=config.lightgbm_learning_rate,
            lightgbm_num_leaves=config.lightgbm_num_leaves,
            xgboost_estimators=config.xgboost_estimators,
            xgboost_learning_rate=config.xgboost_learning_rate,
            xgboost_max_depth=config.xgboost_max_depth,
            xgboost_min_child_weight=config.xgboost_min_child_weight,
            xgboost_subsample=config.xgboost_subsample,
            xgboost_colsample_bytree=config.xgboost_colsample_bytree,
            sample_weight=weight_result.weights if config.use_domain_reweighted_source else None,
        )
        source_pred_train = predict_tree_regressor(weighted_source_model, target_train_df.loc[:, unit_cols])
        source_pred_test = predict_tree_regressor(weighted_source_model, target_test_df.loc[:, unit_cols])
        source_pred_train_by_backend[backend] = source_pred_train
        source_pred_test_by_backend[backend] = source_pred_test
        filtered_source_pred_train = None
        filtered_source_pred_test = None
        if config.use_shift_filtered_variants and backend == "lightgbm" and filtered_unit_cols:
            filtered_weighted_source_model = fit_tree_regressor(
                filtered_source_df.loc[:, filtered_unit_cols],
                y_source,
                backend=backend,
                random_state=config.random_state + 260 + backend_idx,
                target_transform_mode=config.resolved_target_transform(),
                lightgbm_estimators=config.lightgbm_estimators,
                lightgbm_learning_rate=config.lightgbm_learning_rate,
                lightgbm_num_leaves=config.lightgbm_num_leaves,
                xgboost_estimators=config.xgboost_estimators,
                xgboost_learning_rate=config.xgboost_learning_rate,
                xgboost_max_depth=config.xgboost_max_depth,
                xgboost_min_child_weight=config.xgboost_min_child_weight,
                xgboost_subsample=config.xgboost_subsample,
                xgboost_colsample_bytree=config.xgboost_colsample_bytree,
                sample_weight=weight_result.weights if config.use_domain_reweighted_source else None,
            )
            filtered_source_pred_train = predict_tree_regressor(
                filtered_weighted_source_model,
                filtered_target_train_df.loc[:, filtered_unit_cols],
            )
            filtered_source_pred_test = predict_tree_regressor(
                filtered_weighted_source_model,
                filtered_target_test_df.loc[:, filtered_unit_cols],
            )

        candidate_specs[f"source_reweighted_fullctx_{backend}"] = {
            "family": "source_reweighted_stack",
            "backend": backend,
            "source_target": source_target,
            "notes": "domain-reweighted source prediction + full context",
            "feature_components": "full_context+reweighted_source_pred",
            "latent_feature_mode": "none",
            "latent_topk": 0,
            "selected_latent_dims": "[]",
            "train_frame": build_stack_frame(
                target_train_df,
                context_cols=contextual_cols,
                extra_features={"source_pred": source_pred_train},
            ),
            "test_frame": build_stack_frame(
                target_test_df,
                context_cols=contextual_cols,
                extra_features={"source_pred": source_pred_test},
            ),
            "gate_candidate": True,
        }
        overlap_source_pred_train = None
        overlap_source_pred_test = None
        if overlap_weight_result is not None:
            overlap_source_model = fit_tree_regressor(
                source_df.loc[:, unit_cols],
                y_source,
                backend=backend,
                random_state=config.random_state + 230 + backend_idx,
                target_transform_mode=config.resolved_target_transform(),
                lightgbm_estimators=config.lightgbm_estimators,
                lightgbm_learning_rate=config.lightgbm_learning_rate,
                lightgbm_num_leaves=config.lightgbm_num_leaves,
                xgboost_estimators=config.xgboost_estimators,
                xgboost_learning_rate=config.xgboost_learning_rate,
                xgboost_max_depth=config.xgboost_max_depth,
                xgboost_min_child_weight=config.xgboost_min_child_weight,
                xgboost_subsample=config.xgboost_subsample,
                xgboost_colsample_bytree=config.xgboost_colsample_bytree,
                sample_weight=overlap_weight_result.weights,
            )
            overlap_source_pred_train = predict_tree_regressor(overlap_source_model, target_train_df.loc[:, unit_cols])
            overlap_source_pred_test = predict_tree_regressor(overlap_source_model, target_test_df.loc[:, unit_cols])
            candidate_specs[f"source_overlap_fullctx_{backend}"] = {
                "family": "source_overlap_stack",
                "backend": backend,
                "source_target": source_target,
                "notes": "overlap-trimmed density-ratio source prediction + full context",
                "feature_components": "full_context+overlap_source_pred",
                "latent_feature_mode": "none",
                "latent_topk": 0,
                "selected_latent_dims": "[]",
                "train_frame": build_stack_frame(
                    target_train_df,
                    context_cols=contextual_cols,
                    extra_features={"source_pred": overlap_source_pred_train},
                ),
                "test_frame": build_stack_frame(
                    target_test_df,
                    context_cols=contextual_cols,
                    extra_features={"source_pred": overlap_source_pred_test},
                ),
                "gate_candidate": True,
            }
        if (
            config.use_shift_filtered_variants
            and backend == "lightgbm"
            and filtered_source_pred_train is not None
            and filtered_contextual_cols
        ):
            candidate_specs[f"source_reweighted_fullctx_{backend}_filtered"] = {
                "family": "source_reweighted_stack_filtered",
                "backend": backend,
                "source_target": source_target,
                "notes": "filtered domain-reweighted source prediction + filtered full context",
                "feature_components": "filtered_full_context+reweighted_source_pred",
                "latent_feature_mode": "none",
                "latent_topk": 0,
                "selected_latent_dims": "[]",
                "train_frame": build_stack_frame(
                    filtered_target_train_df,
                    context_cols=filtered_contextual_cols,
                    extra_features={"source_pred": filtered_source_pred_train},
                ),
                "test_frame": build_stack_frame(
                    filtered_target_test_df,
                    context_cols=filtered_contextual_cols,
                    extra_features={"source_pred": filtered_source_pred_test},
                ),
                "gate_candidate": True,
            }
        if (
            config.use_target_protected_shift_variants
            and backend == "lightgbm"
            and overlap_source_pred_train is not None
            and target_protected_contextual_cols
        ):
            candidate_specs[f"source_overlap_fullctx_{backend}_target_filtered"] = {
                "family": "source_overlap_stack_target_filtered",
                "backend": backend,
                "source_target": source_target,
                "notes": "overlap-trimmed source prediction + target-protected context filter",
                "feature_components": "target_protected_full_context+overlap_source_pred",
                "latent_feature_mode": "none",
                "latent_topk": 0,
                "selected_latent_dims": "[]",
                "train_frame": build_stack_frame(
                    target_protected_target_train_df,
                    context_cols=target_protected_contextual_cols,
                    extra_features={"source_pred": overlap_source_pred_train},
                ),
                "test_frame": build_stack_frame(
                    target_protected_target_test_df,
                    context_cols=target_protected_contextual_cols,
                    extra_features={"source_pred": overlap_source_pred_test},
                ),
                "gate_candidate": True,
            }
        if config.use_cluster_aware_variants and backend == "lightgbm":
            cluster_k_rows: list[dict[str, Any]] = []
            for n_clusters in config.cluster_candidate_k:
                cluster_model = fit_latent_cluster_model(
                    paired_non_test_latent,
                    n_clusters=int(n_clusters),
                    random_state=config.random_state + 500 + int(n_clusters),
                )
                if int(cluster_model.target_cluster_counts_.min()) < int(config.cluster_min_target_rows):
                    continue
                source_resp = latent_cluster_responsibilities(
                    source_latent,
                    cluster_model,
                    temperature=config.cluster_assignment_temperature,
                )
                train_resp = latent_cluster_responsibilities(
                    embed_train_latent,
                    cluster_model,
                    temperature=config.cluster_assignment_temperature,
                )
                test_resp = latent_cluster_responsibilities(
                    embed_test_latent,
                    cluster_model,
                    temperature=config.cluster_assignment_temperature,
                )
                target_resp = latent_cluster_responsibilities(
                    paired_non_test_latent,
                    cluster_model,
                    temperature=config.cluster_assignment_temperature,
                )
                hard_labels = target_resp.argmax(axis=1)
                pred_train_matrix = np.zeros((len(target_train_df), cluster_model.n_clusters), dtype=float)
                pred_test_matrix = np.zeros((len(target_test_df), cluster_model.n_clusters), dtype=float)
                for cluster_id in range(cluster_model.n_clusters):
                    cluster_weights = weight_result.weights * source_resp[:, cluster_id]
                    cluster_model_reg = fit_tree_regressor(
                        source_df.loc[:, unit_cols],
                        y_source,
                        backend=backend,
                        random_state=config.random_state + 600 + backend_idx * 50 + int(n_clusters) * 10 + cluster_id,
                        target_transform_mode=config.resolved_target_transform(),
                        lightgbm_estimators=config.lightgbm_estimators,
                        lightgbm_learning_rate=config.lightgbm_learning_rate,
                        lightgbm_num_leaves=config.lightgbm_num_leaves,
                        xgboost_estimators=config.xgboost_estimators,
                        xgboost_learning_rate=config.xgboost_learning_rate,
                        xgboost_max_depth=config.xgboost_max_depth,
                        xgboost_min_child_weight=config.xgboost_min_child_weight,
                        xgboost_subsample=config.xgboost_subsample,
                        xgboost_colsample_bytree=config.xgboost_colsample_bytree,
                        sample_weight=cluster_weights if config.use_domain_reweighted_source else None,
                    )
                    pred_train_matrix[:, cluster_id] = predict_tree_regressor(
                        cluster_model_reg,
                        target_train_df.loc[:, unit_cols],
                    )
                    pred_test_matrix[:, cluster_id] = predict_tree_regressor(
                        cluster_model_reg,
                        target_test_df.loc[:, unit_cols],
                    )
                    cluster_study_counts = (
                        split.paired_non_test_all.loc[hard_labels == cluster_id, "study_set"]
                        .astype(str)
                        .value_counts()
                        .head(3)
                    )
                    cluster_rows.append(
                        {
                            "target": target,
                            "variant_id": "cluster_source_fullctx_lightgbm",
                            "backend": backend,
                            "n_clusters": int(cluster_model.n_clusters),
                            "cluster_id": int(cluster_id),
                            "target_non_test_rows": int(cluster_model.target_cluster_counts_[cluster_id]),
                            "source_soft_mass": float(source_resp[:, cluster_id].sum()),
                            "paired_train_soft_mass": float(train_resp[:, cluster_id].sum()),
                            "paired_test_soft_mass": float(test_resp[:, cluster_id].sum()),
                            "top_studies": "|".join(
                                f"{study}:{count}" for study, count in cluster_study_counts.items()
                            ),
                        }
                    )
                cluster_source_pred_train = (pred_train_matrix * train_resp).sum(axis=1)
                cluster_source_pred_test = (pred_test_matrix * test_resp).sum(axis=1)
                cluster_train_frame = build_stack_frame(
                    target_train_df,
                    context_cols=contextual_cols,
                    extra_features={
                        "source_pred": source_pred_train,
                        "cluster_source_pred": cluster_source_pred_train,
                        "cluster_resp": train_resp,
                    },
                )
                cluster_cv_metrics, _ = _score_candidate_cv(
                    train_df=target_train_df,
                    y_train=y_train,
                    feature_train=cluster_train_frame,
                    backend=backend,
                    config=config,
                    random_state_offset=3700 + backend_idx * 100 + int(n_clusters),
                )
                cluster_k_rows.append(
                    {
                        "target": target,
                        "variant_id": "cluster_source_fullctx_lightgbm",
                        "family": "cluster_source_stack",
                        "backend": backend,
                        "feature_components": "full_context+reweighted_source_pred+cluster_source_pred+cluster_resp",
                        "latent_feature_mode": f"cluster_k={int(cluster_model.n_clusters)}",
                        "latent_topk": int(cluster_model.n_clusters),
                        "selected_latent_dims": json.dumps(list(range(cluster_model.n_clusters))),
                        "cv_mae": cluster_cv_metrics["mae"],
                        "cv_rmse": cluster_cv_metrics["rmse"],
                        "cv_r2": cluster_cv_metrics["r2"],
                        "cv_bias": cluster_cv_metrics["bias"],
                        "cv_calibration_slope": cluster_cv_metrics["calibration_slope"],
                        "cv_calibration_intercept": cluster_cv_metrics["calibration_intercept"],
                        "gate_reference_variant_id": reference_variant_id,
                        "gate_reference_cv_mae": np.nan,
                        "gate_reference_cv_r2": np.nan,
                        "passed_cv_gate": False,
                        "holdout_scored": False,
                        "holdout_mae": np.nan,
                        "holdout_r2": np.nan,
                        "cluster_train_frame": cluster_train_frame,
                        "cluster_test_frame": build_stack_frame(
                            target_test_df,
                            context_cols=contextual_cols,
                            extra_features={
                                "source_pred": source_pred_test,
                                "cluster_source_pred": cluster_source_pred_test,
                                "cluster_resp": test_resp,
                            },
                        ),
                    }
                )
            if cluster_k_rows:
                best_cluster = _topk_choice(cluster_k_rows)
                candidate_specs["cluster_source_fullctx_lightgbm"] = {
                    "family": "cluster_source_stack",
                    "backend": backend,
                    "source_target": source_target,
                    "notes": f"cluster-aware source prediction + cluster responsibilities + full context (k={int(best_cluster['latent_topk'])})",
                    "feature_components": "full_context+reweighted_source_pred+cluster_source_pred+cluster_resp",
                    "latent_feature_mode": best_cluster["latent_feature_mode"],
                    "latent_topk": int(best_cluster["latent_topk"]),
                    "selected_latent_dims": best_cluster["selected_latent_dims"],
                    "train_frame": best_cluster["cluster_train_frame"],
                    "test_frame": best_cluster["cluster_test_frame"],
                    "gate_candidate": True,
                }
                for row in cluster_k_rows:
                    row.pop("cluster_train_frame", None)
                    row.pop("cluster_test_frame", None)
                ablation_rows.extend(cluster_k_rows)
        candidate_specs[f"latent_summary_fullctx_{backend}"] = {
            "family": "latent_summary_stack",
            "backend": backend,
            "source_target": source_target,
            "notes": "latent summary statistics + full context",
            "feature_components": "full_context+latent_summary",
            "latent_feature_mode": "summary",
            "latent_topk": 0,
            "selected_latent_dims": "[]",
            "train_frame": _concat_frames(contextual_train, latent_summary_train),
            "test_frame": _concat_frames(contextual_test, latent_summary_test),
            "gate_candidate": True,
        }
        if config.use_subgroup_balanced_variants and backend == "lightgbm" and subgroup_train_weights is not None:
            candidate_specs[f"latent_summary_fullctx_{backend}_subgroup_balanced"] = {
                "family": "latent_summary_stack_subgroup_balanced",
                "backend": backend,
                "source_target": source_target,
                "notes": "latent summary statistics + full context + subgroup-balanced paired fitting",
                "feature_components": "full_context+latent_summary+subgroup_balance",
                "latent_feature_mode": "summary+subgroup_balance",
                "latent_topk": 0,
                "selected_latent_dims": "[]",
                "train_frame": _concat_frames(contextual_train, latent_summary_train),
                "test_frame": _concat_frames(contextual_test, latent_summary_test),
                "train_sample_weight": subgroup_train_weights,
                "gate_candidate": True,
            }
        if config.use_target_protected_shift_variants and backend == "lightgbm" and target_protected_contextual_cols:
            candidate_specs[f"latent_summary_fullctx_{backend}_target_filtered"] = {
                "family": "latent_summary_stack_target_filtered",
                "backend": backend,
                "source_target": source_target,
                "notes": "latent summary statistics + target-protected context filter",
                "feature_components": "target_protected_full_context+latent_summary",
                "latent_feature_mode": "summary",
                "latent_topk": 0,
                "selected_latent_dims": "[]",
                "train_frame": _concat_frames(target_protected_contextual_train, latent_summary_train),
                "test_frame": _concat_frames(target_protected_contextual_test, latent_summary_test),
                "gate_candidate": True,
            }
        if config.use_study_aware_variants and backend == "lightgbm":
            candidate_specs["study_latent_summary_fullctx_lightgbm"] = {
                "family": "study_latent_summary_stack",
                "backend": "lightgbm",
                "source_target": source_target,
                "notes": "latent summary statistics + full context + study one-hot",
                "feature_components": "full_context+latent_summary+study_one_hot",
                "latent_feature_mode": "summary+study_one_hot",
                "latent_topk": 0,
                "selected_latent_dims": "[]",
                "train_frame": _concat_frames(contextual_train, latent_summary_train, study_train),
                "test_frame": _concat_frames(contextual_test, latent_summary_test, study_test),
                "gate_candidate": True,
            }

        topk_rows: list[dict[str, Any]] = []
        filtered_topk_rows: list[dict[str, Any]] = []
        for topk in config.latent_candidate_topk:
            top_dims = ranked_dims[: int(topk)]
            train_frame = build_stack_frame(
                target_train_df,
                context_cols=contextual_cols,
                extra_features={
                    "source_pred": source_pred_train,
                    "latent_sel": embed_train_latent[:, top_dims],
                },
            )
            cv_metrics, _ = _score_candidate_cv(
                train_df=target_train_df,
                y_train=y_train,
                feature_train=train_frame,
                backend=backend,
                config=config,
                random_state_offset=3000 + backend_idx * 100 + int(topk),
            )
            topk_rows.append(
                {
                    "target": target,
                    "variant_id": f"source_plus_reduced_latent_fullctx_{backend}",
                    "family": "reduced_latent_stack",
                    "backend": backend,
                    "feature_components": "full_context+reweighted_source_pred+reduced_latent",
                    "latent_feature_mode": config.latent_selection_mode,
                    "latent_topk": int(topk),
                    "selected_latent_dims": json.dumps(top_dims),
                    "cv_mae": cv_metrics["mae"],
                    "cv_rmse": cv_metrics["rmse"],
                    "cv_r2": cv_metrics["r2"],
                    "cv_bias": cv_metrics["bias"],
                    "cv_calibration_slope": cv_metrics["calibration_slope"],
                    "cv_calibration_intercept": cv_metrics["calibration_intercept"],
                    "gate_reference_variant_id": reference_variant_id,
                    "gate_reference_cv_mae": np.nan,
                    "gate_reference_cv_r2": np.nan,
                    "passed_cv_gate": False,
                    "holdout_scored": False,
                    "holdout_mae": np.nan,
                    "holdout_r2": np.nan,
                }
            )
            if (
                config.use_shift_filtered_variants
                and backend == "lightgbm"
                and filtered_embed_train_latent is not None
                and filtered_source_pred_train is not None
                and filtered_contextual_cols
            ):
                filtered_top_dims = filtered_ranked_dims[: int(topk)]
                filtered_train_frame = build_stack_frame(
                    filtered_target_train_df,
                    context_cols=filtered_contextual_cols,
                    extra_features={
                        "source_pred": filtered_source_pred_train,
                        "latent_sel": filtered_embed_train_latent[:, filtered_top_dims],
                    },
                )
                filtered_cv_metrics, _ = _score_candidate_cv(
                    train_df=target_train_df,
                    y_train=y_train,
                    feature_train=filtered_train_frame,
                    backend=backend,
                    config=config,
                    random_state_offset=3500 + backend_idx * 100 + int(topk),
                )
                filtered_topk_rows.append(
                    {
                        "target": target,
                        "variant_id": f"source_plus_reduced_latent_fullctx_{backend}_filtered",
                        "family": "reduced_latent_stack_filtered",
                        "backend": backend,
                        "feature_components": "filtered_full_context+reweighted_source_pred+reduced_latent",
                        "latent_feature_mode": config.latent_selection_mode,
                        "latent_topk": int(topk),
                        "selected_latent_dims": json.dumps(filtered_top_dims),
                        "cv_mae": filtered_cv_metrics["mae"],
                        "cv_rmse": filtered_cv_metrics["rmse"],
                        "cv_r2": filtered_cv_metrics["r2"],
                        "cv_bias": filtered_cv_metrics["bias"],
                        "cv_calibration_slope": filtered_cv_metrics["calibration_slope"],
                        "cv_calibration_intercept": filtered_cv_metrics["calibration_intercept"],
                        "gate_reference_variant_id": reference_variant_id,
                        "gate_reference_cv_mae": np.nan,
                        "gate_reference_cv_r2": np.nan,
                        "passed_cv_gate": False,
                        "holdout_scored": False,
                        "holdout_mae": np.nan,
                        "holdout_r2": np.nan,
                    }
                )
        best_topk = _topk_choice(topk_rows)
        selected_dims = json.loads(str(best_topk["selected_latent_dims"]))
        candidate_specs[f"source_plus_reduced_latent_fullctx_{backend}"] = {
            "family": "reduced_latent_stack",
            "backend": backend,
            "source_target": source_target,
            "notes": f"domain-reweighted source prediction + top-{best_topk['latent_topk']} latent dims + full context",
            "feature_components": "full_context+reweighted_source_pred+reduced_latent",
            "latent_feature_mode": config.latent_selection_mode,
            "latent_topk": int(best_topk["latent_topk"]),
            "selected_latent_dims": json.dumps(selected_dims),
            "train_frame": build_stack_frame(
                target_train_df,
                context_cols=contextual_cols,
                extra_features={
                    "source_pred": source_pred_train,
                    "latent_sel": embed_train_latent[:, selected_dims],
                },
            ),
            "test_frame": build_stack_frame(
                target_test_df,
                context_cols=contextual_cols,
                extra_features={
                    "source_pred": source_pred_test,
                    "latent_sel": embed_test_latent[:, selected_dims],
                },
            ),
            "gate_candidate": True,
        }
        if overlap_source_pred_train is not None:
            candidate_specs[f"source_overlap_plus_reduced_latent_fullctx_{backend}"] = {
                "family": "reduced_latent_overlap_stack",
                "backend": backend,
                "source_target": source_target,
                "notes": f"overlap-trimmed source prediction + top-{best_topk['latent_topk']} latent dims + full context",
                "feature_components": "full_context+overlap_source_pred+reduced_latent",
                "latent_feature_mode": config.latent_selection_mode,
                "latent_topk": int(best_topk["latent_topk"]),
                "selected_latent_dims": json.dumps(selected_dims),
                "train_frame": build_stack_frame(
                    target_train_df,
                    context_cols=contextual_cols,
                    extra_features={
                        "source_pred": overlap_source_pred_train,
                        "latent_sel": embed_train_latent[:, selected_dims],
                    },
                ),
                "test_frame": build_stack_frame(
                    target_test_df,
                    context_cols=contextual_cols,
                    extra_features={
                        "source_pred": overlap_source_pred_test,
                        "latent_sel": embed_test_latent[:, selected_dims],
                    },
                ),
                "gate_candidate": True,
            }
        if config.use_subgroup_balanced_variants and backend == "lightgbm" and subgroup_train_weights is not None:
            candidate_specs[f"source_plus_reduced_latent_fullctx_{backend}_subgroup_balanced"] = {
                "family": "reduced_latent_stack_subgroup_balanced",
                "backend": backend,
                "source_target": source_target,
                "notes": f"domain-reweighted source prediction + top-{best_topk['latent_topk']} latent dims + full context + subgroup-balanced paired fitting",
                "feature_components": "full_context+reweighted_source_pred+reduced_latent+subgroup_balance",
                "latent_feature_mode": f"{config.latent_selection_mode}+subgroup_balance",
                "latent_topk": int(best_topk["latent_topk"]),
                "selected_latent_dims": json.dumps(selected_dims),
                "train_frame": build_stack_frame(
                    target_train_df,
                    context_cols=contextual_cols,
                    extra_features={
                        "source_pred": source_pred_train,
                        "latent_sel": embed_train_latent[:, selected_dims],
                    },
                ),
                "test_frame": build_stack_frame(
                    target_test_df,
                    context_cols=contextual_cols,
                    extra_features={
                        "source_pred": source_pred_test,
                        "latent_sel": embed_test_latent[:, selected_dims],
                    },
                ),
                "train_sample_weight": subgroup_train_weights,
                "gate_candidate": True,
            }
        if config.use_target_protected_shift_variants and backend == "lightgbm" and target_protected_contextual_cols:
            candidate_specs[f"source_plus_reduced_latent_fullctx_{backend}_target_filtered"] = {
                "family": "reduced_latent_stack_target_filtered",
                "backend": backend,
                "source_target": source_target,
                "notes": f"domain-reweighted source prediction + top-{best_topk['latent_topk']} latent dims + target-protected context filter",
                "feature_components": "target_protected_full_context+reweighted_source_pred+reduced_latent",
                "latent_feature_mode": config.latent_selection_mode,
                "latent_topk": int(best_topk["latent_topk"]),
                "selected_latent_dims": json.dumps(selected_dims),
                "train_frame": build_stack_frame(
                    target_protected_target_train_df,
                    context_cols=target_protected_contextual_cols,
                    extra_features={
                        "source_pred": source_pred_train,
                        "latent_sel": embed_train_latent[:, selected_dims],
                    },
                ),
                "test_frame": build_stack_frame(
                    target_protected_target_test_df,
                    context_cols=target_protected_contextual_cols,
                    extra_features={
                        "source_pred": source_pred_test,
                        "latent_sel": embed_test_latent[:, selected_dims],
                    },
                ),
                "gate_candidate": True,
            }
        if config.use_study_aware_variants and backend == "lightgbm":
            candidate_specs["study_source_plus_reduced_latent_fullctx_lightgbm"] = {
                "family": "study_reduced_latent_stack",
                "backend": "lightgbm",
                "source_target": source_target,
                "notes": f"domain-reweighted source prediction + top-{best_topk['latent_topk']} latent dims + full context + study one-hot",
                "feature_components": "full_context+reweighted_source_pred+reduced_latent+study_one_hot",
                "latent_feature_mode": f"{config.latent_selection_mode}+study_one_hot",
                "latent_topk": int(best_topk["latent_topk"]),
                "selected_latent_dims": json.dumps(selected_dims),
                "train_frame": _concat_frames(
                    build_stack_frame(
                        target_train_df,
                        context_cols=contextual_cols,
                        extra_features={
                            "source_pred": source_pred_train,
                            "latent_sel": embed_train_latent[:, selected_dims],
                        },
                    ),
                    study_train,
                ),
                "test_frame": _concat_frames(
                    build_stack_frame(
                        target_test_df,
                        context_cols=contextual_cols,
                        extra_features={
                            "source_pred": source_pred_test,
                            "latent_sel": embed_test_latent[:, selected_dims],
                        },
                    ),
                    study_test,
                ),
                "gate_candidate": True,
            }
        ablation_rows.extend(topk_rows)
        if filtered_topk_rows:
            best_filtered_topk = _topk_choice(filtered_topk_rows)
            filtered_selected_dims = json.loads(str(best_filtered_topk["selected_latent_dims"]))
            candidate_specs[f"source_plus_reduced_latent_fullctx_{backend}_filtered"] = {
                "family": "reduced_latent_stack_filtered",
                "backend": backend,
                "source_target": source_target,
                "notes": f"filtered domain-reweighted source prediction + top-{best_filtered_topk['latent_topk']} latent dims + filtered full context",
                "feature_components": "filtered_full_context+reweighted_source_pred+reduced_latent",
                "latent_feature_mode": config.latent_selection_mode,
                "latent_topk": int(best_filtered_topk["latent_topk"]),
                "selected_latent_dims": json.dumps(filtered_selected_dims),
                "train_frame": build_stack_frame(
                    filtered_target_train_df,
                    context_cols=filtered_contextual_cols,
                    extra_features={
                        "source_pred": filtered_source_pred_train,
                        "latent_sel": filtered_embed_train_latent[:, filtered_selected_dims],
                    },
                ),
                "test_frame": build_stack_frame(
                    filtered_target_test_df,
                    context_cols=filtered_contextual_cols,
                    extra_features={
                        "source_pred": filtered_source_pred_test,
                        "latent_sel": filtered_embed_test_latent[:, filtered_selected_dims],
                    },
                ),
                "gate_candidate": True,
            }
            ablation_rows.extend(filtered_topk_rows)
    if config.use_study_aware_variants:
        candidate_specs["study_embed_nommd_fullctx_lightgbm"] = {
            "family": "study_embedding_stack",
            "backend": "lightgbm",
            "source_target": source_target,
            "notes": "embed_nommd source prediction + full latent + full context + study one-hot",
            "feature_components": "full_context+source_pred+full_latent+study_one_hot",
            "latent_feature_mode": "full_latent+study_one_hot",
            "latent_topk": int(embed_train_latent.shape[1]),
            "selected_latent_dims": json.dumps(ranked_dims),
            "train_frame": _concat_frames(embed_full_train, study_train),
            "test_frame": _concat_frames(embed_full_test, study_test),
            "gate_candidate": True,
        }
        if (
            config.use_shift_filtered_variants
            and filtered_embed_train_pred is not None
            and filtered_embed_train_latent is not None
            and filtered_contextual_cols
        ):
            filtered_embed_full_train = candidate_specs["embed_nommd_fullctx_lightgbm_filtered"]["train_frame"]
            filtered_embed_full_test = candidate_specs["embed_nommd_fullctx_lightgbm_filtered"]["test_frame"]
            candidate_specs["study_embed_nommd_fullctx_lightgbm_filtered"] = {
                "family": "study_embedding_stack_filtered",
                "backend": "lightgbm",
                "source_target": source_target,
                "notes": "filtered embed_nommd source prediction + full latent + filtered full context + study one-hot",
                "feature_components": "filtered_full_context+source_pred+full_latent+study_one_hot",
                "latent_feature_mode": "full_latent+study_one_hot",
                "latent_topk": int(filtered_embed_train_latent.shape[1]),
                "selected_latent_dims": json.dumps(filtered_ranked_dims),
                "train_frame": _concat_frames(filtered_embed_full_train, study_train),
                "test_frame": _concat_frames(filtered_embed_full_test, study_test),
                "gate_candidate": True,
            }

    cv_summary: dict[str, dict[str, float]] = {}
    cv_oof_pred_by_variant: dict[str, np.ndarray] = {}
    for variant_id, spec in candidate_specs.items():
        _log(config, f"[{target}] CV scoring {variant_id}")
        cv_metrics, oof_pred = _score_candidate_cv(
            train_df=target_train_df,
            y_train=y_train,
            feature_train=spec["train_frame"],
            backend=spec["backend"],
            config=config,
            random_state_offset=4000 + len(cv_summary) * 10,
            sample_weight=spec.get("train_sample_weight"),
        )
        cv_summary[variant_id] = cv_metrics
        cv_oof_pred_by_variant[variant_id] = oof_pred
        ablation_rows.append(
            {
                "target": target,
                "variant_id": variant_id,
                "family": spec["family"],
                "backend": spec["backend"],
                "feature_components": spec["feature_components"],
                "latent_feature_mode": spec["latent_feature_mode"],
                "latent_topk": spec["latent_topk"],
                "selected_latent_dims": spec["selected_latent_dims"],
                "cv_mae": cv_metrics["mae"],
                "cv_rmse": cv_metrics["rmse"],
                "cv_r2": cv_metrics["r2"],
                "cv_bias": cv_metrics["bias"],
                "cv_calibration_slope": cv_metrics["calibration_slope"],
                "cv_calibration_intercept": cv_metrics["calibration_intercept"],
                "gate_reference_variant_id": reference_variant_id,
                "gate_reference_cv_mae": np.nan,
                "gate_reference_cv_r2": np.nan,
                "passed_cv_gate": False,
                "holdout_scored": False,
                "holdout_mae": np.nan,
                "holdout_r2": np.nan,
            }
        )

    if config.use_study_bias_correction_variants and reference_variant_id in cv_oof_pred_by_variant:
        base_variant_id = reference_variant_id
        base_spec = candidate_specs[base_variant_id]
        base_oof_pred = cv_oof_pred_by_variant[base_variant_id]
        residual_oof = y_train - base_oof_pred
        loo_adjustment = leave_one_out_study_bias_adjustment(
            target_train_df["study_set"],
            residual_oof,
            shrinkage=config.study_bias_shrinkage,
        )
        calibrated_oof_pred = np.clip(base_oof_pred + loo_adjustment, 0.0, 1.0)
        calibrated_cv_metrics = score_predictions(y_train, calibrated_oof_pred)
        calibrated_variant_id = f"{base_variant_id}_study_bias"
        candidate_specs[calibrated_variant_id] = {
            "family": f"{base_spec['family']}_study_bias",
            "backend": base_spec["backend"],
            "source_target": base_spec["source_target"],
            "notes": f"{base_spec['notes']} + study-wise residual bias correction",
            "feature_components": f"{base_spec['feature_components']}+study_bias",
            "latent_feature_mode": f"{base_spec['latent_feature_mode']}+study_bias",
            "latent_topk": base_spec["latent_topk"],
            "selected_latent_dims": base_spec["selected_latent_dims"],
            "train_frame": base_spec["train_frame"],
            "test_frame": base_spec["test_frame"],
            "gate_candidate": True,
            "base_variant_id": base_variant_id,
            "is_posthoc_bias": True,
        }
        cv_summary[calibrated_variant_id] = calibrated_cv_metrics
        cv_oof_pred_by_variant[calibrated_variant_id] = calibrated_oof_pred
        ablation_rows.append(
            {
                "target": target,
                "variant_id": calibrated_variant_id,
                "family": f"{base_spec['family']}_study_bias",
                "backend": base_spec["backend"],
                "feature_components": f"{base_spec['feature_components']}+study_bias",
                "latent_feature_mode": f"{base_spec['latent_feature_mode']}+study_bias",
                "latent_topk": base_spec["latent_topk"],
                "selected_latent_dims": base_spec["selected_latent_dims"],
                "cv_mae": calibrated_cv_metrics["mae"],
                "cv_rmse": calibrated_cv_metrics["rmse"],
                "cv_r2": calibrated_cv_metrics["r2"],
                "cv_bias": calibrated_cv_metrics["bias"],
                "cv_calibration_slope": calibrated_cv_metrics["calibration_slope"],
                "cv_calibration_intercept": calibrated_cv_metrics["calibration_intercept"],
                "gate_reference_variant_id": reference_variant_id,
                "gate_reference_cv_mae": np.nan,
                "gate_reference_cv_r2": np.nan,
                "passed_cv_gate": False,
                "holdout_scored": False,
                "holdout_mae": np.nan,
                "holdout_r2": np.nan,
            }
        )

    reference_cv = cv_summary[reference_variant_id]
    ablation_by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in ablation_rows:
        ablation_by_variant.setdefault(str(row["variant_id"]), []).append(row)

    holdout_candidates: set[str] = set(always_score_variants)
    for variant_id, rows in ablation_by_variant.items():
        best_row = max(rows, key=lambda row: float(row["cv_r2"]))
        best_row["gate_reference_variant_id"] = reference_variant_id
        best_row["gate_reference_cv_mae"] = reference_cv["mae"]
        best_row["gate_reference_cv_r2"] = reference_cv["r2"]
        passes = (
            float(best_row["cv_r2"]) >= (float(reference_cv["r2"]) + float(config.min_r2_improvement))
            and float(best_row["cv_mae"]) <= (float(reference_cv["mae"]) * (1.0 + float(config.mae_guardrail_pct)))
        )
        if variant_id in always_score_variants:
            passes = True
        best_row["passed_cv_gate"] = bool(passes)
        if bool(passes) and variant_id in candidate_specs:
            holdout_candidates.add(variant_id)

    holdout_pred_by_variant: dict[str, np.ndarray] = {}
    base_holdout_candidates = [
        variant_id for variant_id in sorted(holdout_candidates) if not candidate_specs[variant_id].get("is_posthoc_bias", False)
    ]
    posthoc_holdout_candidates = [
        variant_id for variant_id in sorted(holdout_candidates) if candidate_specs[variant_id].get("is_posthoc_bias", False)
    ]

    for variant_id in base_holdout_candidates:
        spec = candidate_specs[variant_id]
        _log(config, f"[{target}] holdout scoring {variant_id}")
        pred = _fit_predict_holdout(
            feature_train=spec["train_frame"],
            y_train=y_train,
            feature_test=spec["test_frame"],
            backend=spec["backend"],
            config=config,
            random_state_offset=5000 + len(metric_rows) * 10,
            sample_weight=spec.get("train_sample_weight"),
        )
        holdout_pred_by_variant[variant_id] = pred
        metrics = score_predictions(y_test, pred)
        metric_rows.append(
            make_metric_row(
                target=target,
                variant_id=variant_id,
                family=spec["family"],
                backend=spec["backend"],
                source_target=spec["source_target"],
                metrics=metrics,
                notes=spec["notes"],
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target=target,
                variant_id=variant_id,
                family=spec["family"],
                backend=spec["backend"],
                df=target_test_df,
                y_true=y_test,
                pred=pred,
            )
        )
        if variant_id in ablation_by_variant:
            best_row = max(ablation_by_variant[variant_id], key=lambda row: float(row["cv_r2"]))
            best_row["holdout_scored"] = True
            best_row["holdout_mae"] = metrics["mae"]
            best_row["holdout_r2"] = metrics["r2"]

    for variant_id in posthoc_holdout_candidates:
        spec = candidate_specs[variant_id]
        base_variant_id = str(spec["base_variant_id"])
        if base_variant_id not in holdout_pred_by_variant:
            continue
        _log(config, f"[{target}] holdout scoring {variant_id}")
        calibrator = fit_study_bias_correction(
            target_train_df["study_set"],
            y_train - cv_oof_pred_by_variant[base_variant_id],
            shrinkage=config.study_bias_shrinkage,
        )
        pred = apply_study_bias_correction(
            holdout_pred_by_variant[base_variant_id],
            target_test_df["study_set"],
            calibrator,
        )
        metrics = score_predictions(y_test, pred)
        metric_rows.append(
            make_metric_row(
                target=target,
                variant_id=variant_id,
                family=spec["family"],
                backend=spec["backend"],
                source_target=spec["source_target"],
                metrics=metrics,
                notes=spec["notes"],
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target=target,
                variant_id=variant_id,
                family=spec["family"],
                backend=spec["backend"],
                df=target_test_df,
                y_true=y_test,
                pred=pred,
            )
        )
        if variant_id in ablation_by_variant:
            best_row = max(ablation_by_variant[variant_id], key=lambda row: float(row["cv_r2"]))
            best_row["holdout_scored"] = True
            best_row["holdout_mae"] = metrics["mae"]
            best_row["holdout_r2"] = metrics["r2"]

    metrics_df = _frame_with_columns(metric_rows, _METRIC_COLUMNS)
    recommendation = pick_r2_first_recommendation(
        metrics_df,
        target=target,
        reference_variant_id=reference_variant_id,
        selection_metric=config.selection_metric,
        mae_guardrail_pct=config.mae_guardrail_pct,
        min_r2_improvement=config.min_r2_improvement,
    )
    recommendation["is_breakthrough"] = bool(float(recommendation["recommended_r2"]) >= 0.55)
    recommendation_rows.append(recommendation)

    return (
        metric_rows,
        prediction_frames,
        history_rows,
        weight_rows,
        ablation_rows,
        recommendation_rows,
        cluster_rows,
        target_filter_manifest_rows,
    )


def run_with_config(config: MLStatisticsWorkbenchConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    _log(config, f"Loading parquet from {config.parquet_path}")
    prepared = prepare_feature_table(config.parquet_path)
    split = make_main_split(
        prepared.df,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        random_state=config.random_state,
    )
    if config.use_shift_filtered_variants:
        filtered_manifest_df, unit_drop_cols, full_context_drop_cols = _build_filtered_feature_manifests(
            split=split,
            prepared=prepared,
            config=config,
        )
    else:
        filtered_manifest_df = pd.DataFrame(columns=_FILTERED_MANIFEST_COLUMNS)
        unit_drop_cols = []
        full_context_drop_cols = []

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    target_filter_manifest_rows: list[dict[str, Any]] = []
    for target in config.targets:
        _log(config, f"Running target={target}")
        rows = _run_target(
            target=target,
            split=split,
            prepared=prepared,
            config=config,
            output_dir=output_dir,
            unit_drop_cols=unit_drop_cols,
            full_context_drop_cols=full_context_drop_cols,
        )
        (
            target_metrics,
            target_preds,
            target_history,
            target_weights,
            target_ablations,
            target_recommendations,
            target_clusters,
            target_filter_rows,
        ) = rows
        metric_rows.extend(target_metrics)
        prediction_frames.extend(target_preds)
        history_rows.extend(target_history)
        weight_rows.extend(target_weights)
        ablation_rows.extend(target_ablations)
        recommendation_rows.extend(target_recommendations)
        cluster_rows.extend(target_clusters)
        target_filter_manifest_rows.extend(target_filter_rows)

    if target_filter_manifest_rows:
        filtered_manifest_df = pd.concat(
            [filtered_manifest_df, pd.DataFrame(target_filter_manifest_rows)],
            ignore_index=True,
        )

    split_manifest = {
        **split.manifest,
        "unit_feature_count": int(len(prepared.unit_feature_cols)),
        "shape_feature_count": int(len(prepared.shape_feature_cols)),
        "context_feature_count": int(len(prepared.lightgbm_context_cols)),
        "paired_study_count_non_test": int(split.paired_non_test_all["study_set"].astype(str).nunique()),
        "filtered_unit_drop_count": int(len(unit_drop_cols)),
        "filtered_full_context_drop_count": int(len(full_context_drop_cols)),
        "selection_metric": config.selection_metric,
        "mae_guardrail_pct": config.mae_guardrail_pct,
        "latent_selection_mode": config.latent_selection_mode,
        "study_bias_shrinkage": config.study_bias_shrinkage,
    }

    metrics_df = _frame_with_columns(metric_rows, _METRIC_COLUMNS).sort_values(["target", "r2", "mae"], ascending=[True, False, True]).reset_index(drop=True)
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True).sort_values(["target", "variant_id", "row_uid"]).reset_index(drop=True)
        if prediction_frames
        else pd.DataFrame(columns=_PREDICTION_COLUMNS)
    )
    history_df = _frame_with_columns(history_rows, _HISTORY_COLUMNS).sort_values(["target", "variant_id", "epoch"]).reset_index(drop=True)
    weight_df = _frame_with_columns(weight_rows, _WEIGHT_COLUMNS).sort_values(["target"]).reset_index(drop=True)
    ablation_df = _frame_with_columns(ablation_rows, _ABLATION_COLUMNS).sort_values(["target", "variant_id", "latent_topk"]).reset_index(drop=True)
    recommendation_df = _frame_with_columns(recommendation_rows, _RECOMMEND_COLUMNS).sort_values(["target"]).reset_index(drop=True)
    cluster_df = _frame_with_columns(cluster_rows, _CLUSTER_COLUMNS).sort_values(["target", "n_clusters", "cluster_id"]).reset_index(drop=True)
    study_holdout_df = _build_study_holdout_summary(predictions_df)
    filtered_manifest_df = (
        filtered_manifest_df.loc[:, _FILTERED_MANIFEST_COLUMNS + [c for c in filtered_manifest_df.columns if c not in _FILTERED_MANIFEST_COLUMNS]]
        .sort_values(["feature_set", "selected_for_drop", "abs_smd", "feature"], ascending=[True, False, False, True])
        .reset_index(drop=True)
    )

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"config": config.to_dict(), "main": split_manifest}, f, indent=2, sort_keys=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    weight_df.to_csv(output_dir / "source_reweighting_summary.csv", index=False)
    ablation_df.to_csv(output_dir / "feature_ablation_summary.csv", index=False)
    recommendation_df.to_csv(output_dir / "per_target_recommendation.csv", index=False)
    cluster_df.to_csv(output_dir / "cluster_summary.csv", index=False)
    study_holdout_df.to_csv(output_dir / "study_holdout_summary.csv", index=False)
    filtered_manifest_df.to_csv(output_dir / "filtered_feature_manifest.csv", index=False)
    (output_dir / "README_run.md").write_text(
        _build_readme(
            config=config,
            split_manifest=split_manifest,
            metrics_df=metrics_df,
            recommendations_df=recommendation_df,
        ),
        encoding="utf-8",
    )

    _log(config, f"Saved outputs to {output_dir}")
    if not recommendation_df.empty:
        _log(config, recommendation_df.to_string(index=False))
    return output_dir


def main() -> int:
    args = _parse_args()
    config = _build_config(args)
    run_with_config(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
