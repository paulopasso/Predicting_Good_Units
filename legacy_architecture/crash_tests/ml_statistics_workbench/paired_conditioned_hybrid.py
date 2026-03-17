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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.domain_shift_audit.report import (
    build_domain_classifier_report,
    compute_feature_shift_table,
)
from crash_tests.ml_statistics_workbench.config import MLStatisticsWorkbenchConfig
from crash_tests.ml_statistics_workbench.methods import (
    apply_quantile_transport_model,
    apply_pca_support_projector,
    assign_kmeans_subgroups,
    build_stack_frame,
    compute_importance_weights,
    filter_target_rows,
    fit_kmeans_subgroup_model,
    fit_pca_support_projector,
    fit_quantile_transport_model,
    fit_tree_regressor,
    mask_feature_columns,
    pick_r2_first_recommendation,
    predict_embeddings_from_state,
    predict_tree_regressor,
    prepare_unit_input,
    score_predictions,
    select_shift_filtered_features,
    select_target_transport_features,
    train_zero_shot_mmd_source,
)
from crash_tests.mmd_domain_adaptation.dataset import prepare_feature_table
from crash_tests.self_supervised_domain_adaptation.split_utils import make_main_split


@dataclass
class PairedConditionedHybridConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("crash_tests/ml_statistics_workbench/outputs")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    selection_metric: str = "r2"
    mae_guardrail_pct: float = 0.03
    top_shift_features: int = 40
    assignment_feature_topk: int = 24
    min_abs_smd: float = 0.8
    quantile_points: int = 129
    blend_grid: tuple[float, ...] = (0.5, 0.75, 1.0)
    noise_scale_grid: tuple[float, ...] = (0.0, 0.01, 0.02)
    cluster_candidates: tuple[int, ...] = (2, 3)
    cluster_min_rows: int = 250
    cluster_transport_min_rows: int = 200
    target_transport_topk: int = 18
    target_transport_min_abs_corr: float = 0.06
    manifold_pca_components: int = 8
    manifold_neighbors: int = 16
    manifold_blend: float = 0.5
    manifold_noise_scale: float = 0.0
    max_domain_classifier_rows: int = 3000
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"paired_conditioned_hybrid_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paired-conditioned hybrid benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: PairedConditionedHybridConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _grouped_oof_splits(df: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = df["recording_key"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    effective_splits = min(max(2, int(n_splits)), len(unique_groups))
    splitter = GroupKFold(n_splits=effective_splits)
    return [(np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)) for train_idx, val_idx in splitter.split(df, groups=groups)]


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


def _score_candidate_cv(
    *,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    feature_train: pd.DataFrame,
    base_config: MLStatisticsWorkbenchConfig,
    random_state_offset: int,
) -> dict[str, float]:
    splits = _grouped_oof_splits(train_df, base_config.model_selection_splits)
    oof_pred = np.full(len(train_df), np.nan, dtype=float)
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        model = fit_tree_regressor(
            feature_train.iloc[train_idx],
            y_train[train_idx],
            backend="lightgbm",
            random_state=base_config.random_state + random_state_offset + split_id,
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
        )
        oof_pred[val_idx] = predict_tree_regressor(model, feature_train.iloc[val_idx])
    return score_predictions(y_train, oof_pred)


def _fit_predict_holdout(
    *,
    feature_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_test: pd.DataFrame,
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
    )
    return predict_tree_regressor(model, feature_test)


def _fit_global_transport_candidates(
    *,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    unit_cols: list[str],
    shift_unit: pd.DataFrame,
    config: PairedConditionedHybridConfig,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    translated_frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for blend in config.blend_grid:
        for noise in config.noise_scale_grid:
            variant_id = f"global_b{blend:.2f}_n{noise:.2f}".replace(".", "p")
            model = fit_quantile_transport_model(
                source_df,
                target_df,
                feature_cols=unit_cols,
                shift_df=shift_unit,
                top_n=config.top_shift_features,
                min_abs_smd=config.min_abs_smd,
                quantile_points=config.quantile_points,
                blend=blend,
                noise_scale=noise,
            )
            translated = apply_quantile_transport_model(
                source_df.loc[:, unit_cols],
                model,
                random_state=config.random_state + len(rows) + 1,
            )
            translated_frames[variant_id] = translated
            post_shift = compute_feature_shift_table(
                translated,
                target_df,
                feature_cols=unit_cols,
                comparison=variant_id,
                feature_set="unit",
            )
            summary, _ = build_domain_classifier_report(
                translated.assign(recording_key=source_df["recording_key"].astype(str).to_numpy()),
                target_df,
                feature_cols=unit_cols,
                comparison=variant_id,
                feature_set="unit",
                random_state=config.random_state,
                max_rows_per_domain=config.max_domain_classifier_rows,
            )
            rows.append(
                {
                    "transport_variant": variant_id,
                    "transport_family": "global_quantile_transport",
                    "blend": float(blend),
                    "noise_scale": float(noise),
                    "n_selected_features": int(len(model.selected_features)),
                    "mean_abs_smd_after": float(post_shift["abs_smd"].mean()),
                    "median_abs_smd_after": float(post_shift["abs_smd"].median()),
                    "max_abs_smd_after": float(post_shift["abs_smd"].max()),
                    "cv_auc_after": float(summary["cv_auc_mean"].iloc[0]),
                    "alignment_score": float(summary["cv_auc_mean"].iloc[0]) + float(post_shift["abs_smd"].mean()),
                }
            )
    summary_df = pd.DataFrame(rows).sort_values(
        ["alignment_score", "cv_auc_after", "mean_abs_smd_after", "transport_variant"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return translated_frames, summary_df


def _fit_cluster_translated_source(
    *,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    unit_cols: list[str],
    shift_unit: pd.DataFrame,
    assignment_cols: list[str],
    blend: float,
    noise_scale: float,
    n_clusters: int,
    config: PairedConditionedHybridConfig,
) -> pd.DataFrame:
    subgroup_model = fit_kmeans_subgroup_model(
        target_df,
        feature_cols=assignment_cols,
        n_clusters=n_clusters,
        random_state=config.random_state + n_clusters,
    )
    source_clusters = assign_kmeans_subgroups(source_df, subgroup_model)
    target_clusters = assign_kmeans_subgroups(target_df, subgroup_model)
    translated = source_df.loc[:, unit_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    global_model = fit_quantile_transport_model(
        source_df,
        target_df,
        feature_cols=unit_cols,
        shift_df=shift_unit,
        top_n=config.top_shift_features,
        min_abs_smd=config.min_abs_smd,
        quantile_points=config.quantile_points,
        blend=blend,
        noise_scale=noise_scale,
    )
    global_translated = apply_quantile_transport_model(
        source_df.loc[:, unit_cols].apply(pd.to_numeric, errors="coerce").astype(float),
        global_model,
        random_state=config.random_state + 701,
    )
    for cluster_id in sorted(np.unique(target_clusters).tolist()):
        source_mask = source_clusters == int(cluster_id)
        target_mask = target_clusters == int(cluster_id)
        source_cluster = source_df.loc[source_mask, unit_cols]
        target_cluster = target_df.loc[target_mask, unit_cols]
        if len(source_cluster) < config.cluster_transport_min_rows or len(target_cluster) < config.cluster_transport_min_rows:
            translated.loc[source_mask, :] = global_translated.loc[source_mask, :]
            continue
        cluster_shift = compute_feature_shift_table(
            source_cluster,
            target_cluster,
            feature_cols=unit_cols,
            comparison=f"cluster_{cluster_id}",
            feature_set="unit",
        )
        cluster_model = fit_quantile_transport_model(
            source_cluster,
            target_cluster,
            feature_cols=unit_cols,
            shift_df=cluster_shift,
            top_n=config.top_shift_features,
            min_abs_smd=config.min_abs_smd,
            quantile_points=config.quantile_points,
            blend=blend,
            noise_scale=noise_scale,
            min_feature_rows=max(48, config.cluster_transport_min_rows // 4),
        )
        translated.loc[source_mask, :] = apply_quantile_transport_model(
            source_cluster,
            cluster_model,
            random_state=config.random_state + 800 + int(cluster_id),
        ).to_numpy()
    return translated


def _select_transport_variants(
    *,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    unit_cols: list[str],
    shift_unit: pd.DataFrame,
    config: PairedConditionedHybridConfig,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    assignment_cols = (
        shift_unit[shift_unit["feature"].astype(str).isin(unit_cols)]
        .sort_values(["abs_smd", "ks_stat", "feature"], ascending=[False, False, True])
        .head(int(config.assignment_feature_topk))["feature"]
        .astype(str)
        .tolist()
    )
    translated_frames, global_summary = _fit_global_transport_candidates(
        source_df=source_df,
        target_df=target_df,
        unit_cols=unit_cols,
        shift_unit=shift_unit,
        config=config,
    )
    rows = global_summary.to_dict(orient="records")
    best_global = global_summary.iloc[0]
    best_blend = float(best_global["blend"])
    best_noise = float(best_global["noise_scale"])
    for n_clusters in config.cluster_candidates:
        variant_id = f"cluster_k{int(n_clusters)}_b{best_blend:.2f}_n{best_noise:.2f}".replace(".", "p")
        translated = _fit_cluster_translated_source(
            source_df=source_df,
            target_df=target_df,
            unit_cols=unit_cols,
            shift_unit=shift_unit,
            assignment_cols=assignment_cols,
            blend=best_blend,
            noise_scale=best_noise,
            n_clusters=int(n_clusters),
            config=config,
        )
        translated_frames[variant_id] = translated
        post_shift = compute_feature_shift_table(
            translated,
            target_df,
            feature_cols=unit_cols,
            comparison=variant_id,
            feature_set="unit",
        )
        summary, _ = build_domain_classifier_report(
            translated.assign(recording_key=source_df["recording_key"].astype(str).to_numpy()),
            target_df,
            feature_cols=unit_cols,
            comparison=variant_id,
            feature_set="unit",
            random_state=config.random_state,
            max_rows_per_domain=config.max_domain_classifier_rows,
        )
        rows.append(
            {
                "transport_variant": variant_id,
                "transport_family": "cluster_quantile_transport",
                "blend": best_blend,
                "noise_scale": best_noise,
                "n_clusters": int(n_clusters),
                "assignment_feature_count": int(len(assignment_cols)),
                "mean_abs_smd_after": float(post_shift["abs_smd"].mean()),
                "median_abs_smd_after": float(post_shift["abs_smd"].median()),
                "max_abs_smd_after": float(post_shift["abs_smd"].max()),
                "cv_auc_after": float(summary["cv_auc_mean"].iloc[0]),
                "alignment_score": float(summary["cv_auc_mean"].iloc[0]) + float(post_shift["abs_smd"].mean()),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values(
        ["alignment_score", "cv_auc_after", "mean_abs_smd_after", "transport_variant"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return translated_frames, summary_df


def _source_prediction(
    *,
    source_features: pd.DataFrame,
    y_source: np.ndarray,
    paired_train_features: pd.DataFrame,
    paired_test_features: pd.DataFrame,
    paired_target_df: pd.DataFrame,
    base_config: MLStatisticsWorkbenchConfig,
    random_state_offset: int,
    reweight: bool,
) -> tuple[np.ndarray, np.ndarray]:
    weights = None
    if reweight:
        weight_result = compute_importance_weights(
            source_features,
            paired_target_df,
            feature_cols=source_features.columns.tolist(),
            random_state=base_config.random_state + random_state_offset,
            clip_min=base_config.importance_weight_clip_min,
            clip_max=base_config.importance_weight_clip_max,
            c_value=base_config.domain_classifier_c,
        )
        weights = weight_result.weights
    model = fit_tree_regressor(
        source_features,
        y_source,
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
        sample_weight=weights,
    )
    train_pred = predict_tree_regressor(model, paired_train_features)
    test_pred = predict_tree_regressor(model, paired_test_features)
    return train_pred, test_pred


def _run_with_config(config: PairedConditionedHybridConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

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
    shift_unit = compute_feature_shift_table(
        split.hybrid_source,
        split.paired_non_test_all,
        feature_cols=prepared.unit_feature_cols,
        comparison="hybrid_vs_paired_non_test",
        feature_set="unit",
    )
    raw_alignment_summary, _ = build_domain_classifier_report(
        split.hybrid_source,
        split.paired_non_test_all,
        feature_cols=prepared.unit_feature_cols,
        comparison="raw_hybrid_vs_paired",
        feature_set="unit",
        random_state=config.random_state,
        max_rows_per_domain=config.max_domain_classifier_rows,
    )
    translated_frames, transport_summary = _select_transport_variants(
        source_df=split.hybrid_source,
        target_df=split.paired_non_test_all,
        unit_cols=prepared.unit_feature_cols,
        shift_unit=shift_unit,
        config=config,
    )
    transport_summary = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "transport_variant": "raw_hybrid",
                        "transport_family": "identity",
                        "blend": 0.0,
                        "noise_scale": 0.0,
                        "mean_abs_smd_after": float(shift_unit["abs_smd"].mean()),
                        "median_abs_smd_after": float(shift_unit["abs_smd"].median()),
                        "max_abs_smd_after": float(shift_unit["abs_smd"].max()),
                        "cv_auc_after": float(raw_alignment_summary["cv_auc_mean"].iloc[0]),
                        "alignment_score": float(raw_alignment_summary["cv_auc_mean"].iloc[0]) + float(shift_unit["abs_smd"].mean()),
                    }
                ]
            ),
            transport_summary,
        ],
        ignore_index=True,
    ).sort_values(["alignment_score", "cv_auc_after", "mean_abs_smd_after"]).reset_index(drop=True)
    transport_summary_rows: list[dict[str, Any]] = transport_summary.to_dict(orient="records")

    best_global_variant = transport_summary[transport_summary["transport_family"] == "global_quantile_transport"].iloc[0]["transport_variant"]
    cluster_rows = transport_summary[transport_summary["transport_family"] == "cluster_quantile_transport"]
    best_cluster_variant = str(cluster_rows.iloc[0]["transport_variant"]) if not cluster_rows.empty else str(best_global_variant)

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []
    feature_ablation_rows: list[dict[str, Any]] = []
    transport_feature_rows: list[dict[str, Any]] = []

    for variant_name, translated_unit_df in translated_frames.items():
        post_shift = compute_feature_shift_table(
            translated_unit_df,
            split.paired_non_test_all,
            feature_cols=prepared.unit_feature_cols,
            comparison=variant_name,
            feature_set="unit",
        )
        merged = shift_unit.loc[:, ["feature", "abs_smd", "smd", "mean_a", "mean_b", "std_a", "std_b"]].merge(
            post_shift.loc[:, ["feature", "abs_smd", "smd", "mean_a", "mean_b", "std_a", "std_b"]],
            on="feature",
            suffixes=("_before", "_after"),
            how="left",
        )
        merged["transport_variant"] = variant_name
        transport_feature_rows.extend(merged.to_dict(orient="records"))

    for target in ("fpos", "fmiss"):
        _log(config, f"[{target}] building paired-conditioned hybrid candidates")
        source_target = base.source_fmiss_target if target == "fmiss" else target
        source_df, y_source = filter_target_rows(split.hybrid_source, source_target)
        target_train_df, y_train = filter_target_rows(split.paired_finetune, target)
        target_test_df, y_test = filter_target_rows(split.paired_test, target)

        target_transport_manifest = select_target_transport_features(
            shift_unit,
            train_df=target_train_df,
            target=target,
            allowed_cols=prepared.unit_feature_cols,
            top_k=config.target_transport_topk,
            min_abs_smd=config.min_abs_smd,
            min_abs_corr=config.target_transport_min_abs_corr,
        )
        target_transport_features = (
            target_transport_manifest.loc[target_transport_manifest["selected_for_transport"], "feature"]
            .astype(str)
            .tolist()
        )
        if not target_transport_features:
            target_transport_features = (
                shift_unit.sort_values(["abs_smd", "ks_stat", "feature"], ascending=[False, False, True])
                .head(int(config.target_transport_topk))["feature"]
                .astype(str)
                .tolist()
            )

        target_quantile_source = (
            translated_frames[str(best_global_variant)]
            .loc[source_df.index]
            .reset_index(drop=True)
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        support_projector = fit_pca_support_projector(
            split.paired_non_test_all,
            feature_cols=target_transport_features,
            n_components=config.manifold_pca_components,
            n_neighbors=config.manifold_neighbors,
        )
        target_manifold_source = target_quantile_source.copy()
        target_manifold_source.loc[:, target_transport_features] = apply_pca_support_projector(
            target_quantile_source.loc[:, target_transport_features].copy(),
            support_projector,
            blend=config.manifold_blend,
            noise_scale=config.manifold_noise_scale,
            random_state=config.random_state + (201 if target == "fmiss" else 191),
        ).to_numpy()

        for variant_name, translated_unit_df in (
            (f"{target}_target_quantile", target_quantile_source),
            (f"{target}_target_manifold", target_manifold_source),
        ):
            post_shift = compute_feature_shift_table(
                translated_unit_df,
                split.paired_non_test_all,
                feature_cols=prepared.unit_feature_cols,
                comparison=variant_name,
                feature_set="unit",
            )
            summary, _ = build_domain_classifier_report(
                translated_unit_df.assign(recording_key=source_df["recording_key"].astype(str).to_numpy()),
                split.paired_non_test_all,
                feature_cols=prepared.unit_feature_cols,
                comparison=variant_name,
                feature_set="unit",
                random_state=config.random_state,
                max_rows_per_domain=config.max_domain_classifier_rows,
            )
            transport_summary_rows.append(
                {
                    "transport_variant": variant_name,
                    "transport_family": "target_specific_manifold" if variant_name.endswith("manifold") else "target_specific_global_plus_selection",
                    "target": target,
                    "blend": float(best_global_variant.split("_b")[1].split("_n")[0].replace("p", ".")) if variant_name.endswith("quantile") else config.manifold_blend,
                    "noise_scale": 0.0 if variant_name.endswith("quantile") else config.manifold_noise_scale,
                    "n_selected_features": int(len(target_transport_features)),
                    "mean_abs_smd_after": float(post_shift["abs_smd"].mean()),
                    "median_abs_smd_after": float(post_shift["abs_smd"].median()),
                    "max_abs_smd_after": float(post_shift["abs_smd"].max()),
                    "cv_auc_after": float(summary["cv_auc_mean"].iloc[0]),
                    "alignment_score": float(summary["cv_auc_mean"].iloc[0]) + float(post_shift["abs_smd"].mean()),
                }
            )
            merged = shift_unit.loc[:, ["feature", "abs_smd", "smd", "mean_a", "mean_b", "std_a", "std_b"]].merge(
                post_shift.loc[:, ["feature", "abs_smd", "smd", "mean_a", "mean_b", "std_a", "std_b"]],
                on="feature",
                suffixes=("_before", "_after"),
                how="left",
            )
            merged["transport_variant"] = variant_name
            merged["target"] = target
            transport_feature_rows.extend(merged.to_dict(orient="records"))

        if target == "fpos":
            filtered_drop_cols = set(unit_drop_cols) | set(context_drop_cols)
            filtered_context_cols = [col for col in prepared.lightgbm_context_cols if col not in filtered_drop_cols]
            filtered_target_train_df = mask_feature_columns(target_train_df, filtered_drop_cols)
            filtered_target_test_df = mask_feature_columns(target_test_df, filtered_drop_cols)
            filtered_hybrid_source = mask_feature_columns(split.hybrid_source, unit_drop_cols)
            filtered_paired_non_test_all = mask_feature_columns(split.paired_non_test_all, unit_drop_cols)
            filtered_ssl_pool = mask_feature_columns(split.ssl_pool if not split.ssl_pool.empty else split.paired_non_test_all, unit_drop_cols)
            embed_artifact = train_zero_shot_mmd_source(
                target="fpos",
                hybrid_source=filtered_hybrid_source,
                paired_non_test_all=filtered_paired_non_test_all,
                ssl_pool=filtered_ssl_pool,
                prepared=prepared,
                random_state=config.random_state + 17,
                source_fmiss_target=base.source_fmiss_target,
                mmd_lambda=0.0,
                batch_size=base.mmd_batch_size,
                hidden_dim=base.mmd_hidden_dim,
                latent_dim=base.mmd_latent_dim,
                source_train_epochs=base.mmd_source_train_epochs,
                source_lr=base.mmd_source_lr,
                weight_decay=base.mmd_weight_decay,
                patience=base.mmd_patience,
                verbose=config.verbose,
                checkpoint_path=(output_dir / "checkpoints" / "paired_conditioned_embed_nommd_fpos_source.pt"),
            )
            filtered_train_input = prepare_unit_input(filtered_target_train_df, prepared, embed_artifact.unit_scaler)
            filtered_test_input = prepare_unit_input(filtered_target_test_df, prepared, embed_artifact.unit_scaler)
            embed_train_pred, embed_train_latent = predict_embeddings_from_state(
                model_state=embed_artifact.model_state,
                eval_data=filtered_train_input,
                hidden_dim=base.mmd_hidden_dim,
                latent_dim=base.mmd_latent_dim,
                random_state=config.random_state + 102,
                target_transform_mode=base.resolved_target_transform(),
            )
            embed_test_pred, embed_test_latent = predict_embeddings_from_state(
                model_state=embed_artifact.model_state,
                eval_data=filtered_test_input,
                hidden_dim=base.mmd_hidden_dim,
                latent_dim=base.mmd_latent_dim,
                random_state=config.random_state + 103,
                target_transform_mode=base.resolved_target_transform(),
            )
            orig_source_pred_train, orig_source_pred_test = _source_prediction(
                source_features=source_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                y_source=y_source,
                paired_train_features=target_train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_test_features=target_test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_target_df=split.paired_non_test_all,
                base_config=base,
                random_state_offset=210,
                reweight=True,
            )
            global_source_pred_train, global_source_pred_test = _source_prediction(
                source_features=translated_frames[str(best_global_variant)].loc[source_df.index].reset_index(drop=True),
                y_source=y_source,
                paired_train_features=target_train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_test_features=target_test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_target_df=split.paired_non_test_all,
                base_config=base,
                random_state_offset=310,
                reweight=True,
            )
            cluster_source_pred_train, cluster_source_pred_test = _source_prediction(
                source_features=translated_frames[str(best_cluster_variant)].loc[source_df.index].reset_index(drop=True),
                y_source=y_source,
                paired_train_features=target_train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_test_features=target_test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_target_df=split.paired_non_test_all,
                base_config=base,
                random_state_offset=410,
                reweight=True,
            )
            target_quantile_pred_train, target_quantile_pred_test = _source_prediction(
                source_features=target_quantile_source.reset_index(drop=True),
                y_source=y_source,
                paired_train_features=target_train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_test_features=target_test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_target_df=split.paired_non_test_all,
                base_config=base,
                random_state_offset=430,
                reweight=True,
            )
            target_manifold_pred_train, target_manifold_pred_test = _source_prediction(
                source_features=target_manifold_source.reset_index(drop=True),
                y_source=y_source,
                paired_train_features=target_train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_test_features=target_test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_target_df=split.paired_non_test_all,
                base_config=base,
                random_state_offset=450,
                reweight=True,
            )
            candidate_specs = {
                "contextual_paired_only_lightgbm": {
                    "notes": "paired-only filtered context baseline",
                    "train": target_train_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                    "test": target_test_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                },
                "embed_nommd_fullctx_lightgbm_filtered": {
                    "notes": "current best filtered neural embedding stack",
                    "train": build_stack_frame(
                        filtered_target_train_df,
                        context_cols=filtered_context_cols,
                        extra_features={"source_pred": embed_train_pred, "latent": embed_train_latent},
                    ),
                    "test": build_stack_frame(
                        filtered_target_test_df,
                        context_cols=filtered_context_cols,
                        extra_features={"source_pred": embed_test_pred, "latent": embed_test_latent},
                    ),
                },
                "paired_conditioned_global_fullctx_lightgbm_filtered": {
                    "notes": f"filtered context + globally transported source prediction ({best_global_variant})",
                    "train": build_stack_frame(filtered_target_train_df, context_cols=filtered_context_cols, extra_features={"source_pred": global_source_pred_train}),
                    "test": build_stack_frame(filtered_target_test_df, context_cols=filtered_context_cols, extra_features={"source_pred": global_source_pred_test}),
                },
                "paired_conditioned_dual_fullctx_lightgbm_filtered": {
                    "notes": f"filtered context + original weighted source + globally transported source ({best_global_variant})",
                    "train": build_stack_frame(filtered_target_train_df, context_cols=filtered_context_cols, extra_features={"source_pred_orig": orig_source_pred_train, "source_pred_transport": global_source_pred_train}),
                    "test": build_stack_frame(filtered_target_test_df, context_cols=filtered_context_cols, extra_features={"source_pred_orig": orig_source_pred_test, "source_pred_transport": global_source_pred_test}),
                },
                "paired_conditioned_cluster_fullctx_lightgbm_filtered": {
                    "notes": f"filtered context + cluster-transported source prediction ({best_cluster_variant})",
                    "train": build_stack_frame(filtered_target_train_df, context_cols=filtered_context_cols, extra_features={"source_pred": cluster_source_pred_train}),
                    "test": build_stack_frame(filtered_target_test_df, context_cols=filtered_context_cols, extra_features={"source_pred": cluster_source_pred_test}),
                },
                "paired_conditioned_target_quantile_fullctx_lightgbm_filtered": {
                    "notes": f"filtered context + target-specific quantile transported source prediction ({len(target_transport_features)} features)",
                    "train": build_stack_frame(filtered_target_train_df, context_cols=filtered_context_cols, extra_features={"source_pred": target_quantile_pred_train}),
                    "test": build_stack_frame(filtered_target_test_df, context_cols=filtered_context_cols, extra_features={"source_pred": target_quantile_pred_test}),
                },
                "paired_conditioned_target_manifold_fullctx_lightgbm_filtered": {
                    "notes": f"filtered context + target-specific manifold transported source prediction ({len(target_transport_features)} features)",
                    "train": build_stack_frame(filtered_target_train_df, context_cols=filtered_context_cols, extra_features={"source_pred": target_manifold_pred_train}),
                    "test": build_stack_frame(filtered_target_test_df, context_cols=filtered_context_cols, extra_features={"source_pred": target_manifold_pred_test}),
                },
            }
            reference_variant = "embed_nommd_fullctx_lightgbm_filtered"
        else:
            orig_source_pred_train, orig_source_pred_test = _source_prediction(
                source_features=source_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                y_source=y_source,
                paired_train_features=target_train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_test_features=target_test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_target_df=split.paired_non_test_all,
                base_config=base,
                random_state_offset=510,
                reweight=True,
            )
            global_source_pred_train, global_source_pred_test = _source_prediction(
                source_features=translated_frames[str(best_global_variant)].loc[source_df.index].reset_index(drop=True),
                y_source=y_source,
                paired_train_features=target_train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_test_features=target_test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_target_df=split.paired_non_test_all,
                base_config=base,
                random_state_offset=610,
                reweight=True,
            )
            cluster_source_pred_train, cluster_source_pred_test = _source_prediction(
                source_features=translated_frames[str(best_cluster_variant)].loc[source_df.index].reset_index(drop=True),
                y_source=y_source,
                paired_train_features=target_train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_test_features=target_test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_target_df=split.paired_non_test_all,
                base_config=base,
                random_state_offset=710,
                reweight=True,
            )
            target_quantile_pred_train, target_quantile_pred_test = _source_prediction(
                source_features=target_quantile_source.reset_index(drop=True),
                y_source=y_source,
                paired_train_features=target_train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_test_features=target_test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_target_df=split.paired_non_test_all,
                base_config=base,
                random_state_offset=730,
                reweight=True,
            )
            target_manifold_pred_train, target_manifold_pred_test = _source_prediction(
                source_features=target_manifold_source.reset_index(drop=True),
                y_source=y_source,
                paired_train_features=target_train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_test_features=target_test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce"),
                paired_target_df=split.paired_non_test_all,
                base_config=base,
                random_state_offset=750,
                reweight=True,
            )
            candidate_specs = {
                "contextual_paired_only_lightgbm": {
                    "notes": "paired-only full context baseline",
                    "train": target_train_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                    "test": target_test_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
                },
                "source_reweighted_fullctx_lightgbm": {
                    "notes": "current best domain-reweighted source prediction + full context",
                    "train": build_stack_frame(target_train_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred": orig_source_pred_train}),
                    "test": build_stack_frame(target_test_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred": orig_source_pred_test}),
                },
                "paired_conditioned_global_fullctx_lightgbm": {
                    "notes": f"full context + globally transported source prediction ({best_global_variant})",
                    "train": build_stack_frame(target_train_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred": global_source_pred_train}),
                    "test": build_stack_frame(target_test_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred": global_source_pred_test}),
                },
                "paired_conditioned_dual_fullctx_lightgbm": {
                    "notes": f"full context + original weighted source + globally transported source ({best_global_variant})",
                    "train": build_stack_frame(target_train_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred_orig": orig_source_pred_train, "source_pred_transport": global_source_pred_train}),
                    "test": build_stack_frame(target_test_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred_orig": orig_source_pred_test, "source_pred_transport": global_source_pred_test}),
                },
                "paired_conditioned_cluster_fullctx_lightgbm": {
                    "notes": f"full context + cluster-transported source prediction ({best_cluster_variant})",
                    "train": build_stack_frame(target_train_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred": cluster_source_pred_train}),
                    "test": build_stack_frame(target_test_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred": cluster_source_pred_test}),
                },
                "paired_conditioned_target_quantile_fullctx_lightgbm": {
                    "notes": f"full context + target-specific quantile transported source prediction ({len(target_transport_features)} features)",
                    "train": build_stack_frame(target_train_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred": target_quantile_pred_train}),
                    "test": build_stack_frame(target_test_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred": target_quantile_pred_test}),
                },
                "paired_conditioned_target_manifold_fullctx_lightgbm": {
                    "notes": f"full context + target-specific manifold transported source prediction ({len(target_transport_features)} features)",
                    "train": build_stack_frame(target_train_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred": target_manifold_pred_train}),
                    "test": build_stack_frame(target_test_df, context_cols=prepared.lightgbm_context_cols, extra_features={"source_pred": target_manifold_pred_test}),
                },
            }
            reference_variant = "source_reweighted_fullctx_lightgbm"

        holdout_metrics_rows: list[dict[str, Any]] = []
        for idx, (variant_id, spec) in enumerate(candidate_specs.items(), start=1):
            cv_metrics = _score_candidate_cv(
                train_df=target_train_df,
                y_train=y_train,
                feature_train=spec["train"],
                base_config=base,
                random_state_offset=1000 * idx,
            )
            holdout_pred = _fit_predict_holdout(
                feature_train=spec["train"],
                y_train=y_train,
                feature_test=spec["test"],
                base_config=base,
                random_state_offset=2000 * idx,
            )
            holdout_metrics = score_predictions(y_test, holdout_pred)
            metric_rows.append(
                {
                    "protocol": "main",
                    "target": target,
                    "variant_id": variant_id,
                    "family": "paired_conditioned_hybrid" if "paired_conditioned" in variant_id else "reference",
                    "backend": "lightgbm",
                    "source_target": source_target if "contextual_paired_only" not in variant_id else np.nan,
                    "notes": spec["notes"],
                    **holdout_metrics,
                }
            )
            feature_ablation_rows.append(
                {
                    "target": target,
                    "variant_id": variant_id,
                    "cv_mae": float(cv_metrics["mae"]),
                    "cv_r2": float(cv_metrics["r2"]),
                    "holdout_mae": float(holdout_metrics["mae"]),
                    "holdout_r2": float(holdout_metrics["r2"]),
                    "notes": spec["notes"],
                }
            )
            holdout_metrics_rows.append({"target": target, "variant_id": variant_id, "mae": holdout_metrics["mae"], "r2": holdout_metrics["r2"]})
            for row_uid, recording_key, study_set, truth, pred in zip(
                target_test_df["row_uid"].astype(str),
                target_test_df["recording_key"].astype(str),
                target_test_df["study_set"].astype(str),
                y_test,
                holdout_pred,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "protocol": "main",
                        "target": target,
                        "variant_id": variant_id,
                        "row_uid": row_uid,
                        "recording_key": recording_key,
                        "study_set": study_set,
                        "true": float(truth),
                        "pred": float(pred),
                        "abs_error": float(abs(truth - pred)),
                    }
                )
        recommendations = pick_r2_first_recommendation(
            pd.DataFrame(holdout_metrics_rows),
            target=target,
            reference_variant_id=reference_variant,
            selection_metric=config.selection_metric,
            mae_guardrail_pct=config.mae_guardrail_pct,
            min_r2_improvement=0.0,
        )
        recommendation_rows.append(recommendations)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["target", "r2", "mae"], ascending=[True, False, True]).reset_index(drop=True)
    predictions_df = pd.DataFrame(prediction_rows).sort_values(["target", "variant_id", "row_uid"]).reset_index(drop=True)
    recommendations_df = pd.DataFrame(recommendation_rows)
    feature_ablation_df = pd.DataFrame(feature_ablation_rows)
    transport_feature_df = pd.DataFrame(transport_feature_rows).sort_values(["transport_variant", "abs_smd_after", "feature"], ascending=[True, False, True]).reset_index(drop=True)
    transport_summary_df = pd.DataFrame(transport_summary_rows).sort_values(
        ["alignment_score", "cv_auc_after", "mean_abs_smd_after", "transport_variant"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    split_manifest = {
        **split.manifest,
        "top_shift_features": int(config.top_shift_features),
        "assignment_feature_topk": int(config.assignment_feature_topk),
        "min_abs_smd": float(config.min_abs_smd),
        "best_global_transport_variant": str(best_global_variant),
        "best_cluster_transport_variant": str(best_cluster_variant),
    }

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    recommendations_df.to_csv(output_dir / "per_target_recommendation.csv", index=False)
    feature_ablation_df.to_csv(output_dir / "feature_ablation_summary.csv", index=False)
    transport_summary_df.to_csv(output_dir / "transport_alignment_summary.csv", index=False)
    transport_feature_df.to_csv(output_dir / "transport_feature_manifest.csv", index=False)
    filtered_manifest.to_csv(output_dir / "filtered_feature_manifest.csv", index=False)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    lines = [
        "# Paired-Conditioned Hybrid Benchmark",
        "",
        "This run transforms hybrid unit features toward the paired domain using measured feature-shift diagnostics, then uses those translated source models inside the strict paired holdout stack.",
        "",
        "## Transport Selection",
        f"- best_global_transport_variant: `{best_global_variant}`",
        f"- best_cluster_transport_variant: `{best_cluster_variant}`",
        "",
        "## Recommendations",
        "",
    ]
    for _, row in recommendations_df.iterrows():
        lines.extend(
            [
                f"### {row['target']}",
                f"- reference: `{row['reference_variant_id']}`",
                f"- recommendation: `{row['recommended_variant_id']}`",
                f"- recommended R²: `{row['recommended_r2']:.4f}`",
                f"- recommended MAE: `{row['recommended_mae']:.4f}`",
                "",
            ]
        )
    (output_dir / "README_run.md").write_text("\n".join(lines) + "\n")
    _log(config, f"Saved outputs to {output_dir}")
    if not recommendations_df.empty:
        print(recommendations_df.to_string(index=False))
    return output_dir


def run_with_config(config: PairedConditionedHybridConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = PairedConditionedHybridConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.blend_grid = (0.75, 1.0)
        cli_config.noise_scale_grid = (0.0, 0.01)
        cli_config.cluster_candidates = (2,)
        cli_config.top_shift_features = 24
    _run_with_config(cli_config)
