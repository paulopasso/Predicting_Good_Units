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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.ml_statistics_workbench.config import MLStatisticsWorkbenchConfig
from crash_tests.ml_statistics_workbench.label_budget import (
    _aggregate_budget_curve,
    _choose_budget_recommendations,
    _grouped_budget_subset,
)
from crash_tests.ml_statistics_workbench.methods import (
    build_stack_frame,
    compute_importance_weights,
    filter_target_rows,
    fit_tree_regressor,
    make_local_regression_mixup,
    mask_feature_columns,
    predict_embeddings_from_state,
    predict_tree_regressor,
    prepare_unit_input,
    score_predictions,
    select_shift_filtered_features,
    train_zero_shot_mmd_source,
)
from crash_tests.mmd_domain_adaptation.dataset import prepare_feature_table
from crash_tests.self_supervised_domain_adaptation.split_utils import make_main_split
from crash_tests.domain_shift_audit.report import compute_feature_shift_table


@dataclass
class SyntheticAugmentationConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("crash_tests/ml_statistics_workbench/outputs")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    budgets: tuple[int, ...] = (40, 60, 80, 107)
    repeats: int = 3
    augmentation_factor: float = 1.0
    augmentation_weight: float = 0.35
    augmentation_noise_scale: float = 0.01
    augmentation_k_neighbors: int = 3
    augmentation_alpha_range: tuple[float, float] = (0.25, 0.75)
    subgroup_column: str = "study_set"
    budget_selection_r2_frac: float = 0.98
    budget_selection_mae_frac: float = 1.03
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"synthetic_augmentation_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic paired augmentation benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: SyntheticAugmentationConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _metric_row(
    *,
    target: str,
    variant_id: str,
    budget_rows: int,
    repeat_id: int,
    subset_recordings: int,
    synthetic_rows: int,
    synthetic_weight: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "target": target,
        "variant_id": variant_id,
        "budget_rows": int(budget_rows),
        "repeat_id": int(repeat_id),
        "subset_recordings": int(subset_recordings),
        "synthetic_rows": int(synthetic_rows),
        "synthetic_weight": float(synthetic_weight),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "bias": float(metrics["bias"]),
        "calibration_slope": float(metrics["calibration_slope"]),
        "calibration_intercept": float(metrics["calibration_intercept"]),
    }


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


def _augment_subset(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    *,
    subgroup_labels: pd.Series,
    config: SyntheticAugmentationConfig,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n_synth = int(round(len(feature_frame) * float(config.augmentation_factor)))
    if len(feature_frame) < 4 or n_synth <= 0:
        return feature_frame.reset_index(drop=True), np.asarray(target, dtype=float), np.ones(len(feature_frame), dtype=np.float32)
    synth_frame, synth_target = make_local_regression_mixup(
        feature_frame,
        target,
        subgroup_labels=subgroup_labels,
        n_synthetic=n_synth,
        k_neighbors=config.augmentation_k_neighbors,
        alpha_range=config.augmentation_alpha_range,
        random_state=random_state,
        noise_scale=config.augmentation_noise_scale,
    )
    base_frame = feature_frame.reset_index(drop=True)
    base_target = np.asarray(target, dtype=float)
    if synth_frame.empty:
        return base_frame, base_target, np.ones(len(base_frame), dtype=np.float32)
    full_frame = pd.concat([base_frame, synth_frame.reset_index(drop=True)], ignore_index=True)
    full_target = np.concatenate([base_target, synth_target.astype(float)])
    sample_weight = np.concatenate(
        [
            np.ones(len(base_frame), dtype=np.float32),
            np.full(len(synth_frame), float(config.augmentation_weight), dtype=np.float32),
        ]
    )
    return full_frame, full_target, sample_weight


def _prepare_variants(
    *,
    split: Any,
    prepared: Any,
    base: MLStatisticsWorkbenchConfig,
    config: SyntheticAugmentationConfig,
    output_dir: Path,
    unit_drop_cols: list[str],
    context_drop_cols: list[str],
) -> dict[str, tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, dict[str, Any]]]]:
    unlabeled_target_df = split.ssl_pool if not split.ssl_pool.empty else split.paired_non_test_all

    filtered_drop_cols = set(unit_drop_cols) | set(context_drop_cols)
    filtered_context_cols = [col for col in prepared.lightgbm_context_cols if col not in filtered_drop_cols]

    # fpos
    fpos_train_df, y_fpos_train = filter_target_rows(split.paired_finetune, "fpos")
    fpos_test_df, y_fpos_test = filter_target_rows(split.paired_test, "fpos")
    filtered_fpos_train_df = mask_feature_columns(fpos_train_df, filtered_drop_cols)
    filtered_fpos_test_df = mask_feature_columns(fpos_test_df, filtered_drop_cols)
    filtered_hybrid_source = mask_feature_columns(split.hybrid_source, unit_drop_cols)
    filtered_paired_non_test_all = mask_feature_columns(split.paired_non_test_all, unit_drop_cols)
    filtered_ssl_pool = mask_feature_columns(unlabeled_target_df, unit_drop_cols)

    _log(config, "[fpos] preparing filtered source embedding stack")
    fpos_embed_artifact = train_zero_shot_mmd_source(
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
        checkpoint_path=(output_dir / "checkpoints" / "synthetic_embed_nommd_filtered_fpos_source.pt"),
    )
    filtered_fpos_train_input = prepare_unit_input(filtered_fpos_train_df, prepared, fpos_embed_artifact.unit_scaler)
    filtered_fpos_test_input = prepare_unit_input(filtered_fpos_test_df, prepared, fpos_embed_artifact.unit_scaler)
    fpos_embed_train_pred, fpos_embed_train_latent = predict_embeddings_from_state(
        model_state=fpos_embed_artifact.model_state,
        eval_data=filtered_fpos_train_input,
        hidden_dim=base.mmd_hidden_dim,
        latent_dim=base.mmd_latent_dim,
        random_state=config.random_state + 102,
        target_transform_mode=base.resolved_target_transform(),
    )
    fpos_embed_test_pred, fpos_embed_test_latent = predict_embeddings_from_state(
        model_state=fpos_embed_artifact.model_state,
        eval_data=filtered_fpos_test_input,
        hidden_dim=base.mmd_hidden_dim,
        latent_dim=base.mmd_latent_dim,
        random_state=config.random_state + 103,
        target_transform_mode=base.resolved_target_transform(),
    )
    fpos_variants = {
        "contextual_paired_only_lightgbm": {
            "backend": "lightgbm",
            "train_frame": fpos_train_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
            "test_frame": fpos_test_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
            "subgroup_labels": fpos_train_df[config.subgroup_column].astype(str).reset_index(drop=True),
            "family": "paired_context",
        },
        "embed_nommd_fullctx_lightgbm_filtered": {
            "backend": "lightgbm",
            "train_frame": build_stack_frame(
                filtered_fpos_train_df,
                context_cols=filtered_context_cols,
                extra_features={"source_pred": fpos_embed_train_pred, "latent": fpos_embed_train_latent},
            ),
            "test_frame": build_stack_frame(
                filtered_fpos_test_df,
                context_cols=filtered_context_cols,
                extra_features={"source_pred": fpos_embed_test_pred, "latent": fpos_embed_test_latent},
            ),
            "subgroup_labels": fpos_train_df[config.subgroup_column].astype(str).reset_index(drop=True),
            "family": "embedding_stack_filtered",
        },
    }

    # fmiss
    fmiss_train_df, y_fmiss_train = filter_target_rows(split.paired_finetune, "fmiss")
    fmiss_test_df, y_fmiss_test = filter_target_rows(split.paired_test, "fmiss")
    fmiss_source_df, y_fmiss_source = filter_target_rows(split.hybrid_source, base.source_fmiss_target)
    _log(config, "[fmiss] preparing source-reweighted stack")
    fmiss_weight_result = compute_importance_weights(
        fmiss_source_df,
        unlabeled_target_df,
        feature_cols=prepared.unit_feature_cols,
        random_state=config.random_state,
        clip_min=base.importance_weight_clip_min,
        clip_max=base.importance_weight_clip_max,
        c_value=base.domain_classifier_c,
    )
    fmiss_source_model = fit_tree_regressor(
        fmiss_source_df.loc[:, prepared.unit_feature_cols],
        y_fmiss_source,
        backend="lightgbm",
        random_state=config.random_state + 200,
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
        sample_weight=fmiss_weight_result.weights,
    )
    fmiss_source_pred_train = predict_tree_regressor(fmiss_source_model, fmiss_train_df.loc[:, prepared.unit_feature_cols])
    fmiss_source_pred_test = predict_tree_regressor(fmiss_source_model, fmiss_test_df.loc[:, prepared.unit_feature_cols])
    fmiss_variants = {
        "contextual_paired_only_lightgbm": {
            "backend": "lightgbm",
            "train_frame": fmiss_train_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
            "test_frame": fmiss_test_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
            "subgroup_labels": fmiss_train_df[config.subgroup_column].astype(str).reset_index(drop=True),
            "family": "paired_context",
        },
        "source_reweighted_fullctx_lightgbm": {
            "backend": "lightgbm",
            "train_frame": build_stack_frame(
                fmiss_train_df,
                context_cols=prepared.lightgbm_context_cols,
                extra_features={"source_pred": fmiss_source_pred_train},
            ),
            "test_frame": build_stack_frame(
                fmiss_test_df,
                context_cols=prepared.lightgbm_context_cols,
                extra_features={"source_pred": fmiss_source_pred_test},
            ),
            "subgroup_labels": fmiss_train_df[config.subgroup_column].astype(str).reset_index(drop=True),
            "family": "source_reweighted_stack",
        },
    }
    return {
        "fpos": (fpos_train_df, y_fpos_train, y_fpos_test, fpos_variants),
        "fmiss": (fmiss_train_df, y_fmiss_train, y_fmiss_test, fmiss_variants),
    }


def _build_readme(
    *,
    config: SyntheticAugmentationConfig,
    split_manifest: dict[str, Any],
    summary_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
) -> str:
    lines = [
        "# Synthetic Paired Augmentation Benchmark",
        "",
        "This run keeps the paired holdout fixed and tests whether leakage-safe synthetic paired rows help the current best context-first tabular models.",
        "",
        "## Config",
        f"- budgets: `{list(sorted(set(int(b) for b in config.budgets)))}`",
        f"- repeats: `{config.repeats}`",
        f"- augmentation_factor: `{config.augmentation_factor}`",
        f"- augmentation_weight: `{config.augmentation_weight}`",
        f"- subgroup_column: `{config.subgroup_column}`",
        "",
        "## Split",
        f"- paired_test_rows: `{split_manifest['paired_test_rows']}`",
        f"- paired_finetune_rows: `{split_manifest['paired_finetune_rows']}`",
        f"- paired_test_recordings: `{split_manifest['paired_test_recordings']}`",
        "",
        "## Recommendations",
        "",
    ]
    for _, row in recommendations_df.iterrows():
        lines.extend(
            [
                f"### {row['target']}",
                f"- recommended variant: `{row['recommended_variant_id']}`",
                f"- recommended budget rows: `{int(row['recommended_budget_rows'])}`",
                f"- recommended mean MAE: `{row['recommended_mean_mae']:.4f}`",
                f"- recommended mean R²: `{row['recommended_mean_r2']:.4f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _run_with_config(config: SyntheticAugmentationConfig) -> Path:
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

    _log(config, f"Loading parquet from {config.parquet_path}")
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
    target_payloads = _prepare_variants(
        split=split,
        prepared=prepared,
        base=base,
        config=config,
        output_dir=output_dir,
        unit_drop_cols=unit_drop_cols,
        context_drop_cols=context_drop_cols,
    )

    metric_rows: list[dict[str, Any]] = []
    synth_rows_summary: list[dict[str, Any]] = []
    for target, (train_df, y_train, y_test, feature_variants) in target_payloads.items():
        for variant_id, spec in feature_variants.items():
            backend = str(spec["backend"])
            train_frame = spec["train_frame"]
            test_frame = spec["test_frame"]
            subgroup_labels = pd.Series(spec["subgroup_labels"]).astype(str).reset_index(drop=True)
            family = str(spec["family"])
            _log(config, f"[{target}] synthetic benchmark for {variant_id}")
            for budget in sorted(set(int(b) for b in config.budgets)):
                budget_eff = min(int(budget), len(train_df))
                repeat_count = 1 if budget_eff >= len(train_df) else int(config.repeats)
                for repeat_id in range(repeat_count):
                    subset_idx = _grouped_budget_subset(
                        train_df,
                        target_rows=budget_eff,
                        random_state=config.random_state + (1000 * repeat_id) + budget_eff + (17 if target == "fmiss" else 0),
                    )
                    subset_groups = int(train_df.iloc[subset_idx]["recording_key"].astype(str).nunique())
                    subset_frame = train_frame.iloc[subset_idx].reset_index(drop=True)
                    subset_y = y_train[subset_idx]
                    subset_subgroups = subgroup_labels.iloc[subset_idx].reset_index(drop=True)

                    baseline_model = fit_tree_regressor(
                        subset_frame,
                        subset_y,
                        backend=backend,
                        random_state=config.random_state + 5000 + repeat_id + budget_eff,
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
                    baseline_pred = predict_tree_regressor(baseline_model, test_frame)
                    baseline_metrics = score_predictions(y_test, baseline_pred)
                    metric_rows.append(
                        _metric_row(
                            target=target,
                            variant_id=variant_id,
                            budget_rows=len(subset_idx),
                            repeat_id=repeat_id,
                            subset_recordings=subset_groups,
                            synthetic_rows=0,
                            synthetic_weight=0.0,
                            metrics=baseline_metrics,
                        )
                    )

                    aug_frame, aug_y, aug_weights = _augment_subset(
                        subset_frame,
                        subset_y,
                        subgroup_labels=subset_subgroups,
                        config=config,
                        random_state=config.random_state + 9000 + repeat_id + budget_eff + (31 if target == "fmiss" else 0),
                    )
                    aug_variant_id = f"{variant_id}_synthetic"
                    synth_rows_summary.append(
                        {
                            "target": target,
                            "variant_id": aug_variant_id,
                            "budget_rows": int(len(subset_idx)),
                            "repeat_id": int(repeat_id),
                            "subset_recordings": int(subset_groups),
                            "synthetic_rows": int(max(len(aug_frame) - len(subset_frame), 0)),
                            "synthetic_fraction_vs_real": float(max(len(aug_frame) - len(subset_frame), 0) / max(len(subset_frame), 1)),
                            "augmentation_weight": float(config.augmentation_weight),
                            "subgroup_column": config.subgroup_column,
                        }
                    )
                    aug_model = fit_tree_regressor(
                        aug_frame,
                        aug_y,
                        backend=backend,
                        random_state=config.random_state + 7000 + repeat_id + budget_eff,
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
                        sample_weight=aug_weights,
                    )
                    aug_pred = predict_tree_regressor(aug_model, test_frame)
                    aug_metrics = score_predictions(y_test, aug_pred)
                    metric_rows.append(
                        _metric_row(
                            target=target,
                            variant_id=aug_variant_id,
                            budget_rows=len(subset_idx),
                            repeat_id=repeat_id,
                            subset_recordings=subset_groups,
                            synthetic_rows=int(max(len(aug_frame) - len(subset_frame), 0)),
                            synthetic_weight=config.augmentation_weight,
                            metrics=aug_metrics,
                        )
                    )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["target", "variant_id", "budget_rows", "repeat_id"]).reset_index(drop=True)
    summary_df = _aggregate_budget_curve(metrics_df)
    recommendations_df = _choose_budget_recommendations(
        summary_df,
        r2_frac=config.budget_selection_r2_frac,
        mae_frac=config.budget_selection_mae_frac,
    )
    synth_summary_df = pd.DataFrame(synth_rows_summary).sort_values(
        ["target", "variant_id", "budget_rows", "repeat_id"]
    ).reset_index(drop=True)

    split_manifest = {
        **split.manifest,
        "budgets": list(sorted(set(int(b) for b in config.budgets))),
        "repeats": int(config.repeats),
        "augmentation_factor": float(config.augmentation_factor),
        "augmentation_weight": float(config.augmentation_weight),
        "augmentation_noise_scale": float(config.augmentation_noise_scale),
        "augmentation_k_neighbors": int(config.augmentation_k_neighbors),
        "subgroup_column": config.subgroup_column,
        "filtered_unit_drop_count": int(len(unit_drop_cols)),
        "filtered_context_drop_count": int(len(context_drop_cols)),
    }

    metrics_df.to_csv(output_dir / "synthetic_curve_per_repeat.csv", index=False)
    summary_df.to_csv(output_dir / "synthetic_curve_summary.csv", index=False)
    recommendations_df.to_csv(output_dir / "synthetic_recommendation.csv", index=False)
    synth_summary_df.to_csv(output_dir / "synthetic_generation_summary.csv", index=False)
    filtered_manifest.to_csv(output_dir / "filtered_feature_manifest.csv", index=False)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    (output_dir / "README_run.md").write_text(
        _build_readme(
            config=config,
            split_manifest=split_manifest,
            summary_df=summary_df,
            recommendations_df=recommendations_df,
        )
    )
    _log(config, f"Saved outputs to {output_dir}")
    if not recommendations_df.empty:
        print(recommendations_df.to_string(index=False))
    return output_dir


def run_with_config(config: SyntheticAugmentationConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = SyntheticAugmentationConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.budgets = (40, 107)
        cli_config.repeats = 2
        cli_config.augmentation_factor = 0.75
    _run_with_config(cli_config)
