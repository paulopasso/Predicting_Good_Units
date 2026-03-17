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
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.domain_shift_audit.report import compute_feature_shift_table
from crash_tests.ml_statistics_workbench.config import MLStatisticsWorkbenchConfig
from crash_tests.ml_statistics_workbench.methods import (
    build_stack_frame,
    compute_importance_weights,
    filter_target_rows,
    fit_tree_regressor,
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


@dataclass
class LabelBudgetConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("crash_tests/ml_statistics_workbench/outputs")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    budgets: tuple[int, ...] = (20, 40, 60, 80, 107)
    repeats: int = 5
    budget_selection_r2_frac: float = 0.98
    budget_selection_mae_frac: float = 1.03
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"label_budget_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paired label-budget benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: LabelBudgetConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _grouped_budget_subset(
    df: pd.DataFrame,
    *,
    target_rows: int,
    random_state: int,
    n_candidates: int = 128,
) -> np.ndarray:
    if target_rows >= len(df):
        return np.arange(len(df), dtype=int)
    target_rows = max(1, min(int(target_rows), len(df)))
    test_size = max(0.05, min(0.95, target_rows / max(len(df), 1)))
    splitter = GroupShuffleSplit(
        n_splits=n_candidates,
        test_size=test_size,
        random_state=random_state,
    )
    best_idx: np.ndarray | None = None
    best_gap: int | None = None
    groups = df["recording_key"].astype(str)
    dummy = np.zeros(len(df), dtype=int)
    for _, subset_idx in splitter.split(dummy, groups=groups):
        subset_idx = np.asarray(subset_idx, dtype=int)
        gap = abs(len(subset_idx) - target_rows)
        if best_gap is None or gap < best_gap:
            best_idx = subset_idx
            best_gap = gap
            if gap == 0:
                break
    assert best_idx is not None
    return np.sort(best_idx)


def _metric_row(
    *,
    target: str,
    variant_id: str,
    budget_rows: int,
    repeat_id: int,
    subset_recordings: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "target": target,
        "variant_id": variant_id,
        "budget_rows": int(budget_rows),
        "repeat_id": int(repeat_id),
        "subset_recordings": int(subset_recordings),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "bias": float(metrics["bias"]),
        "calibration_slope": float(metrics["calibration_slope"]),
        "calibration_intercept": float(metrics["calibration_intercept"]),
    }


def _aggregate_budget_curve(metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics_df.groupby(["target", "variant_id", "budget_rows"], dropna=False)
    rows: list[dict[str, Any]] = []
    for (target, variant_id, budget_rows), group in grouped:
        rows.append(
            {
                "target": target,
                "variant_id": variant_id,
                "budget_rows": int(budget_rows),
                "repeats": int(len(group)),
                "mean_mae": float(group["mae"].mean()),
                "std_mae": float(group["mae"].std(ddof=0)),
                "mean_r2": float(group["r2"].mean()),
                "std_r2": float(group["r2"].std(ddof=0)),
                "mean_bias": float(group["bias"].mean()),
                "mean_subset_recordings": float(group["subset_recordings"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "variant_id", "budget_rows"]).reset_index(drop=True)


def _choose_budget_recommendations(
    curve_df: pd.DataFrame,
    *,
    r2_frac: float,
    mae_frac: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, target_df in curve_df.groupby("target", dropna=False):
        target_df = target_df.copy()
        best_r2 = float(target_df["mean_r2"].max())
        best_mae = float(target_df["mean_mae"].min())
        eligible = target_df[
            (target_df["mean_r2"] >= best_r2 * float(r2_frac))
            & (target_df["mean_mae"] <= best_mae * float(mae_frac))
        ].copy()
        if eligible.empty:
            chosen = target_df.sort_values(["mean_r2", "mean_mae", "budget_rows"], ascending=[False, True, True]).iloc[0]
        else:
            chosen = eligible.sort_values(["budget_rows", "mean_mae", "mean_r2"], ascending=[True, True, False]).iloc[0]
        rows.append(
            {
                "target": target,
                "recommended_variant_id": str(chosen["variant_id"]),
                "recommended_budget_rows": int(chosen["budget_rows"]),
                "recommended_mean_mae": float(chosen["mean_mae"]),
                "recommended_mean_r2": float(chosen["mean_r2"]),
                "max_mean_r2_observed": best_r2,
                "min_mean_mae_observed": best_mae,
                "budget_selection_r2_frac": float(r2_frac),
                "budget_selection_mae_frac": float(mae_frac),
            }
        )
    return pd.DataFrame(rows)


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


def _run_with_config(config: LabelBudgetConfig) -> Path:
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

    unlabeled_target_df = split.ssl_pool if not split.ssl_pool.empty else split.paired_non_test_all
    metric_rows: list[dict[str, Any]] = []

    # fpos models
    _log(config, "[fpos] preparing current best and baseline models")
    fpos_train_df, y_fpos_train = filter_target_rows(split.paired_finetune, "fpos")
    fpos_test_df, y_fpos_test = filter_target_rows(split.paired_test, "fpos")
    fpos_source_df, y_fpos_source = filter_target_rows(split.hybrid_source, "fpos")

    filtered_drop_cols = set(unit_drop_cols) | set(context_drop_cols)
    filtered_context_cols = [col for col in prepared.lightgbm_context_cols if col not in filtered_drop_cols]
    filtered_unit_cols = [col for col in prepared.unit_feature_cols if col not in set(unit_drop_cols)]
    filtered_fpos_train_df = mask_feature_columns(fpos_train_df, filtered_drop_cols)
    filtered_fpos_test_df = mask_feature_columns(fpos_test_df, filtered_drop_cols)
    filtered_hybrid_source = mask_feature_columns(split.hybrid_source, unit_drop_cols)
    filtered_paired_non_test_all = mask_feature_columns(split.paired_non_test_all, unit_drop_cols)
    filtered_ssl_pool = mask_feature_columns(unlabeled_target_df, unit_drop_cols)

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
        checkpoint_path=(output_dir / "checkpoints" / "budget_embed_nommd_filtered_fpos_source.pt"),
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
    fpos_feature_variants = {
        "contextual_paired_only_lightgbm": {
            "backend": "lightgbm",
            "train_frame": fpos_train_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
            "test_frame": fpos_test_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
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
        },
    }

    # fmiss models
    _log(config, "[fmiss] preparing current best and baseline models")
    fmiss_train_df, y_fmiss_train = filter_target_rows(split.paired_finetune, "fmiss")
    fmiss_test_df, y_fmiss_test = filter_target_rows(split.paired_test, "fmiss")
    fmiss_source_df, y_fmiss_source = filter_target_rows(split.hybrid_source, base.source_fmiss_target)
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
    fmiss_feature_variants = {
        "contextual_paired_only_lightgbm": {
            "backend": "lightgbm",
            "train_frame": fmiss_train_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
            "test_frame": fmiss_test_df.loc[:, prepared.lightgbm_context_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True),
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
        },
    }

    target_payloads = {
        "fpos": (fpos_train_df, y_fpos_train, y_fpos_test, fpos_feature_variants),
        "fmiss": (fmiss_train_df, y_fmiss_train, y_fmiss_test, fmiss_feature_variants),
    }

    for target, (train_df, y_train, y_test, feature_variants) in target_payloads.items():
        for variant_id, spec in feature_variants.items():
            _log(config, f"[{target}] budget curve for {variant_id}")
            train_frame = spec["train_frame"]
            test_frame = spec["test_frame"]
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
                    model = fit_tree_regressor(
                        train_frame.iloc[subset_idx],
                        y_train[subset_idx],
                        backend=str(spec["backend"]),
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
                    pred = predict_tree_regressor(model, test_frame)
                    metrics = score_predictions(y_test, pred)
                    metric_rows.append(
                        _metric_row(
                            target=target,
                            variant_id=variant_id,
                            budget_rows=len(subset_idx),
                            repeat_id=repeat_id,
                            subset_recordings=subset_groups,
                            metrics=metrics,
                        )
                    )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["target", "variant_id", "budget_rows", "repeat_id"]).reset_index(drop=True)
    curve_df = _aggregate_budget_curve(metrics_df)
    recommended_df = _choose_budget_recommendations(
        curve_df,
        r2_frac=config.budget_selection_r2_frac,
        mae_frac=config.budget_selection_mae_frac,
    )

    split_manifest = {
        **split.manifest,
        "budgets": list(sorted(set(int(b) for b in config.budgets))),
        "repeats": int(config.repeats),
        "fpos_variants": list(fpos_feature_variants.keys()),
        "fmiss_variants": list(fmiss_feature_variants.keys()),
        "filtered_unit_drop_count": int(len(unit_drop_cols)),
        "filtered_context_drop_count": int(len(context_drop_cols)),
    }

    metrics_df.to_csv(output_dir / "budget_curve_per_repeat.csv", index=False)
    curve_df.to_csv(output_dir / "budget_curve_summary.csv", index=False)
    recommended_df.to_csv(output_dir / "budget_recommendation.csv", index=False)
    filtered_manifest.to_csv(output_dir / "filtered_feature_manifest.csv", index=False)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    readme_lines = [
        "# Paired Label-Budget Benchmark",
        "",
        "This run keeps the paired holdout fixed and varies only the non-test paired adaptation label budget.",
        "",
        "## Budgets",
        f"- budgets: `{list(sorted(set(int(b) for b in config.budgets)))}`",
        f"- repeats per non-full budget: `{config.repeats}`",
        "",
        "## Recommendations",
        "",
    ]
    for _, row in recommended_df.iterrows():
        readme_lines.extend(
            [
                f"### {row['target']}",
                f"- recommended variant: `{row['recommended_variant_id']}`",
                f"- recommended budget rows: `{int(row['recommended_budget_rows'])}`",
                f"- recommended mean MAE: `{row['recommended_mean_mae']:.4f}`",
                f"- recommended mean R²: `{row['recommended_mean_r2']:.4f}`",
                "",
            ]
        )
    (output_dir / "README_run.md").write_text("\n".join(readme_lines) + "\n")
    _log(config, f"Saved outputs to {output_dir}")
    if not recommended_df.empty:
        print(recommended_df.to_string(index=False))
    return output_dir


def run_with_config(config: LabelBudgetConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = LabelBudgetConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.budgets = (20, 60, 107)
        cli_config.repeats = 2
    _run_with_config(cli_config)
