from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lightgbm import LGBMRegressor

from crash_tests.mmd_domain_adaptation.config import MMDDomainAdaptationConfig
from crash_tests.mmd_domain_adaptation.dataset import (
    PreparedFeatureTable,
    PreparedModelInput,
    fit_training_side_scalers,
    prepare_feature_table,
    prepare_model_input,
    target_name_for_source,
)
from crash_tests.mmd_domain_adaptation.train import (
    evaluate_state,
    finetune_labeled_target,
    score_predictions,
    train_source_with_mmd,
)
from crash_tests.self_supervised_domain_adaptation.split_utils import (
    MainExperimentSplit,
    make_grouped_validation_splits,
    make_main_split,
)

_METRIC_COLUMNS = [
    "protocol",
    "target",
    "variant_id",
    "family",
    "model_id",
    "source_target",
    "eval_stage",
    "lambda",
    "selected_lambda",
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
    "eval_stage",
    "lambda",
    "row_uid",
    "recording_key",
    "study_set",
    "true",
    "pred",
    "abs_error",
]
_LAMBDA_SWEEP_COLUMNS = [
    "target",
    "eval_stage",
    "lambda",
    "cv_mae_mean",
    "cv_rmse_mean",
    "cv_r2_mean",
    "cv_bias_mean",
    "cv_calibration_slope_mean",
    "cv_calibration_intercept_mean",
    "n_splits",
]
_HISTORY_COLUMNS = [
    "protocol",
    "target",
    "variant_id",
    "eval_stage",
    "lambda",
    "stage",
    "split_id",
    "epoch",
    "train_task_loss",
    "train_mmd_loss",
    "train_total_loss",
    "train_loss",
    "val_mae",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MMD domain adaptation crash test")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet", help="Path to the main QC parquet")
    parser.add_argument("--output-root", default="crash_tests/mmd_domain_adaptation/outputs", help="Root folder for run outputs")
    parser.add_argument("--run-name", default=None, help="Optional explicit run directory name")
    parser.add_argument("--smoke", action="store_true", help="Run a 1-epoch smoke configuration")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip paired fine-tuning variants")
    parser.add_argument("--quiet", action="store_true", help="Reduce runner logging")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> MMDDomainAdaptationConfig:
    config = MMDDomainAdaptationConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        run_finetune_stage=not args.skip_finetune,
        verbose=not args.quiet,
    )
    if args.smoke:
        config.source_train_epochs = 1
        config.finetune_epochs = 1
        config.paired_finetune_val_splits = 2
        config.patience = 1
    return config


def _log(config: MMDDomainAdaptationConfig, message: str) -> None:
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


def _make_lightgbm(config: MMDDomainAdaptationConfig) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    n_estimators=config.lightgbm_estimators,
                    learning_rate=config.lightgbm_learning_rate,
                    num_leaves=config.lightgbm_num_leaves,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=config.random_state,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )


def _filter_target_rows(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    return df.loc[mask].copy(), y.loc[mask].to_numpy(dtype=float)


def _prepare_source_train_val_frames(
    hybrid_df: pd.DataFrame,
    *,
    source_target: str,
    random_state: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_df, _ = _filter_target_rows(hybrid_df, source_target)
    splits = make_grouped_validation_splits(
        target_df,
        n_splits=1,
        test_size=val_fraction,
        random_state=random_state,
    )
    if not splits:
        raise RuntimeError(f"Unable to make grouped source validation split for target={source_target}")
    return splits[0]


def _prepare_unit_input(
    df: pd.DataFrame,
    prepared: PreparedFeatureTable,
    unit_scaler: Any,
) -> PreparedModelInput:
    return prepare_model_input(
        df,
        prepared,
        unit_scaler,
        None,
        use_context=False,
    )


def _run_lightgbm_baseline(
    *,
    model_id: str,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    config: MMDDomainAdaptationConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_t, y_train = _filter_target_rows(train_df, target)
    test_t, y_test = _filter_target_rows(test_df, target)
    model = _make_lightgbm(config)
    model.fit(train_t[feature_cols], y_train)
    pred = np.clip(model.predict(test_t[feature_cols]), 0.0, 1.0)
    metrics = {
        "protocol": "main",
        "target": target,
        "variant_id": model_id,
        "family": "baseline",
        "model_id": model_id,
        "source_target": np.nan,
        "eval_stage": "holdout",
        "lambda": np.nan,
        "selected_lambda": np.nan,
        **score_predictions(y_test, pred),
    }
    predictions = pd.DataFrame(
        {
            "protocol": "main",
            "target": target,
            "variant_id": model_id,
            "family": "baseline",
            "eval_stage": "holdout",
            "lambda": np.nan,
            "row_uid": test_t["row_uid"].astype(str).to_numpy(),
            "recording_key": test_t["recording_key"].astype(str).to_numpy(),
            "study_set": test_t["study_set"].astype(str).to_numpy(),
            "true": y_test,
            "pred": pred,
            "abs_error": np.abs(pred - y_test),
        }
    )
    return metrics, predictions


def _summarize_cv_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    return {
        "cv_mae_mean": float(frame["mae"].mean()),
        "cv_rmse_mean": float(frame["rmse"].mean()),
        "cv_r2_mean": float(frame["r2"].mean()),
        "cv_bias_mean": float(frame["bias"].mean()),
        "cv_calibration_slope_mean": float(frame["calibration_slope"].mean()),
        "cv_calibration_intercept_mean": float(frame["calibration_intercept"].mean()),
        "n_splits": int(len(frame)),
    }


def _select_best_positive_lambda(lambda_rows: list[dict[str, Any]]) -> float:
    frame = pd.DataFrame(lambda_rows)
    positive = frame[frame["lambda"] > 0.0].copy()
    if positive.empty:
        return 0.0
    positive = positive.sort_values(["cv_mae_mean", "lambda"]).reset_index(drop=True)
    return float(positive.iloc[0]["lambda"])


def _append_prediction_frame(
    pred_rows: list[pd.DataFrame],
    *,
    variant_id: str,
    family: str,
    target: str,
    eval_stage: str,
    selected_lambda: float | None,
    prepared_df: pd.DataFrame,
    y_true: np.ndarray,
    pred: np.ndarray,
) -> None:
    pred_rows.append(
        pd.DataFrame(
            {
                "protocol": "main",
                "target": target,
                "variant_id": variant_id,
                "family": family,
                "eval_stage": eval_stage,
                "lambda": selected_lambda,
                "row_uid": prepared_df["row_uid"].astype(str).to_numpy(),
                "recording_key": prepared_df["recording_key"].astype(str).to_numpy(),
                "study_set": prepared_df["study_set"].astype(str).to_numpy(),
                "true": y_true,
                "pred": pred,
                "abs_error": np.abs(pred - y_true),
            }
        )
    )


def _run_target(
    *,
    target: str,
    split: MainExperimentSplit,
    prepared: PreparedFeatureTable,
    config: MMDDomainAdaptationConfig,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    source_target = target_name_for_source(target, config.source_fmiss_target)
    target_train_df, target_train_y = _filter_target_rows(split.paired_finetune, target)
    target_test_df, target_test_y = _filter_target_rows(split.paired_test, target)
    if target_train_df.empty or target_test_df.empty:
        raise RuntimeError(f"Target {target} does not have enough paired labeled rows")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []

    source_train_df, source_val_df = _prepare_source_train_val_frames(
        split.hybrid_source,
        source_target=source_target,
        random_state=config.random_state,
        val_fraction=config.source_val_fraction,
    )

    paired_cv_splits = make_grouped_validation_splits(
        target_train_df,
        n_splits=config.paired_finetune_val_splits,
        test_size=config.paired_finetune_val_fraction,
        random_state=config.random_state + 200,
    )
    if not paired_cv_splits:
        raise RuntimeError(f"Unable to build paired validation splits for target={target}")

    zero_shot_lambda_rows: list[dict[str, Any]] = []
    finetune_lambda_rows: list[dict[str, Any]] = []
    for mmd_lambda in config.lambda_grid:
        zero_metrics: list[dict[str, Any]] = []
        finetune_metrics: list[dict[str, Any]] = []
        for split_id, (fold_train_df, fold_val_df) in enumerate(paired_cv_splits, start=1):
            held_out_recordings = set(fold_val_df["recording_key"].astype(str))
            paired_non_val_all = split.paired_non_test_all[
                ~split.paired_non_test_all["recording_key"].astype(str).isin(held_out_recordings)
            ].copy()
            unlabeled_df = paired_non_val_all[paired_non_val_all["matched_gt_unit_id"].isna()].copy()
            if unlabeled_df.empty:
                unlabeled_df = split.ssl_pool.copy()

            scaler_fit_df = pd.concat([split.hybrid_source, paired_non_val_all], ignore_index=True)
            unit_scaler, _ = fit_training_side_scalers(scaler_fit_df, prepared)

            source_train_input = _prepare_unit_input(source_train_df, prepared, unit_scaler)
            source_val_input = _prepare_unit_input(source_val_df, prepared, unit_scaler)
            unlabeled_input = _prepare_unit_input(unlabeled_df, prepared, unit_scaler)
            fold_val_input = _prepare_unit_input(fold_val_df, prepared, unit_scaler)
            fold_train_input = _prepare_unit_input(fold_train_df, prepared, unit_scaler)
            source_train_target = pd.to_numeric(source_train_df[source_target], errors="coerce").to_numpy(dtype=float)
            source_val_target = pd.to_numeric(source_val_df[source_target], errors="coerce").to_numpy(dtype=float)
            fold_train_target = pd.to_numeric(fold_train_df[target], errors="coerce").to_numpy(dtype=float)
            fold_val_target = pd.to_numeric(fold_val_df[target], errors="coerce").to_numpy(dtype=float)

            source_result = train_source_with_mmd(
                source_train_data=source_train_input,
                source_train_target=source_train_target,
                source_val_data=source_val_input,
                source_val_target=source_val_target,
                target_unlabeled_data=unlabeled_input,
                mmd_lambda=float(mmd_lambda),
                config=config,
                target_name=target,
                checkpoint_path=None,
                progress_prefix=f"{target}:lambda={mmd_lambda:.2f}:cv{split_id}:source",
            )
            history_rows.extend(
                {
                    "protocol": "main",
                    "target": target,
                    "variant_id": "lambda_sweep",
                    "eval_stage": "selection_zero_shot",
                    "lambda": float(mmd_lambda),
                    "stage": "source",
                    "split_id": split_id,
                    **row,
                }
                for row in source_result.history
            )
            zero_metrics.append(evaluate_state(
                model_state=source_result.model_state,
                eval_data=fold_val_input,
                eval_target=fold_val_target,
                config=config,
            )[0])

            if config.run_finetune_stage:
                finetune_result = finetune_labeled_target(
                    source_state=source_result.model_state,
                    train_data=fold_train_input,
                    train_target=fold_train_target,
                    config=config,
                    target_name=target,
                    random_state=config.random_state + split_id,
                    checkpoint_path=None,
                    progress_prefix=f"{target}:lambda={mmd_lambda:.2f}:cv{split_id}:finetune",
                )
                history_rows.extend(
                    {
                        "protocol": "main",
                        "target": target,
                        "variant_id": "lambda_sweep",
                        "eval_stage": "selection_finetuned",
                        "lambda": float(mmd_lambda),
                        "stage": "finetune",
                        "split_id": split_id,
                        **row,
                    }
                    for row in finetune_result.history
                )
                finetune_metrics.append(evaluate_state(
                    model_state=finetune_result.model_state,
                    eval_data=fold_val_input,
                    eval_target=fold_val_target,
                    config=config,
                )[0])

        zero_row = {
            "target": target,
            "eval_stage": "zero_shot",
            "lambda": float(mmd_lambda),
            **_summarize_cv_metrics(zero_metrics),
        }
        lambda_rows.append(zero_row)
        zero_shot_lambda_rows.append(zero_row)
        if config.run_finetune_stage:
            finetune_row = {
                "target": target,
                "eval_stage": "finetuned",
                "lambda": float(mmd_lambda),
                **_summarize_cv_metrics(finetune_metrics),
            }
            lambda_rows.append(finetune_row)
            finetune_lambda_rows.append(finetune_row)

    selected_zero_lambda = _select_best_positive_lambda(zero_shot_lambda_rows)
    selected_finetune_lambda = _select_best_positive_lambda(finetune_lambda_rows) if config.run_finetune_stage else 0.0

    train_side_df = pd.concat([split.hybrid_source, split.paired_non_test_all], ignore_index=True)
    unit_scaler, _ = fit_training_side_scalers(train_side_df, prepared)
    source_train_input = _prepare_unit_input(source_train_df, prepared, unit_scaler)
    source_val_input = _prepare_unit_input(source_val_df, prepared, unit_scaler)
    unlabeled_input = _prepare_unit_input(split.ssl_pool, prepared, unit_scaler)
    paired_train_input = _prepare_unit_input(target_train_df, prepared, unit_scaler)
    paired_test_input = _prepare_unit_input(target_test_df, prepared, unit_scaler)
    source_train_target = pd.to_numeric(source_train_df[source_target], errors="coerce").to_numpy(dtype=float)
    source_val_target = pd.to_numeric(source_val_df[source_target], errors="coerce").to_numpy(dtype=float)

    final_variants = [
        ("neural_no_mmd_zero_shot", 0.0, "zero_shot"),
        ("neural_mmd_zero_shot", selected_zero_lambda, "zero_shot"),
    ]
    if config.run_finetune_stage:
        final_variants.extend(
            [
                ("neural_no_mmd_finetuned", 0.0, "finetuned"),
                ("neural_mmd_finetuned", selected_finetune_lambda, "finetuned"),
            ]
        )

    for variant_id, mmd_lambda, eval_stage in final_variants:
        source_result = train_source_with_mmd(
            source_train_data=source_train_input,
            source_train_target=source_train_target,
            source_val_data=source_val_input,
            source_val_target=source_val_target,
            target_unlabeled_data=unlabeled_input,
            mmd_lambda=float(mmd_lambda),
            config=config,
            target_name=target,
            checkpoint_path=(output_dir / "checkpoints" / f"{variant_id}_{target}_source.pt") if config.save_checkpoints else None,
            progress_prefix=f"{variant_id}:{target}:source",
        )
        history_rows.extend(
            {
                "protocol": "main",
                "target": target,
                "variant_id": variant_id,
                "eval_stage": eval_stage,
                "lambda": float(mmd_lambda),
                "stage": "source",
                "split_id": pd.NA,
                **row,
            }
            for row in source_result.history
        )
        final_state = source_result.model_state

        if eval_stage == "finetuned":
            finetune_result = finetune_labeled_target(
                source_state=source_result.model_state,
                train_data=paired_train_input,
                train_target=target_train_y,
                config=config,
                target_name=target,
                random_state=config.random_state + 500,
                checkpoint_path=(output_dir / "checkpoints" / f"{variant_id}_{target}_final.pt") if config.save_checkpoints else None,
                progress_prefix=f"{variant_id}:{target}:finetune",
            )
            history_rows.extend(
                {
                    "protocol": "main",
                    "target": target,
                    "variant_id": variant_id,
                    "eval_stage": eval_stage,
                    "lambda": float(mmd_lambda),
                    "stage": "finetune",
                    "split_id": pd.NA,
                    **row,
                }
                for row in finetune_result.history
            )
            final_state = finetune_result.model_state

        metrics, pred = evaluate_state(
            model_state=final_state,
            eval_data=paired_test_input,
            eval_target=target_test_y,
            config=config,
        )
        metric_rows.append(
            {
                "protocol": "main",
                "target": target,
                "variant_id": variant_id,
                "family": "neural_mmd" if "mmd" in variant_id and "no_mmd" not in variant_id else "neural_no_mmd",
                "model_id": variant_id,
                "source_target": source_target,
                "eval_stage": "holdout",
                "lambda": float(mmd_lambda),
                "selected_lambda": float(mmd_lambda),
                **metrics,
            }
        )
        _append_prediction_frame(
            prediction_frames,
            variant_id=variant_id,
            family="neural",
            target=target,
            eval_stage="holdout",
            selected_lambda=float(mmd_lambda),
            prepared_df=target_test_df,
            y_true=target_test_y,
            pred=pred,
        )

    paired_shape_metrics, paired_shape_pred = _run_lightgbm_baseline(
        model_id="paired_only_lightgbm_shape",
        feature_cols=prepared.shape_feature_cols,
        train_df=target_train_df,
        test_df=target_test_df,
        target=target,
        config=config,
    )
    metric_rows.append(paired_shape_metrics)
    prediction_frames.append(paired_shape_pred)

    contextual_metrics, contextual_pred = _run_lightgbm_baseline(
        model_id="contextual_lightgbm_no_study_id",
        feature_cols=prepared.lightgbm_context_cols,
        train_df=target_train_df,
        test_df=target_test_df,
        target=target,
        config=config,
    )
    metric_rows.append(contextual_metrics)
    prediction_frames.append(contextual_pred)

    return metric_rows, prediction_frames, history_rows, lambda_rows


def _build_readme(
    *,
    config: MMDDomainAdaptationConfig,
    split_manifest: dict[str, Any],
    metrics_df: pd.DataFrame,
    lambda_df: pd.DataFrame,
) -> str:
    lines = [
        "# MMD Domain Adaptation Run",
        "",
        "## Config",
        f"- parquet_path: `{config.parquet_path}`",
        f"- targets: `{', '.join(config.targets)}`",
        f"- lambda_grid: `{', '.join(str(x) for x in config.lambda_grid)}`",
        f"- run_finetune_stage: `{config.run_finetune_stage}`",
        "",
        "## Main Split",
        f"- paired_test_rows: `{split_manifest['paired_test_rows']}`",
        f"- paired_finetune_rows: `{split_manifest['paired_finetune_rows']}`",
        f"- ssl_pool_rows: `{split_manifest['ssl_pool_rows']}`",
        f"- paired_test_recordings: `{split_manifest['paired_test_recordings']}`",
        "",
        "## Holdout Results",
    ]
    if metrics_df.empty:
        lines.append("- No metrics were produced.")
    else:
        for _, row in metrics_df.sort_values(["target", "mae", "variant_id"]).iterrows():
            lines.append(
                f"- {row['target']} | {row['variant_id']} | MAE `{row['mae']:.4f}` | "
                f"R2 `{row['r2']:.4f}` | lambda `{row['selected_lambda']}`"
            )
    lines.extend(["", "## Lambda Sweep"])
    if lambda_df.empty:
        lines.append("- No lambda sweep rows were produced.")
    else:
        for _, row in lambda_df.sort_values(["target", "eval_stage", "cv_mae_mean", "lambda"]).iterrows():
            lines.append(
                f"- {row['target']} | {row['eval_stage']} | lambda `{row['lambda']}` | "
                f"CV MAE `{row['cv_mae_mean']:.4f}`"
            )
    return "\n".join(lines) + "\n"


def run_with_config(config: MMDDomainAdaptationConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(config, f"Loading parquet from {config.parquet_path}")
    prepared = prepare_feature_table(config.parquet_path)
    split = make_main_split(
        prepared.df,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        random_state=config.random_state,
    )

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    for target in config.targets:
        _log(config, f"Running target={target}")
        target_metrics, target_preds, target_history, target_lambdas = _run_target(
            target=target,
            split=split,
            prepared=prepared,
            config=config,
            output_dir=output_dir,
        )
        metrics_rows.extend(target_metrics)
        prediction_frames.extend(target_preds)
        history_rows.extend(target_history)
        lambda_rows.extend(target_lambdas)

    split_manifest = {
        **split.manifest,
        "training_side_scaler_rows": int(len(pd.concat([split.hybrid_source, split.paired_non_test_all], ignore_index=True))),
        "unit_feature_count": int(len(prepared.unit_feature_cols)),
        "shape_feature_count": int(len(prepared.shape_feature_cols)),
        "context_baseline_feature_count": int(len(prepared.lightgbm_context_cols)),
    }
    metrics_df = _frame_with_columns(metrics_rows, _METRIC_COLUMNS).sort_values(["target", "mae", "variant_id"]).reset_index(drop=True)
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True).sort_values(["target", "variant_id", "row_uid"]).reset_index(drop=True)
        if prediction_frames
        else pd.DataFrame(columns=_PREDICTION_COLUMNS)
    )
    lambda_df = _frame_with_columns(lambda_rows, _LAMBDA_SWEEP_COLUMNS).sort_values(
        ["target", "eval_stage", "cv_mae_mean", "lambda"]
    ).reset_index(drop=True)
    history_df = _frame_with_columns(history_rows, _HISTORY_COLUMNS)
    baseline_df = metrics_df.copy()

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"config": config.to_dict(), "main": split_manifest}, f, indent=2, sort_keys=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    baseline_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    lambda_df.to_csv(output_dir / "lambda_sweep.csv", index=False)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "README_run.md").write_text(
        _build_readme(
            config=config,
            split_manifest=split_manifest,
            metrics_df=metrics_df,
            lambda_df=lambda_df,
        ),
        encoding="utf-8",
    )

    _log(config, f"Saved outputs to {output_dir}")
    if not metrics_df.empty:
        _log(config, metrics_df.to_string(index=False))
    return output_dir


def main() -> int:
    args = _parse_args()
    config = _build_config(args)
    run_with_config(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
