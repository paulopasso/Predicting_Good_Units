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

from crash_tests.self_supervised_domain_adaptation.config import SSLDomainAdaptationConfig
from crash_tests.self_supervised_domain_adaptation.dataset import (
    PreparedFeatureTable,
    PreparedModelInput,
    fit_training_side_scalers,
    numeric_target,
    prepare_feature_table,
    prepare_model_input,
    target_name_for_source,
)
from crash_tests.self_supervised_domain_adaptation.split_utils import (
    MainExperimentSplit,
    StressStudySplit,
    build_stress_splits,
    make_grouped_validation_splits,
    make_main_split,
)
from crash_tests.self_supervised_domain_adaptation.train import (
    SSLPretrainResult,
    SupervisedTrainResult,
    pretrain_ssl_autoencoder,
    run_supervised_experiment,
    score_predictions,
)

_METRIC_COLUMNS = [
    "protocol",
    "family",
    "model_id",
    "source_target",
    "target",
    "arm",
    "selected_finetune_epochs",
    "mae",
    "rmse",
    "r2",
    "bias",
    "calibration_slope",
    "calibration_intercept",
]
_PREDICTION_COLUMNS = [
    "protocol",
    "family",
    "row_uid",
    "recording_key",
    "study_set",
    "model_id",
    "target",
    "true",
    "pred",
    "abs_error",
]
_STRESS_COLUMNS = ["held_out_study"] + _METRIC_COLUMNS
_HISTORY_COLUMNS = ["protocol", "model_id", "stage", "split_id", "epoch", "train_loss", "val_loss", "val_mae"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SSL + context paired QC crash-test experiment")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet", help="Path to the main QC parquet")
    parser.add_argument("--output-root", default="crash_tests/self_supervised_domain_adaptation/outputs", help="Root folder for run outputs")
    parser.add_argument("--run-name", default=None, help="Optional explicit run directory name")
    parser.add_argument("--smoke", action="store_true", help="Run a 1-epoch smoke configuration")
    parser.add_argument("--skip-stress", action="store_true", help="Skip the leave-one-study-out stress-test runs")
    parser.add_argument("--max-stress-studies", type=int, default=None, help="Cap the number of leave-one-study-out studies")
    parser.add_argument("--quiet", action="store_true", help="Reduce runner logging")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> SSLDomainAdaptationConfig:
    config = SSLDomainAdaptationConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        run_stress_test=not args.skip_stress,
        max_stress_studies=args.max_stress_studies,
        verbose=not args.quiet,
    )
    if args.smoke:
        config.ssl_pretrain_epochs = 1
        config.source_warmup_epochs = 1
        config.source_train_epochs = 1
        config.finetune_epochs = 1
        config.paired_finetune_val_splits = 2
        config.max_stress_studies = 1 if config.max_stress_studies is None else config.max_stress_studies
    return config


def _log(config: SSLDomainAdaptationConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _make_lightgbm(config: SSLDomainAdaptationConfig) -> Pipeline:
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


def _prepare_source_train_val(
    df: pd.DataFrame,
    *,
    target: str,
    use_context: bool,
    prepared: PreparedFeatureTable,
    unit_scaler: Any,
    context_scaler: Any,
    config: SSLDomainAdaptationConfig,
) -> tuple[PreparedModelInput, np.ndarray, PreparedModelInput, np.ndarray]:
    target_df, target_values = _filter_target_rows(df, target)
    splits = make_grouped_validation_splits(
        target_df,
        n_splits=1,
        test_size=config.source_val_fraction,
        random_state=config.random_state,
    )
    if not splits:
        raise RuntimeError(f"Unable to make grouped source validation split for target={target}")
    train_df, val_df = splits[0]
    train_y = pd.to_numeric(train_df[target], errors="coerce").to_numpy(dtype=float)
    val_y = pd.to_numeric(val_df[target], errors="coerce").to_numpy(dtype=float)
    return (
        prepare_model_input(train_df, prepared, unit_scaler, context_scaler, use_context=use_context),
        train_y,
        prepare_model_input(val_df, prepared, unit_scaler, context_scaler, use_context=use_context),
        val_y,
    )


def _prepare_paired_cv_splits(
    df: pd.DataFrame,
    *,
    target: str,
    use_context: bool,
    prepared: PreparedFeatureTable,
    unit_scaler: Any,
    context_scaler: Any,
    config: SSLDomainAdaptationConfig,
) -> list[tuple[PreparedModelInput, np.ndarray, PreparedModelInput, np.ndarray]]:
    target_df, _ = _filter_target_rows(df, target)
    folds = make_grouped_validation_splits(
        target_df,
        n_splits=config.paired_finetune_val_splits,
        test_size=config.paired_finetune_val_fraction,
        random_state=config.random_state + 200,
    )
    prepared_folds: list[tuple[PreparedModelInput, np.ndarray, PreparedModelInput, np.ndarray]] = []
    for train_df, val_df in folds:
        train_y = pd.to_numeric(train_df[target], errors="coerce").to_numpy(dtype=float)
        val_y = pd.to_numeric(val_df[target], errors="coerce").to_numpy(dtype=float)
        prepared_folds.append(
            (
                prepare_model_input(train_df, prepared, unit_scaler, context_scaler, use_context=use_context),
                train_y,
                prepare_model_input(val_df, prepared, unit_scaler, context_scaler, use_context=use_context),
                val_y,
            )
        )
    return prepared_folds


def _run_lightgbm_baseline(
    *,
    model_id: str,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    protocol_name: str,
    config: SSLDomainAdaptationConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_t, y_train = _filter_target_rows(train_df, target)
    test_t, y_test = _filter_target_rows(test_df, target)
    model = _make_lightgbm(config)
    model.fit(train_t[feature_cols], y_train)
    pred = np.clip(model.predict(test_t[feature_cols]), 0.0, 1.0)
    metrics = {
        "protocol": protocol_name,
        "family": "baseline",
        "model_id": model_id,
        "target": target,
        **score_predictions(y_test, pred),
    }
    predictions = pd.DataFrame(
        {
            "protocol": protocol_name,
            "row_uid": test_t["row_uid"].astype(str).to_numpy(),
            "recording_key": test_t["recording_key"].astype(str).to_numpy(),
            "study_set": test_t["study_set"].astype(str).to_numpy(),
            "model_id": model_id,
            "target": target,
            "true": y_test,
            "pred": pred,
            "abs_error": np.abs(pred - y_test),
        }
    )
    return metrics, predictions


def _run_single_protocol(
    *,
    protocol_name: str,
    split: MainExperimentSplit | StressStudySplit,
    prepared: PreparedFeatureTable,
    config: SSLDomainAdaptationConfig,
    output_dir: Path,
    include_baselines: bool,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]], dict[str, Any]]:
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []

    train_side_df = pd.concat([split.hybrid_source, split.paired_non_test_all], ignore_index=True)
    unit_scaler, _ = fit_training_side_scalers(train_side_df, prepared)
    ssl_splits = make_grouped_validation_splits(
        split.ssl_pool,
        n_splits=1,
        test_size=config.ssl_val_recording_frac,
        random_state=config.random_state + 100,
    )
    if not ssl_splits:
        raise RuntimeError(f"Unable to build SSL train/validation split for protocol={protocol_name}")
    ssl_train_df, ssl_val_df = ssl_splits[0]
    ssl_train_input = prepare_model_input(ssl_train_df, prepared, unit_scaler, None, use_context=False)
    ssl_val_input = prepare_model_input(ssl_val_df, prepared, unit_scaler, None, use_context=False)

    ssl_checkpoint = output_dir / "checkpoints" / f"{protocol_name}_ssl_encoder.pt"
    _log(config, f"{protocol_name}: SSL pretraining on {len(split.ssl_pool)} unlabeled paired rows")
    ssl_result = pretrain_ssl_autoencoder(
        train_data=ssl_train_input,
        val_data=ssl_val_input,
        config=config,
        checkpoint_path=ssl_checkpoint if config.save_checkpoints else None,
    )
    history_rows.extend({"protocol": protocol_name, "model_id": "ssl_pretrain", **row} for row in ssl_result.history)

    for target in config.targets:
        source_target = target_name_for_source(target, config.source_fmiss_target)
        target_context_cols = config.target_context_columns(target, prepared.context_feature_cols_by_target)
        _, context_scaler = fit_training_side_scalers(
            train_side_df,
            prepared,
            context_columns=target_context_cols,
        )
        _log(config, f"{protocol_name}: target={target} source_target={source_target}")
        if target_context_cols:
            preview = ", ".join(target_context_cols[:6])
            suffix = " ..." if len(target_context_cols) > 6 else ""
            _log(config, f"{protocol_name}: target={target} contextual_cols={len(target_context_cols)} ({preview}{suffix})")

        source_train_unit, y_source_train, source_val_unit, y_source_val = _prepare_source_train_val(
            split.hybrid_source,
            target=source_target,
            use_context=False,
            prepared=prepared,
            unit_scaler=unit_scaler,
            context_scaler=context_scaler,
            config=config,
        )
        paired_full_df, paired_full_target = _filter_target_rows(split.paired_finetune, target)
        paired_test_df, paired_test_target = _filter_target_rows(split.paired_test, target)
        paired_full_unit = prepare_model_input(
            paired_full_df,
            prepared,
            unit_scaler,
            context_scaler,
            use_context=False,
        )
        paired_test_unit = prepare_model_input(
            paired_test_df,
            prepared,
            unit_scaler,
            context_scaler,
            use_context=False,
        )
        paired_cv_unit = _prepare_paired_cv_splits(
            split.paired_finetune,
            target=target,
            use_context=False,
            prepared=prepared,
            unit_scaler=unit_scaler,
            context_scaler=context_scaler,
            config=config,
        )

        if config.run_unit_only_arm:
            ssl_unit = run_supervised_experiment(
                arm="unit_only",
                target_name=target,
                source_target_name=source_target,
                source_train_data=source_train_unit,
                source_train_target=y_source_train,
                source_val_data=source_val_unit,
                source_val_target=y_source_val,
                paired_train_splits=paired_cv_unit,
                paired_full_train_data=paired_full_unit,
                paired_full_train_target=paired_full_target,
                paired_test_data=paired_test_unit,
                paired_test_target=paired_test_target,
                output_checkpoint_dir=output_dir / "checkpoints",
                model_stem=f"{protocol_name}_ssl_{target}_unit_only",
                config=config,
                pretrained_encoder_state=ssl_result.encoder_state,
            )
            metrics_rows.append(
                {
                    "protocol": protocol_name,
                    "family": "ssl",
                    "model_id": f"ssl_{target}_unit_only",
                    "source_target": source_target,
                    **ssl_unit.metrics,
                }
            )
            prediction_frames.append(ssl_unit.predictions.assign(protocol=protocol_name, family="ssl"))
            history_rows.extend(
                {"protocol": protocol_name, "model_id": f"ssl_{target}_unit_only", **row}
                for row in ssl_unit.history
            )

            if include_baselines:
                no_ssl_unit = run_supervised_experiment(
                    arm="unit_only",
                    target_name=target,
                    source_target_name=source_target,
                    source_train_data=source_train_unit,
                    source_train_target=y_source_train,
                    source_val_data=source_val_unit,
                    source_val_target=y_source_val,
                    paired_train_splits=paired_cv_unit,
                    paired_full_train_data=paired_full_unit,
                    paired_full_train_target=paired_full_target,
                    paired_test_data=paired_test_unit,
                    paired_test_target=paired_test_target,
                    output_checkpoint_dir=output_dir / "checkpoints",
                    model_stem=f"{protocol_name}_neural_no_ssl_{target}_unit_only",
                    config=config,
                    pretrained_encoder_state=None,
                )
                metrics_rows.append(
                    {
                        "protocol": protocol_name,
                        "family": "baseline",
                        "model_id": f"neural_no_ssl_{target}_unit_only",
                        "source_target": source_target,
                        **no_ssl_unit.metrics,
                    }
                )
                prediction_frames.append(no_ssl_unit.predictions.assign(protocol=protocol_name, family="baseline"))
                history_rows.extend(
                    {"protocol": protocol_name, "model_id": f"neural_no_ssl_{target}_unit_only", **row}
                    for row in no_ssl_unit.history
                )

        if config.run_contextual_arm:
            source_train_ctx, _, source_val_ctx, _ = _prepare_source_train_val(
                split.hybrid_source,
                target=source_target,
                use_context=True,
                prepared=prepared,
                unit_scaler=unit_scaler,
                context_scaler=context_scaler,
                config=config,
            )
            paired_full_ctx = prepare_model_input(
                paired_full_df,
                prepared,
                unit_scaler,
                context_scaler,
                use_context=True,
            )
            paired_test_ctx = prepare_model_input(
                paired_test_df,
                prepared,
                unit_scaler,
                context_scaler,
                use_context=True,
            )
            paired_cv_ctx = _prepare_paired_cv_splits(
                split.paired_finetune,
                target=target,
                use_context=True,
                prepared=prepared,
                unit_scaler=unit_scaler,
                context_scaler=context_scaler,
                config=config,
            )
            ssl_contextual = run_supervised_experiment(
                arm="contextual",
                target_name=target,
                source_target_name=source_target,
                source_train_data=source_train_ctx,
                source_train_target=y_source_train,
                source_val_data=source_val_ctx,
                source_val_target=y_source_val,
                paired_train_splits=paired_cv_ctx,
                paired_full_train_data=paired_full_ctx,
                paired_full_train_target=paired_full_target,
                paired_test_data=paired_test_ctx,
                paired_test_target=paired_test_target,
                output_checkpoint_dir=output_dir / "checkpoints",
                model_stem=f"{protocol_name}_ssl_{target}_contextual",
                config=config,
                pretrained_encoder_state=ssl_result.encoder_state,
            )
            metrics_rows.append(
                {
                    "protocol": protocol_name,
                    "family": "ssl",
                    "model_id": f"ssl_{target}_contextual",
                    "source_target": source_target,
                    **ssl_contextual.metrics,
                }
            )
            prediction_frames.append(ssl_contextual.predictions.assign(protocol=protocol_name, family="ssl"))
            history_rows.extend(
                {"protocol": protocol_name, "model_id": f"ssl_{target}_contextual", **row}
                for row in ssl_contextual.history
            )

            if include_baselines:
                no_ssl_contextual = run_supervised_experiment(
                    arm="contextual",
                    target_name=target,
                    source_target_name=source_target,
                    source_train_data=source_train_ctx,
                    source_train_target=y_source_train,
                    source_val_data=source_val_ctx,
                    source_val_target=y_source_val,
                    paired_train_splits=paired_cv_ctx,
                    paired_full_train_data=paired_full_ctx,
                    paired_full_train_target=paired_full_target,
                    paired_test_data=paired_test_ctx,
                    paired_test_target=paired_test_target,
                    output_checkpoint_dir=output_dir / "checkpoints",
                    model_stem=f"{protocol_name}_neural_no_ssl_{target}_contextual",
                    config=config,
                    pretrained_encoder_state=None,
                )
                metrics_rows.append(
                    {
                        "protocol": protocol_name,
                        "family": "baseline",
                        "model_id": f"neural_no_ssl_{target}_contextual",
                        "source_target": source_target,
                        **no_ssl_contextual.metrics,
                    }
                )
                prediction_frames.append(no_ssl_contextual.predictions.assign(protocol=protocol_name, family="baseline"))
                history_rows.extend(
                    {"protocol": protocol_name, "model_id": f"neural_no_ssl_{target}_contextual", **row}
                    for row in no_ssl_contextual.history
                )

        if include_baselines:
            paired_shape_metrics, paired_shape_pred = _run_lightgbm_baseline(
                model_id="paired_only_lightgbm_shape",
                feature_cols=prepared.shape_feature_cols,
                train_df=split.paired_finetune,
                test_df=split.paired_test,
                target=target,
                protocol_name=protocol_name,
                config=config,
            )
            metrics_rows.append(paired_shape_metrics)
            prediction_frames.append(paired_shape_pred.assign(family="baseline"))

            contextual_lgbm_metrics, contextual_lgbm_pred = _run_lightgbm_baseline(
                model_id="contextual_lightgbm_no_study_id",
                feature_cols=prepared.lightgbm_context_cols,
                train_df=split.paired_finetune,
                test_df=split.paired_test,
                target=target,
                protocol_name=protocol_name,
                config=config,
            )
            metrics_rows.append(contextual_lgbm_metrics)
            prediction_frames.append(contextual_lgbm_pred.assign(family="baseline"))

    split_manifest = {
        **split.manifest,
        "training_side_scaler_rows": int(len(train_side_df)),
        "context_feature_count": int(len(prepared.context_feature_cols)),
        "context_feature_count_by_target": {
            target: int(len(prepared.context_feature_cols_by_target.get(target, ())))
            for target in config.targets
        },
        "unit_feature_count": int(len(prepared.unit_feature_cols)),
        "shape_feature_count": int(len(prepared.shape_feature_cols)),
    }
    return metrics_rows, prediction_frames, history_rows, split_manifest


def _build_readme(
    *,
    config: SSLDomainAdaptationConfig,
    main_manifest: dict[str, Any],
    metrics_df: pd.DataFrame,
    stress_df: pd.DataFrame,
) -> str:
    lines = [
        "# SSL + Context Domain Adaptation Run",
        "",
        "## Config",
        f"- parquet_path: `{config.parquet_path}`",
        f"- main_protocol: `{config.main_protocol}`",
        f"- stress_protocol: `{config.stress_protocol}`",
        f"- targets: `{', '.join(config.targets)}`",
        f"- ssl_objective: `{config.ssl_objective}`",
        f"- allow_test_time_unlabeled_context: `{config.allow_test_time_unlabeled_context}`",
        "",
        "## Main Split",
        f"- paired_test_rows: `{main_manifest['paired_test_rows']}`",
        f"- paired_finetune_rows: `{main_manifest['paired_finetune_rows']}`",
        f"- ssl_pool_rows: `{main_manifest['ssl_pool_rows']}`",
        f"- paired_test_recordings: `{main_manifest['paired_test_recordings']}`",
        "",
        "## Main Results",
    ]
    if metrics_df.empty:
        lines.append("- No metrics were produced.")
    else:
        for _, row in metrics_df.sort_values(["target", "mae", "model_id"]).iterrows():
            lines.append(
                f"- {row['target']} | {row['model_id']} | MAE `{row['mae']:.4f}` | R2 `{row['r2']:.4f}`"
            )
    lines.append("")
    lines.append("## Stress Test")
    if stress_df.empty:
        lines.append("- Stress-test skipped or produced no results.")
    else:
        for _, row in stress_df.sort_values(["held_out_study", "target", "mae"]).iterrows():
            lines.append(
                f"- {row['held_out_study']} | {row['target']} | {row['model_id']} | "
                f"MAE `{row['mae']:.4f}` | R2 `{row['r2']:.4f}`"
            )
    return "\n".join(lines) + "\n"


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns + [c for c in frame.columns if c not in columns]]


def run_with_config(config: SSLDomainAdaptationConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(config, f"Loading parquet from {config.parquet_path}")
    prepared = prepare_feature_table(config.parquet_path)
    main_split = make_main_split(
        prepared.df,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        random_state=config.random_state,
    )
    metrics_rows, prediction_frames, history_rows, main_manifest = _run_single_protocol(
        protocol_name="main",
        split=main_split,
        prepared=prepared,
        config=config,
        output_dir=output_dir,
        include_baselines=True,
    )

    stress_rows: list[dict[str, Any]] = []
    stress_manifests: list[dict[str, Any]] = []
    if config.run_stress_test:
        for stress_split in build_stress_splits(
            prepared.df,
            max_stress_studies=config.max_stress_studies,
        ):
            _log(config, f"Stress test: held_out_study={stress_split.held_out_study}")
            split_metrics, split_predictions, split_history, split_manifest = _run_single_protocol(
                protocol_name=f"stress:{stress_split.held_out_study}",
                split=stress_split,
                prepared=prepared,
                config=config,
                output_dir=output_dir / "stress" / stress_split.held_out_study.lower(),
                include_baselines=False,
            )
            prediction_frames.extend(split_predictions)
            history_rows.extend(split_history)
            stress_manifests.append(split_manifest)
            for row in split_metrics:
                stress_rows.append(
                    {
                        "held_out_study": stress_split.held_out_study,
                        **row,
                    }
                )

    metrics_df = _frame_with_columns(metrics_rows, _METRIC_COLUMNS)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["target", "mae", "model_id"]).reset_index(drop=True)
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True).sort_values(["protocol", "target", "model_id", "row_uid"])
        if prediction_frames
        else pd.DataFrame(columns=_PREDICTION_COLUMNS)
    )
    stress_df = _frame_with_columns(stress_rows, _STRESS_COLUMNS)
    if not stress_df.empty:
        stress_df = stress_df.sort_values(["held_out_study", "target", "mae", "model_id"]).reset_index(drop=True)
    baseline_df = (
        metrics_df[metrics_df["family"] == "baseline"].copy()
        if "family" in metrics_df.columns
        else pd.DataFrame(columns=_METRIC_COLUMNS)
    )
    history_df = _frame_with_columns(history_rows, _HISTORY_COLUMNS)

    split_manifest = {
        "config": config.to_dict(),
        "main": main_manifest,
        "stress": stress_manifests,
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2, sort_keys=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    baseline_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    stress_df.to_csv(output_dir / "stress_test_results.csv", index=False)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "README_run.md").write_text(
        _build_readme(
            config=config,
            main_manifest=main_manifest,
            metrics_df=metrics_df,
            stress_df=stress_df,
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
