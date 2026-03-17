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
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qc_thesis.modeling.stacking.config import MLStatisticsWorkbenchConfig
from qc_thesis.modeling.stacking.fpos_anchor_stack import _fit_anchor_scores
from qc_thesis.modeling.stacking.fpos_cluster_trust_pseudo import (
    FPosClusterTrustPseudoConfig,
    _apply_trust_selection,
    _trust_from_group_stats,
)
from qc_thesis.modeling.stacking.methods import (
    build_stack_frame,
    make_metric_row,
    make_prediction_frame,
    mask_feature_columns,
    pick_r2_first_recommendation,
    score_predictions,
)
from qc_thesis.modeling.stacking.pseudo_label_manifold import (
    PseudoLabelManifoldConfig,
    _assemble_student_training_data,
    _build_filtered_drop_cols,
    _build_support_artifacts,
    _manifold_support_table,
    _select_pseudo_labels,
    _teacher_frames_for_target,
    _teacher_predictions,
)
from qc_thesis.modeling.neural.mmd.dataset import prepare_feature_table
from qc_thesis.modeling.neural.mmd.model import target_inverse, target_transform
from qc_thesis.modeling.neural.ssl.split_utils import make_main_split


@dataclass
class FPosXGBoostTuningConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("artifacts/modeling/stacking")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    max_pseudo_multiplier: float = 1.5
    max_pseudo_per_recording: int = 48
    max_pseudo_per_cluster: int = 128
    max_pseudo_per_labeled_ratio: float = 1.5
    cluster_trust_power: float = 1.35
    study_trust_weight: float = 0.30
    min_trust: float = 0.25
    selection_metric: str = "r2"
    mae_guardrail_pct: float = 0.03
    min_r2_improvement: float = 0.0
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_xgboost_tuning_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fpos XGBoost tuning benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosXGBoostTuningConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _grouped_oof_splits(df: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = df["recording_key"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(max(2, int(n_splits)), len(unique_groups))
    fold_sizes = np.full(n_splits, len(unique_groups) // n_splits, dtype=int)
    fold_sizes[: len(unique_groups) % n_splits] += 1
    rng = np.random.default_rng(42)
    permuted = unique_groups.copy()
    rng.shuffle(permuted)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for fold_size in fold_sizes:
        val_groups = set(permuted[start : start + fold_size].tolist())
        start += fold_size
        val_idx = np.flatnonzero(np.isin(groups, list(val_groups)))
        train_idx = np.flatnonzero(~np.isin(groups, list(val_groups)))
        splits.append((train_idx.astype(int), val_idx.astype(int)))
    return splits


def _fit_xgb_predict(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    sample_weight: np.ndarray | None,
    target_transform_mode: str,
    params: dict[str, Any],
    random_state: int,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_eval_imp = imputer.transform(X_eval)
    y_train_t = target_transform(y_train, mode=target_transform_mode)
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train_imp, y_train_t, **fit_kwargs)
    pred_t = np.asarray(model.predict(X_eval_imp), dtype=float)
    return np.clip(target_inverse(pred_t, mode=target_transform_mode), 0.0, 1.0)


def _score_xgb_cv(
    *,
    labeled_meta_df: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    params: dict[str, Any],
    base: MLStatisticsWorkbenchConfig,
    random_state_offset: int,
) -> dict[str, float]:
    splits = _grouped_oof_splits(labeled_meta_df, base.model_selection_splits)
    oof = np.full(len(labeled_meta_df), np.nan, dtype=float)
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
        oof[val_idx] = _fit_xgb_predict(
            X_train=train_frame,
            y_train=train_y,
            X_eval=labeled_frame.iloc[val_idx].reset_index(drop=True),
            sample_weight=train_weight,
            target_transform_mode=base.resolved_target_transform(),
            params=params,
            random_state=base.random_state + random_state_offset + split_id,
        )
    return score_predictions(labeled_y, oof)


def _fit_xgb_holdout(
    *,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    params: dict[str, Any],
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
    return _fit_xgb_predict(
        X_train=train_frame,
        y_train=train_y,
        X_eval=test_frame.reset_index(drop=True),
        sample_weight=train_weight,
        target_transform_mode=base.resolved_target_transform(),
        params=params,
        random_state=base.random_state + random_state_offset,
    )


def _run_with_config(config: FPosXGBoostTuningConfig) -> Path:
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
        student_templates,
        _,
    ) = _teacher_frames_for_target(
        target="fpos",
        split=split,
        prepared=prepared,
        base=base,
        config=pseudo_base,
        unit_drop_cols=unit_drop_cols,
        context_drop_cols=context_drop_cols,
    )
    teacher_train_df, teacher_unlabeled_df, _, _ = _teacher_predictions(
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
    train_trust_df["teacher_abs_error"] = np.abs(
        y_train - teacher_train_df["teacher_mean"].to_numpy(dtype=float)
    )
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
    base_row = student_templates.loc[
        student_templates["variant_id"].astype(str) == "source_reweighted_pseudo_lightgbm_filtered"
    ].iloc[0]
    base_train = base_row["train_frame"]
    base_unlabeled = base_row["unlabeled_frame"]
    base_test = base_row["test_frame"]
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
    support_test_df = _manifold_support_table(
        unlabeled_df=target_test_df,
        labeled_train_df=target_train_df,
        support_artifacts=support_artifacts,
    )
    anchor_train = build_stack_frame(
        target_train_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_train_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_oof,
            "matched_support_conf": support_train_df["support_confidence"].to_numpy(dtype=float),
            "anchor_support_margin": anchor_oof - support_train_df["support_confidence"].to_numpy(dtype=float),
        },
    )
    anchor_unlabeled_frame = build_stack_frame(
        unlabeled_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_unlabeled_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_unlabeled,
            "matched_support_conf": support_df["support_confidence"].to_numpy(dtype=float),
            "anchor_support_margin": anchor_unlabeled - support_df["support_confidence"].to_numpy(dtype=float),
        },
    )
    anchor_test_frame = build_stack_frame(
        target_test_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teachers[1].test_frame["source_pred"].to_numpy(dtype=float),
            "anchor_fp_score": anchor_test,
            "matched_support_conf": support_test_df["support_confidence"].to_numpy(dtype=float),
            "anchor_support_margin": anchor_test - support_test_df["support_confidence"].to_numpy(dtype=float),
        },
    )

    presets = {
        "default": {
            "n_estimators": 350,
            "learning_rate": 0.04,
            "max_depth": 4,
            "min_child_weight": 4.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
        "deep": {
            "n_estimators": 450,
            "learning_rate": 0.035,
            "max_depth": 5,
            "min_child_weight": 2.0,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "reg_lambda": 1.5,
        },
        "regularized": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 6.0,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "reg_lambda": 3.0,
            "reg_alpha": 0.1,
        },
        "wide_low_lr": {
            "n_estimators": 700,
            "learning_rate": 0.02,
            "max_depth": 4,
            "min_child_weight": 4.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
        "shallow": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 6.0,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.5,
        },
        "balanced": {
            "n_estimators": 450,
            "learning_rate": 0.035,
            "max_depth": 4,
            "min_child_weight": 5.0,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "reg_lambda": 2.0,
        },
        "subsampled": {
            "n_estimators": 550,
            "learning_rate": 0.025,
            "max_depth": 5,
            "min_child_weight": 4.0,
            "subsample": 0.65,
            "colsample_bytree": 0.65,
            "reg_lambda": 2.0,
        },
    }
    feature_sets = [
        ("source_reweighted_cluster_trust_pseudo", base_train, base_unlabeled, base_test),
        ("source_reweighted_anchor_stack_cluster_trust_pseudo", anchor_train, anchor_unlabeled_frame, anchor_test_frame),
    ]

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []
    for feature_name, train_frame, unlabeled_frame, test_frame in feature_sets:
        _log(config, f"[{feature_name}] xgboost presets")
        for idx, (preset_name, params) in enumerate(presets.items(), start=1):
            cv_metrics = _score_xgb_cv(
                labeled_meta_df=target_train_df,
                labeled_frame=train_frame,
                labeled_y=y_train,
                pseudo_selected=trust_selected,
                unlabeled_frame=unlabeled_frame,
                params=params,
                base=base,
                random_state_offset=2000 + (100 * idx),
            )
            holdout_pred = _fit_xgb_holdout(
                labeled_frame=train_frame,
                labeled_y=y_train,
                pseudo_selected=trust_selected,
                unlabeled_frame=unlabeled_frame,
                test_frame=test_frame,
                params=params,
                base=base,
                random_state_offset=4000 + (100 * idx),
            )
            variant_id = f"{feature_name}_xgb_{preset_name}"
            holdout_metrics = score_predictions(y_test, holdout_pred)
            metric_rows.append(
                make_metric_row(
                    target="fpos",
                    variant_id=variant_id,
                    family="xgboost_tuning",
                    backend="xgboost",
                    source_target="fpos",
                    metrics=holdout_metrics,
                    notes=variant_id,
                )
            )
            prediction_frames.append(
                make_prediction_frame(
                    target="fpos",
                    variant_id=variant_id,
                    family="xgboost_tuning",
                    backend="xgboost",
                    df=target_test_df,
                    y_true=y_test,
                    pred=holdout_pred,
                )
            )
            cv_rows.append(
                {
                    "target": "fpos",
                    "variant_id": variant_id,
                    "family": "xgboost_tuning",
                    "feature_set": feature_name,
                    "preset_name": preset_name,
                    "cv_mae": float(cv_metrics["mae"]),
                    "cv_r2": float(cv_metrics["r2"]),
                    "selected_rows": int(len(trust_selected)),
                    "mean_selected_weight": float(trust_selected["pseudo_weight"].mean()) if not trust_selected.empty else 0.0,
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows)
    reference_variant_id = "source_reweighted_anchor_stack_cluster_trust_pseudo_xgb_default"
    cv_for_picker = cv_df.rename(columns={"cv_mae": "mae", "cv_r2": "r2"})
    recommendation = pick_r2_first_recommendation(
        cv_for_picker,
        target="fpos",
        reference_variant_id=reference_variant_id,
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
    holdout_best_df = pd.DataFrame(
        [
            {
                "target": "fpos",
                "best_holdout_variant_id": str(metrics_df.iloc[0]["variant_id"]),
                "best_holdout_mae": float(metrics_df.iloc[0]["mae"]),
                "best_holdout_r2": float(metrics_df.iloc[0]["r2"]),
            }
        ]
    )

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    recommendation_df.to_csv(output_dir / "per_target_recommendation.csv", index=False)
    holdout_best_df.to_csv(output_dir / "holdout_best_by_target.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split.manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosXGBoostTuningConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosXGBoostTuningConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.max_pseudo_multiplier = 1.0
        cli_config.max_pseudo_per_recording = 24
        cli_config.max_pseudo_per_cluster = 64
    _run_with_config(cli_config)
