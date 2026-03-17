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
from crash_tests.ml_statistics_workbench.fpos_anchor_stack import _fit_anchor_scores
from crash_tests.ml_statistics_workbench.fpos_backend_sweep import _fit_backend_predict, _grouped_oof_splits
from crash_tests.ml_statistics_workbench.fpos_cluster_trust_pseudo import (
    FPosClusterTrustPseudoConfig,
    _apply_trust_selection,
    _trust_from_group_stats,
)
from crash_tests.ml_statistics_workbench.methods import (
    build_stack_frame,
    make_metric_row,
    make_prediction_frame,
    mask_feature_columns,
    score_predictions,
)
from crash_tests.ml_statistics_workbench.pseudo_label_manifold import (
    PseudoLabelManifoldConfig,
    _build_filtered_drop_cols,
    _build_support_artifacts,
    _manifold_support_table,
    _select_pseudo_labels,
    _teacher_frames_for_target,
    _teacher_predictions,
)
from crash_tests.mmd_domain_adaptation.dataset import prepare_feature_table
from crash_tests.self_supervised_domain_adaptation.split_utils import make_main_split


@dataclass
class FPosFamilyRobustnessConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("crash_tests/ml_statistics_workbench/outputs")
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
    mae_guardrail_pct: float = 0.03
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_family_robustness_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fpos family robustness benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosFamilyRobustnessConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _family_weights(study_set: pd.Series, *, alpha: float, min_weight: float = 0.6, max_weight: float = 2.5) -> np.ndarray:
    counts = study_set.astype(str).value_counts().to_dict()
    raw = np.array([counts[str(v)] for v in study_set.astype(str)], dtype=float)
    weights = np.power(raw / max(float(np.mean(raw)), 1e-6), -float(alpha))
    weights = weights / max(float(np.mean(weights)), 1e-6)
    return np.clip(weights, min_weight, max_weight).astype(np.float32)


def _assemble_training_data(
    *,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    labeled_meta_df: pd.DataFrame,
    labeled_indices: np.ndarray | None,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    family_alpha: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if labeled_indices is None:
        base_frame = labeled_frame.reset_index(drop=True)
        base_y = np.asarray(labeled_y, dtype=float)
        base_meta = labeled_meta_df.reset_index(drop=True)
    else:
        base_frame = labeled_frame.iloc[labeled_indices].reset_index(drop=True)
        base_y = np.asarray(labeled_y[labeled_indices], dtype=float)
        base_meta = labeled_meta_df.iloc[labeled_indices].reset_index(drop=True)

    labeled_weight = _family_weights(base_meta["study_set"], alpha=family_alpha)
    if pseudo_selected.empty:
        return base_frame, base_y, labeled_weight

    pseudo_idx = pseudo_selected["unlabeled_row_idx"].to_numpy(dtype=int)
    pseudo_frame = unlabeled_frame.iloc[pseudo_idx].reset_index(drop=True)
    full_frame = pd.concat([base_frame, pseudo_frame], ignore_index=True)
    full_y = np.concatenate([base_y, pseudo_selected["pseudo_label"].to_numpy(dtype=float)])
    full_weight = np.concatenate(
        [
            labeled_weight,
            pseudo_selected["pseudo_weight"].to_numpy(dtype=np.float32),
        ]
    )
    return full_frame, full_y, full_weight.astype(np.float32)


def _per_family_summary(y_true: np.ndarray, pred: np.ndarray, study_set: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "study_set": study_set.astype(str).to_numpy(),
            "y_true": np.asarray(y_true, dtype=float),
            "pred": np.asarray(pred, dtype=float),
        }
    )
    rows: list[dict[str, Any]] = []
    for study, group in frame.groupby("study_set", dropna=False):
        metrics = score_predictions(
            group["y_true"].to_numpy(dtype=float),
            group["pred"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "study_set": str(study),
                "n_rows": int(len(group)),
                "mae": float(metrics["mae"]),
                "r2": float(metrics["r2"]),
                "bias": float(metrics["bias"]),
            }
        )
    return pd.DataFrame(rows).sort_values("study_set").reset_index(drop=True)


def _cv_variant(
    *,
    labeled_meta_df: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    base: MLStatisticsWorkbenchConfig,
    backend_id: str,
    family_alpha: float,
    random_state_offset: int,
) -> tuple[np.ndarray, dict[str, float], pd.DataFrame]:
    splits = _grouped_oof_splits(labeled_meta_df, base.model_selection_splits)
    oof = np.full(len(labeled_meta_df), np.nan, dtype=float)
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        val_recordings = set(labeled_meta_df.iloc[val_idx]["recording_key"].astype(str))
        fold_pseudo = pseudo_selected[
            ~pseudo_selected["recording_key"].astype(str).isin(val_recordings)
        ].reset_index(drop=True)
        train_frame, train_y, train_weight = _assemble_training_data(
            labeled_frame=labeled_frame,
            labeled_y=labeled_y,
            labeled_meta_df=labeled_meta_df,
            labeled_indices=train_idx,
            pseudo_selected=fold_pseudo,
            unlabeled_frame=unlabeled_frame,
            family_alpha=family_alpha,
        )
        fold_pred = _fit_backend_predict(
            X_train=train_frame,
            y_train=train_y,
            X_eval=labeled_frame.iloc[val_idx].reset_index(drop=True),
            sample_weight=train_weight,
            target_transform_mode=base.resolved_target_transform(),
            backend_id=backend_id,
            random_state=base.random_state + random_state_offset + split_id,
        )
        oof[val_idx] = fold_pred

    summary = _per_family_summary(labeled_y, oof, labeled_meta_df["study_set"])
    metrics = score_predictions(labeled_y, oof)
    metrics["worst_family_r2"] = float(summary["r2"].min()) if not summary.empty else float("nan")
    metrics["mean_family_r2"] = float(summary["r2"].mean()) if not summary.empty else float("nan")
    metrics["worst_family_mae"] = float(summary["mae"].max()) if not summary.empty else float("nan")
    return oof, metrics, summary


def _fit_holdout_variant(
    *,
    labeled_meta_df: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    base: MLStatisticsWorkbenchConfig,
    backend_id: str,
    family_alpha: float,
    random_state_offset: int,
) -> np.ndarray:
    train_frame, train_y, train_weight = _assemble_training_data(
        labeled_frame=labeled_frame,
        labeled_y=labeled_y,
        labeled_meta_df=labeled_meta_df,
        labeled_indices=None,
        pseudo_selected=pseudo_selected,
        unlabeled_frame=unlabeled_frame,
        family_alpha=family_alpha,
    )
    return _fit_backend_predict(
        X_train=train_frame,
        y_train=train_y,
        X_eval=test_frame.reset_index(drop=True),
        sample_weight=train_weight,
        target_transform_mode=base.resolved_target_transform(),
        backend_id=backend_id,
        random_state=base.random_state + random_state_offset,
    )


def _pick_robust_variant(cv_df: pd.DataFrame, *, reference_variant_id: str, mae_guardrail_pct: float) -> dict[str, Any]:
    ref = cv_df.loc[cv_df["variant_id"].astype(str) == str(reference_variant_id)].iloc[0]
    mae_limit = float(ref["cv_mae"]) * (1.0 + float(mae_guardrail_pct))
    candidates = cv_df.loc[cv_df["cv_mae"] <= mae_limit].copy()
    if candidates.empty:
        candidates = cv_df.copy()
    candidates = candidates.sort_values(
        ["cv_worst_family_r2", "cv_r2", "cv_mae"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    rec = candidates.iloc[0]
    return {
        "target": "fpos",
        "selection_metric": "worst_family_r2",
        "reference_variant_id": str(reference_variant_id),
        "reference_cv_mae": float(ref["cv_mae"]),
        "reference_cv_r2": float(ref["cv_r2"]),
        "reference_cv_worst_family_r2": float(ref["cv_worst_family_r2"]),
        "mae_guardrail_limit": float(mae_limit),
        "recommended_variant_id": str(rec["variant_id"]),
        "recommended_cv_mae": float(rec["cv_mae"]),
        "recommended_cv_r2": float(rec["cv_r2"]),
        "recommended_cv_worst_family_r2": float(rec["cv_worst_family_r2"]),
        "worst_family_r2_improvement": float(rec["cv_worst_family_r2"] - ref["cv_worst_family_r2"]),
        "global_r2_improvement": float(rec["cv_r2"] - ref["cv_r2"]),
        "mae_regression_pct": float((rec["cv_mae"] - ref["cv_mae"]) / max(float(ref["cv_mae"]), 1e-9)),
    }


def _prepare_frames(config: FPosFamilyRobustnessConfig):
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
        _student_templates,
        _transport_manifest,
    ) = _teacher_frames_for_target(
        target="fpos",
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
    matched_support_conf_train = support_train_df["support_confidence"].to_numpy(dtype=float)
    matched_support_conf_unlabeled = support_df["support_confidence"].to_numpy(dtype=float)
    matched_support_conf_test = _manifold_support_table(
        unlabeled_df=target_test_df,
        labeled_train_df=target_train_df,
        support_artifacts=support_artifacts,
    )["support_confidence"].to_numpy(dtype=float)

    train_frame = build_stack_frame(
        target_train_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_train_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_oof,
            "matched_support_conf": matched_support_conf_train,
            "anchor_support_margin": anchor_oof - matched_support_conf_train,
        },
    )
    unlabeled_frame = build_stack_frame(
        unlabeled_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_unlabeled_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_unlabeled,
            "matched_support_conf": matched_support_conf_unlabeled,
            "anchor_support_margin": anchor_unlabeled - matched_support_conf_unlabeled,
        },
    )
    test_frame = build_stack_frame(
        target_test_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_test_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_test,
            "matched_support_conf": matched_support_conf_test,
            "anchor_support_margin": anchor_test - matched_support_conf_test,
        },
    )
    return base, split, target_train_df, y_train, target_test_df, y_test, trust_selected, train_frame, unlabeled_frame, test_frame


def _run_with_config(config: FPosFamilyRobustnessConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        base,
        split,
        target_train_df,
        y_train,
        target_test_df,
        y_test,
        trust_selected,
        train_frame,
        unlabeled_frame,
        test_frame,
    ) = _prepare_frames(config)

    variants = [
        ("reference_xgboost", "xgboost_default", 0.0),
        ("family_balanced_mild_xgboost", "xgboost_default", 0.5),
        ("family_balanced_strong_xgboost", "xgboost_default", 1.0),
        ("family_balanced_shallow_xgboost", "xgboost_shallow", 1.0),
    ]

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []
    family_frames: list[pd.DataFrame] = []
    oof_store: dict[str, np.ndarray] = {}
    holdout_store: dict[str, np.ndarray] = {}

    backend_offset_map = {
        "xgboost_default": 2200,
        "xgboost_shallow": 2300,
    }

    for idx, (variant_id, backend_id, family_alpha) in enumerate(variants, start=1):
        _log(config, f"[family-robust] {variant_id}")
        cv_offset = backend_offset_map.get(backend_id, 3000 + (100 * idx))
        holdout_offset = cv_offset + 2000
        oof_pred, cv_metrics, cv_family_df = _cv_variant(
            labeled_meta_df=target_train_df,
            labeled_frame=train_frame,
            labeled_y=y_train,
            pseudo_selected=trust_selected,
            unlabeled_frame=unlabeled_frame,
            base=base,
            backend_id=backend_id,
            family_alpha=family_alpha,
            random_state_offset=cv_offset,
        )
        holdout_pred = _fit_holdout_variant(
            labeled_meta_df=target_train_df,
            labeled_frame=train_frame,
            labeled_y=y_train,
            pseudo_selected=trust_selected,
            unlabeled_frame=unlabeled_frame,
            test_frame=test_frame,
            base=base,
            backend_id=backend_id,
            family_alpha=family_alpha,
            random_state_offset=holdout_offset,
        )
        holdout_metrics = score_predictions(y_test, holdout_pred)
        holdout_family_df = _per_family_summary(y_test, holdout_pred, target_test_df["study_set"])
        holdout_worst_family_r2 = float(holdout_family_df["r2"].min()) if not holdout_family_df.empty else float("nan")
        oof_store[variant_id] = oof_pred.copy()
        holdout_store[variant_id] = holdout_pred.copy()

        metric_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family="family_robustness",
                backend=backend_id,
                source_target="fpos",
                metrics=holdout_metrics,
                notes=f"family_alpha={family_alpha:.2f}",
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family="family_robustness",
                backend=backend_id,
                df=target_test_df,
                y_true=y_test,
                pred=holdout_pred,
            )
        )
        cv_rows.append(
            {
                "target": "fpos",
                "variant_id": variant_id,
                "backend": backend_id,
                "family_alpha": float(family_alpha),
                "cv_mae": float(cv_metrics["mae"]),
                "cv_r2": float(cv_metrics["r2"]),
                "cv_worst_family_r2": float(cv_metrics["worst_family_r2"]),
                "cv_mean_family_r2": float(cv_metrics["mean_family_r2"]),
                "cv_worst_family_mae": float(cv_metrics["worst_family_mae"]),
                "holdout_worst_family_r2": float(holdout_worst_family_r2),
                "selected_rows": int(len(trust_selected)),
                "mean_selected_weight": float(trust_selected["pseudo_weight"].mean()) if not trust_selected.empty else 0.0,
            }
        )
        family_frames.append(cv_family_df.assign(variant_id=variant_id, split="cv"))
        family_frames.append(holdout_family_df.assign(variant_id=variant_id, split="holdout"))

    ref_oof = oof_store["reference_xgboost"]
    ref_holdout = holdout_store["reference_xgboost"]
    strong_oof = oof_store["family_balanced_strong_xgboost"]
    strong_holdout = holdout_store["family_balanced_strong_xgboost"]
    for alpha in (0.25, 0.50, 0.75):
        variant_id = f"reference_plus_strong_blend_{int(round(alpha * 100)):03d}"
        oof_pred = ((1.0 - alpha) * ref_oof) + (alpha * strong_oof)
        holdout_pred = ((1.0 - alpha) * ref_holdout) + (alpha * strong_holdout)
        cv_metrics = score_predictions(y_train, oof_pred)
        cv_family_df = _per_family_summary(y_train, oof_pred, target_train_df["study_set"])
        cv_metrics["worst_family_r2"] = float(cv_family_df["r2"].min()) if not cv_family_df.empty else float("nan")
        cv_metrics["mean_family_r2"] = float(cv_family_df["r2"].mean()) if not cv_family_df.empty else float("nan")
        cv_metrics["worst_family_mae"] = float(cv_family_df["mae"].max()) if not cv_family_df.empty else float("nan")
        holdout_metrics = score_predictions(y_test, holdout_pred)
        holdout_family_df = _per_family_summary(y_test, holdout_pred, target_test_df["study_set"])
        holdout_worst_family_r2 = float(holdout_family_df["r2"].min()) if not holdout_family_df.empty else float("nan")
        metric_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family="family_robustness_blend",
                backend="blend",
                source_target="fpos",
                metrics=holdout_metrics,
                notes=f"reference/strong blend alpha={alpha:.2f}",
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family="family_robustness_blend",
                backend="blend",
                df=target_test_df,
                y_true=y_test,
                pred=holdout_pred,
            )
        )
        cv_rows.append(
            {
                "target": "fpos",
                "variant_id": variant_id,
                "backend": "blend",
                "family_alpha": float(alpha),
                "cv_mae": float(cv_metrics["mae"]),
                "cv_r2": float(cv_metrics["r2"]),
                "cv_worst_family_r2": float(cv_metrics["worst_family_r2"]),
                "cv_mean_family_r2": float(cv_metrics["mean_family_r2"]),
                "cv_worst_family_mae": float(cv_metrics["worst_family_mae"]),
                "holdout_worst_family_r2": float(holdout_worst_family_r2),
                "selected_rows": int(len(trust_selected)),
                "mean_selected_weight": float(trust_selected["pseudo_weight"].mean()) if not trust_selected.empty else 0.0,
            }
        )
        family_frames.append(cv_family_df.assign(variant_id=variant_id, split="cv"))
        family_frames.append(holdout_family_df.assign(variant_id=variant_id, split="holdout"))

    metrics_df = pd.DataFrame(metric_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows).sort_values(["cv_worst_family_r2", "cv_r2"], ascending=[False, False]).reset_index(drop=True)
    recommendation = _pick_robust_variant(
        cv_df,
        reference_variant_id="reference_xgboost",
        mae_guardrail_pct=config.mae_guardrail_pct,
    )
    recommended_holdout = metrics_df.loc[
        metrics_df["variant_id"].astype(str) == str(recommendation["recommended_variant_id"])
    ].iloc[0]
    recommendation["recommended_holdout_mae"] = float(recommended_holdout["mae"])
    recommendation["recommended_holdout_r2"] = float(recommended_holdout["r2"])

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    pd.DataFrame([recommendation]).to_csv(output_dir / "per_target_recommendation.csv", index=False)
    pd.concat(family_frames, ignore_index=True).to_csv(output_dir / "family_performance.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split.manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosFamilyRobustnessConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosFamilyRobustnessConfig(
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
