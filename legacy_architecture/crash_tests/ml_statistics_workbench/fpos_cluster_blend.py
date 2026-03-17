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
from crash_tests.ml_statistics_workbench.fpos_backend_sweep import (
    _fit_backend_holdout,
    _score_backend_cv,
)
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
    pick_r2_first_recommendation,
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
class FPosClusterBlendConfig:
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
    alpha_grid: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21).tolist())
    cluster_blend_shrinkage: float = 12.0
    min_cluster_rows_for_alpha: int = 8
    selection_metric: str = "r2"
    mae_guardrail_pct: float = 0.03
    min_r2_improvement: float = 0.0
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_cluster_blend_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fpos cluster-aware blend benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosClusterBlendConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _best_alpha(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, alpha_grid: tuple[float, ...]) -> float:
    best_alpha = 0.5
    best_r2 = -np.inf
    for alpha in alpha_grid:
        pred = ((1.0 - alpha) * pred_a) + (alpha * pred_b)
        r2 = float(score_predictions(y_true, pred)["r2"])
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = float(alpha)
    return best_alpha


def _run_with_config(config: FPosClusterBlendConfig) -> Path:
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
    support_test_df = _manifold_support_table(
        unlabeled_df=target_test_df,
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

    backend_id = "xgboost_default"
    base_cv = _score_backend_cv(
        labeled_meta_df=target_train_df,
        labeled_frame=base_train,
        labeled_y=y_train,
        pseudo_selected=trust_selected,
        unlabeled_frame=base_unlabeled,
        backend_id=backend_id,
        base=base,
        random_state_offset=7000,
    )
    anchor_cv = _score_backend_cv(
        labeled_meta_df=target_train_df,
        labeled_frame=anchor_train,
        labeled_y=y_train,
        pseudo_selected=trust_selected,
        unlabeled_frame=anchor_unlabeled_frame,
        backend_id=backend_id,
        base=base,
        random_state_offset=7100,
    )
    base_holdout = _fit_backend_holdout(
        labeled_frame=base_train,
        labeled_y=y_train,
        pseudo_selected=trust_selected,
        unlabeled_frame=base_unlabeled,
        test_frame=base_test,
        backend_id=backend_id,
        base=base,
        random_state_offset=9000,
    )
    anchor_holdout = _fit_backend_holdout(
        labeled_frame=anchor_train,
        labeled_y=y_train,
        pseudo_selected=trust_selected,
        unlabeled_frame=anchor_unlabeled_frame,
        test_frame=anchor_test_frame,
        backend_id=backend_id,
        base=base,
        random_state_offset=9100,
    )

    global_alpha = _best_alpha(y_train, base_cv["pred"], anchor_cv["pred"], config.alpha_grid) if "pred" in base_cv else None
    # _score_backend_cv returns metrics only, so build OOF predictions explicitly below.
    # Re-run OOF predictions to support blending diagnostics.
    from crash_tests.ml_statistics_workbench.fpos_backend_sweep import _grouped_oof_splits as _backend_splits
    from crash_tests.ml_statistics_workbench.pseudo_label_manifold import _assemble_student_training_data

    splits = _backend_splits(target_train_df, base.model_selection_splits)
    base_oof = np.full(len(target_train_df), np.nan, dtype=float)
    anchor_oof_pred = np.full(len(target_train_df), np.nan, dtype=float)
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        val_recordings = set(target_train_df.iloc[val_idx]["recording_key"].astype(str))
        fold_pseudo = trust_selected[
            ~trust_selected["recording_key"].astype(str).isin(val_recordings)
        ].reset_index(drop=True)
        train_frame, train_y, train_weight = _assemble_student_training_data(
            labeled_frame=base_train,
            labeled_y=y_train,
            labeled_indices=train_idx,
            pseudo_selected=fold_pseudo,
            unlabeled_frame=base_unlabeled,
        )
        base_oof[val_idx] = _fit_backend_holdout(
            labeled_frame=base_train.iloc[train_idx].reset_index(drop=True),
            labeled_y=y_train[train_idx],
            pseudo_selected=fold_pseudo,
            unlabeled_frame=base_unlabeled,
            test_frame=base_train.iloc[val_idx].reset_index(drop=True),
            backend_id=backend_id,
            base=base,
            random_state_offset=9500 + split_id,
        )
        anchor_oof_pred[val_idx] = _fit_backend_holdout(
            labeled_frame=anchor_train.iloc[train_idx].reset_index(drop=True),
            labeled_y=y_train[train_idx],
            pseudo_selected=fold_pseudo,
            unlabeled_frame=anchor_unlabeled_frame,
            test_frame=anchor_train.iloc[val_idx].reset_index(drop=True),
            backend_id=backend_id,
            base=base,
            random_state_offset=9600 + split_id,
        )
    global_alpha = _best_alpha(y_train, base_oof, anchor_oof_pred, config.alpha_grid)
    global_holdout = ((1.0 - global_alpha) * base_holdout) + (global_alpha * anchor_holdout)

    cluster_rows: list[dict[str, float]] = []
    cluster_alpha_map: dict[int, float] = {}
    global_alpha_float = float(global_alpha)
    for cluster_id, subset in support_train_df.groupby("cluster_id", dropna=False):
        idx = subset.index.to_numpy(dtype=int)
        y_cluster = y_train[idx]
        if len(idx) < int(config.min_cluster_rows_for_alpha):
            alpha = global_alpha_float
        else:
            raw_alpha = _best_alpha(y_cluster, base_oof[idx], anchor_oof_pred[idx], config.alpha_grid)
            shrink = len(idx) / (len(idx) + float(config.cluster_blend_shrinkage))
            alpha = ((1.0 - shrink) * global_alpha_float) + (shrink * float(raw_alpha))
        cluster_alpha_map[int(cluster_id)] = float(alpha)
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "labeled_rows": int(len(idx)),
                "cluster_alpha": float(alpha),
            }
        )
    holdout_cluster_ids = support_test_df["cluster_id"].to_numpy(dtype=int)
    cluster_alpha = np.asarray([cluster_alpha_map.get(int(cid), global_alpha_float) for cid in holdout_cluster_ids], dtype=float)
    cluster_holdout = ((1.0 - cluster_alpha) * base_holdout) + (cluster_alpha * anchor_holdout)
    global_oof = ((1.0 - global_alpha_float) * base_oof) + (global_alpha_float * anchor_oof_pred)
    cluster_train_alpha = np.asarray(
        [cluster_alpha_map.get(int(cid), global_alpha_float) for cid in support_train_df["cluster_id"].to_numpy(dtype=int)],
        dtype=float,
    )
    cluster_oof = ((1.0 - cluster_train_alpha) * base_oof) + (cluster_train_alpha * anchor_oof_pred)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []
    variants = [
        ("source_reweighted_cluster_trust_pseudo_xgboost_default", base_holdout, score_predictions(y_train, base_oof)),
        ("source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default", anchor_holdout, score_predictions(y_train, anchor_oof_pred)),
        ("source_reweighted_global_blend_cluster_trust_pseudo_xgboost_default", global_holdout, score_predictions(y_train, global_oof)),
        ("source_reweighted_cluster_blend_cluster_trust_pseudo_xgboost_default", cluster_holdout, score_predictions(y_train, cluster_oof)),
    ]
    for variant_id, holdout_pred, cv_metrics in variants:
        metrics = score_predictions(y_test, holdout_pred)
        metric_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family="cluster_blend",
                backend="xgboost_default",
                source_target="fpos",
                metrics=metrics,
                notes=variant_id,
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family="cluster_blend",
                backend="xgboost_default",
                df=target_test_df,
                y_true=y_test,
                pred=holdout_pred,
            )
        )
        cv_rows.append(
            {
                "target": "fpos",
                "variant_id": variant_id,
                "family": "cluster_blend",
                "cv_mae": float(cv_metrics["mae"]),
                "cv_r2": float(cv_metrics["r2"]),
                "selected_rows": int(len(trust_selected)),
                "mean_selected_weight": float(trust_selected["pseudo_weight"].mean()) if not trust_selected.empty else 0.0,
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows)
    reference_variant_id = "source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default"
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
    blend_summary_df = pd.DataFrame(
        [
            {
                "target": "fpos",
                "global_alpha": float(global_alpha_float),
            }
        ]
    )
    cluster_alpha_df = pd.DataFrame(cluster_rows)

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    recommendation_df.to_csv(output_dir / "per_target_recommendation.csv", index=False)
    holdout_best_df.to_csv(output_dir / "holdout_best_by_target.csv", index=False)
    blend_summary_df.to_csv(output_dir / "blend_summary.csv", index=False)
    cluster_alpha_df.to_csv(output_dir / "cluster_alpha_summary.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split.manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosClusterBlendConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosClusterBlendConfig(
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
