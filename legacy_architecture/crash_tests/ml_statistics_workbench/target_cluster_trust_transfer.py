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
from crash_tests.ml_statistics_workbench.fpos_cluster_trust_pseudo import (
    _apply_trust_selection,
    _trust_from_group_stats,
)
from crash_tests.ml_statistics_workbench.methods import (
    make_metric_row,
    make_prediction_frame,
    pick_r2_first_recommendation,
    score_predictions,
)
from crash_tests.ml_statistics_workbench.pseudo_label_manifold import (
    PseudoLabelManifoldConfig,
    _build_filtered_drop_cols,
    _build_support_artifacts,
    _fit_student_holdout,
    _manifold_support_table,
    _score_student_cv,
    _select_pseudo_labels,
    _teacher_frames_for_target,
    _teacher_predictions,
)
from crash_tests.mmd_domain_adaptation.dataset import prepare_feature_table
from crash_tests.self_supervised_domain_adaptation.split_utils import make_main_split


@dataclass
class TargetClusterTrustTransferConfig:
    target: str = "fmiss"
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
    real_candidate_support_quantile: float = 0.35
    real_label_weight: float = 1.0
    selection_metric: str = "r2"
    mae_guardrail_pct: float = 0.03
    min_r2_improvement: float = 0.0
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"target_cluster_trust_{self.target}_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trust-aware target transfer on a strict paired holdout")
    parser.add_argument("--target", choices=["fmiss", "accuracy", "fpos"], default="fmiss")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: TargetClusterTrustTransferConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _teacher_reference_variant(
    target: str,
    teacher_perf_df: pd.DataFrame,
    base: MLStatisticsWorkbenchConfig,
) -> str:
    configured = base.reference_variant_by_target.get(target)
    if configured and configured in set(teacher_perf_df["variant_id"].astype(str)):
        return configured
    teacher_rows = teacher_perf_df.sort_values(["oof_r2", "oof_mae"], ascending=[False, True]).reset_index(drop=True)
    return str(teacher_rows.iloc[0]["variant_id"])


def _build_real_candidate_pool(
    *,
    target: str,
    unlabeled_df: pd.DataFrame,
    teacher_unlabeled_df: pd.DataFrame,
    support_df: pd.DataFrame,
    config: TargetClusterTrustTransferConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_df = pd.concat(
        [
            unlabeled_df.loc[:, ["row_uid", "recording_key", "study_set"]].reset_index(drop=True),
            teacher_unlabeled_df.reset_index(drop=True),
            support_df.loc[
                :,
                [
                    "unlabeled_row_idx",
                    "support_distance",
                    "support_confidence",
                    "cluster_id",
                    "cluster_labeled_count",
                    "cluster_labeled_share",
                ],
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    candidate_df["pseudo_label"] = pd.to_numeric(unlabeled_df[target], errors="coerce").to_numpy(dtype=float)
    support_cutoff = float(np.nanquantile(candidate_df["support_confidence"].astype(float), config.real_candidate_support_quantile))
    candidate_df["selected_candidate"] = (
        candidate_df["pseudo_label"].notna()
        & (candidate_df["cluster_labeled_count"].astype(float) > 0.0)
        & (candidate_df["support_confidence"].astype(float) >= support_cutoff)
    )
    all_real = candidate_df[candidate_df["pseudo_label"].notna()].copy().reset_index(drop=True)
    all_real["pseudo_weight"] = float(config.real_label_weight)
    return candidate_df, all_real


def _run_with_config(config: TargetClusterTrustTransferConfig) -> Path:
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
    (
        target_train_df,
        y_train,
        target_test_df,
        y_test,
        unlabeled_df,
        teachers,
        student_templates,
        transport_manifest,
    ) = _teacher_frames_for_target(
        target=config.target,
        split=split,
        prepared=prepared,
        base=base,
        config=pseudo_base,
        unit_drop_cols=unit_drop_cols,
        context_drop_cols=context_drop_cols,
    )
    teacher_train_df, teacher_unlabeled_df, teacher_test_df, teacher_perf_df = _teacher_predictions(
        teachers=teachers,
        labeled_df=target_train_df,
        y_train=y_train,
        base=base,
    )
    teacher_perf_df["target"] = config.target

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

    has_real_unlabeled_target = bool(pd.to_numeric(unlabeled_df[config.target], errors="coerce").notna().all())
    if has_real_unlabeled_target:
        candidate_df, baseline_selected = _build_real_candidate_pool(
            target=config.target,
            unlabeled_df=unlabeled_df,
            teacher_unlabeled_df=teacher_unlabeled_df,
            support_df=support_df,
            config=config,
        )
        trust_selected = _apply_trust_selection(
            candidate_df=candidate_df,
            cluster_trust_df=cluster_trust_df,
            study_trust_df=study_trust_df,
            max_rows=int(round(len(target_train_df) * float(config.max_pseudo_multiplier))),
            config=config,
        )
        if not trust_selected.empty:
            trust_selected["pseudo_weight"] = float(config.real_label_weight) * np.clip(
                trust_selected["pseudo_weight"].astype(float) / max(float(trust_selected["pseudo_weight"].max()), 1e-6),
                0.35,
                1.0,
            )
    else:
        candidate_df, baseline_selected = _select_pseudo_labels(
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
            config=config,
        )

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []

    reference_variant_id = _teacher_reference_variant(config.target, teacher_perf_df, base)

    for teacher in teachers:
        pred = teacher_test_df[teacher.variant_id].to_numpy(dtype=float)
        metrics = score_predictions(y_test, pred)
        metrics_rows.append(
            make_metric_row(
                target=config.target,
                variant_id=teacher.variant_id,
                family=teacher.family,
                backend=teacher.backend,
                source_target=teacher.source_target,
                metrics=metrics,
                notes=teacher.notes,
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target=config.target,
                variant_id=teacher.variant_id,
                family=teacher.family,
                backend=teacher.backend,
                df=target_test_df,
                y_true=y_test,
                pred=pred,
            )
        )
        teacher_cv = teacher_perf_df.loc[teacher_perf_df["variant_id"].astype(str) == str(teacher.variant_id)].iloc[0]
        cv_rows.append(
            {
                "target": config.target,
                "variant_id": str(teacher.variant_id),
                "family": teacher.family,
                "cv_mae": float(teacher_cv["oof_mae"]),
                "cv_r2": float(teacher_cv["oof_r2"]),
                "selected_rows": 0,
                "mean_selected_weight": 0.0,
            }
        )

    for _, row in student_templates.iterrows():
        base_variant_id = str(row["variant_id"])
        if has_real_unlabeled_target:
            variants = [
                (f"{base_variant_id}_all_real", baseline_selected, f"{row['notes']} + all real non-test paired labels"),
                (f"{base_variant_id}_cluster_trust_real", trust_selected, f"{row['notes']} + study/cluster trust-selected real paired labels"),
            ]
        else:
            variants = [
                (base_variant_id, baseline_selected, str(row["notes"])),
                (base_variant_id.replace("_pseudo_", "_cluster_trust_pseudo_"), trust_selected, f"{row['notes']} + study/cluster trust-aware pseudo selection"),
            ]
        for variant_id, selected_df, notes in variants:
            cv_metrics = _score_student_cv(
                labeled_meta_df=target_train_df,
                labeled_frame=row["train_frame"],
                labeled_y=y_train,
                pseudo_selected=selected_df,
                unlabeled_frame=row["unlabeled_frame"],
                base=base,
                random_state_offset=5000 + abs(hash((config.target, variant_id))) % 2000,
            )
            holdout_pred = _fit_student_holdout(
                labeled_frame=row["train_frame"],
                labeled_y=y_train,
                pseudo_selected=selected_df,
                unlabeled_frame=row["unlabeled_frame"],
                test_frame=row["test_frame"],
                base=base,
                random_state_offset=9000 + abs(hash((config.target, variant_id))) % 2000,
            )
            holdout_metrics = score_predictions(y_test, holdout_pred)
            metrics_rows.append(
                make_metric_row(
                    target=config.target,
                    variant_id=variant_id,
                    family="real_label_student" if has_real_unlabeled_target else "pseudo_label_student",
                    backend="lightgbm",
                    source_target=base.source_fmiss_target if config.target == "fmiss" else config.target,
                    metrics=holdout_metrics,
                    notes=notes,
                )
            )
            prediction_frames.append(
                make_prediction_frame(
                    target=config.target,
                    variant_id=variant_id,
                    family="real_label_student" if has_real_unlabeled_target else "pseudo_label_student",
                    backend="lightgbm",
                    df=target_test_df,
                    y_true=y_test,
                    pred=holdout_pred,
                )
            )
            cv_rows.append(
                {
                    "target": config.target,
                    "variant_id": variant_id,
                    "family": "real_label_student" if has_real_unlabeled_target else "pseudo_label_student",
                    "cv_mae": float(cv_metrics["mae"]),
                    "cv_r2": float(cv_metrics["r2"]),
                    "selected_rows": int(len(selected_df)),
                    "mean_selected_weight": float(selected_df["pseudo_weight"].mean()) if not selected_df.empty else 0.0,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows)
    cv_for_picker = cv_df.rename(columns={"cv_mae": "mae", "cv_r2": "r2"})
    recommendation = pick_r2_first_recommendation(
        cv_for_picker,
        target=config.target,
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
                "target": config.target,
                "best_holdout_variant_id": str(metrics_df.iloc[0]["variant_id"]),
                "best_holdout_mae": float(metrics_df.iloc[0]["mae"]),
                "best_holdout_r2": float(metrics_df.iloc[0]["r2"]),
            }
        ]
    )
    selection_summary_df = pd.DataFrame(
        [
            {
                "target": config.target,
                "selection_mode": "real_label_augmentation" if has_real_unlabeled_target else "pseudo_label_augmentation",
                "baseline_selected_rows": int(len(baseline_selected)),
                "trust_selected_rows": int(len(trust_selected)),
                "trust_selected_recordings": int(trust_selected["recording_key"].astype(str).nunique()) if not trust_selected.empty else 0,
                "trust_mean_weight": float(trust_selected["pseudo_weight"].mean()) if not trust_selected.empty else 0.0,
            }
        ]
    )

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    recommendation_df.to_csv(output_dir / "per_target_recommendation.csv", index=False)
    holdout_best_df.to_csv(output_dir / "holdout_best_by_target.csv", index=False)
    selection_summary_df.to_csv(output_dir / "selection_summary.csv", index=False)
    candidate_df.to_csv(output_dir / "candidate_table.csv", index=False)
    baseline_selected.to_csv(output_dir / "selected_baseline.csv", index=False)
    trust_selected.to_csv(output_dir / "selected_trust.csv", index=False)
    cluster_trust_df.to_csv(output_dir / "cluster_trust_summary.csv", index=False)
    study_trust_df.to_csv(output_dir / "study_trust_summary.csv", index=False)
    filtered_manifest.to_csv(output_dir / "filtered_feature_manifest.csv", index=False)
    transport_manifest.to_csv(output_dir / "transport_shift_summary.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split.manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: TargetClusterTrustTransferConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = TargetClusterTrustTransferConfig(
        target=args.target,
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
