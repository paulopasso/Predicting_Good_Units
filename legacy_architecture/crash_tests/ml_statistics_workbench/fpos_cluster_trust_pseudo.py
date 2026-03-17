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
class FPosClusterTrustPseudoConfig:
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
    selection_metric: str = "r2"
    mae_guardrail_pct: float = 0.03
    min_r2_improvement: float = 0.0
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_cluster_trust_pseudo_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fpos cluster-trust pseudo-label benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosClusterTrustPseudoConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _trust_from_group_stats(
    group_df: pd.DataFrame,
    *,
    group_col: str,
) -> pd.DataFrame:
    overall_mae = float(group_df["teacher_abs_error"].mean())
    max_n = max(int(group_df[group_col].value_counts().max()), 1)
    stats = (
        group_df.groupby(group_col, dropna=False)
        .agg(
            labeled_rows=("row_uid", "size"),
            teacher_mae=("teacher_abs_error", "mean"),
            mean_support_confidence=("support_confidence", "mean"),
        )
        .reset_index()
    )
    err_score = np.exp(-stats["teacher_mae"].astype(float) / max(overall_mae, 1e-6))
    size_score = np.sqrt(stats["labeled_rows"].astype(float) / float(max_n))
    support_score = np.clip(stats["mean_support_confidence"].astype(float), 0.0, 1.0)
    trust = 0.60 * err_score + 0.25 * support_score + 0.15 * size_score
    trust = trust / max(float(np.nanmax(trust)), 1e-6)
    stats["trust_score"] = trust.astype(float)
    return stats


def _apply_trust_selection(
    *,
    candidate_df: pd.DataFrame,
    cluster_trust_df: pd.DataFrame,
    study_trust_df: pd.DataFrame,
    max_rows: int,
    config: FPosClusterTrustPseudoConfig,
) -> pd.DataFrame:
    pool = candidate_df[candidate_df["selected_candidate"].astype(bool)].copy()
    if pool.empty:
        return pool

    pool = pool.merge(
        cluster_trust_df.loc[:, ["cluster_id", "trust_score"]].rename(columns={"trust_score": "cluster_trust"}),
        on="cluster_id",
        how="left",
    )
    pool = pool.merge(
        study_trust_df.loc[:, ["study_set", "trust_score"]].rename(columns={"trust_score": "study_trust"}),
        on="study_set",
        how="left",
    )
    pool["cluster_trust"] = pool["cluster_trust"].fillna(float(config.min_trust))
    pool["study_trust"] = pool["study_trust"].fillna(float(config.min_trust))
    pool["combined_trust"] = (
        ((1.0 - float(config.study_trust_weight)) * pool["cluster_trust"].astype(float))
        + (float(config.study_trust_weight) * pool["study_trust"].astype(float))
    ).clip(lower=float(config.min_trust), upper=1.0)
    std_denom = max(float(np.nanquantile(pool["teacher_std"].astype(float), 0.75)), 1e-6)
    base_conf = (
        0.65 * np.exp(-pool["teacher_std"].astype(float) / std_denom)
        + 0.35 * np.clip(pool["support_confidence"].astype(float), 0.0, 1.0)
    )
    pool["base_pseudo_weight"] = np.clip(0.25 * base_conf, 0.08, 0.25)
    pool["trust_adjusted_weight"] = (
        pool["base_pseudo_weight"].astype(float)
        * np.power(pool["combined_trust"].astype(float), float(config.cluster_trust_power))
    ).clip(lower=0.08, upper=0.25)
    cluster_caps = (
        pool.groupby("cluster_id")["combined_trust"]
        .mean()
        .map(
            lambda trust: max(
                1,
                min(
                    int(config.max_pseudo_per_cluster),
                    int(
                        round(
                            min(
                                float(config.max_pseudo_per_cluster),
                                float(config.max_pseudo_per_cluster) * (max(float(trust), float(config.min_trust)) ** float(config.cluster_trust_power)),
                            )
                        )
                    ),
                ),
            )
        )
        .to_dict()
    )
    cluster_ratio_caps = (
        pool.groupby("cluster_id")["cluster_labeled_count"]
        .max()
        .map(
            lambda labeled_count: max(
                1,
                int(round(float(labeled_count) * float(config.max_pseudo_per_labeled_ratio))),
            )
        )
        .to_dict()
    )

    selected_rows: list[pd.Series] = []
    recording_counts: dict[str, int] = {}
    cluster_counts: dict[int, int] = {}
    pool = pool.sort_values(
        ["trust_adjusted_weight", "combined_trust", "support_confidence", "base_pseudo_weight", "teacher_std"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    for _, row in pool.iterrows():
        if len(selected_rows) >= int(max_rows):
            break
        recording_key = str(row["recording_key"])
        cluster_id = int(row["cluster_id"])
        if recording_counts.get(recording_key, 0) >= int(config.max_pseudo_per_recording):
            continue
        cluster_cap = min(cluster_caps.get(cluster_id, int(config.max_pseudo_per_cluster)), cluster_ratio_caps.get(cluster_id, int(config.max_pseudo_per_cluster)))
        if cluster_counts.get(cluster_id, 0) >= int(cluster_cap):
            continue
        selected_rows.append(row)
        recording_counts[recording_key] = recording_counts.get(recording_key, 0) + 1
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    if not selected_rows:
        return pool.head(0).copy()

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    if "pseudo_label" not in selected_df.columns:
        selected_df["pseudo_label"] = selected_df["teacher_mean"].astype(float)
    selected_df["pseudo_weight"] = selected_df["trust_adjusted_weight"].astype(np.float32)
    return selected_df


def _build_readme(
    *,
    config: FPosClusterTrustPseudoConfig,
    metrics_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    pseudo_summary_df: pd.DataFrame,
) -> str:
    lines = [
        "# FPos Cluster-Trust Pseudo Benchmark",
        "",
        "This run keeps the strong filtered-context/source student, but changes pseudo-label selection to trust only the subgroup regions where the paired teacher is locally reliable.",
        "",
        "## Config",
        f"- max_pseudo_multiplier: `{config.max_pseudo_multiplier}`",
        f"- max_pseudo_per_recording: `{config.max_pseudo_per_recording}`",
        f"- cluster_trust_power: `{config.cluster_trust_power}`",
        f"- study_trust_weight: `{config.study_trust_weight}`",
        f"- min_trust: `{config.min_trust}`",
        "",
        "## Holdout Results",
        "",
    ]
    for _, row in metrics_df.sort_values(["r2", "mae"], ascending=[False, True]).iterrows():
        lines.append(f"- {row['variant_id']} | MAE `{row['mae']:.4f}` | R² `{row['r2']:.4f}`")
    lines.append("")
    if not pseudo_summary_df.empty:
        row = pseudo_summary_df.iloc[0]
        lines.extend(
            [
                "## Pseudo Selection",
                f"- baseline selected pseudo rows: `{int(row['baseline_selected_rows'])}`",
                f"- trust-selected pseudo rows: `{int(row['trust_selected_rows'])}`",
                f"- trust-selected recordings: `{int(row['trust_selected_recordings'])}`",
                f"- mean trust-adjusted weight: `{row['trust_mean_weight']:.4f}`",
                "",
            ]
        )
    if not recommendation_df.empty:
        row = recommendation_df.iloc[0]
        lines.extend(
            [
                "## Recommendation",
                f"- reference CV variant: `{row['reference_variant_id']}`",
                f"- recommended CV variant: `{row['recommended_variant_id']}`",
                f"- recommended holdout MAE: `{row['recommended_holdout_mae']:.4f}`",
                f"- recommended holdout R²: `{row['recommended_holdout_r2']:.4f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _run_with_config(config: FPosClusterTrustPseudoConfig) -> Path:
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
        target="fpos",
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
    teacher_perf_df["target"] = "fpos"

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

    candidate_df, baseline_selected = _select_pseudo_labels(
        unlabeled_df=unlabeled_df,
        unlabeled_teacher_df=teacher_unlabeled_df,
        support_df=support_df,
        config=pseudo_base,
        max_rows=int(round(len(target_train_df) * float(config.max_pseudo_multiplier))),
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

    source_row = student_templates.loc[
        student_templates["variant_id"].astype(str) == "source_reweighted_pseudo_lightgbm_filtered"
    ].iloc[0]
    source_train = source_row["train_frame"]
    source_unlabeled = source_row["unlabeled_frame"]
    source_test = source_row["test_frame"]

    def _record(
        variant_id: str,
        family: str,
        pred: np.ndarray,
        notes: str,
    ) -> None:
        metrics = score_predictions(y_test, pred)
        metrics_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family=family,
                backend="lightgbm" if family != "teacher_ensemble" else "ensemble",
                source_target="fpos",
                metrics=metrics,
                notes=notes,
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family=family,
                backend="lightgbm" if family != "teacher_ensemble" else "ensemble",
                df=target_test_df,
                y_true=y_test,
                pred=pred,
            )
        )

    reference_variant_id = "teacher_paired_conditioned_global_fullctx_lightgbm_filtered"
    reference_cv = teacher_perf_df.loc[
        teacher_perf_df["variant_id"].astype(str) == reference_variant_id
    ].iloc[0]
    cv_rows.append(
        {
            "target": "fpos",
            "variant_id": reference_variant_id,
            "family": "transport_teacher",
            "cv_mae": float(reference_cv["oof_mae"]),
            "cv_r2": float(reference_cv["oof_r2"]),
            "selected_pseudo_rows": 0,
            "mean_pseudo_weight": 0.0,
        }
    )

    _record(
        reference_variant_id,
        "transport_teacher",
        teacher_test_df[reference_variant_id].to_numpy(dtype=float),
        "filtered context + globally transported hybrid source prediction",
    )

    for variant_id, selected_df, notes in [
        (
            "source_reweighted_pseudo_lightgbm_filtered",
            baseline_selected,
            "baseline pseudo-label student with global pseudo selection",
        ),
        (
            "source_reweighted_cluster_trust_pseudo_lightgbm_filtered",
            trust_selected,
            "source pseudo student with study+cluster trust-aware pseudo selection",
        ),
    ]:
        cv_metrics = _score_student_cv(
            labeled_meta_df=target_train_df,
            labeled_frame=source_train,
            labeled_y=y_train,
            pseudo_selected=selected_df,
            unlabeled_frame=source_unlabeled,
            base=base,
            random_state_offset=6100 if "cluster_trust" not in variant_id else 7100,
        )
        holdout_pred = _fit_student_holdout(
            labeled_frame=source_train,
            labeled_y=y_train,
            pseudo_selected=selected_df,
            unlabeled_frame=source_unlabeled,
            test_frame=source_test,
            base=base,
            random_state_offset=9100 if "cluster_trust" not in variant_id else 10100,
        )
        _record(variant_id, "pseudo_label_student", holdout_pred, notes)
        cv_rows.append(
            {
                "target": "fpos",
                "variant_id": variant_id,
                "family": "pseudo_label_student",
                "cv_mae": float(cv_metrics["mae"]),
                "cv_r2": float(cv_metrics["r2"]),
                "selected_pseudo_rows": int(len(selected_df)),
                "mean_pseudo_weight": float(selected_df["pseudo_weight"].mean()) if not selected_df.empty else 0.0,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows)
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
    pseudo_summary_df = pd.DataFrame(
        [
            {
                "target": "fpos",
                "baseline_selected_rows": int(len(baseline_selected)),
                "trust_selected_rows": int(len(trust_selected)),
                "trust_selected_recordings": int(trust_selected["recording_key"].astype(str).nunique()) if not trust_selected.empty else 0,
                "trust_mean_weight": float(trust_selected["pseudo_weight"].mean()) if not trust_selected.empty else 0.0,
            }
        ]
    )
    cluster_summary_df = cluster_trust_df.copy()
    cluster_summary_df["selected_baseline_rows"] = cluster_summary_df["cluster_id"].map(
        baseline_selected["cluster_id"].value_counts().to_dict()
    ).fillna(0).astype(int)
    cluster_summary_df["selected_trust_rows"] = cluster_summary_df["cluster_id"].map(
        trust_selected["cluster_id"].value_counts().to_dict()
    ).fillna(0).astype(int)
    study_summary_df = study_trust_df.copy()
    study_summary_df["selected_baseline_rows"] = study_summary_df["study_set"].map(
        baseline_selected["study_set"].astype(str).value_counts().to_dict()
    ).fillna(0).astype(int)
    study_summary_df["selected_trust_rows"] = study_summary_df["study_set"].map(
        trust_selected["study_set"].astype(str).value_counts().to_dict()
    ).fillna(0).astype(int)

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    recommendation_df.to_csv(output_dir / "per_target_recommendation.csv", index=False)
    holdout_best_df.to_csv(output_dir / "holdout_best_by_target.csv", index=False)
    pseudo_summary_df.to_csv(output_dir / "pseudo_label_summary.csv", index=False)
    baseline_selected.to_csv(output_dir / "pseudo_selected_baseline.csv", index=False)
    trust_selected.to_csv(output_dir / "pseudo_selected_trust.csv", index=False)
    cluster_summary_df.to_csv(output_dir / "cluster_trust_summary.csv", index=False)
    study_summary_df.to_csv(output_dir / "study_trust_summary.csv", index=False)
    filtered_manifest.to_csv(output_dir / "filtered_feature_manifest.csv", index=False)
    transport_manifest.to_csv(output_dir / "transport_shift_summary.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split.manifest, f, indent=2)
    (output_dir / "README_run.md").write_text(
        _build_readme(
            config=config,
            metrics_df=metrics_df,
            recommendation_df=recommendation_df,
            pseudo_summary_df=pseudo_summary_df,
        )
    )
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosClusterTrustPseudoConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosClusterTrustPseudoConfig(
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
