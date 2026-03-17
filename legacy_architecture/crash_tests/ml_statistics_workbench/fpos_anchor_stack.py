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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.ml_statistics_workbench.config import MLStatisticsWorkbenchConfig
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
class FPosAnchorStackConfig:
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
    anchor_max_depth: int = 4
    anchor_learning_rate: float = 0.06
    anchor_max_iter: int = 250
    selection_metric: str = "r2"
    mae_guardrail_pct: float = 0.03
    min_r2_improvement: float = 0.0
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_anchor_stack_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fpos anchor-stack benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosAnchorStackConfig, message: str) -> None:
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


def _fit_anchor_scores(
    *,
    matched_train_df: pd.DataFrame,
    matched_test_df: pd.DataFrame,
    unmatched_df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int,
    config: FPosAnchorStackConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_groups = matched_train_df["recording_key"].astype(str).to_numpy()
    full_feature_source = pd.concat(
        [
            matched_train_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce"),
            unmatched_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce"),
        ],
        ignore_index=True,
    )
    imputer = SimpleImputer(strategy="median")
    full_feature_source = pd.DataFrame(imputer.fit_transform(full_feature_source), columns=feature_cols)
    matched_train_x = full_feature_source.iloc[: len(matched_train_df)].reset_index(drop=True)
    unmatched_x = full_feature_source.iloc[len(matched_train_df) :].reset_index(drop=True)
    matched_test_x = pd.DataFrame(
        imputer.transform(matched_test_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")),
        columns=feature_cols,
    )

    oof = np.full(len(matched_train_df), np.nan, dtype=float)
    for fold_id, (train_idx, val_idx) in enumerate(_grouped_oof_splits(matched_train_df, n_splits), start=1):
        val_recordings = set(matched_train_df.iloc[val_idx]["recording_key"].astype(str))
        fold_unmatched = unmatched_df[~unmatched_df["recording_key"].astype(str).isin(val_recordings)].copy()
        fold_unmatched_x = pd.DataFrame(
            imputer.transform(fold_unmatched.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")),
            columns=feature_cols,
        )
        X_train = pd.concat(
            [
                matched_train_x.iloc[train_idx].reset_index(drop=True),
                fold_unmatched_x.reset_index(drop=True),
            ],
            ignore_index=True,
        )
        y_train = np.concatenate(
            [
                np.zeros(len(train_idx), dtype=int),
                np.ones(len(fold_unmatched_x), dtype=int),
            ]
        )
        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=config.anchor_learning_rate,
            max_depth=config.anchor_max_depth,
            max_iter=config.anchor_max_iter,
            random_state=config.random_state + fold_id,
        )
        clf.fit(X_train, y_train)
        oof[val_idx] = clf.predict_proba(matched_train_x.iloc[val_idx])[:, 1]

    X_full = pd.concat([matched_train_x, unmatched_x], ignore_index=True)
    y_full = np.concatenate(
        [
            np.zeros(len(matched_train_x), dtype=int),
            np.ones(len(unmatched_x), dtype=int),
        ]
    )
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=config.anchor_learning_rate,
        max_depth=config.anchor_max_depth,
        max_iter=config.anchor_max_iter,
        random_state=config.random_state + 999,
    )
    clf.fit(X_full, y_full)
    unmatched_pred = clf.predict_proba(unmatched_x)[:, 1]
    test_pred = clf.predict_proba(matched_test_x)[:, 1]
    return np.clip(oof, 0.0, 1.0), np.clip(unmatched_pred, 0.0, 1.0), np.clip(test_pred, 0.0, 1.0)


def _run_with_config(config: FPosAnchorStackConfig) -> Path:
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
        config=config,
    )
    matched_support_conf_train = support_train_df["support_confidence"].to_numpy(dtype=float)
    matched_support_conf_unlabeled = support_df["support_confidence"].to_numpy(dtype=float)
    matched_support_conf_test = _manifold_support_table(
        unlabeled_df=target_test_df,
        labeled_train_df=target_train_df,
        support_artifacts=support_artifacts,
    )["support_confidence"].to_numpy(dtype=float)

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []

    base_row = student_templates.loc[
        student_templates["variant_id"].astype(str) == "source_reweighted_pseudo_lightgbm_filtered"
    ].iloc[0]
    base_train = base_row["train_frame"]
    base_unlabeled = base_row["unlabeled_frame"]
    base_test = base_row["test_frame"]

    anchor_train = build_stack_frame(
        target_train_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_train_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_oof,
            "matched_support_conf": matched_support_conf_train,
            "anchor_support_margin": anchor_oof - matched_support_conf_train,
        },
    )
    anchor_unlabeled_frame = build_stack_frame(
        unlabeled_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_unlabeled_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_unlabeled,
            "matched_support_conf": matched_support_conf_unlabeled,
            "anchor_support_margin": anchor_unlabeled - matched_support_conf_unlabeled,
        },
    )
    anchor_test_frame = build_stack_frame(
        target_test_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_test_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_test,
            "matched_support_conf": matched_support_conf_test,
            "anchor_support_margin": anchor_test - matched_support_conf_test,
        },
    )

    reference_variant_id = "source_reweighted_cluster_trust_pseudo_lightgbm_filtered"

    baseline_cv = _score_student_cv(
        labeled_meta_df=target_train_df,
        labeled_frame=base_train,
        labeled_y=y_train,
        pseudo_selected=trust_selected,
        unlabeled_frame=base_unlabeled,
        base=base,
        random_state_offset=6100,
    )
    baseline_holdout = _fit_student_holdout(
        labeled_frame=base_train,
        labeled_y=y_train,
        pseudo_selected=trust_selected,
        unlabeled_frame=base_unlabeled,
        test_frame=base_test,
        base=base,
        random_state_offset=9100,
    )
    anchor_cv = _score_student_cv(
        labeled_meta_df=target_train_df,
        labeled_frame=anchor_train,
        labeled_y=y_train,
        pseudo_selected=trust_selected,
        unlabeled_frame=anchor_unlabeled_frame,
        base=base,
        random_state_offset=6200,
    )
    anchor_holdout = _fit_student_holdout(
        labeled_frame=anchor_train,
        labeled_y=y_train,
        pseudo_selected=trust_selected,
        unlabeled_frame=anchor_unlabeled_frame,
        test_frame=anchor_test_frame,
        base=base,
        random_state_offset=9200,
    )

    for variant_id, family, pred, notes, cv_metrics in [
        (
            reference_variant_id,
            "pseudo_label_student",
            baseline_holdout,
            "filtered context + raw weighted hybrid source prediction student with study/cluster trust-aware pseudo selection",
            baseline_cv,
        ),
        (
            "source_reweighted_anchor_stack_cluster_trust_pseudo_lightgbm_filtered",
            "anchor_stack_student",
            anchor_holdout,
            "trust-aware pseudo student + anchor false-positive regime score + support margin",
            anchor_cv,
        ),
    ]:
        metrics = score_predictions(y_test, pred)
        metrics_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family=family,
                backend="lightgbm",
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
                backend="lightgbm",
                df=target_test_df,
                y_true=y_test,
                pred=pred,
            )
        )
        cv_rows.append(
            {
                "target": "fpos",
                "variant_id": variant_id,
                "family": family,
                "cv_mae": float(cv_metrics["mae"]),
                "cv_r2": float(cv_metrics["r2"]),
                "selected_rows": int(len(trust_selected)),
                "mean_selected_weight": float(trust_selected["pseudo_weight"].mean()) if not trust_selected.empty else 0.0,
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
    anchor_summary_df = pd.DataFrame(
        [
            {
                "target": "fpos",
                "mean_anchor_score_train": float(np.mean(anchor_oof)),
                "mean_anchor_score_unlabeled": float(np.mean(anchor_unlabeled)),
                "mean_anchor_score_test": float(np.mean(anchor_test)),
                "selected_rows": int(len(trust_selected)),
            }
        ]
    )

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    recommendation_df.to_csv(output_dir / "per_target_recommendation.csv", index=False)
    holdout_best_df.to_csv(output_dir / "holdout_best_by_target.csv", index=False)
    anchor_summary_df.to_csv(output_dir / "anchor_summary.csv", index=False)
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


def run_with_config(config: FPosAnchorStackConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosAnchorStackConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.anchor_max_iter = 120
        cli_config.max_pseudo_multiplier = 1.0
        cli_config.max_pseudo_per_recording = 24
        cli_config.max_pseudo_per_cluster = 64
    _run_with_config(cli_config)
