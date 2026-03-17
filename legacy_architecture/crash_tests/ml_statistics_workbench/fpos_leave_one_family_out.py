from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.ml_statistics_workbench.config import MLStatisticsWorkbenchConfig
from crash_tests.ml_statistics_workbench.fpos_anchor_stack import _fit_anchor_scores
from crash_tests.ml_statistics_workbench.fpos_backend_sweep import _fit_backend_predict
from crash_tests.ml_statistics_workbench.fpos_cluster_trust_pseudo import (
    FPosClusterTrustPseudoConfig,
    _apply_trust_selection,
    _trust_from_group_stats,
)
from crash_tests.ml_statistics_workbench.fpos_family_robustness import _family_weights
from crash_tests.ml_statistics_workbench.fpos_signal_embedding_stack import (
    FPosSignalEmbeddingStackConfig,
    _signal_embeddings,
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
from crash_tests.self_supervised_domain_adaptation.split_utils import build_stress_splits


@dataclass
class FPosLeaveOneFamilyOutConfig(FPosSignalEmbeddingStackConfig):
    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_leave_one_family_out_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leave-one-family-out benchmark for best fpos model")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosLeaveOneFamilyOutConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _assemble_training_data(
    *,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    labeled_meta_df: pd.DataFrame,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    family_alpha: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    labeled_weight = _family_weights(labeled_meta_df["study_set"], alpha=family_alpha)
    if pseudo_selected.empty:
        return labeled_frame.reset_index(drop=True), np.asarray(labeled_y, dtype=float), labeled_weight
    pseudo_idx = pseudo_selected["unlabeled_row_idx"].to_numpy(dtype=int)
    pseudo_frame = unlabeled_frame.iloc[pseudo_idx].reset_index(drop=True)
    full_frame = pd.concat([labeled_frame.reset_index(drop=True), pseudo_frame], ignore_index=True)
    full_y = np.concatenate([np.asarray(labeled_y, dtype=float), pseudo_selected["pseudo_label"].to_numpy(dtype=float)])
    full_weight = np.concatenate([labeled_weight, pseudo_selected["pseudo_weight"].to_numpy(dtype=np.float32)])
    return full_frame, full_y, full_weight


def _fit_holdout(
    *,
    labeled_meta_df: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: FPosLeaveOneFamilyOutConfig,
    random_state_offset: int,
) -> np.ndarray:
    full_train, full_y, full_weight = _assemble_training_data(
        labeled_frame=labeled_frame,
        labeled_y=labeled_y,
        labeled_meta_df=labeled_meta_df,
        pseudo_selected=pseudo_selected,
        unlabeled_frame=unlabeled_frame,
        family_alpha=1.0,
    )
    return _fit_backend_predict(
        X_train=full_train,
        y_train=full_y,
        X_eval=test_frame.reset_index(drop=True),
        sample_weight=full_weight,
        target_transform_mode="logit_clip",
        backend_id="xgboost_default",
        random_state=config.random_state + random_state_offset,
    )


def _prepare_family_frames(
    *,
    prepared: Any,
    split: Any,
    config: FPosLeaveOneFamilyOutConfig,
) -> dict[str, Any]:
    base = MLStatisticsWorkbenchConfig(
        parquet_path=config.parquet_path,
        output_root=config.output_root,
        run_name=config.run_name,
        random_state=config.random_state,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        verbose=False,
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
        verbose=False,
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
        verbose=False,
    )

    unit_drop_cols, context_drop_cols, _ = _build_filtered_drop_cols(split=split, prepared=prepared, base_config=base)
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
    support_df = _manifold_support_table(unlabeled_df=unlabeled_df, labeled_train_df=target_train_df, support_artifacts=support_artifacts)
    support_train_df = _manifold_support_table(unlabeled_df=target_train_df, labeled_train_df=target_train_df, support_artifacts=support_artifacts)
    train_trust_df = pd.concat(
        [
            target_train_df.loc[:, ["row_uid", "recording_key", "study_set"]].reset_index(drop=True),
            support_train_df.loc[:, ["cluster_id", "support_confidence"]].reset_index(drop=True),
        ],
        axis=1,
    )
    train_trust_df["teacher_abs_error"] = np.abs(y_train - teacher_train_df["teacher_mean"].to_numpy(dtype=float))
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
        n_splits=3,
        config=type(
            "AnchorCfg",
            (),
            {"anchor_learning_rate": 0.06, "anchor_max_depth": 4, "anchor_max_iter": 250, "random_state": config.random_state},
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
    return {
        "target_train_df": target_train_df,
        "target_test_df": target_test_df,
        "unlabeled_df": unlabeled_df,
        "y_train": y_train,
        "y_test": y_test,
        "trust_selected": trust_selected,
        "train_frame": train_frame,
        "unlabeled_frame": unlabeled_frame,
        "test_frame": test_frame,
    }


def _run_with_config(config: FPosLeaveOneFamilyOutConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_feature_table(config.parquet_path)
    stress_splits = build_stress_splits(prepared.df)
    if config.universe_max_rows is not None:
        config.universe_max_rows = int(config.universe_max_rows)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    family_rows: list[dict[str, Any]] = []

    for split_idx, split in enumerate(stress_splits, start=1):
        held_out = split.held_out_study
        _log(config, f"[family] held-out {held_out}")
        state = _prepare_family_frames(prepared=prepared, split=split, config=config)
        target_train_df = state["target_train_df"]
        target_test_df = state["target_test_df"]
        unlabeled_df = state["unlabeled_df"]
        y_train = state["y_train"]
        y_test = state["y_test"]
        trust_selected = state["trust_selected"]
        train_frame = state["train_frame"]
        unlabeled_frame = state["unlabeled_frame"]
        test_frame = state["test_frame"]

        universe_df = pd.concat(
            [
                split.hybrid_source.reset_index(drop=True),
                split.paired_non_test_all.reset_index(drop=True),
                split.ssl_pool.reset_index(drop=True),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["row_uid"]).reset_index(drop=True)
        if config.universe_max_rows is not None and len(universe_df) > config.universe_max_rows:
            universe_df = universe_df.sample(n=config.universe_max_rows, random_state=config.random_state).reset_index(drop=True)

        wf_train_emb, wf_unlabeled_emb, wf_test_emb, wf_hist = _signal_embeddings(
            train_signal=target_train_df.loc[:, prepared.wf_feature_cols],
            train_universe_signal=universe_df.loc[:, prepared.wf_feature_cols],
            unlabeled_signal=unlabeled_df.loc[:, prepared.wf_feature_cols],
            test_signal=target_test_df.loc[:, prepared.wf_feature_cols],
            prefix="wf",
            config=config,
        )

        variants = [
            ("reference_anchor_stack_xgboost", train_frame, unlabeled_frame, test_frame),
            (
                "wf_embed_anchor_stack_xgboost",
                pd.concat([train_frame.reset_index(drop=True), wf_train_emb], axis=1),
                pd.concat([unlabeled_frame.reset_index(drop=True), wf_unlabeled_emb], axis=1),
                pd.concat([test_frame.reset_index(drop=True), wf_test_emb], axis=1),
            ),
        ]

        for variant_offset, (variant_id, train_variant, unlabeled_variant, test_variant) in enumerate(variants, start=1):
            pred = _fit_holdout(
                labeled_meta_df=target_train_df,
                labeled_frame=train_variant,
                labeled_y=y_train,
                pseudo_selected=trust_selected,
                unlabeled_frame=unlabeled_variant,
                test_frame=test_variant,
                config=config,
                random_state_offset=(1000 * split_idx) + (100 * variant_offset),
            )
            metrics = score_predictions(y_test, pred)
            metric_rows.append(
                make_metric_row(
                    target="fpos",
                    variant_id=variant_id,
                    family="leave_one_family_out",
                    backend="xgboost_default",
                    source_target="fpos",
                    metrics=metrics,
                    notes=f"held-out family {held_out}",
                )
                | {"held_out_family": held_out, "n_test_rows": int(len(y_test)), "n_test_recordings": int(target_test_df['recording_key'].nunique()), "pseudo_rows": int(len(trust_selected)), "embed_recon_loss": float(wf_hist["train_recon_loss"])}
            )
            prediction_frames.append(
                make_prediction_frame(
                    target="fpos",
                    variant_id=variant_id,
                    family="leave_one_family_out",
                    backend="xgboost_default",
                    df=target_test_df,
                    y_true=y_test,
                    pred=pred,
                ).assign(held_out_family=held_out)
            )
            family_rows.append(
                {
                    "held_out_family": held_out,
                    "variant_id": variant_id,
                    "mae": float(metrics["mae"]),
                    "r2": float(metrics["r2"]),
                    "bias": float(metrics["bias"]),
                    "n_test_rows": int(len(y_test)),
                    "n_test_recordings": int(target_test_df["recording_key"].nunique()),
                    "pseudo_rows": int(len(trust_selected)),
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["variant_id", "held_out_family"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    family_df = pd.DataFrame(family_rows).sort_values(["variant_id", "held_out_family"]).reset_index(drop=True)
    aggregate_df = family_df.groupby("variant_id", dropna=False).apply(
        lambda g: pd.Series(
            {
                "macro_mae": float(g["mae"].mean()),
                "macro_r2": float(g["r2"].mean()),
                "weighted_mae": float(np.average(g["mae"], weights=g["n_test_rows"])),
                "weighted_r2": float(np.average(g["r2"], weights=g["n_test_rows"])),
                "worst_family_r2": float(g["r2"].min()),
                "best_family_r2": float(g["r2"].max()),
            }
        )
    ).reset_index()

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_df = family_df.pivot(index="held_out_family", columns="variant_id", values="r2").reset_index()
    ax = plot_df.plot(
        x="held_out_family",
        y=[c for c in plot_df.columns if c != "held_out_family"],
        kind="bar",
        figsize=(10, 5),
    )
    ax.set_ylabel("Holdout R²")
    ax.set_title("Leave-One-Family-Out fpos Performance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plot_path = output_dir / "leave_one_family_r2.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    family_df.to_csv(output_dir / "family_summary.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_summary.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config) | {"parquet_path": str(config.parquet_path), "output_root": str(config.output_root)}, f, indent=2)
    with open(output_dir / "stress_manifest.json", "w") as f:
        json.dump([split.manifest for split in stress_splits], f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosLeaveOneFamilyOutConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosLeaveOneFamilyOutConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.max_pseudo_multiplier = 1.0
        cli_config.max_pseudo_per_recording = 24
        cli_config.max_pseudo_per_cluster = 64
        cli_config.ae_epochs = 4
        cli_config.conv_embed_dim = 6
        cli_config.universe_max_rows = 4000
    _run_with_config(cli_config)
