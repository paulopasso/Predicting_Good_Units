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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qc_thesis.modeling.stacking.config import MLStatisticsWorkbenchConfig
from qc_thesis.modeling.stacking.fpos_anchor_stack import _fit_anchor_scores
from qc_thesis.modeling.stacking.fpos_backend_sweep import _fit_backend_holdout, _grouped_oof_splits
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
    _build_filtered_drop_cols,
    _build_support_artifacts,
    _manifold_support_table,
    _select_pseudo_labels,
    _teacher_frames_for_target,
    _teacher_predictions,
)
from qc_thesis.modeling.neural.mmd.dataset import prepare_feature_table
from qc_thesis.modeling.neural.ssl.split_utils import make_main_split


@dataclass
class FPosResidualCorrectionConfig:
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
        return f"fpos_residual_correction_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


TOP_FEATURE_CANDIDATES = [
    "source_pred",
    "amp_bimodality",
    "acg_26",
    "isi_violation_rate",
    "si_amplitude_cv_range",
    "n_waveform_confusable_recording_pct_recording_iqr",
    "acg_19",
    "si_isi_violations_count",
    "si_amplitude_cv_median",
    "amp_mean_recording_pct",
    "amp_p5_recording_pct",
    "si_amplitude_cv_median_recording_pct",
    "si_amplitude_cv_median_recording_delta",
    "presence_ratio",
    "amp_std_recording_pct",
    "isi_std_ms",
    "matched_support_conf",
    "anchor_fp_score",
    "anchor_support_margin",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run residual correction on top of the fpos winner")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosResidualCorrectionConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _prepare_frames(config: FPosResidualCorrectionConfig):
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
    matched_support_conf_test = support_test_df["support_confidence"].to_numpy(dtype=float)

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


def _winner_oof_and_holdout(
    *,
    base: MLStatisticsWorkbenchConfig,
    target_train_df: pd.DataFrame,
    y_train: np.ndarray,
    trust_selected: pd.DataFrame,
    train_frame: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    splits = _grouped_oof_splits(target_train_df, base.model_selection_splits)
    oof = np.full(len(target_train_df), np.nan, dtype=float)
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        val_recordings = set(target_train_df.iloc[val_idx]["recording_key"].astype(str))
        fold_pseudo = trust_selected[
            ~trust_selected["recording_key"].astype(str).isin(val_recordings)
        ].reset_index(drop=True)
        oof[val_idx] = _fit_backend_holdout(
            labeled_frame=train_frame.iloc[train_idx].reset_index(drop=True),
            labeled_y=y_train[train_idx],
            pseudo_selected=fold_pseudo,
            unlabeled_frame=unlabeled_frame,
            test_frame=train_frame.iloc[val_idx].reset_index(drop=True),
            backend_id="xgboost_default",
            base=base,
            random_state_offset=4200 + split_id,
        )
    holdout = _fit_backend_holdout(
        labeled_frame=train_frame,
        labeled_y=y_train,
        pseudo_selected=trust_selected,
        unlabeled_frame=unlabeled_frame,
        test_frame=test_frame,
        backend_id="xgboost_default",
        base=base,
        random_state_offset=4200,
    )
    return oof, holdout


def _build_residual_frame(
    anchor_frame: pd.DataFrame,
    *,
    base_pred: np.ndarray,
    target_df: pd.DataFrame,
    include_study: bool,
) -> pd.DataFrame:
    cols = [c for c in TOP_FEATURE_CANDIDATES if c in anchor_frame.columns]
    frame = anchor_frame.loc[:, cols].copy()
    frame["winner_pred"] = base_pred
    frame["winner_uncentered"] = base_pred - np.mean(base_pred)
    if include_study:
        study_df = pd.get_dummies(target_df["study_set"].astype(str), prefix="study").reset_index(drop=True)
        frame = pd.concat([frame.reset_index(drop=True), study_df], axis=1)
    return frame.reset_index(drop=True)


def _fit_residual_model(
    X_train: pd.DataFrame,
    residual_train: np.ndarray,
    model_id: str,
    random_state: int,
):
    if model_id == "ridge":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=2.0, random_state=random_state)),
            ]
        )
    elif model_id == "histgb":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=3,
                        max_iter=220,
                        min_samples_leaf=10,
                        l2_regularization=0.05,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    else:
        raise KeyError(model_id)
    model.fit(X_train, residual_train)
    return model


def _run_with_config(config: FPosResidualCorrectionConfig) -> Path:
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

    base_oof, base_holdout = _winner_oof_and_holdout(
        base=base,
        target_train_df=target_train_df,
        y_train=y_train,
        trust_selected=trust_selected,
        train_frame=train_frame,
        unlabeled_frame=unlabeled_frame,
        test_frame=test_frame,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []

    reference_cv = score_predictions(y_train, base_oof)
    reference_holdout = score_predictions(y_test, base_holdout)
    ref_variant = "source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default"
    metric_rows.append(
        make_metric_row(
            target="fpos",
            variant_id=ref_variant,
            family="residual_correction_reference",
            backend="xgboost_default",
            source_target="fpos",
            metrics=reference_holdout,
            notes="winner reference",
        )
    )
    prediction_frames.append(
        make_prediction_frame(
            target="fpos",
            variant_id=ref_variant,
            family="residual_correction_reference",
            backend="xgboost_default",
            df=target_test_df,
            y_true=y_test,
            pred=base_holdout,
        )
    )
    cv_rows.append(
        {
            "target": "fpos",
            "variant_id": ref_variant,
            "family": "residual_correction_reference",
            "cv_mae": float(reference_cv["mae"]),
            "cv_r2": float(reference_cv["r2"]),
            "selected_rows": int(len(trust_selected)),
            "mean_selected_weight": float(trust_selected["pseudo_weight"].mean()) if not trust_selected.empty else 0.0,
        }
    )

    variants = [
        ("global_ridge", False, "ridge"),
        ("global_histgb", False, "histgb"),
        ("study_ridge", True, "ridge"),
        ("study_histgb", True, "histgb"),
    ]

    splits = _grouped_oof_splits(target_train_df, base.model_selection_splits)
    for idx, (variant_name, include_study, model_id) in enumerate(variants, start=1):
        _log(config, f"[residual] {variant_name}")
        oof = np.full(len(target_train_df), np.nan, dtype=float)
        for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
            train_residual_frame = _build_residual_frame(
                train_frame.iloc[train_idx].reset_index(drop=True),
                base_pred=base_oof[train_idx],
                target_df=target_train_df.iloc[train_idx].reset_index(drop=True),
                include_study=include_study,
            )
            val_residual_frame = _build_residual_frame(
                train_frame.iloc[val_idx].reset_index(drop=True),
                base_pred=base_oof[val_idx],
                target_df=target_train_df.iloc[val_idx].reset_index(drop=True),
                include_study=include_study,
            ).reindex(columns=train_residual_frame.columns, fill_value=0.0)
            residual_train = y_train[train_idx] - base_oof[train_idx]
            model = _fit_residual_model(
                X_train=train_residual_frame,
                residual_train=residual_train,
                model_id=model_id,
                random_state=base.random_state + 5000 + (100 * idx) + split_id,
            )
            residual_pred = np.asarray(model.predict(val_residual_frame), dtype=float)
            oof[val_idx] = np.clip(base_oof[val_idx] + residual_pred, 0.0, 1.0)

        holdout_residual_frame = _build_residual_frame(
            test_frame.reset_index(drop=True),
            base_pred=base_holdout,
            target_df=target_test_df.reset_index(drop=True),
            include_study=include_study,
        )
        full_residual_frame = _build_residual_frame(
            train_frame.reset_index(drop=True),
            base_pred=base_oof,
            target_df=target_train_df.reset_index(drop=True),
            include_study=include_study,
        )
        holdout_residual_frame = holdout_residual_frame.reindex(columns=full_residual_frame.columns, fill_value=0.0)
        model = _fit_residual_model(
            X_train=full_residual_frame,
            residual_train=y_train - base_oof,
            model_id=model_id,
            random_state=base.random_state + 8000 + idx,
        )
        holdout_pred = np.clip(base_holdout + np.asarray(model.predict(holdout_residual_frame), dtype=float), 0.0, 1.0)
        cv_metrics = score_predictions(y_train, oof)
        holdout_metrics = score_predictions(y_test, holdout_pred)
        variant_id = f"source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default_plus_{variant_name}"
        metric_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family="residual_correction",
                backend=model_id,
                source_target="fpos",
                metrics=holdout_metrics,
                notes=f"winner + residual correction ({variant_name})",
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family="residual_correction",
                backend=model_id,
                df=target_test_df,
                y_true=y_test,
                pred=holdout_pred,
            )
        )
        cv_rows.append(
            {
                "target": "fpos",
                "variant_id": variant_id,
                "family": "residual_correction",
                "cv_mae": float(cv_metrics["mae"]),
                "cv_r2": float(cv_metrics["r2"]),
                "selected_rows": int(len(trust_selected)),
                "mean_selected_weight": float(trust_selected["pseudo_weight"].mean()) if not trust_selected.empty else 0.0,
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows)
    cv_for_picker = cv_df.rename(columns={"cv_mae": "mae", "cv_r2": "r2"})
    recommendation = pick_r2_first_recommendation(
        cv_for_picker,
        target="fpos",
        reference_variant_id=ref_variant,
        selection_metric=config.selection_metric,
        mae_guardrail_pct=config.mae_guardrail_pct,
        min_r2_improvement=config.min_r2_improvement,
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
    pd.DataFrame(
        [
            {
                "target": "fpos",
                "best_holdout_variant_id": str(metrics_df.iloc[0]["variant_id"]),
                "best_holdout_mae": float(metrics_df.iloc[0]["mae"]),
                "best_holdout_r2": float(metrics_df.iloc[0]["r2"]),
            }
        ]
    ).to_csv(output_dir / "holdout_best_by_target.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split.manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosResidualCorrectionConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosResidualCorrectionConfig(
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
