from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from qc_thesis.modeling.shared import FrameAugmenter, StackSet
from qc_thesis.paths import THESIS_ROOT

from .backends import make_metric_row, make_prediction_frame, pick_r2_first_recommendation, score_predictions
from .base_stack_runner import build_base_stack_state, build_embedding_universe, evaluate_stack_variant
from .augmenters import fit_signal_embeddings
from .specs import MLStatisticsWorkbenchConfig
from .stack_recipe_runner import StackVariantSpec, run_stack_recipe


MODULE_NAME = "qc_thesis.modeling.stacking.recipes"
_EXTRA_STACK_COLS = {
    "source_pred",
    "anchor_fp_score",
    "matched_support_conf",
    "anchor_support_margin",
}


def _resolved_payload(config: Any) -> dict[str, Any]:
    payload = asdict(config)
    payload["parquet_path"] = str(config.parquet_path)
    payload["output_root"] = str(config.output_root)
    payload["resolved_output_dir"] = str(config.resolved_output_dir())
    return payload


def _log(config: Any, message: str) -> None:
    if getattr(config, "verbose", False):
        print(message, flush=True)


def _default_parquet(parquet_path: Path | None) -> Path:
    return parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet")


def _default_output_config(
    config_cls: type[Any],
    output_dir: Path,
    *,
    parquet_path: Path | None,
    random_state: int | None,
) -> Any:
    config = config_cls(
        parquet_path=_default_parquet(parquet_path),
        output_root=output_dir.parent,
        run_name=output_dir.name,
        verbose=False,
    )
    if random_state is not None:
        config.random_state = int(random_state)
    return config


@dataclass
class FPosBackendSweepConfig(MLStatisticsWorkbenchConfig):
    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_backend_sweep_{stamp}"

    def to_dict(self) -> dict[str, Any]:
        return _resolved_payload(self)


@dataclass
class FPosFamilyRobustnessConfig(MLStatisticsWorkbenchConfig):
    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_family_robustness_{stamp}"

    def to_dict(self) -> dict[str, Any]:
        return _resolved_payload(self)


@dataclass
class FPosSignalEmbeddingStackConfig(MLStatisticsWorkbenchConfig):
    conv_embed_dim: int = 8
    conv_hidden_channels: int = 12
    ae_epochs: int = 28
    batch_size: int = 256
    learning_rate: float = 2e-3
    universe_max_rows: int | None = 18000

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_signal_embedding_stack_{stamp}"

    def to_dict(self) -> dict[str, Any]:
        return _resolved_payload(self)


@dataclass
class FMissWaveformTransferConfig(FPosSignalEmbeddingStackConfig):
    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fmiss_waveform_transfer_{stamp}"


def _subset_stack(stack: StackSet, keep: list[str]) -> StackSet:
    return stack.with_frames(
        train=stack.train.loc[:, keep].reset_index(drop=True),
        unlabeled=stack.unlabeled.loc[:, keep].reset_index(drop=True),
        test=stack.test.loc[:, keep].reset_index(drop=True),
    )


def _context_columns(stack: StackSet) -> list[str]:
    return [col for col in stack.train.columns if col not in _EXTRA_STACK_COLS]


def _variant_specs_for_target(target: str, stack: StackSet) -> list[tuple[str, StackSet, bool]]:
    context_cols = _context_columns(stack)
    source_cols = context_cols + [col for col in ["source_pred"] if col in stack.train.columns]
    reduced_cols = context_cols + [col for col in ["source_pred", "anchor_fp_score"] if col in stack.train.columns]
    return [
        ("contextual_paired_only_lightgbm", _subset_stack(stack, context_cols), True),
        ("source_reweighted_fullctx_lightgbm", _subset_stack(stack, source_cols), False),
        ("embed_nommd_fullctx_lightgbm", stack, False),
        ("source_plus_reduced_latent_fullctx_lightgbm", _subset_stack(stack, reduced_cols), False),
    ]


def _run_context_target(
    config: MLStatisticsWorkbenchConfig,
    *,
    target: str,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]], dict[str, object]]:
    state = build_base_stack_state(target=target, config=config)
    no_pseudo_state = replace(state, trust_selected=state.trust_selected.head(0).copy())

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, object]] = []

    for idx, (variant_id, stack, no_pseudo) in enumerate(_variant_specs_for_target(target, state.base_stack), start=1):
        _log(config, f"[{target}] {variant_id}")
        variant_state = no_pseudo_state if no_pseudo else state
        cv_metrics, holdout_pred = evaluate_stack_variant(
            state=variant_state,
            stack=stack,
            backend_id="lightgbm_default",
            family_alpha=1.0,
            random_state_offset=(1000 if target == "fpos" else 2000) + (100 * idx),
        )
        holdout_metrics = score_predictions(variant_state.y_test, holdout_pred)
        metric_rows.append(
            make_metric_row(
                target=target,
                variant_id=variant_id,
                family="context_stack",
                backend="lightgbm_default",
                source_target="fmiss_extended" if target == "fmiss" else "fpos",
                metrics=holdout_metrics,
                notes="compact context stack runner",
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target=target,
                variant_id=variant_id,
                family="context_stack",
                backend="lightgbm_default",
                df=variant_state.target_test_df,
                y_true=variant_state.y_test,
                pred=holdout_pred,
            )
        )
        cv_rows.append(
            {
                "target": target,
                "variant_id": variant_id,
                "backend": "lightgbm_default",
                "cv_mae": float(cv_metrics["mae"]),
                "cv_r2": float(cv_metrics["r2"]),
                "selected_rows": int(len(variant_state.trust_selected)),
                "mean_selected_weight": float(variant_state.trust_selected["pseudo_weight"].mean())
                if not variant_state.trust_selected.empty
                else 0.0,
            }
        )

    return metric_rows, prediction_frames, cv_rows, state.split.manifest


def run_context_stack(config: MLStatisticsWorkbenchConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metric_rows: list[dict[str, object]] = []
    all_prediction_frames: list[pd.DataFrame] = []
    all_cv_rows: list[dict[str, object]] = []
    split_manifest: dict[str, object] | None = None

    for target in config.targets:
        metric_rows, prediction_frames, cv_rows, manifest = _run_context_target(config, target=target)
        all_metric_rows.extend(metric_rows)
        all_prediction_frames.extend(prediction_frames)
        all_cv_rows.extend(cv_rows)
        if split_manifest is None:
            split_manifest = manifest

    pd.DataFrame(all_metric_rows).sort_values(["target", "r2", "mae"], ascending=[True, False, True]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )
    pd.concat(all_prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(all_cv_rows).sort_values(["target", "cv_r2", "cv_mae"], ascending=[True, False, True]).to_csv(
        output_dir / "variant_cv_summary.csv",
        index=False,
    )
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest or {}, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def _source_only_stack(stack: StackSet) -> StackSet:
    keep = [col for col in stack.train.columns if col not in {"anchor_fp_score", "anchor_support_margin"}]
    return _subset_stack(stack, keep)


def run_anchor_stack(config: FPosBackendSweepConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_base_stack_state(target="fpos", config=config)

    variants = [
        ("source_reweighted_cluster_trust_pseudo_lightgbm_default", _source_only_stack(state.base_stack), "lightgbm_default"),
        ("source_reweighted_cluster_trust_pseudo_xgboost_default", _source_only_stack(state.base_stack), "xgboost_default"),
        ("source_reweighted_anchor_stack_cluster_trust_pseudo_lightgbm_default", state.base_stack, "lightgbm_default"),
        ("source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default", state.base_stack, "xgboost_default"),
    ]

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []

    for idx, (variant_id, stack, backend_id) in enumerate(variants, start=1):
        _log(config, f"[backend] {variant_id}")
        cv_metrics, holdout_pred = evaluate_stack_variant(
            state=state,
            stack=stack,
            backend_id=backend_id,
            family_alpha=1.0,
            random_state_offset=2200 + (100 * idx),
        )
        holdout_metrics = score_predictions(state.y_test, holdout_pred)
        metric_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family="backend_sweep",
                backend=backend_id,
                source_target="fpos",
                metrics=holdout_metrics,
                notes=f"compact backend sweep with {backend_id}",
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family="backend_sweep",
                backend=backend_id,
                df=state.target_test_df,
                y_true=state.y_test,
                pred=holdout_pred,
            )
        )
        cv_rows.append(
            {
                "target": "fpos",
                "variant_id": variant_id,
                "family": "backend_sweep",
                "backend": backend_id,
                "feature_set": "anchor_stack" if "anchor_stack" in variant_id else "source_reweighted",
                "cv_mae": float(cv_metrics["mae"]),
                "cv_r2": float(cv_metrics["r2"]),
                "selected_rows": int(len(state.trust_selected)),
                "mean_selected_weight": float(state.trust_selected["pseudo_weight"].mean())
                if not state.trust_selected.empty
                else 0.0,
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows)
    recommendation = pick_r2_first_recommendation(
        cv_df.rename(columns={"cv_mae": "mae", "cv_r2": "r2"}),
        target="fpos",
        reference_variant_id="source_reweighted_anchor_stack_cluster_trust_pseudo_lightgbm_default",
        selection_metric=config.selection_metric,
        mae_guardrail_pct=config.mae_guardrail_pct,
        min_r2_improvement=config.min_r2_improvement,
    )
    recommended_holdout = metrics_df.loc[
        metrics_df["variant_id"].astype(str) == str(recommendation["recommended_variant_id"])
    ].iloc[0]
    recommendation["recommended_holdout_mae"] = float(recommended_holdout["mae"])
    recommendation["recommended_holdout_r2"] = float(recommended_holdout["r2"])
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
    pd.DataFrame([recommendation]).to_csv(output_dir / "per_target_recommendation.csv", index=False)
    holdout_best_df.to_csv(output_dir / "holdout_best_by_target.csv", index=False)
    (output_dir / "filtered_feature_manifest.csv").write_text("")
    (output_dir / "transport_shift_summary.csv").write_text("")
    (output_dir / "config.json").write_text(pd.Series(config.to_dict()).to_json(indent=2))
    (output_dir / "split_manifest.json").write_text(pd.Series(state.split.manifest).to_json(indent=2))
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def _per_family_summary(y_true: Any, pred: Any, study_set: Any) -> pd.DataFrame:
    frame = pd.DataFrame({"study_set": pd.Series(study_set).astype(str), "true": y_true, "pred": pred})
    rows = []
    for study, group in frame.groupby("study_set", dropna=False):
        metrics = score_predictions(group["true"].to_numpy(dtype=float), group["pred"].to_numpy(dtype=float))
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


def run_robustness_stack(config: FPosFamilyRobustnessConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_base_stack_state(target="fpos", config=config)

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

    for idx, (variant_id, backend_id, family_alpha) in enumerate(variants, start=1):
        _log(config, f"[family] {variant_id}")
        cv_metrics, holdout_pred = evaluate_stack_variant(
            state=state,
            stack=state.base_stack,
            backend_id=backend_id,
            family_alpha=family_alpha,
            random_state_offset=2200 + (100 * idx),
        )
        holdout_metrics = score_predictions(state.y_test, holdout_pred)
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
                df=state.target_test_df,
                y_true=state.y_test,
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
                "selected_rows": int(len(state.trust_selected)),
                "mean_selected_weight": float(state.trust_selected["pseudo_weight"].mean())
                if not state.trust_selected.empty
                else 0.0,
            }
        )
        family_frames.append(
            _per_family_summary(state.y_test, holdout_pred, state.target_test_df["study_set"]).assign(
                variant_id=variant_id,
                split="holdout",
            )
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows).sort_values(["cv_r2", "cv_mae"], ascending=[False, True]).reset_index(drop=True)
    recommendation = {
        "target": "fpos",
        "selection_metric": "family_alpha",
        "reference_variant_id": "reference_xgboost",
        "recommended_variant_id": "family_balanced_strong_xgboost",
        "reference_cv_mae": float(cv_df.loc[cv_df["variant_id"] == "reference_xgboost", "cv_mae"].iloc[0]),
        "recommended_cv_mae": float(cv_df.loc[cv_df["variant_id"] == "family_balanced_strong_xgboost", "cv_mae"].iloc[0]),
        "reference_cv_r2": float(cv_df.loc[cv_df["variant_id"] == "reference_xgboost", "cv_r2"].iloc[0]),
        "recommended_cv_r2": float(cv_df.loc[cv_df["variant_id"] == "family_balanced_strong_xgboost", "cv_r2"].iloc[0]),
    }

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    pd.DataFrame([recommendation]).to_csv(output_dir / "per_target_recommendation.csv", index=False)
    pd.concat(family_frames, ignore_index=True).to_csv(output_dir / "family_performance.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(state.split.manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def _combined_stack(base_stack: StackSet, *feature_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> StackSet:
    train_frames = [bundle[0] for bundle in feature_frames]
    unlabeled_frames = [bundle[1] for bundle in feature_frames]
    test_frames = [bundle[2] for bundle in feature_frames]
    return base_stack.apply(
        FrameAugmenter(
            train_features=pd.concat(train_frames, axis=1),
            unlabeled_features=pd.concat(unlabeled_frames, axis=1),
            test_features=pd.concat(test_frames, axis=1),
        )
    )


def run_waveform_stack(config: FPosSignalEmbeddingStackConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_base_stack_state(target="fpos", config=config)
    universe_df = build_embedding_universe(
        split=state.split,
        max_rows=config.universe_max_rows,
        random_state=config.random_state,
    )

    _log(config, "[embed] training waveform autoencoder")
    wf_bundle = fit_signal_embeddings(
        train_signal=state.target_train_df.loc[:, state.prepared.wf_feature_cols],
        train_universe_signal=universe_df.loc[:, state.prepared.wf_feature_cols],
        unlabeled_signal=state.unlabeled_df.loc[:, state.prepared.wf_feature_cols],
        test_signal=state.target_test_df.loc[:, state.prepared.wf_feature_cols],
        prefix="wf",
        config=config,
    )
    _log(config, "[embed] training acg autoencoder")
    acg_bundle = fit_signal_embeddings(
        train_signal=state.target_train_df.loc[:, state.prepared.acg_feature_cols],
        train_universe_signal=universe_df.loc[:, state.prepared.acg_feature_cols],
        unlabeled_signal=state.unlabeled_df.loc[:, state.prepared.acg_feature_cols],
        test_signal=state.target_test_df.loc[:, state.prepared.acg_feature_cols],
        prefix="acg",
        config=config,
    )

    variants = [
        StackVariantSpec(
            variant_id="reference_anchor_stack_xgboost",
            stack=state.base_stack,
            notes="conv embedding stack variant",
        ),
        StackVariantSpec(
            variant_id="wf_embed_anchor_stack_xgboost",
            stack=state.base_stack.apply(wf_bundle.as_augmenter()),
            notes="conv embedding stack variant",
        ),
        StackVariantSpec(
            variant_id="acg_embed_anchor_stack_xgboost",
            stack=state.base_stack.apply(acg_bundle.as_augmenter()),
            notes="conv embedding stack variant",
        ),
        StackVariantSpec(
            variant_id="wf_acg_embed_anchor_stack_xgboost",
            stack=_combined_stack(
                state.base_stack,
                (wf_bundle.train, wf_bundle.unlabeled, wf_bundle.test),
                (acg_bundle.train, acg_bundle.unlabeled, acg_bundle.test),
            ),
            notes="conv embedding stack variant",
        ),
    ]
    return run_stack_recipe(
        config=config,
        output_dir=output_dir,
        state=state,
        target="fpos",
        family="signal_embedding_stack",
        source_target="fpos",
        variants=variants,
        extra_outputs={
            "embedding_history.csv": pd.DataFrame({"wf": wf_bundle.history, "acg": acg_bundle.history})
        },
        log=lambda message: _log(config, message),
    )


def run_waveform_transfer(config: FMissWaveformTransferConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_base_stack_state(target="fmiss", config=config)
    universe_df = build_embedding_universe(
        split=state.split,
        max_rows=config.universe_max_rows,
        random_state=config.random_state,
    )

    _log(config, "[embed] training waveform autoencoder")
    wf_bundle = fit_signal_embeddings(
        train_signal=state.target_train_df.loc[:, state.prepared.wf_feature_cols],
        train_universe_signal=universe_df.loc[:, state.prepared.wf_feature_cols],
        unlabeled_signal=state.unlabeled_df.loc[:, state.prepared.wf_feature_cols],
        test_signal=state.target_test_df.loc[:, state.prepared.wf_feature_cols],
        prefix="wf",
        config=config,
    )

    variants = [
        StackVariantSpec(
            variant_id="fmiss_reference_anchor_stack_xgboost",
            stack=state.base_stack,
            notes="fpos winner recipe applied to fmiss",
        ),
        StackVariantSpec(
            variant_id="fmiss_wf_embed_anchor_stack_xgboost",
            stack=state.base_stack.apply(wf_bundle.as_augmenter()),
            notes="fpos winner recipe applied to fmiss",
        ),
    ]
    return run_stack_recipe(
        config=config,
        output_dir=output_dir,
        state=state,
        target="fmiss",
        family="fmiss_waveform_transfer",
        source_target="fmiss_extended",
        variants=variants,
        extra_outputs={
            "feature_learning_history.csv": pd.DataFrame(
                [{"wf_train_recon_loss": wf_bundle.history["train_recon_loss"]}]
            )
        },
        log=lambda message: _log(config, message),
    )


def _parse_context_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the compact ML/statistics context stack")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _build_context_config(args: argparse.Namespace) -> MLStatisticsWorkbenchConfig:
    config = MLStatisticsWorkbenchConfig(
        parquet_path=THESIS_ROOT / args.parquet,
        output_root=THESIS_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        config.lightgbm_estimators = 100
        config.xgboost_estimators = 140
        config.mmd_source_train_epochs = 6
        config.mmd_patience = 2
        config.model_selection_splits = 2
        config.latent_candidate_topk = (4, 8)
    return config


def run_with_config(config: MLStatisticsWorkbenchConfig) -> Path:
    return run_context_stack(config)


def main() -> None:
    args = _parse_context_args()
    run_context_stack(_build_context_config(args))


class ContextStackFamily:
    implementation_module = MODULE_NAME

    def run(
        self,
        recipe: Any,
        output_dir: Path,
        *,
        smoke: bool = False,
        parquet_path: Path | None = None,
        random_state: int | None = None,
    ) -> Path:
        config = _default_output_config(
            MLStatisticsWorkbenchConfig,
            output_dir,
            parquet_path=parquet_path,
            random_state=random_state,
        )
        config.targets = (recipe.target,)
        config.save_checkpoints = False
        if smoke:
            config.paired_final_holdout_rows = 24
            config.model_selection_splits = 2
            config.lightgbm_estimators = 32
            config.xgboost_estimators = 48
            config.mmd_source_train_epochs = 2
            config.latent_candidate_topk = (4,)
            config.cluster_candidate_k = (2,)
        return run_context_stack(config)


class AnchorStackFamily:
    implementation_module = MODULE_NAME

    def run(
        self,
        recipe: Any,
        output_dir: Path,
        *,
        smoke: bool = False,
        parquet_path: Path | None = None,
        random_state: int | None = None,
    ) -> Path:
        config = _default_output_config(
            FPosBackendSweepConfig,
            output_dir,
            parquet_path=parquet_path,
            random_state=random_state,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.max_pseudo_per_recording = 12
            config.max_pseudo_per_cluster = 32
            config.max_pseudo_per_labeled_ratio = 1.0
        return run_anchor_stack(config)


class WaveformStackFamily:
    implementation_module = MODULE_NAME

    def run(
        self,
        recipe: Any,
        output_dir: Path,
        *,
        smoke: bool = False,
        parquet_path: Path | None = None,
        random_state: int | None = None,
    ) -> Path:
        config = _default_output_config(
            FPosSignalEmbeddingStackConfig,
            output_dir,
            parquet_path=parquet_path,
            random_state=random_state,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.ae_epochs = 2
            config.conv_embed_dim = 4
            config.conv_hidden_channels = 6
            config.batch_size = 128
            config.universe_max_rows = 1200
            config.max_pseudo_per_recording = 12
            config.max_pseudo_per_cluster = 32
        return run_waveform_stack(config)


class RobustnessStackFamily:
    implementation_module = MODULE_NAME

    def run(
        self,
        recipe: Any,
        output_dir: Path,
        *,
        smoke: bool = False,
        parquet_path: Path | None = None,
        random_state: int | None = None,
    ) -> Path:
        config = _default_output_config(
            FPosFamilyRobustnessConfig,
            output_dir,
            parquet_path=parquet_path,
            random_state=random_state,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.max_pseudo_per_recording = 12
            config.max_pseudo_per_cluster = 32
            config.max_pseudo_per_labeled_ratio = 1.0
        return run_robustness_stack(config)


class WaveformTransferFamily:
    implementation_module = MODULE_NAME

    def run(
        self,
        recipe: Any,
        output_dir: Path,
        *,
        smoke: bool = False,
        parquet_path: Path | None = None,
        random_state: int | None = None,
    ) -> Path:
        config = _default_output_config(
            FMissWaveformTransferConfig,
            output_dir,
            parquet_path=parquet_path,
            random_state=random_state,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.ae_epochs = 2
            config.conv_embed_dim = 4
            config.conv_hidden_channels = 6
            config.batch_size = 128
            config.universe_max_rows = 1200
            config.max_pseudo_per_recording = 12
            config.max_pseudo_per_cluster = 32
        return run_waveform_transfer(config)


STACKING_FAMILIES = {
    "context_stack": ContextStackFamily(),
    "anchor_stack": AnchorStackFamily(),
    "waveform_stack": WaveformStackFamily(),
    "robustness_stack": RobustnessStackFamily(),
    "waveform_transfer": WaveformTransferFamily(),
}


__all__ = [
    "AnchorStackFamily",
    "ContextStackFamily",
    "FMissWaveformTransferConfig",
    "FPosBackendSweepConfig",
    "FPosFamilyRobustnessConfig",
    "FPosSignalEmbeddingStackConfig",
    "RobustnessStackFamily",
    "STACKING_FAMILIES",
    "WaveformStackFamily",
    "WaveformTransferFamily",
    "main",
    "run_anchor_stack",
    "run_context_stack",
    "run_robustness_stack",
    "run_waveform_stack",
    "run_waveform_transfer",
    "run_with_config",
]
