from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qc_thesis.modeling.core import StackSet

from .backends import make_metric_row, make_prediction_frame, score_predictions
from .base_stack_runner import build_base_stack_state, evaluate_stack_variant
from .config import MLStatisticsWorkbenchConfig


_EXTRA_STACK_COLS = {
    "source_pred",
    "anchor_fp_score",
    "matched_support_conf",
    "anchor_support_margin",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the compact ML/statistics context stack")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> MLStatisticsWorkbenchConfig:
    config = MLStatisticsWorkbenchConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
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


def _log(config: MLStatisticsWorkbenchConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


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


def _run_target(config: MLStatisticsWorkbenchConfig, *, target: str) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]], dict[str, object]]:
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


def run_with_config(config: MLStatisticsWorkbenchConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metric_rows: list[dict[str, object]] = []
    all_prediction_frames: list[pd.DataFrame] = []
    all_cv_rows: list[dict[str, object]] = []
    split_manifest: dict[str, object] | None = None

    for target in config.targets:
        metric_rows, prediction_frames, cv_rows, manifest = _run_target(config, target=target)
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


def main() -> None:
    args = _parse_args()
    run_with_config(_build_config(args))


if __name__ == "__main__":
    main()
