from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qc_thesis.modeling.core import StackSet

from .backends import make_metric_row, make_prediction_frame, pick_r2_first_recommendation, score_predictions
from .base_stack_runner import build_base_stack_state, evaluate_stack_variant


@dataclass
class FPosBackendSweepConfig:
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
        return f"fpos_backend_sweep_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the compact fpos backend sweep benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosBackendSweepConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _source_only_stack(stack: StackSet) -> StackSet:
    keep = [col for col in stack.train.columns if col not in {"anchor_fp_score", "anchor_support_margin"}]
    return stack.with_frames(
        train=stack.train.loc[:, keep].reset_index(drop=True),
        unlabeled=stack.unlabeled.loc[:, keep].reset_index(drop=True),
        test=stack.test.loc[:, keep].reset_index(drop=True),
    )


def _run_with_config(config: FPosBackendSweepConfig) -> Path:
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


def run_with_config(config: FPosBackendSweepConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosBackendSweepConfig(
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
