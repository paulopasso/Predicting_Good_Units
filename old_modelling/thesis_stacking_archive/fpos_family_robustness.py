from __future__ import annotations

import argparse
import json
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

from .backends import make_metric_row, make_prediction_frame, score_predictions
from .base_stack_runner import build_base_stack_state, evaluate_stack_variant


@dataclass
class FPosFamilyRobustnessConfig:
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
    parser = argparse.ArgumentParser(description="Run the compact fpos family robustness benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosFamilyRobustnessConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _per_family_summary(y_true, pred, study_set) -> pd.DataFrame:
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


def _run_with_config(config: FPosFamilyRobustnessConfig) -> Path:
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
    oof_store: dict[str, Any] = {}
    holdout_store: dict[str, Any] = {}

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
        family_frames.append(_per_family_summary(state.y_test, holdout_pred, state.target_test_df["study_set"]).assign(variant_id=variant_id, split="holdout"))
        holdout_store[variant_id] = holdout_pred
        oof_store[variant_id] = cv_metrics

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
