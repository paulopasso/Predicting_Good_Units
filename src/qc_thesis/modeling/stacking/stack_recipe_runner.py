from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import json
import pandas as pd

from qc_thesis.modeling.shared import StackSet

from .backends import make_metric_row, make_prediction_frame, score_predictions
from .base_stack_runner import BaseStackState, evaluate_stack_variant


@dataclass(frozen=True)
class StackVariantSpec:
    variant_id: str
    stack: StackSet
    notes: str
    backend_id: str = "xgboost_default"
    family_alpha: float = 1.0


def _config_payload(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if is_dataclass(config):
        payload = asdict(config)
    else:
        payload = dict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def run_stack_recipe(
    *,
    config: Any,
    output_dir: Path,
    state: BaseStackState,
    target: str,
    family: str,
    source_target: str,
    variants: list[StackVariantSpec],
    extra_outputs: Mapping[str, pd.DataFrame] | None = None,
    log: Callable[[str], None] | None = None,
    random_state_seed: int = 2200,
) -> Path:
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []

    for idx, variant in enumerate(variants, start=1):
        if log is not None:
            log(f"[variant] {variant.variant_id}")
        cv_metrics, holdout_pred = evaluate_stack_variant(
            state=state,
            stack=variant.stack,
            backend_id=variant.backend_id,
            family_alpha=variant.family_alpha,
            random_state_offset=random_state_seed + (100 * idx),
        )
        holdout_metrics = score_predictions(state.y_test, holdout_pred)
        metric_rows.append(
            make_metric_row(
                target=target,
                variant_id=variant.variant_id,
                family=family,
                backend=variant.backend_id,
                source_target=source_target,
                metrics=holdout_metrics,
                notes=variant.notes,
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target=target,
                variant_id=variant.variant_id,
                family=family,
                backend=variant.backend_id,
                df=state.target_test_df,
                y_true=state.y_test,
                pred=holdout_pred,
            )
        )
        cv_rows.append(
            {
                "target": target,
                "variant_id": variant.variant_id,
                "cv_mae": float(cv_metrics["mae"]),
                "cv_r2": float(cv_metrics["r2"]),
                "selected_rows": int(len(state.trust_selected)),
                "mean_selected_weight": float(state.trust_selected["pseudo_weight"].mean())
                if not state.trust_selected.empty
                else 0.0,
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows).sort_values(["cv_r2", "cv_mae"], ascending=[False, True]).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    if extra_outputs:
        for filename, frame in extra_outputs.items():
            frame.to_csv(output_dir / filename, index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(_config_payload(config), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(state.split.manifest, f, indent=2)
    if log is not None:
        log(f"Saved outputs to {output_dir}")
    return output_dir
