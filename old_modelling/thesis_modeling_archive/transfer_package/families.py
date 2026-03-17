from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from qc_thesis.paths import THESIS_ROOT

from .transfer import QCTransferBenchmark, TransferBenchmarkConfig


def _legacy_feature_view_sets(feature_view: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    mapping = {
        "none": (("none", tuple()),),
        "unit_raw": (("unit_raw", ("unit_raw",)),),
        "shape_only": (("shape_only", ("shape_only",)),),
        "full_context": (
            (
                "full_context",
                ("norm_swap", "recording_relative", "recording_context", "study_identity"),
            ),
        ),
    }
    return mapping.get(feature_view, (("unit_raw", ("unit_raw",)),))


def _legacy_backend_from_result_key(result_key: str) -> tuple[str, ...]:
    if "xgboost" in result_key:
        return ("xgboost",)
    if "lightgbm" in result_key:
        return ("lightgbm",)
    if "catboost" in result_key:
        return ("catboost",)
    return ("lightgbm",)


class LegacyTransferFamily:
    implementation_module = "qc_thesis.modeling.transfer.transfer"

    def _build_config(self, recipe: Any, output_dir: Path, *, smoke: bool, parquet_path: Path | None) -> TransferBenchmarkConfig:
        config = TransferBenchmarkConfig(
            parquet_path=parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet"),
            primary_targets=(recipe.target,),
            secondary_targets=(),
            feature_view_sets=_legacy_feature_view_sets(recipe.feature_view),
            model_families=(recipe.model_family,),
            protocol_modes=("inductive",),
            boosting_backends=_legacy_backend_from_result_key(recipe.result_key),
            paired_weight_grid=(25.0,) if "w25" in recipe.result_key else (10.0,),
            include_ceiling_diagnostics=False,
            deploy_best_per_target=False,
            verbose=False,
            neural_verbose=False,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.model_selection_splits = 2
            config.lightgbm_estimators = 24
            config.xgboost_estimators = 24
            config.neural_pretrain_epochs = 2
            config.neural_finetune_epochs = 2
        return config

    def run(self, recipe: Any, output_dir: Path, *, smoke: bool = False, parquet_path: Path | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        benchmark = QCTransferBenchmark(self._build_config(recipe, output_dir, smoke=smoke, parquet_path=parquet_path))
        artifacts = benchmark.run()

        metrics_df = artifacts.get("final_holdout_results", pd.DataFrame())
        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            metrics_df.to_csv(output_dir / "metrics.csv", index=False)

        for name, frame in artifacts.items():
            if name == "final_holdout_results":
                continue
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frame.to_csv(output_dir / f"{name}.csv", index=False)

        config_payload = benchmark._config_dict()
        (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
        return output_dir
