from __future__ import annotations

from pathlib import Path

import pandas as pd

from qc_framework.transfer import QCTransferBenchmark, TransferBenchmarkConfig


def _benchmark() -> QCTransferBenchmark:
    config = TransferBenchmarkConfig(
        parquet_path=Path("data/raw/33000_ROWS.parquet"),
        deploy_best_per_target=True,
        model_families=("paired_only", "hybrid_calibrated"),
        protocol_modes=("inductive",),
    )
    return QCTransferBenchmark(config)


def test_select_finalists_keeps_separate_target_candidates() -> None:
    benchmark = _benchmark()
    selection_summary = pd.DataFrame(
        [
            {
                "protocol_mode": "inductive",
                "candidate_id": "inductive|hybrid_calibrated|fpos_model|contextual_full",
                "model_family": "hybrid_calibrated",
                "model_name": "fpos_model",
                "feature_view": "contextual_full",
                "target": "fpos",
                "is_ceiling_model": False,
                "mae": 0.10,
                "advances_target": True,
                "advances": True,
            },
            {
                "protocol_mode": "inductive",
                "candidate_id": "inductive|hybrid_calibrated|fmiss_model|contextual_full",
                "model_family": "hybrid_calibrated",
                "model_name": "fmiss_model",
                "feature_view": "contextual_full",
                "target": "fmiss",
                "is_ceiling_model": False,
                "mae": 0.12,
                "advances_target": True,
                "advances": True,
            },
            {
                "protocol_mode": "inductive",
                "candidate_id": "inductive|paired_only|shape|shape_only",
                "model_family": "paired_only",
                "model_name": "shape",
                "feature_view": "shape_only",
                "target": "fpos",
                "is_ceiling_model": False,
                "mae": 0.15,
                "advances_target": True,
                "advances": True,
            },
            {
                "protocol_mode": "inductive",
                "candidate_id": "inductive|paired_only|shape|shape_only",
                "model_family": "paired_only",
                "model_name": "shape",
                "feature_view": "shape_only",
                "target": "fmiss",
                "is_ceiling_model": False,
                "mae": 0.18,
                "advances_target": True,
                "advances": True,
            },
        ]
    )

    finalists = benchmark._select_finalists(selection_summary)

    hybrid = finalists[finalists["model_family"] == "hybrid_calibrated"].sort_values("target")
    assert hybrid["target"].tolist() == ["fmiss", "fpos"]
    assert hybrid["candidate_id"].tolist() == [
        "inductive|hybrid_calibrated|fmiss_model|contextual_full",
        "inductive|hybrid_calibrated|fpos_model|contextual_full",
    ]


def test_per_target_recommendation_prefers_target_specific_eligible_models() -> None:
    benchmark = _benchmark()
    holdout_summary = pd.DataFrame(
        [
            {
                "protocol_mode": "inductive",
                "candidate_id": "inductive|paired_only|shape|shape_only",
                "model_family": "paired_only",
                "model_name": "shape",
                "feature_view": "shape_only",
                "target": "fpos",
                "is_ceiling_model": False,
                "mae": 0.14,
                "eligible_target_default": True,
            },
            {
                "protocol_mode": "inductive",
                "candidate_id": "inductive|hybrid_calibrated|fpos_model|contextual_full",
                "model_family": "hybrid_calibrated",
                "model_name": "fpos_model",
                "feature_view": "contextual_full",
                "target": "fpos",
                "is_ceiling_model": False,
                "mae": 0.11,
                "eligible_target_default": True,
            },
            {
                "protocol_mode": "inductive",
                "candidate_id": "inductive|paired_only|shape|shape_only",
                "model_family": "paired_only",
                "model_name": "shape",
                "feature_view": "shape_only",
                "target": "fmiss",
                "is_ceiling_model": False,
                "mae": 0.18,
                "eligible_target_default": True,
            },
            {
                "protocol_mode": "inductive",
                "candidate_id": "inductive|hybrid_stack|fmiss_model|contextual_full",
                "model_family": "hybrid_stack",
                "model_name": "fmiss_model",
                "feature_view": "contextual_full",
                "target": "fmiss",
                "is_ceiling_model": False,
                "mae": 0.15,
                "eligible_target_default": True,
            },
        ]
    )

    recommendation = benchmark._per_target_recommendation(
        holdout_summary,
        protocol_mode="inductive",
        eligible_only=True,
    )

    assert recommendation["target"].tolist() == ["fmiss", "fpos"]
    assert recommendation["candidate_id"].tolist() == [
        "inductive|hybrid_stack|fmiss_model|contextual_full",
        "inductive|hybrid_calibrated|fpos_model|contextual_full",
    ]
