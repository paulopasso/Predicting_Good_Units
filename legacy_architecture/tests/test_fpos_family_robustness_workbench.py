from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.ml_statistics_workbench.fpos_family_robustness import (
    _family_weights,
    _pick_robust_variant,
)


def test_family_weights_upweight_smaller_families() -> None:
    study_set = pd.Series(["A", "A", "A", "B", "B", "C"])
    weights = _family_weights(study_set, alpha=1.0)
    assert weights.shape == (6,)
    assert np.all(np.isfinite(weights))
    assert weights[-1] > weights[0]
    assert weights[3] > weights[0]


def test_pick_robust_variant_prefers_better_worst_family_within_mae_guardrail() -> None:
    cv_df = pd.DataFrame(
        {
            "variant_id": ["reference_xgboost", "balanced_ok", "balanced_too_costly"],
            "cv_mae": [0.12, 0.123, 0.14],
            "cv_r2": [0.45, 0.44, 0.50],
            "cv_worst_family_r2": [0.10, 0.18, 0.25],
        }
    )
    recommendation = _pick_robust_variant(
        cv_df,
        reference_variant_id="reference_xgboost",
        mae_guardrail_pct=0.03,
    )
    assert recommendation["recommended_variant_id"] == "balanced_ok"
    assert recommendation["worst_family_r2_improvement"] > 0.0
