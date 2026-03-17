from __future__ import annotations

import pandas as pd

from crash_tests.ml_statistics_workbench.label_budget import (
    _aggregate_budget_curve,
    _choose_budget_recommendations,
    _grouped_budget_subset,
)


def test_grouped_budget_subset_hits_target_without_group_duplicates() -> None:
    df = pd.DataFrame(
        {
            "recording_key": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "x": range(8),
        }
    )
    idx = _grouped_budget_subset(df, target_rows=4, random_state=42, n_candidates=32)
    subset = df.iloc[idx]
    assert len(idx) == 4
    counts = subset["recording_key"].value_counts()
    assert set(counts.tolist()) == {2}


def test_choose_budget_recommendation_prefers_smallest_near_optimal_budget() -> None:
    curve = pd.DataFrame(
        [
            {"target": "fmiss", "variant_id": "a", "budget_rows": 20, "mean_mae": 0.20, "mean_r2": 0.40},
            {"target": "fmiss", "variant_id": "a", "budget_rows": 40, "mean_mae": 0.18, "mean_r2": 0.47},
            {"target": "fmiss", "variant_id": "a", "budget_rows": 60, "mean_mae": 0.17, "mean_r2": 0.49},
            {"target": "fmiss", "variant_id": "a", "budget_rows": 80, "mean_mae": 0.169, "mean_r2": 0.50},
        ]
    )
    rec = _choose_budget_recommendations(curve, r2_frac=0.98, mae_frac=1.03)
    assert int(rec.iloc[0]["recommended_budget_rows"]) == 60


def test_aggregate_budget_curve_computes_means() -> None:
    metrics = pd.DataFrame(
        [
            {"target": "fpos", "variant_id": "v1", "budget_rows": 20, "repeat_id": 0, "subset_recordings": 2, "mae": 0.2, "rmse": 0.3, "r2": 0.4, "bias": 0.0, "calibration_slope": 1.0, "calibration_intercept": 0.0},
            {"target": "fpos", "variant_id": "v1", "budget_rows": 20, "repeat_id": 1, "subset_recordings": 2, "mae": 0.1, "rmse": 0.2, "r2": 0.6, "bias": 0.1, "calibration_slope": 0.9, "calibration_intercept": 0.0},
        ]
    )
    curve = _aggregate_budget_curve(metrics)
    row = curve.iloc[0]
    assert abs(float(row["mean_mae"]) - 0.15) < 1e-9
    assert abs(float(row["mean_r2"]) - 0.5) < 1e-9
