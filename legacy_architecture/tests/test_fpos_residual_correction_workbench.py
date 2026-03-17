from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.ml_statistics_workbench.fpos_residual_correction import (
    _build_residual_frame,
    _fit_residual_model,
)


def test_build_residual_frame_adds_study_dummies_and_winner_features() -> None:
    anchor_frame = pd.DataFrame(
        {
            "source_pred": [0.2, 0.4],
            "amp_bimodality": [0.1, 0.3],
            "matched_support_conf": [0.9, 0.7],
        }
    )
    target_df = pd.DataFrame({"study_set": ["S1", "S2"]})
    base_pred = np.array([0.25, 0.45], dtype=float)

    frame = _build_residual_frame(
        anchor_frame,
        base_pred=base_pred,
        target_df=target_df,
        include_study=True,
    )

    assert "winner_pred" in frame.columns
    assert "winner_uncentered" in frame.columns
    assert "study_S1" in frame.columns
    assert "study_S2" in frame.columns
    assert frame.shape[0] == 2


def test_fit_residual_model_returns_finite_predictions_for_ridge_and_histgb() -> None:
    X_train = pd.DataFrame(
        {
            "winner_pred": [0.1, 0.2, 0.7, 0.8],
            "source_pred": [0.12, 0.18, 0.68, 0.82],
            "amp_bimodality": [0.2, 0.25, 0.7, 0.75],
        }
    )
    residual_train = np.array([0.02, -0.01, 0.04, -0.03], dtype=float)
    X_eval = pd.DataFrame(
        {
            "winner_pred": [0.15, 0.75],
            "source_pred": [0.14, 0.78],
            "amp_bimodality": [0.22, 0.72],
        }
    )

    ridge = _fit_residual_model(X_train, residual_train, model_id="ridge", random_state=42)
    histgb = _fit_residual_model(X_train, residual_train, model_id="histgb", random_state=42)

    ridge_pred = np.asarray(ridge.predict(X_eval), dtype=float)
    histgb_pred = np.asarray(histgb.predict(X_eval), dtype=float)

    assert ridge_pred.shape == (2,)
    assert histgb_pred.shape == (2,)
    assert np.all(np.isfinite(ridge_pred))
    assert np.all(np.isfinite(histgb_pred))
