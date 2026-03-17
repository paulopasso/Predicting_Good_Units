from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.ml_statistics_workbench.fpos_expert_stack import (
    _fit_meta_predict,
    _make_meta_frame,
)


def test_make_meta_frame_builds_expected_columns() -> None:
    frame = _make_meta_frame(
        pred_base=np.array([0.1, 0.2], dtype=float),
        pred_anchor=np.array([0.2, 0.4], dtype=float),
        support_conf=np.array([0.8, 0.6], dtype=float),
        anchor_fp_score=np.array([0.3, 0.7], dtype=float),
        anchor_support_margin=np.array([-0.5, 0.1], dtype=float),
        cluster_ids=np.array([0, 1], dtype=int),
        study_sets=pd.Series(["PAIRED_A", "PAIRED_B"]),
    )

    assert "pred_base" in frame.columns
    assert "pred_anchor" in frame.columns
    assert "anchor_support_margin" in frame.columns
    assert any(col.startswith("cluster_") for col in frame.columns)
    assert any(col.startswith("study_") for col in frame.columns)
    assert frame.shape[0] == 2


def test_fit_meta_predict_returns_bounded_values() -> None:
    X_train = pd.DataFrame(
        {
            "pred_base": [0.1, 0.2, 0.7, 0.8],
            "pred_anchor": [0.15, 0.25, 0.65, 0.85],
            "support_conf": [0.9, 0.8, 0.4, 0.3],
            "anchor_fp_score": [0.2, 0.3, 0.7, 0.8],
            "anchor_support_margin": [-0.7, -0.5, 0.3, 0.5],
        }
    )
    y_train = np.array([0.05, 0.1, 0.75, 0.9], dtype=float)
    X_eval = pd.DataFrame(
        {
            "pred_base": [0.12, 0.78],
            "pred_anchor": [0.16, 0.82],
            "support_conf": [0.85, 0.35],
            "anchor_fp_score": [0.25, 0.75],
            "anchor_support_margin": [-0.6, 0.4],
        }
    )

    pred = _fit_meta_predict(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        target_transform_mode="logit",
        model_id="ridge_1",
        random_state=42,
    )

    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))
    assert np.all((pred >= 0.0) & (pred <= 1.0))
