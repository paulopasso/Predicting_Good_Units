from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.ml_statistics_workbench.fpos_manifold_context import (
    _curriculum_refine_pseudo,
    _distance_weighted_knn_predict,
    _pseudo_weight_from_error,
)


def test_distance_weighted_knn_predict_respects_local_targets() -> None:
    train_features = pd.DataFrame({"x": [0.0, 0.1, 0.9, 1.0]})
    train_target = np.array([0.0, 0.1, 0.9, 1.0], dtype=float)
    eval_features = pd.DataFrame({"x": [0.05, 0.95]})

    pred = _distance_weighted_knn_predict(
        train_features,
        train_target,
        eval_features,
        n_neighbors=2,
    )

    assert pred.shape == (2,)
    assert float(pred[0]) < 0.2
    assert float(pred[1]) > 0.8


def test_pseudo_weight_from_error_decreases_with_error_and_increases_with_support() -> None:
    weight = _pseudo_weight_from_error(
        predicted_error=np.array([0.01, 0.20, 0.01], dtype=float),
        support_confidence=np.array([0.9, 0.9, 0.2], dtype=float),
        scale=0.25,
        floor=0.08,
    )

    assert float(weight[0]) > float(weight[1])
    assert float(weight[0]) > float(weight[2])
    assert np.all(weight >= 0.08)
    assert np.all(weight <= 0.25)


def test_curriculum_refine_pseudo_blends_and_clips() -> None:
    selected = pd.DataFrame(
        {
            "pseudo_label": [0.2, 0.8],
            "pseudo_weight": [0.1, 0.2],
        }
    )

    refined = _curriculum_refine_pseudo(
        selected=selected,
        student_pred=np.array([1.5, -0.5], dtype=float),
        blend=0.5,
    )

    assert np.allclose(refined["pseudo_weight"], selected["pseudo_weight"])
    assert np.allclose(refined["pseudo_label"], [0.85, 0.15])
