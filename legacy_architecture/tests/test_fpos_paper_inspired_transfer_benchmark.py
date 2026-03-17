from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.ml_statistics_workbench.fpos_paper_inspired_transfer_benchmark import (
    _fewshot_family_score,
    _labeled_neighborhood_features,
    _unlabeled_manifold_features,
)


def test_fewshot_family_score_returns_finite_predictions() -> None:
    train_meta = pd.DataFrame({"study_set": ["A", "A", "B", "B"]})
    Z_train = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=float)
    y_train = np.array([0.1, 0.2, 0.7, 0.8], dtype=float)
    eval_meta = pd.DataFrame({"study_set": ["A", "B"]})
    Z_eval = np.array([[0.05], [1.05]], dtype=float)

    pred = _fewshot_family_score(train_meta, Z_train, y_train, eval_meta, Z_eval, k=2)

    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))
    assert np.all((pred >= 0.0) & (pred <= 1.0))


def test_neighborhood_feature_helpers_return_expected_columns() -> None:
    Z_train = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=float)
    y_train = np.array([0.1, 0.2, 0.7, 0.8], dtype=float)
    Z_eval = np.array([[0.05], [1.05]], dtype=float)
    unlabeled = pd.DataFrame(
        {
            "source_pred": [0.15, 0.2, 0.65, 0.75],
            "anchor_fp_score": [0.2, 0.25, 0.7, 0.8],
            "matched_support_conf": [0.9, 0.85, 0.7, 0.65],
        }
    )

    lab = _labeled_neighborhood_features(Z_train, y_train, Z_eval, k=2, prefix="lab")
    unlab = _unlabeled_manifold_features(Z_train, unlabeled, Z_eval, k=2, prefix="u")

    assert list(lab.columns) == ["lab_nbr_mean", "lab_nbr_std", "lab_nbr_min_dist", "lab_nbr_density"]
    assert list(unlab.columns) == ["u_u_dist", "u_u_source_mean", "u_u_anchor_mean", "u_u_support_mean"]
    assert lab.shape[0] == 2
    assert unlab.shape[0] == 2
