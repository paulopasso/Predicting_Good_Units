from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.ml_statistics_workbench.fpos_anchor_stack import _fit_anchor_scores


def test_fit_anchor_scores_separates_unmatched_positive_regime() -> None:
    matched_train = pd.DataFrame(
        {
            "recording_key": ["r1", "r1", "r2", "r2"],
            "feat": [0.1, 0.2, 0.15, 0.25],
        }
    )
    matched_test = pd.DataFrame(
        {
            "recording_key": ["r3", "r3"],
            "feat": [0.2, 0.22],
        }
    )
    unmatched = pd.DataFrame(
        {
            "recording_key": ["u1", "u2", "u3", "u4"],
            "feat": [1.0, 0.95, 0.9, 1.1],
        }
    )

    train_score, unlabeled_score, test_score = _fit_anchor_scores(
        matched_train_df=matched_train,
        matched_test_df=matched_test,
        unmatched_df=unmatched,
        feature_cols=["feat"],
        n_splits=2,
        config=type(
            "Cfg",
            (),
            {
                "anchor_learning_rate": 0.1,
                "anchor_max_depth": 3,
                "anchor_max_iter": 80,
                "random_state": 42,
            },
        )(),
    )

    assert train_score.shape == (4,)
    assert unlabeled_score.shape == (4,)
    assert test_score.shape == (2,)
    assert np.all(np.isfinite(train_score))
    assert np.all(np.isfinite(unlabeled_score))
    assert np.all(np.isfinite(test_score))
    assert np.all((train_score >= 0.0) & (train_score <= 1.0))
    assert np.all((unlabeled_score >= 0.0) & (unlabeled_score <= 1.0))
    assert np.all((test_score >= 0.0) & (test_score <= 1.0))
