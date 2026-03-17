from __future__ import annotations

import numpy as np

from crash_tests.ml_statistics_workbench.fpos_xgb_bagging import _best_alpha


def test_best_alpha_prefers_better_expert() -> None:
    y_true = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    pred_a = np.array([0.1, 0.2, 0.7, 0.8], dtype=float)
    pred_b = np.array([0.3, 0.4, 0.5, 0.6], dtype=float)

    alpha = _best_alpha(y_true, pred_a, pred_b, (0.0, 0.25, 0.5, 0.75, 1.0))

    assert alpha == 0.0
