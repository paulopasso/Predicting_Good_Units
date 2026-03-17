from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.ml_statistics_workbench.fpos_backend_sweep import _fit_backend_predict


def test_fit_backend_predict_returns_bounded_values_for_histgb() -> None:
    X_train = pd.DataFrame({"x1": [0.0, 0.2, 0.8, 1.0], "x2": [1.0, 0.9, 0.2, 0.1]})
    y_train = np.array([0.05, 0.1, 0.7, 0.9], dtype=float)
    X_eval = pd.DataFrame({"x1": [0.1, 0.9], "x2": [0.95, 0.15]})

    pred = _fit_backend_predict(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        sample_weight=None,
        target_transform_mode="logit",
        backend_id="histgb",
        random_state=42,
    )

    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))
    assert np.all((pred >= 0.0) & (pred <= 1.0))
