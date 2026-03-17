from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.ml_statistics_workbench.fpos_tree_interaction_stack import (
    FPosTreeInteractionStackConfig,
    _fit_leaf_encoder,
    _interaction_feature_subset,
    _transform_leaf_features,
)


def test_interaction_feature_subset_prefers_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "source_pred": [0.1, 0.2],
            "anchor_fp_score": [0.3, 0.4],
            "wf_embed_00": [0.5, 0.6],
            "unused": [1.0, 2.0],
        }
    )
    cols = _interaction_feature_subset(frame)
    assert cols == ["source_pred", "anchor_fp_score", "wf_embed_00"]


def test_leaf_encoder_returns_stable_feature_matrix() -> None:
    X_train = pd.DataFrame(
        {
            "source_pred": [0.1, 0.2, 0.7, 0.8],
            "anchor_fp_score": [0.3, 0.25, 0.9, 0.85],
            "wf_embed_00": [0.0, 0.1, 0.8, 0.9],
        }
    )
    y_train = np.array([0.05, 0.1, 0.75, 0.85], dtype=float)
    weight = np.ones(len(X_train), dtype=np.float32)
    config = FPosTreeInteractionStackConfig(interaction_trees=6, interaction_max_depth=3, verbose=False)

    model, imputer, encoder = _fit_leaf_encoder(
        X_train,
        y_train,
        weight,
        config=config,
        random_state=42,
    )
    leaf_df = _transform_leaf_features(model, imputer, encoder, X_train, prefix="tree")

    assert leaf_df.shape[0] == len(X_train)
    assert leaf_df.shape[1] > 0
    assert np.all(np.isfinite(leaf_df.to_numpy(dtype=float)))
