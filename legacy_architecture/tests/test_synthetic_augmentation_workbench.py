from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.ml_statistics_workbench.synthetic_augmentation import (
    SyntheticAugmentationConfig,
    _augment_subset,
)


def test_augment_subset_returns_weighted_augmented_rows() -> None:
    feature_frame = pd.DataFrame(
        {
            "x1": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1],
            "x2": [1.0, 1.1, 1.2, 1.3, 2.0, 2.1],
        }
    )
    target = np.array([0.0, 0.05, 0.1, 0.12, 0.8, 0.85], dtype=float)
    subgroup = pd.Series(["A", "A", "A", "A", "B", "B"])
    config = SyntheticAugmentationConfig(
        augmentation_factor=1.0,
        augmentation_weight=0.4,
        augmentation_noise_scale=0.0,
        augmentation_k_neighbors=2,
        augmentation_alpha_range=(0.25, 0.75),
    )
    aug_frame, aug_target, aug_weight = _augment_subset(
        feature_frame,
        target,
        subgroup_labels=subgroup,
        config=config,
        random_state=42,
    )
    assert len(aug_frame) > len(feature_frame)
    assert len(aug_frame) == len(aug_target) == len(aug_weight)
    assert np.allclose(aug_weight[: len(feature_frame)], 1.0)
    assert np.allclose(aug_weight[len(feature_frame) :], 0.4)


def test_augment_subset_skips_tiny_frames() -> None:
    feature_frame = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 2.0, 3.0]})
    target = np.array([0.1, 0.2, 0.3], dtype=float)
    subgroup = pd.Series(["A", "A", "A"])
    config = SyntheticAugmentationConfig(augmentation_factor=1.0)
    aug_frame, aug_target, aug_weight = _augment_subset(
        feature_frame,
        target,
        subgroup_labels=subgroup,
        config=config,
        random_state=0,
    )
    assert len(aug_frame) == len(feature_frame)
    assert np.allclose(aug_target, target)
    assert np.allclose(aug_weight, 1.0)
