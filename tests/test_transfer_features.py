from __future__ import annotations

import numpy as np
import pandas as pd

from qc_framework.features import FeatureSetBuilder, FeatureSpec, RecordingRelativeFeatureAugmenter


def test_norm_swap_excludes_raw_normalized_columns() -> None:
    df = pd.DataFrame(
        {
            "recording_key": ["r1", "r1"],
            "snr": [1.0, 2.0],
            "snr_z": [-0.7, 0.7],
            "wf_bin_00": [0.1, 0.2],
            "acg_00": [0.3, 0.4],
            "other_feature": [5.0, 7.0],
        }
    )
    spec = FeatureSpec(
        mode="unit_only",
        numeric_cols=["snr", "wf_bin_00", "acg_00", "other_feature"],
        categorical_cols=[],
    )

    cols = FeatureSetBuilder.transfer_feature_columns(
        df,
        spec,
        view_names=["norm_swap"],
        normalized_raw_cols=["snr"],
    )

    assert "snr" not in cols
    assert "snr_z" in cols
    assert "other_feature" in cols


def test_recording_relative_feature_percentiles_are_centered() -> None:
    df = pd.DataFrame(
        {
            "recording_key": ["a", "a", "a", "b"],
            "snr": [1.0, 2.0, 3.0, 10.0],
        }
    )
    augmented = RecordingRelativeFeatureAugmenter(cols_to_rank=["snr"]).fit_transform(df)

    vals = augmented["snr_recording_pct"].round(4).tolist()
    assert vals[:3] == [0.1667, 0.5, 0.8333]
    assert vals[3] == 0.5
