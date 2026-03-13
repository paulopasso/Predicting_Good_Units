from __future__ import annotations

import numpy as np
import pandas as pd

from qc_framework.features import (
    FeatureSetBuilder,
    FeatureSpec,
    RecordingContextFeatureAugmenter,
    RecordingRelativeFeatureAugmenter,
    StudyIndicatorAugmenter,
)


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


def test_recording_context_and_study_features_are_generated() -> None:
    df = pd.DataFrame(
        {
            "recording_key": ["a", "a", "b"],
            "study_set": ["PAIRED_ENGLISH", "PAIRED_ENGLISH", "PAIRED_KAMPFF"],
            "snr": [1.0, 3.0, 10.0],
            "wf_bin_00": [0.1, 0.2, 0.3],
            "acg_00": [0.4, 0.5, 0.6],
        }
    )
    context_df = RecordingContextFeatureAugmenter(cols_to_summarize=["snr"]).fit_transform(df)
    full_df = StudyIndicatorAugmenter().fit_transform(context_df)

    assert "snr_recording_mean" in full_df.columns
    assert "snr_recording_median" in full_df.columns
    assert "snr_recording_iqr" in full_df.columns
    assert "snr_recording_delta" in full_df.columns
    assert "study_indicator__paired_english" in full_df.columns
    assert "study_indicator__paired_kampff" in full_df.columns


def test_transfer_feature_columns_include_context_blocks() -> None:
    df = pd.DataFrame(
        {
            "recording_key": ["r1", "r1"],
            "snr": [1.0, 2.0],
            "snr_z": [-0.7, 0.7],
            "snr_recording_pct": [0.25, 0.75],
            "snr_recording_mean": [1.5, 1.5],
            "study_indicator__paired_english": [1.0, 1.0],
            "wf_bin_00": [0.1, 0.2],
            "acg_00": [0.3, 0.4],
        }
    )
    spec = FeatureSpec(
        mode="unit_only",
        numeric_cols=["snr", "wf_bin_00", "acg_00"],
        categorical_cols=[],
    )

    cols = FeatureSetBuilder.transfer_feature_columns(
        df,
        spec,
        view_names=["unit_raw", "recording_relative", "recording_context", "study_identity"],
        normalized_raw_cols=["snr"],
    )

    assert "snr" in cols
    assert "snr_recording_pct" in cols
    assert "snr_recording_mean" in cols
    assert "study_indicator__paired_english" in cols
