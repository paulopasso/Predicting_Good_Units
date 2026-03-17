from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.domain_shift_audit.report import (
    build_pca_projection,
    build_domain_classifier_report,
    compute_feature_shift_table,
    feature_group_for_column,
)


def test_feature_group_for_column_maps_expected_prefixes() -> None:
    assert feature_group_for_column("wf_bin_01") == "waveform_bins"
    assert feature_group_for_column("acg_03") == "acg_bins"
    assert feature_group_for_column("amp_mean") == "amplitude"
    assert feature_group_for_column("si_drift_ptp") == "spikeinterface"
    assert feature_group_for_column("max_cosine") == "confusability"
    assert feature_group_for_column("snr_recording_mean") == "recording_context"
    assert feature_group_for_column("snr_recording_pct") == "recording_relative"


def test_compute_feature_shift_table_reports_stronger_shift_for_changed_feature() -> None:
    a = pd.DataFrame({"x": [0.0, 0.1, 0.2, 0.3], "y": [1.0, 1.1, 1.0, 1.1]})
    b = pd.DataFrame({"x": [2.0, 2.1, 2.2, 2.3], "y": [1.0, 1.1, 1.0, 1.1]})
    shift = compute_feature_shift_table(a, b, feature_cols=["x", "y"], comparison="a_vs_b", feature_set="unit")
    top = shift.iloc[0]
    assert top["feature"] == "x"
    assert top["abs_smd"] > shift.loc[shift["feature"] == "y", "abs_smd"].iloc[0]


def test_build_domain_classifier_report_separates_easy_domains() -> None:
    a = pd.DataFrame(
        {
            "recording_key": ["a1", "a2", "a3", "a4"],
            "row_uid": ["a1", "a2", "a3", "a4"],
            "study_set": ["hybrid"] * 4,
            "dataset_type": ["hybrid"] * 4,
            "f1": [0.0, 0.1, 0.2, 0.3],
            "f2": [0.0, 0.1, 0.0, 0.1],
        }
    )
    b = pd.DataFrame(
        {
            "recording_key": ["b1", "b2", "b3", "b4"],
            "row_uid": ["b1", "b2", "b3", "b4"],
            "study_set": ["paired"] * 4,
            "dataset_type": ["paired"] * 4,
            "f1": [3.0, 3.1, 3.2, 3.3],
            "f2": [1.0, 1.1, 1.0, 1.1],
        }
    )
    summary, coefs = build_domain_classifier_report(
        a,
        b,
        feature_cols=["f1", "f2"],
        comparison="hybrid_vs_paired",
        feature_set="unit",
        random_state=42,
        max_rows_per_domain=10,
    )
    assert float(summary.iloc[0]["cv_auc_mean"]) > 0.9
    assert set(coefs["feature"]) == {"f1", "f2"}


def test_build_pca_projection_supports_three_components() -> None:
    a = pd.DataFrame(
        {
            "recording_key": ["a1", "a2", "a3", "a4"],
            "row_uid": ["a1", "a2", "a3", "a4"],
            "study_set": ["hybrid"] * 4,
            "dataset_type": ["hybrid"] * 4,
            "f1": [0.0, 0.1, 0.2, 0.3],
            "f2": [0.0, 0.1, 0.0, 0.1],
            "f3": [1.0, 1.1, 1.2, 1.3],
        }
    )
    b = pd.DataFrame(
        {
            "recording_key": ["b1", "b2", "b3", "b4"],
            "row_uid": ["b1", "b2", "b3", "b4"],
            "study_set": ["paired"] * 4,
            "dataset_type": ["paired"] * 4,
            "f1": [3.0, 3.1, 3.2, 3.3],
            "f2": [1.0, 1.1, 1.0, 1.1],
            "f3": [2.0, 2.1, 2.2, 2.3],
        }
    )
    projection = build_pca_projection(
        a,
        b,
        feature_cols=["f1", "f2", "f3"],
        comparison="hybrid_vs_paired",
        random_state=42,
        rows_per_domain=10,
        n_components=3,
    )
    assert {"pc1", "pc2", "pc3"}.issubset(projection.columns)
    assert len(projection) == 8
