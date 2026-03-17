from __future__ import annotations

import pandas as pd

from crash_tests.ml_statistics_workbench.fpos_cluster_trust_pseudo import (
    FPosClusterTrustPseudoConfig,
    _apply_trust_selection,
    _trust_from_group_stats,
)


def test_trust_from_group_stats_penalizes_higher_teacher_error() -> None:
    train_df = pd.DataFrame(
        {
            "row_uid": ["a", "b", "c", "d"],
            "cluster_id": [0, 0, 1, 1],
            "support_confidence": [0.8, 0.8, 0.8, 0.8],
            "teacher_abs_error": [0.05, 0.06, 0.20, 0.22],
        }
    )

    trust_df = _trust_from_group_stats(train_df, group_col="cluster_id")
    trust = trust_df.set_index("cluster_id")["trust_score"].to_dict()

    assert trust[0] > trust[1]


def test_apply_trust_selection_reallocates_away_from_low_trust_cluster() -> None:
    candidate_df = pd.DataFrame(
        {
            "selected_candidate": [True] * 6,
            "row_uid": [f"u{i}" for i in range(6)],
            "recording_key": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "study_set": ["S1", "S1", "S2", "S2", "S3", "S3"],
            "cluster_id": [0, 0, 1, 1, 2, 2],
            "cluster_labeled_count": [10, 10, 10, 10, 2, 2],
            "support_confidence": [0.9, 0.8, 0.7, 0.7, 0.4, 0.3],
            "teacher_std": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "teacher_mean": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        }
    )
    cluster_trust_df = pd.DataFrame(
        {
            "cluster_id": [0, 1, 2],
            "trust_score": [1.0, 0.7, 0.25],
        }
    )
    study_trust_df = pd.DataFrame(
        {
            "study_set": ["S1", "S2", "S3"],
            "trust_score": [1.0, 0.8, 0.3],
        }
    )
    config = FPosClusterTrustPseudoConfig(
        max_pseudo_per_recording=2,
        max_pseudo_per_cluster=4,
        max_pseudo_per_labeled_ratio=1.0,
        cluster_trust_power=1.5,
        study_trust_weight=0.3,
        min_trust=0.25,
        verbose=False,
    )

    selected = _apply_trust_selection(
        candidate_df=candidate_df,
        cluster_trust_df=cluster_trust_df,
        study_trust_df=study_trust_df,
        max_rows=4,
        config=config,
    )

    counts = selected["cluster_id"].value_counts().to_dict()
    assert counts.get(0, 0) >= counts.get(2, 0)
    assert "pseudo_label" in selected.columns
    assert "pseudo_weight" in selected.columns
