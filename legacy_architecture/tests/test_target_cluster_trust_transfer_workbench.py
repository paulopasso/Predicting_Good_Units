from __future__ import annotations

import pandas as pd

from crash_tests.ml_statistics_workbench.target_cluster_trust_transfer import (
    TargetClusterTrustTransferConfig,
    _build_real_candidate_pool,
)


def test_build_real_candidate_pool_uses_observed_target_values() -> None:
    unlabeled_df = pd.DataFrame(
        {
            "row_uid": ["u1", "u2", "u3"],
            "recording_key": ["r1", "r2", "r3"],
            "study_set": ["S1", "S1", "S2"],
            "accuracy": [0.0, 0.5, 1.0],
        }
    )
    teacher_unlabeled_df = pd.DataFrame(
        {
            "teacher_mean": [0.1, 0.4, 0.8],
            "teacher_std": [0.02, 0.03, 0.04],
        }
    )
    support_df = pd.DataFrame(
        {
            "unlabeled_row_idx": [0, 1, 2],
            "support_distance": [1.0, 0.5, 0.2],
            "support_confidence": [0.3, 0.8, 0.9],
            "cluster_id": [0, 0, 1],
            "cluster_labeled_count": [5, 5, 6],
            "cluster_labeled_share": [0.4, 0.4, 0.6],
        }
    )
    config = TargetClusterTrustTransferConfig(
        target="accuracy",
        real_candidate_support_quantile=0.5,
        real_label_weight=1.0,
        verbose=False,
    )

    candidate_df, all_real = _build_real_candidate_pool(
        target="accuracy",
        unlabeled_df=unlabeled_df,
        teacher_unlabeled_df=teacher_unlabeled_df,
        support_df=support_df,
        config=config,
    )

    assert candidate_df["pseudo_label"].tolist() == [0.0, 0.5, 1.0]
    assert all_real["pseudo_weight"].tolist() == [1.0, 1.0, 1.0]
    assert candidate_df["selected_candidate"].sum() == 2
