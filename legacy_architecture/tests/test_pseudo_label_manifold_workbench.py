from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.ml_statistics_workbench.pseudo_label_manifold import (
    PseudoLabelManifoldConfig,
    _assemble_student_training_data,
    _select_pseudo_labels,
    _teacher_weight_table,
)


def test_assemble_student_training_data_uses_original_unlabeled_indices() -> None:
    labeled_frame = pd.DataFrame({"feat": [1.0, 2.0]})
    labeled_y = np.array([0.1, 0.2], dtype=float)
    unlabeled_frame = pd.DataFrame({"feat": [10.0, 11.0, 12.0, 13.0]})
    pseudo_selected = pd.DataFrame(
        {
            "unlabeled_row_idx": [2, 0],
            "pseudo_label": [0.7, 0.8],
            "pseudo_weight": [0.2, 0.3],
        }
    )

    train_frame, train_y, train_weight = _assemble_student_training_data(
        labeled_frame=labeled_frame,
        labeled_y=labeled_y,
        labeled_indices=None,
        pseudo_selected=pseudo_selected,
        unlabeled_frame=unlabeled_frame,
    )

    assert train_frame["feat"].tolist() == [1.0, 2.0, 12.0, 10.0]
    assert np.allclose(train_y, [0.1, 0.2, 0.7, 0.8])
    assert np.allclose(train_weight, [1.0, 1.0, 0.2, 0.3])


def test_select_pseudo_labels_respects_recording_and_cluster_caps() -> None:
    unlabeled_df = pd.DataFrame(
        {
            "row_uid": [f"u{i}" for i in range(6)],
            "recording_key": ["r1", "r1", "r2", "r2", "r3", "r3"],
            "study_set": ["A", "A", "A", "A", "B", "B"],
        }
    )
    unlabeled_teacher_df = pd.DataFrame(
        {
            "teacher_mean": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "teacher_std": [0.01, 0.02, 0.01, 0.02, 0.01, 0.02],
        }
    )
    support_df = pd.DataFrame(
        {
            "unlabeled_row_idx": np.arange(6, dtype=int),
            "row_uid": unlabeled_df["row_uid"],
            "recording_key": unlabeled_df["recording_key"],
            "study_set": unlabeled_df["study_set"],
            "support_distance": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            "support_confidence": [0.9, 0.85, 0.9, 0.85, 0.9, 0.85],
            "cluster_id": [0, 0, 1, 1, 1, 1],
            "cluster_labeled_count": [5, 5, 10, 10, 10, 10],
            "cluster_labeled_share": [0.33, 0.33, 0.67, 0.67, 0.67, 0.67],
        }
    )
    config = PseudoLabelManifoldConfig(
        support_quantile=1.0,
        agreement_quantile=1.0,
        cluster_coverage_quantile=0.0,
        min_labeled_cluster_rows=5,
        max_pseudo_per_recording=1,
        max_pseudo_per_cluster=2,
        max_pseudo_per_labeled_ratio=0.5,
        pseudo_weight_scale=0.3,
        pseudo_weight_floor=0.1,
        verbose=False,
    )

    candidate_df, selected_df = _select_pseudo_labels(
        unlabeled_df=unlabeled_df,
        unlabeled_teacher_df=unlabeled_teacher_df,
        support_df=support_df,
        config=config,
        max_rows=6,
    )

    assert int(selected_df["recording_key"].nunique()) == len(selected_df)
    assert selected_df["cluster_id"].value_counts().max() <= 2
    assert selected_df.groupby("cluster_id")["unlabeled_row_idx"].count().to_dict()[0] <= 2
    assert candidate_df["selected_final"].sum() == len(selected_df)


def test_teacher_weight_table_falls_back_to_inverse_mae_when_r2_is_negative() -> None:
    perf = pd.DataFrame(
        {
            "variant_id": ["a", "b", "c"],
            "oof_r2": [-0.2, -0.1, -0.3],
            "oof_mae": [0.30, 0.20, 0.40],
        }
    )
    weighted = _teacher_weight_table(perf)
    weights = weighted.set_index("variant_id")["ensemble_weight"].to_dict()

    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["b"] > weights["a"] > weights["c"]
