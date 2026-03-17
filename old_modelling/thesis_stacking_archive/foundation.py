from __future__ import annotations

import numpy as np
import pandas as pd

from .methods import (
    build_stack_frame,
    build_stack_set,
    filter_target_rows,
    mask_feature_columns,
    select_shift_filtered_features,
    select_shift_protected_features,
)
from .pseudo_label_manifold import (
    PseudoLabelManifoldConfig,
    SupportArtifacts,
    TeacherBundle,
    _aggregate_teacher_predictions,
    _build_filtered_drop_cols,
    _build_support_artifacts,
    _fit_global_transport,
    _fit_source_teacher_predictions,
    _manifold_support_table,
    _select_pseudo_labels,
    _teacher_frames_for_target,
    _teacher_predictions,
    _teacher_weight_table,
)
from .specs import FPosClusterTrustPseudoConfig


def _trust_from_group_stats(
    group_df: pd.DataFrame,
    *,
    group_col: str,
) -> pd.DataFrame:
    overall_mae = float(group_df["teacher_abs_error"].mean())
    max_n = max(int(group_df[group_col].value_counts().max()), 1)
    stats = (
        group_df.groupby(group_col, dropna=False)
        .agg(
            labeled_rows=("row_uid", "size"),
            teacher_mae=("teacher_abs_error", "mean"),
            mean_support_confidence=("support_confidence", "mean"),
        )
        .reset_index()
    )
    err_score = np.exp(-stats["teacher_mae"].astype(float) / max(overall_mae, 1e-6))
    size_score = np.sqrt(stats["labeled_rows"].astype(float) / float(max_n))
    support_score = np.clip(stats["mean_support_confidence"].astype(float), 0.0, 1.0)
    trust = 0.60 * err_score + 0.25 * support_score + 0.15 * size_score
    trust = trust / max(float(np.nanmax(trust)), 1e-6)
    stats["trust_score"] = trust.astype(float)
    return stats


def _apply_trust_selection(
    *,
    candidate_df: pd.DataFrame,
    cluster_trust_df: pd.DataFrame,
    study_trust_df: pd.DataFrame,
    max_rows: int,
    config: FPosClusterTrustPseudoConfig,
) -> pd.DataFrame:
    pool = candidate_df[candidate_df["selected_candidate"].astype(bool)].copy()
    if pool.empty:
        return pool

    pool = pool.merge(
        cluster_trust_df.loc[:, ["cluster_id", "trust_score"]].rename(columns={"trust_score": "cluster_trust"}),
        on="cluster_id",
        how="left",
    )
    pool = pool.merge(
        study_trust_df.loc[:, ["study_set", "trust_score"]].rename(columns={"trust_score": "study_trust"}),
        on="study_set",
        how="left",
    )
    pool["cluster_trust"] = pool["cluster_trust"].fillna(float(config.min_trust))
    pool["study_trust"] = pool["study_trust"].fillna(float(config.min_trust))
    pool["combined_trust"] = (
        ((1.0 - float(config.study_trust_weight)) * pool["cluster_trust"].astype(float))
        + (float(config.study_trust_weight) * pool["study_trust"].astype(float))
    ).clip(lower=float(config.min_trust), upper=1.0)
    std_denom = max(float(np.nanquantile(pool["teacher_std"].astype(float), 0.75)), 1e-6)
    base_conf = (
        0.65 * np.exp(-pool["teacher_std"].astype(float) / std_denom)
        + 0.35 * np.clip(pool["support_confidence"].astype(float), 0.0, 1.0)
    )
    pool["base_pseudo_weight"] = np.clip(0.25 * base_conf, 0.08, 0.25)
    pool["trust_adjusted_weight"] = (
        pool["base_pseudo_weight"].astype(float)
        * np.power(pool["combined_trust"].astype(float), float(config.cluster_trust_power))
    ).clip(lower=0.08, upper=0.25)
    cluster_caps = (
        pool.groupby("cluster_id")["combined_trust"]
        .mean()
        .map(
            lambda trust: max(
                1,
                min(
                    int(config.max_pseudo_per_cluster),
                    int(
                        round(
                            min(
                                float(config.max_pseudo_per_cluster),
                                float(config.max_pseudo_per_cluster)
                                * (max(float(trust), float(config.min_trust)) ** float(config.cluster_trust_power)),
                            )
                        )
                    ),
                ),
            )
        )
        .to_dict()
    )
    cluster_ratio_caps = (
        pool.groupby("cluster_id")["cluster_labeled_count"]
        .max()
        .map(
            lambda labeled_count: max(
                1,
                int(round(float(labeled_count) * float(config.max_pseudo_per_labeled_ratio))),
            )
        )
        .to_dict()
    )

    selected_rows: list[pd.Series] = []
    recording_counts: dict[str, int] = {}
    cluster_counts: dict[int, int] = {}
    pool = pool.sort_values(
        ["trust_adjusted_weight", "combined_trust", "support_confidence", "base_pseudo_weight", "teacher_std"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    for _, row in pool.iterrows():
        if len(selected_rows) >= int(max_rows):
            break
        recording_key = str(row["recording_key"])
        cluster_id = int(row["cluster_id"])
        if recording_counts.get(recording_key, 0) >= int(config.max_pseudo_per_recording):
            continue
        cluster_cap = min(
            cluster_caps.get(cluster_id, int(config.max_pseudo_per_cluster)),
            cluster_ratio_caps.get(cluster_id, int(config.max_pseudo_per_cluster)),
        )
        if cluster_counts.get(cluster_id, 0) >= int(cluster_cap):
            continue
        selected_rows.append(row)
        recording_counts[recording_key] = recording_counts.get(recording_key, 0) + 1
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    if not selected_rows:
        return pool.head(0).copy()

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    if "pseudo_label" not in selected_df.columns:
        selected_df["pseudo_label"] = selected_df["teacher_mean"].astype(float)
    selected_df["pseudo_weight"] = selected_df["trust_adjusted_weight"].astype(np.float32)
    return selected_df

__all__ = [
    "FPosClusterTrustPseudoConfig",
    "PseudoLabelManifoldConfig",
    "SupportArtifacts",
    "TeacherBundle",
    "_apply_trust_selection",
    "_aggregate_teacher_predictions",
    "_build_filtered_drop_cols",
    "_build_support_artifacts",
    "_fit_global_transport",
    "_fit_source_teacher_predictions",
    "_manifold_support_table",
    "_select_pseudo_labels",
    "_teacher_frames_for_target",
    "_teacher_predictions",
    "_teacher_weight_table",
    "_trust_from_group_stats",
    "build_stack_frame",
    "build_stack_set",
    "filter_target_rows",
    "mask_feature_columns",
    "select_shift_filtered_features",
    "select_shift_protected_features",
]
