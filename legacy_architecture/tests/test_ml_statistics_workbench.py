from __future__ import annotations

import numpy as np
import pandas as pd

from crash_tests.ml_statistics_workbench.methods import (
    apply_quantile_transport_model,
    apply_pca_support_projector,
    apply_study_bias_correction,
    assign_kmeans_subgroups,
    CORALAligner,
    build_stack_frame,
    compute_overlap_importance_weights,
    compute_soft_group_balance_weights,
    compute_importance_weights,
    fit_kmeans_subgroup_model,
    fit_pca_support_projector,
    fit_latent_cluster_model,
    fit_quantile_transport_model,
    fit_study_bias_correction,
    latent_cluster_responsibilities,
    leave_one_out_study_bias_adjustment,
    make_local_regression_mixup,
    mask_feature_columns,
    make_latent_summary_frame,
    make_study_indicator_frame,
    pick_r2_first_recommendation,
    rank_latent_dims_by_correlation,
    select_shift_filtered_features,
    select_shift_protected_features,
    select_target_transport_features,
)


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, columns=["a", "b", "c"])


def test_coral_alignment_reduces_source_target_mean_gap() -> None:
    source = _frame(np.array([[0.0, 0.5, 1.0], [0.2, 0.4, 1.1], [0.1, 0.6, 0.9]]))
    target = _frame(np.array([[3.0, 3.5, 4.0], [3.2, 3.4, 4.2], [2.8, 3.6, 3.8]]))
    aligner = CORALAligner().fit(source, target)
    aligned_source = aligner.transform_source(source)
    raw_gap = np.abs(source.mean(axis=0).to_numpy() - target.mean(axis=0).to_numpy()).mean()
    aligned_gap = np.abs(aligned_source.mean(axis=0) - target.mean(axis=0).to_numpy()).mean()
    assert aligned_gap < raw_gap


def test_importance_weights_are_positive_and_clipped() -> None:
    source = _frame(np.array([[0.0, 0.0, 0.0], [0.2, 0.1, 0.3], [0.4, 0.2, 0.1], [0.5, 0.4, 0.2]]))
    target = _frame(np.array([[1.0, 1.1, 0.9], [0.9, 1.2, 1.0], [1.1, 0.8, 1.2], [1.2, 1.0, 1.1]]))
    result = compute_importance_weights(
        source,
        target,
        feature_cols=["a", "b", "c"],
        random_state=42,
        clip_min=0.3,
        clip_max=3.0,
        c_value=1.0,
    )
    assert np.isfinite(result.weights).all()
    assert np.all(result.weights >= 0.3)
    assert np.all(result.weights <= 3.0)


def test_overlap_importance_weights_downweight_out_of_support_rows() -> None:
    source = _frame(np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.2], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]))
    target = _frame(np.array([[0.0, 0.1, 0.0], [0.1, 0.0, 0.2], [0.2, 0.1, 0.1], [0.05, 0.0, 0.15]]))
    result = compute_overlap_importance_weights(
        source,
        target,
        feature_cols=["a", "b", "c"],
        random_state=42,
        clip_min=0.2,
        clip_max=4.0,
        c_value=1.0,
        support_quantiles=(0.05, 0.95),
        outside_support_scale=0.1,
    )
    assert np.isfinite(result.weights).all()
    assert result.summary["source_in_support_frac"] < 1.0
    assert np.all(result.weights >= 0.2)
    assert np.all(result.weights <= 4.0)


def test_build_stack_frame_expands_vector_features() -> None:
    df = pd.DataFrame({"ctx_1": [1.0, 2.0], "ctx_2": [3.0, 4.0]})
    stacked = build_stack_frame(
        df,
        context_cols=["ctx_1", "ctx_2"],
        extra_features={
            "source_pred": np.array([0.2, 0.8]),
            "latent": np.array([[1.0, 2.0], [3.0, 4.0]]),
        },
    )
    assert list(stacked.columns) == ["ctx_1", "ctx_2", "source_pred", "latent_00", "latent_01"]
    assert stacked.shape == (2, 5)


def test_make_study_indicator_frame_aligns_categories() -> None:
    train = pd.DataFrame({"study_set": ["PAIRED_A", "PAIRED_B", "PAIRED_A"]})
    test = pd.DataFrame({"study_set": ["PAIRED_B", "PAIRED_C"]})
    categories = ["PAIRED_A", "PAIRED_B", "PAIRED_C"]
    train_frame = make_study_indicator_frame(train, categories=categories)
    test_frame = make_study_indicator_frame(test, categories=categories)
    assert list(train_frame.columns) == ["study_PAIRED_A", "study_PAIRED_B", "study_PAIRED_C"]
    assert list(test_frame.columns) == ["study_PAIRED_A", "study_PAIRED_B", "study_PAIRED_C"]
    assert train_frame.iloc[0].tolist() == [1.0, 0.0, 0.0]
    assert test_frame.iloc[1].tolist() == [0.0, 0.0, 1.0]


def test_study_bias_correction_shrinks_toward_global_mean() -> None:
    studies = np.array(["A", "A", "B", "B", "B"])
    residuals = np.array([0.10, 0.14, -0.20, -0.18, -0.22], dtype=float)
    model = fit_study_bias_correction(studies, residuals, shrinkage=4.0)
    corrected = apply_study_bias_correction(
        np.array([0.50, 0.50], dtype=float),
        np.array(["A", "B"]),
        model,
    )
    assert corrected[0] > corrected[1]
    assert model.study_offsets["A"] < residuals[:2].mean()


def test_leave_one_out_study_bias_adjustment_returns_finite_offsets() -> None:
    studies = np.array(["A", "A", "A", "B", "B"])
    residuals = np.array([0.10, 0.20, 0.00, -0.30, -0.10], dtype=float)
    adjustment = leave_one_out_study_bias_adjustment(studies, residuals, shrinkage=2.0)
    assert adjustment.shape == residuals.shape
    assert np.isfinite(adjustment).all()
    assert adjustment[0] > adjustment[3]


def test_make_latent_summary_frame_builds_expected_columns() -> None:
    latent = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=float)
    frame = make_latent_summary_frame(latent, prefix="latent", quantiles=(0.25, 0.50))
    assert list(frame.columns) == [
        "latent_mean",
        "latent_std",
        "latent_min",
        "latent_max",
        "latent_l2",
        "latent_q25",
        "latent_q50",
    ]
    assert frame.shape == (2, 7)


def test_rank_latent_dims_by_correlation_prefers_signal_dimension() -> None:
    latent = np.array(
        [
            [0.0, 10.0],
            [0.1, 9.8],
            [0.2, 9.9],
            [0.9, 0.1],
            [1.0, 0.2],
            [1.1, 0.3],
        ],
        dtype=float,
    )
    target = np.array([0.0, 0.1, 0.2, 0.9, 1.0, 1.1], dtype=float)
    groups = np.array(["a", "a", "b", "c", "d", "e"])
    ranked = rank_latent_dims_by_correlation(
        latent,
        target,
        groups=groups,
        n_splits=2,
        test_size=0.4,
        random_state=42,
    )
    assert int(ranked.iloc[0]["latent_dim"]) == 0


def test_pick_r2_first_recommendation_obeys_guardrail() -> None:
    metrics = pd.DataFrame(
        [
            {"target": "fpos", "variant_id": "ref", "mae": 0.100, "r2": 0.45},
            {"target": "fpos", "variant_id": "better_r2", "mae": 0.102, "r2": 0.49},
            {"target": "fpos", "variant_id": "too_high_mae", "mae": 0.120, "r2": 0.60},
        ]
    )
    rec = pick_r2_first_recommendation(
        metrics,
        target="fpos",
        reference_variant_id="ref",
        selection_metric="r2",
        mae_guardrail_pct=0.03,
        min_r2_improvement=0.0,
    )
    assert rec["recommended_variant_id"] == "better_r2"


def test_select_shift_filtered_features_preserves_shape_columns() -> None:
    shift = pd.DataFrame(
        {
            "feature": ["snr", "wf_bin_01", "amp_mean", "acg_01"],
            "feature_group": ["other", "waveform_bins", "amplitude", "acg_bins"],
            "abs_smd": [2.0, 3.0, 1.5, 2.5],
            "ks_stat": [0.8, 0.9, 0.7, 0.6],
            "missing_gap": [0.0, 0.0, 0.0, 0.0],
        }
    )
    manifest, drop_cols = select_shift_filtered_features(
        shift,
        allowed_cols=["snr", "wf_bin_01", "amp_mean", "acg_01"],
        top_n=3,
        min_abs_smd=1.0,
        preserve_shape_features=True,
        feature_set="unit",
    )
    assert drop_cols == ["amp_mean", "snr"]
    assert manifest.loc[manifest["feature"] == "wf_bin_01", "protected_reason"].iloc[0] == "preserve_shape_feature"


def test_select_shift_protected_features_preserves_predictive_shifted_feature() -> None:
    shift = pd.DataFrame(
        {
            "feature": ["snr", "amp_mean", "wf_bin_01"],
            "feature_group": ["other", "amplitude", "waveform_bins"],
            "abs_smd": [2.5, 2.2, 3.0],
            "ks_stat": [0.8, 0.7, 0.9],
            "missing_gap": [0.0, 0.0, 0.0],
        }
    )
    train = pd.DataFrame(
        {
            "snr": [0.0, 0.1, 0.2, 0.25, 0.35, 0.45, 0.65, 0.8, 0.9, 1.0],
            "amp_mean": [1.0] * 10,
            "wf_bin_01": [0.2, 0.25, 0.3, 0.35, 0.37, 0.41, 0.44, 0.47, 0.5, 0.54],
            "fmiss": [0.0, 0.08, 0.15, 0.2, 0.28, 0.4, 0.62, 0.78, 0.9, 1.0],
        }
    )
    manifest, drop_cols = select_shift_protected_features(
        shift,
        train_df=train,
        target="fmiss",
        allowed_cols=["snr", "amp_mean", "wf_bin_01"],
        top_n=2,
        min_abs_smd=1.0,
        preserve_shape_features=True,
        protect_top_k=1,
        min_abs_corr_to_protect=0.05,
        feature_set="full_context_target_protected",
    )
    assert "snr" not in drop_cols
    protected_reason = manifest.loc[manifest["feature"] == "snr", "protected_reason"].iloc[0]
    assert "preserve_target_predictive_feature" in protected_reason


def test_mask_feature_columns_sets_selected_features_to_nan() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    masked = mask_feature_columns(df, ["b"])
    assert masked["a"].tolist() == [1.0, 2.0]
    assert masked["b"].isna().all()


def test_latent_cluster_responsibilities_sum_to_one() -> None:
    latent = np.array(
        [
            [0.0, 0.1],
            [0.1, 0.0],
            [3.0, 3.1],
            [3.2, 2.9],
        ],
        dtype=float,
    )
    cluster_model = fit_latent_cluster_model(latent, n_clusters=2, random_state=42)
    resp = latent_cluster_responsibilities(latent, cluster_model)
    assert resp.shape == (4, 2)
    assert np.allclose(resp.sum(axis=1), 1.0)
    assert np.all(resp >= 0.0)


def test_compute_soft_group_balance_weights_upweights_minor_group() -> None:
    resp = np.array(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.10, 0.90],
        ],
        dtype=float,
    )
    weights = compute_soft_group_balance_weights(resp, reference_mass=np.array([10.0, 2.0]), power=1.0)
    assert weights.shape == (3,)
    assert np.isfinite(weights).all()
    assert float(weights[2]) > float(weights[0])


def test_make_local_regression_mixup_returns_bounded_targets() -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 0.1, 0.9, 1.0],
            "x2": [0.0, 0.2, 0.8, 1.0],
        }
    )
    y = np.array([0.0, 0.1, 0.9, 1.0], dtype=float)
    synth_x, synth_y = make_local_regression_mixup(
        frame,
        y,
        subgroup_labels=np.array(["A", "A", "B", "B"]),
        n_synthetic=6,
        k_neighbors=2,
        alpha_range=(0.3, 0.7),
        random_state=42,
    )
    assert len(synth_x) == len(synth_y)
    assert len(synth_x) > 0
    assert list(synth_x.columns) == ["x1", "x2"]
    assert np.all(np.isfinite(synth_x.to_numpy()))
    assert np.all((synth_y >= 0.0) & (synth_y <= 1.0))


def test_quantile_transport_moves_source_toward_target_distribution() -> None:
    source = pd.DataFrame({"x": np.linspace(0.0, 1.0, 200), "y": np.linspace(-1.0, 0.0, 200)})
    target = pd.DataFrame({"x": np.linspace(4.0, 5.0, 200), "y": np.linspace(2.0, 3.0, 200)})
    shift = pd.DataFrame(
        {
            "feature": ["x", "y"],
            "abs_smd": [2.0, 2.0],
            "ks_stat": [0.9, 0.9],
        }
    )
    model = fit_quantile_transport_model(
        source,
        target,
        feature_cols=["x", "y"],
        shift_df=shift,
        top_n=2,
        min_abs_smd=0.5,
        quantile_points=65,
        blend=1.0,
    )
    transformed = apply_quantile_transport_model(source, model, random_state=42)
    raw_gap = abs(float(source["x"].mean()) - float(target["x"].mean())) + abs(float(source["y"].mean()) - float(target["y"].mean()))
    new_gap = abs(float(transformed["x"].mean()) - float(target["x"].mean())) + abs(float(transformed["y"].mean()) - float(target["y"].mean()))
    assert new_gap < raw_gap


def test_kmeans_subgroup_assigner_returns_valid_cluster_ids() -> None:
    df = pd.DataFrame(
        {
            "a": [0.0, 0.1, 0.2, 3.0, 3.1, 3.2],
            "b": [0.0, 0.2, 0.1, 3.1, 3.0, 3.2],
        }
    )
    model = fit_kmeans_subgroup_model(
        df,
        feature_cols=["a", "b"],
        n_clusters=2,
        random_state=42,
    )
    labels = assign_kmeans_subgroups(df, model)
    assert labels.shape == (len(df),)
    assert set(labels.tolist()) <= {0, 1}


def test_select_target_transport_features_prefers_shifted_predictive_columns() -> None:
    shift = pd.DataFrame(
        {
            "feature": ["snr", "amp_mean", "wf_bin_17"],
            "abs_smd": [2.5, 1.5, 2.0],
            "ks_stat": [0.9, 0.7, 0.8],
        }
    )
    train = pd.DataFrame(
        {
            "snr": [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2],
            "amp_mean": [1.0, 1.1, 1.0, 1.2, 1.1, 1.0, 1.1, 1.0],
            "wf_bin_17": [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.5, 0.55],
            "fmiss": [0.0, 0.1, 0.15, 0.2, 0.45, 0.55, 0.8, 0.9],
        }
    )
    scored = select_target_transport_features(
        shift,
        train_df=train,
        target="fmiss",
        allowed_cols=["snr", "amp_mean", "wf_bin_17"],
        top_k=2,
        min_abs_smd=1.0,
        min_abs_corr=0.05,
    )
    selected = scored.loc[scored["selected_for_transport"], "feature"].astype(str).tolist()
    assert "snr" in selected
    assert "wf_bin_17" in selected


def test_pca_support_projector_moves_rows_toward_target_neighbors() -> None:
    target = pd.DataFrame(
        {
            "x": [2.0, 2.1, 1.9, 2.2, 2.05],
            "y": [3.0, 3.1, 2.9, 3.2, 3.05],
        }
    )
    source = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2],
            "y": [0.0, 0.1, 0.2],
        }
    )
    projector = fit_pca_support_projector(
        target,
        feature_cols=["x", "y"],
        n_components=2,
        n_neighbors=3,
    )
    moved = apply_pca_support_projector(
        source,
        projector,
        blend=0.5,
        noise_scale=0.0,
        random_state=42,
    )
    raw_gap = np.abs(source.mean(axis=0).to_numpy() - target.mean(axis=0).to_numpy()).sum()
    new_gap = np.abs(moved.mean(axis=0).to_numpy() - target.mean(axis=0).to_numpy()).sum()
    assert new_gap < raw_gap
