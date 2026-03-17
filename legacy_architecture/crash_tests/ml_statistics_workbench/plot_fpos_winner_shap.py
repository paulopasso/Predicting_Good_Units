from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.ml_statistics_workbench.config import MLStatisticsWorkbenchConfig
from crash_tests.ml_statistics_workbench.fpos_anchor_stack import _fit_anchor_scores
from crash_tests.ml_statistics_workbench.fpos_cluster_trust_pseudo import (
    FPosClusterTrustPseudoConfig,
    _apply_trust_selection,
    _trust_from_group_stats,
)
from crash_tests.ml_statistics_workbench.methods import build_stack_frame, mask_feature_columns
from crash_tests.ml_statistics_workbench.pseudo_label_manifold import (
    PseudoLabelManifoldConfig,
    _assemble_student_training_data,
    _build_filtered_drop_cols,
    _build_support_artifacts,
    _manifold_support_table,
    _select_pseudo_labels,
    _teacher_frames_for_target,
    _teacher_predictions,
)
from crash_tests.mmd_domain_adaptation.dataset import prepare_feature_table
from crash_tests.mmd_domain_adaptation.model import target_transform
from crash_tests.self_supervised_domain_adaptation.split_utils import make_main_split


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SHAP diagnostics for the fpos XGBoost winner")
    parser.add_argument(
        "--parquet",
        default="data/raw/33000_ROWS.parquet",
        help="Path to parquet dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="crash_tests/ml_statistics_workbench/outputs/fpos_backend_sweep_full_v1",
        help="Run directory where plots should be written",
    )
    return parser.parse_args()


def _refit_winner(
    parquet_path: Path,
) -> tuple[XGBRegressor, SimpleImputer, pd.DataFrame, pd.DataFrame, np.ndarray]:
    base = MLStatisticsWorkbenchConfig(
        parquet_path=parquet_path,
        output_root=Path("crash_tests/ml_statistics_workbench/outputs"),
        run_name="fpos_shap_refit",
        random_state=42,
        paired_final_holdout_rows=100,
        verbose=False,
    )
    pseudo_base = PseudoLabelManifoldConfig(
        parquet_path=parquet_path,
        output_root=Path("crash_tests/ml_statistics_workbench/outputs"),
        run_name="fpos_shap_refit",
        random_state=42,
        paired_final_holdout_rows=100,
        max_pseudo_multiplier=1.5,
        max_pseudo_per_recording=48,
        max_pseudo_per_cluster=128,
        max_pseudo_per_labeled_ratio=1.5,
        verbose=False,
    )
    trust_base = FPosClusterTrustPseudoConfig(
        parquet_path=parquet_path,
        output_root=Path("crash_tests/ml_statistics_workbench/outputs"),
        run_name="fpos_shap_refit",
        random_state=42,
        paired_final_holdout_rows=100,
        max_pseudo_multiplier=1.5,
        max_pseudo_per_recording=48,
        max_pseudo_per_cluster=128,
        max_pseudo_per_labeled_ratio=1.5,
        cluster_trust_power=1.35,
        study_trust_weight=0.30,
        min_trust=0.25,
        verbose=False,
    )

    prepared = prepare_feature_table(parquet_path)
    split = make_main_split(prepared.df, paired_final_holdout_rows=100, random_state=42)
    unit_drop_cols, context_drop_cols, _ = _build_filtered_drop_cols(
        split=split,
        prepared=prepared,
        base_config=base,
    )
    (
        target_train_df,
        y_train,
        target_test_df,
        y_test,
        unlabeled_df,
        teachers,
        _student_templates,
        _transport_manifest,
    ) = _teacher_frames_for_target(
        target="fpos",
        split=split,
        prepared=prepared,
        base=base,
        config=pseudo_base,
        unit_drop_cols=unit_drop_cols,
        context_drop_cols=context_drop_cols,
    )
    teacher_train_df, teacher_unlabeled_df, teacher_test_df, _ = _teacher_predictions(
        teachers=teachers,
        labeled_df=target_train_df,
        y_train=y_train,
        base=base,
    )

    support_artifacts = _build_support_artifacts(
        paired_non_test_all=split.paired_non_test_all,
        labeled_train_df=target_train_df,
        unit_cols=prepared.unit_feature_cols,
        config=pseudo_base,
    )
    support_df = _manifold_support_table(
        unlabeled_df=unlabeled_df,
        labeled_train_df=target_train_df,
        support_artifacts=support_artifacts,
    )
    support_train_df = _manifold_support_table(
        unlabeled_df=target_train_df,
        labeled_train_df=target_train_df,
        support_artifacts=support_artifacts,
    )
    support_test_df = _manifold_support_table(
        unlabeled_df=target_test_df,
        labeled_train_df=target_train_df,
        support_artifacts=support_artifacts,
    )

    train_trust_df = pd.concat(
        [
            target_train_df.loc[:, ["row_uid", "recording_key", "study_set"]].reset_index(drop=True),
            support_train_df.loc[:, ["cluster_id", "support_confidence"]].reset_index(drop=True),
        ],
        axis=1,
    )
    train_trust_df["teacher_abs_error"] = np.abs(
        y_train - teacher_train_df["teacher_mean"].to_numpy(dtype=float)
    )
    cluster_trust_df = _trust_from_group_stats(train_trust_df, group_col="cluster_id")
    study_trust_df = _trust_from_group_stats(train_trust_df, group_col="study_set")
    candidate_df, _ = _select_pseudo_labels(
        unlabeled_df=unlabeled_df,
        unlabeled_teacher_df=teacher_unlabeled_df,
        support_df=support_df,
        config=pseudo_base,
        max_rows=int(round(len(target_train_df) * float(pseudo_base.max_pseudo_multiplier))),
    )
    trust_selected = _apply_trust_selection(
        candidate_df=candidate_df,
        cluster_trust_df=cluster_trust_df,
        study_trust_df=study_trust_df,
        max_rows=int(round(len(target_train_df) * float(pseudo_base.max_pseudo_multiplier))),
        config=trust_base,
    )

    filtered_drop_cols = set(unit_drop_cols) | set(context_drop_cols)
    filtered_context_cols = [col for col in prepared.lightgbm_context_cols if col not in filtered_drop_cols]
    filtered_unit_cols = [col for col in prepared.unit_feature_cols if col not in set(unit_drop_cols)]
    target_train_masked = mask_feature_columns(target_train_df, filtered_drop_cols)
    target_test_masked = mask_feature_columns(target_test_df, filtered_drop_cols)
    unlabeled_masked = mask_feature_columns(unlabeled_df, filtered_drop_cols)

    anchor_feature_cols = sorted(set(filtered_context_cols + filtered_unit_cols))
    anchor_oof, anchor_unlabeled, anchor_test = _fit_anchor_scores(
        matched_train_df=target_train_masked,
        matched_test_df=target_test_masked,
        unmatched_df=unlabeled_masked,
        feature_cols=anchor_feature_cols,
        n_splits=base.model_selection_splits,
        config=type(
            "AnchorCfg",
            (),
            {
                "anchor_learning_rate": 0.06,
                "anchor_max_depth": 4,
                "anchor_max_iter": 250,
                "random_state": 42,
            },
        )(),
    )
    matched_support_conf_train = support_train_df["support_confidence"].to_numpy(dtype=float)
    matched_support_conf_unlabeled = support_df["support_confidence"].to_numpy(dtype=float)
    matched_support_conf_test = support_test_df["support_confidence"].to_numpy(dtype=float)

    train_frame = build_stack_frame(
        target_train_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_train_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_oof,
            "matched_support_conf": matched_support_conf_train,
            "anchor_support_margin": anchor_oof - matched_support_conf_train,
        },
    )
    unlabeled_frame = build_stack_frame(
        unlabeled_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_unlabeled_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_unlabeled,
            "matched_support_conf": matched_support_conf_unlabeled,
            "anchor_support_margin": anchor_unlabeled - matched_support_conf_unlabeled,
        },
    )
    test_frame = build_stack_frame(
        target_test_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_test_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_test,
            "matched_support_conf": matched_support_conf_test,
            "anchor_support_margin": anchor_test - matched_support_conf_test,
        },
    )

    full_train_frame, full_train_y, full_train_weight = _assemble_student_training_data(
        labeled_frame=train_frame,
        labeled_y=y_train,
        labeled_indices=None,
        pseudo_selected=trust_selected,
        unlabeled_frame=unlabeled_frame,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(full_train_frame)
    X_test_imp = imputer.transform(test_frame)
    y_train_t = target_transform(full_train_y, mode=base.resolved_target_transform())

    model = XGBRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=4.0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42 + 4000 + 200,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train_imp, y_train_t, sample_weight=full_train_weight)

    test_frame = test_frame.reset_index(drop=True)
    test_frame["true"] = y_test
    test_frame["study_set"] = target_test_df["study_set"].astype(str).reset_index(drop=True)
    return model, imputer, full_train_frame.reset_index(drop=True), test_frame, X_test_imp


def make_shap_plots(parquet_path: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, imputer, train_frame, test_frame, X_test_imp = _refit_winner(parquet_path)
    explainer = shap.TreeExplainer(model)

    feature_names = train_frame.columns.astype(str).tolist()
    shap_values = explainer.shap_values(X_test_imp)
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_path = output_dir / "winner_fpos_shap_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")

    bar_path = output_dir / "winner_fpos_shap_bar.png"
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        pd.DataFrame(X_test_imp, columns=feature_names),
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.title("Top SHAP features for fpos winner")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close()

    beeswarm_path = output_dir / "winner_fpos_shap_beeswarm.png"
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        pd.DataFrame(X_test_imp, columns=feature_names),
        max_display=20,
        show=False,
    )
    plt.title("SHAP beeswarm for fpos winner")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=180, bbox_inches="tight")
    plt.close()

    top_features = importance_df["feature"].head(3).tolist()
    dependence_paths: dict[str, str] = {}
    X_test_df = pd.DataFrame(X_test_imp, columns=feature_names)
    for feat in top_features:
        safe_name = feat.replace("/", "_")
        out_path = output_dir / f"winner_fpos_shap_dependence_{safe_name}.png"
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            feat,
            shap_values,
            X_test_df,
            interaction_index=None,
            show=False,
        )
        plt.title(f"SHAP dependence: {feat}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        dependence_paths[feat] = str(out_path)

    study_summary = (
        test_frame.assign(pred=model.predict(X_test_imp))
        .groupby("study_set", dropna=False)
        .agg(
            n=("true", "size"),
            true_mean=("true", "mean"),
            pred_logit_mean=("pred", "mean"),
        )
        .reset_index()
    )
    study_summary.to_csv(output_dir / "winner_fpos_shap_study_summary.csv", index=False)

    manifest = {
        "bar_plot": str(bar_path),
        "beeswarm_plot": str(beeswarm_path),
        "importance_csv": str(importance_path),
        "dependence_plots": dependence_paths,
    }
    with open(output_dir / "winner_fpos_shap_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = make_shap_plots(
        parquet_path=(REPO_ROOT / args.parquet).resolve(),
        output_dir=(REPO_ROOT / args.output_dir).resolve(),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
