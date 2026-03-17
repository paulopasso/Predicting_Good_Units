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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.ml_statistics_workbench.fpos_signal_embedding_stack import (
    FPosSignalEmbeddingStackConfig,
    _assemble_training_data,
    _prepare_base_frames,
    _signal_embeddings,
)
from crash_tests.ml_statistics_workbench.fpos_backend_sweep import _grouped_oof_splits
from crash_tests.ml_statistics_workbench.methods import score_predictions
from crash_tests.mmd_domain_adaptation.model import target_inverse, target_transform


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain the current best fpos model")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument(
        "--output-dir",
        default="crash_tests/ml_statistics_workbench/outputs/fpos_signal_embedding_stack_full_v1/explainability_v1",
    )
    return parser.parse_args()


def _fit_xgb(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    random_state: int,
) -> tuple[XGBRegressor, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    y_train_t = target_transform(y_train, mode="logit_clip")
    model = XGBRegressor(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=4.0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train_imp, y_train_t, sample_weight=sample_weight)
    return model, imputer


def _predict_xgb(model: XGBRegressor, imputer: SimpleImputer, X_eval: pd.DataFrame) -> np.ndarray:
    pred_t = np.asarray(model.predict(imputer.transform(X_eval)), dtype=float)
    return np.clip(target_inverse(pred_t, mode="logit_clip"), 0.0, 1.0)


def _refit_current_winner(
    parquet_path: Path,
) -> dict[str, object]:
    config = FPosSignalEmbeddingStackConfig(
        parquet_path=parquet_path,
        output_root=Path("crash_tests/ml_statistics_workbench/outputs"),
        run_name="fpos_explain_refit",
        random_state=42,
        paired_final_holdout_rows=100,
        verbose=False,
    )
    state = _prepare_base_frames(config)
    prepared = state["prepared"]
    target_train_df = state["target_train_df"]
    target_test_df = state["target_test_df"]
    unlabeled_df = state["unlabeled_df"]
    y_train = state["y_train"]
    y_test = state["y_test"]
    trust_selected = state["trust_selected"]
    train_frame = state["train_frame"]
    unlabeled_frame = state["unlabeled_frame"]
    test_frame = state["test_frame"]

    universe_df = pd.concat(
        [
            state["split"].hybrid_source.reset_index(drop=True),
            state["split"].paired_non_test_all.reset_index(drop=True),
            state["split"].ssl_pool.reset_index(drop=True),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["row_uid"]).reset_index(drop=True)
    if config.universe_max_rows is not None and len(universe_df) > int(config.universe_max_rows):
        universe_df = universe_df.sample(
            n=int(config.universe_max_rows),
            random_state=config.random_state,
        ).reset_index(drop=True)

    wf_train_emb, wf_unlabeled_emb, wf_test_emb, wf_hist = _signal_embeddings(
        train_signal=target_train_df.loc[:, prepared.wf_feature_cols],
        train_universe_signal=universe_df.loc[:, prepared.wf_feature_cols],
        unlabeled_signal=unlabeled_df.loc[:, prepared.wf_feature_cols],
        test_signal=target_test_df.loc[:, prepared.wf_feature_cols],
        prefix="wf",
        config=config,
    )

    train_variant = pd.concat([train_frame.reset_index(drop=True), wf_train_emb], axis=1)
    unlabeled_variant = pd.concat([unlabeled_frame.reset_index(drop=True), wf_unlabeled_emb], axis=1)
    test_variant = pd.concat([test_frame.reset_index(drop=True), wf_test_emb], axis=1)

    oof_pred = np.full(len(target_train_df), np.nan, dtype=float)
    splits = _grouped_oof_splits(target_train_df, 3)
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        val_recordings = set(target_train_df.iloc[val_idx]["recording_key"].astype(str))
        fold_pseudo = trust_selected[
            ~trust_selected["recording_key"].astype(str).isin(val_recordings)
        ].reset_index(drop=True)
        train_fold, fold_y, fold_weight = _assemble_training_data(
            labeled_frame=train_variant.iloc[train_idx].reset_index(drop=True),
            labeled_y=y_train[train_idx],
            labeled_meta_df=target_train_df.iloc[train_idx].reset_index(drop=True),
            pseudo_selected=fold_pseudo,
            unlabeled_frame=unlabeled_variant,
            family_alpha=1.0,
        )
        model, imputer = _fit_xgb(
            X_train=train_fold,
            y_train=fold_y,
            sample_weight=fold_weight,
            random_state=42 + split_id,
        )
        oof_pred[val_idx] = _predict_xgb(model, imputer, train_variant.iloc[val_idx].reset_index(drop=True))

    full_train, full_y, full_weight = _assemble_training_data(
        labeled_frame=train_variant,
        labeled_y=y_train,
        labeled_meta_df=target_train_df,
        pseudo_selected=trust_selected,
        unlabeled_frame=unlabeled_variant,
        family_alpha=1.0,
    )
    model, imputer = _fit_xgb(
        X_train=full_train,
        y_train=full_y,
        sample_weight=full_weight,
        random_state=42 + 100,
    )
    holdout_pred = _predict_xgb(model, imputer, test_variant)

    return {
        "model": model,
        "imputer": imputer,
        "train_variant": train_variant,
        "test_variant": test_variant,
        "target_train_df": target_train_df.reset_index(drop=True),
        "target_test_df": target_test_df.reset_index(drop=True),
        "y_train": y_train,
        "y_test": y_test,
        "oof_pred": oof_pred,
        "holdout_pred": holdout_pred,
        "embedding_history": wf_hist,
    }


def _calibration_bins(true: np.ndarray, pred: np.ndarray, bins: int = 10) -> pd.DataFrame:
    order = np.argsort(pred)
    edges = np.linspace(0, len(pred), bins + 1, dtype=int)
    rows: list[dict[str, float | int]] = []
    for idx in range(bins):
        lo, hi = edges[idx], edges[idx + 1]
        batch = order[lo:hi]
        if len(batch) == 0:
            continue
        rows.append(
            {
                "bin": idx,
                "n": int(len(batch)),
                "pred_mean": float(np.mean(pred[batch])),
                "true_mean": float(np.mean(true[batch])),
                "abs_error_mean": float(np.mean(np.abs(true[batch] - pred[batch]))),
            }
        )
    return pd.DataFrame(rows)


def _fit_residual_probe(
    *,
    X: pd.DataFrame,
    residuals: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    splits = _grouped_oof_splits(pd.DataFrame({"recording_key": groups}), 3)
    oof = np.full(len(residuals), np.nan, dtype=float)
    importances: list[np.ndarray] = []
    feature_names = X.columns.astype(str).tolist()
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X.iloc[train_idx])
        X_val = imputer.transform(X.iloc[val_idx])
        model = XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=4.0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=100 + split_id,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, residuals[train_idx])
        oof[val_idx] = np.asarray(model.predict(X_val), dtype=float)
        importances.append(np.asarray(model.feature_importances_, dtype=float))
    metrics = {
        "residual_r2": float(r2_score(residuals, oof)),
        "residual_mae": float(mean_absolute_error(residuals, oof)),
        "residual_std": float(np.std(residuals)),
    }
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.mean(np.vstack(importances), axis=0) if importances else np.zeros(len(feature_names)),
        }
    ).sort_values("importance", ascending=False)
    return metrics, importance_df


def _write_report(output_dir: Path, report_lines: list[str]) -> None:
    (output_dir / "EXPLAINABILITY_REPORT.md").write_text("\n".join(report_lines) + "\n")


def run_report(parquet_path: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _refit_current_winner(parquet_path)
    model = state["model"]
    imputer = state["imputer"]
    train_variant = state["train_variant"]
    test_variant = state["test_variant"]
    target_train_df = state["target_train_df"]
    target_test_df = state["target_test_df"]
    y_train = np.asarray(state["y_train"], dtype=float)
    y_test = np.asarray(state["y_test"], dtype=float)
    oof_pred = np.asarray(state["oof_pred"], dtype=float)
    holdout_pred = np.asarray(state["holdout_pred"], dtype=float)

    train_metrics = score_predictions(y_train, oof_pred)
    holdout_metrics = score_predictions(y_test, holdout_pred)

    X_test_imp = imputer.transform(test_variant)
    X_test_df = pd.DataFrame(X_test_imp, columns=test_variant.columns.astype(str))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_imp)
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame(
        {"feature": X_test_df.columns.astype(str), "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance_shap.csv", index=False)

    holdout_frame = target_test_df.loc[:, ["row_uid", "recording_key", "study_set", "sorter_name"]].copy()
    holdout_frame["true"] = y_test
    holdout_frame["pred"] = holdout_pred
    holdout_frame["residual"] = y_test - holdout_pred
    holdout_frame["abs_error"] = np.abs(holdout_frame["residual"])
    holdout_frame.to_csv(output_dir / "holdout_predictions_enriched.csv", index=False)

    study_summary = holdout_frame.groupby("study_set", dropna=False).agg(
        n=("true", "size"),
        true_mean=("true", "mean"),
        pred_mean=("pred", "mean"),
        mae=("abs_error", "mean"),
        bias=("residual", "mean"),
    ).reset_index()
    study_summary["r2"] = [
        float(r2_score(group["true"], group["pred"])) if len(group) > 1 else np.nan
        for _, group in holdout_frame.groupby("study_set", dropna=False)
    ]
    study_summary.to_csv(output_dir / "study_summary.csv", index=False)

    sorter_summary = holdout_frame.groupby("sorter_name", dropna=False).agg(
        n=("true", "size"),
        true_mean=("true", "mean"),
        pred_mean=("pred", "mean"),
        mae=("abs_error", "mean"),
        bias=("residual", "mean"),
    ).reset_index()
    sorter_summary["r2"] = [
        float(r2_score(group["true"], group["pred"])) if len(group) > 1 else np.nan
        for _, group in holdout_frame.groupby("sorter_name", dropna=False)
    ]
    sorter_summary.to_csv(output_dir / "sorter_summary.csv", index=False)

    recording_summary = holdout_frame.groupby("recording_key", dropna=False).agg(
        n=("true", "size"),
        mae=("abs_error", "mean"),
        bias=("residual", "mean"),
    ).reset_index().sort_values(["mae", "n"], ascending=[False, False])
    recording_summary.to_csv(output_dir / "recording_summary.csv", index=False)

    calibration_df = _calibration_bins(y_test, holdout_pred, bins=10)
    calibration_df.to_csv(output_dir / "calibration_bins.csv", index=False)

    oracle_lin = LinearRegression().fit(holdout_pred.reshape(-1, 1), y_test)
    oracle_lin_pred = np.clip(oracle_lin.predict(holdout_pred.reshape(-1, 1)), 0.0, 1.0)
    oracle_lin_metrics = score_predictions(y_test, oracle_lin_pred)

    family_bias = holdout_frame.groupby("study_set", dropna=False)["residual"].mean().to_dict()
    oracle_family_pred = np.clip(
        holdout_frame["pred"].to_numpy(dtype=float)
        + holdout_frame["study_set"].map(family_bias).to_numpy(dtype=float),
        0.0,
        1.0,
    )
    oracle_family_metrics = score_predictions(y_test, oracle_family_pred)

    residual_metrics, residual_importance = _fit_residual_probe(
        X=train_variant,
        residuals=y_train - oof_pred,
        groups=target_train_df["recording_key"].astype(str).to_numpy(),
    )
    residual_importance.to_csv(output_dir / "residual_probe_importance.csv", index=False)

    top_features = importance_df["feature"].head(5).tolist()
    plt.style.use("seaborn-v0_8-whitegrid")

    bar_path = output_dir / "shap_bar.png"
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", max_display=20, show=False)
    plt.title("Top SHAP Features: Best fpos Model")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close()

    beeswarm_path = output_dir / "shap_beeswarm.png"
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, max_display=20, show=False)
    plt.title("SHAP Beeswarm: Best fpos Model")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=180, bbox_inches="tight")
    plt.close()

    dep_paths: list[str] = []
    for feat in top_features[:4]:
        feat_safe = feat.replace("/", "_")
        dep_path = output_dir / f"dependence_{feat_safe}.png"
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat, shap_values, X_test_df, interaction_index=None, show=False)
        plt.title(f"SHAP Dependence: {feat}")
        plt.tight_layout()
        plt.savefig(dep_path, dpi=180, bbox_inches="tight")
        plt.close()
        dep_paths.append(str(dep_path))

    scatter_path = output_dir / "pred_vs_true.png"
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, holdout_pred, alpha=0.7, s=28)
    lo = min(float(y_test.min()), float(holdout_pred.min()))
    hi = max(float(y_test.max()), float(holdout_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    plt.xlabel("True fpos")
    plt.ylabel("Predicted fpos")
    plt.title("Holdout Predictions vs True Values")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=180, bbox_inches="tight")
    plt.close()

    calib_path = output_dir / "calibration_curve.png"
    plt.figure(figsize=(7, 6))
    plt.plot(calibration_df["pred_mean"], calibration_df["true_mean"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Predicted bin mean")
    plt.ylabel("True bin mean")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(calib_path, dpi=180, bbox_inches="tight")
    plt.close()

    study_path = output_dir / "study_performance.png"
    study_plot = study_summary.sort_values("mae", ascending=False)
    plt.figure(figsize=(9, 5))
    plt.bar(study_plot["study_set"], study_plot["mae"], color="steelblue")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MAE")
    plt.title("Holdout MAE by Study Family")
    plt.tight_layout()
    plt.savefig(study_path, dpi=180, bbox_inches="tight")
    plt.close()

    decile_path = output_dir / "error_by_true_decile.png"
    decile_df = holdout_frame.copy()
    decile_df["true_decile"] = pd.qcut(decile_df["true"], q=min(10, decile_df["true"].nunique()), duplicates="drop")
    decile_summary = decile_df.groupby("true_decile", dropna=False).agg(
        true_mean=("true", "mean"),
        pred_mean=("pred", "mean"),
        mae=("abs_error", "mean"),
    ).reset_index(drop=True)
    decile_summary.to_csv(output_dir / "error_by_true_decile.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(decile_summary["true_mean"], decile_summary["mae"], marker="o")
    plt.xlabel("True fpos decile mean")
    plt.ylabel("MAE")
    plt.title("Error by True fpos Level")
    plt.tight_layout()
    plt.savefig(decile_path, dpi=180, bbox_inches="tight")
    plt.close()

    report_lines = [
        "# Explainability Report: Best fpos Model",
        "",
        "## Model",
        "",
        "- variant: `wf_embed_anchor_stack_xgboost`",
        "- run: `crash_tests/ml_statistics_workbench/outputs/fpos_signal_embedding_stack_full_v1`",
        "",
        "## Main Metrics",
        "",
        f"- training OOF MAE: `{train_metrics['mae']:.4f}`",
        f"- training OOF R²: `{train_metrics['r2']:.4f}`",
        f"- holdout MAE: `{holdout_metrics['mae']:.4f}`",
        f"- holdout R²: `{holdout_metrics['r2']:.4f}`",
        f"- holdout bias: `{holdout_metrics['bias']:.4f}`",
        f"- holdout calibration slope: `{holdout_metrics['calibration_slope']:.4f}`",
        "",
        "## Top SHAP Features",
        "",
    ]
    for _, row in importance_df.head(12).iterrows():
        report_lines.append(f"- `{row['feature']}`: mean |SHAP| `{row['mean_abs_shap']:.5f}`")
    report_lines.extend(
        [
            "",
            "## Missing Signal Diagnostics",
            "",
            f"- oracle global linear recalibration holdout R²: `{oracle_lin_metrics['r2']:.4f}`",
            f"- oracle per-study intercept correction holdout R²: `{oracle_family_metrics['r2']:.4f}`",
            f"- residual-probe CV R² on training residuals: `{residual_metrics['residual_r2']:.4f}`",
            f"- residual-probe CV MAE on training residuals: `{residual_metrics['residual_mae']:.4f}`",
            "",
            "Interpretation:",
            f"- linear calibration alone could recover about `{oracle_lin_metrics['r2'] - holdout_metrics['r2']:.4f}` R²",
            f"- perfect study-specific bias correction would recover about `{oracle_family_metrics['r2'] - holdout_metrics['r2']:.4f}` R²",
            f"- residual-probe predictability suggests there is still about `{max(residual_metrics['residual_r2'], 0.0):.4f}` structured residual signal left in the current feature set",
            "",
            "## Worst Families",
            "",
        ]
    )
    for _, row in study_summary.sort_values("mae", ascending=False).iterrows():
        report_lines.append(
            f"- `{row['study_set']}`: n=`{int(row['n'])}`, MAE=`{row['mae']:.4f}`, R²=`{row['r2']:.4f}`, bias=`{row['bias']:.4f}`"
        )
    report_lines.extend(
        [
            "",
            "## Residual-Probe Top Features",
            "",
        ]
    )
    for _, row in residual_importance.head(10).iterrows():
        report_lines.append(f"- `{row['feature']}`: residual importance `{row['importance']:.5f}`")

    _write_report(output_dir, report_lines)

    manifest = {
        "report_md": str(output_dir / "EXPLAINABILITY_REPORT.md"),
        "shap_bar": str(bar_path),
        "shap_beeswarm": str(beeswarm_path),
        "dependence_plots": dep_paths,
        "pred_vs_true": str(scatter_path),
        "calibration_curve": str(calib_path),
        "study_performance": str(study_path),
        "error_by_true_decile": str(decile_path),
        "feature_importance_csv": str(output_dir / "feature_importance_shap.csv"),
        "study_summary_csv": str(output_dir / "study_summary.csv"),
        "sorter_summary_csv": str(output_dir / "sorter_summary.csv"),
        "recording_summary_csv": str(output_dir / "recording_summary.csv"),
        "residual_probe_importance_csv": str(output_dir / "residual_probe_importance.csv"),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = run_report(
        parquet_path=(REPO_ROOT / args.parquet).resolve(),
        output_dir=(REPO_ROOT / args.output_dir).resolve(),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
