from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.domain_shift_audit.config import DomainShiftAuditConfig
from crash_tests.self_supervised_domain_adaptation.dataset import prepare_feature_table
from crash_tests.self_supervised_domain_adaptation.split_utils import make_main_split


def _log(config: DomainShiftAuditConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def feature_group_for_column(column: str) -> str:
    name = str(column)
    if name.startswith("wf_bin_"):
        return "waveform_bins"
    if name.startswith("acg_"):
        return "acg_bins"
    if any(suffix in name for suffix in ("_recording_mean", "_recording_std", "_recording_median", "_recording_iqr", "_recording_delta")):
        return "recording_context"
    if name.endswith("_recording_pct"):
        return "recording_relative"
    if name.startswith("amp_"):
        return "amplitude"
    if name.startswith("si_"):
        return "spikeinterface"
    if "cosine" in name or "confusable" in name or "neighbor_dist" in name or "isolation" in name:
        return "confusability"
    if name.startswith("rec_"):
        return "recording_metadata"
    return "other"


def _sample_frame(
    df: pd.DataFrame,
    *,
    n_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if len(df) <= n_rows:
        return df.copy()
    return df.sample(n=n_rows, random_state=random_state).copy()


def _numeric_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _safe_smd(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) == 0 or len(values_b) == 0:
        return float("nan")
    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    var_a = float(np.var(values_a))
    var_b = float(np.var(values_b))
    pooled = np.sqrt(max((var_a + var_b) / 2.0, 1e-12))
    return float((mean_b - mean_a) / pooled)


def compute_feature_shift_table(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    feature_cols: list[str],
    comparison: str,
    feature_set: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in feature_cols:
        a = _numeric_series(df_a, column)
        b = _numeric_series(df_b, column)
        missing_a = 1.0 - (len(a) / max(len(df_a), 1))
        missing_b = 1.0 - (len(b) / max(len(df_b), 1))
        ks_stat = float("nan")
        wasser = float("nan")
        if len(a) > 0 and len(b) > 0:
            ks_stat = float(ks_2samp(a, b).statistic)
            wasser = float(wasserstein_distance(a, b))
        smd = _safe_smd(a, b)
        rows.append(
            {
                "comparison": comparison,
                "feature_set": feature_set,
                "feature": column,
                "feature_group": feature_group_for_column(column),
                "rows_a": int(len(df_a)),
                "rows_b": int(len(df_b)),
                "mean_a": float(np.mean(a)) if len(a) else float("nan"),
                "mean_b": float(np.mean(b)) if len(b) else float("nan"),
                "median_a": float(np.median(a)) if len(a) else float("nan"),
                "median_b": float(np.median(b)) if len(b) else float("nan"),
                "std_a": float(np.std(a)) if len(a) else float("nan"),
                "std_b": float(np.std(b)) if len(b) else float("nan"),
                "missing_a": float(missing_a),
                "missing_b": float(missing_b),
                "missing_gap": float(missing_b - missing_a),
                "smd": smd,
                "abs_smd": abs(smd) if np.isfinite(smd) else float("nan"),
                "ks_stat": ks_stat,
                "wasserstein": wasser,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["comparison", "feature_set", "abs_smd", "ks_stat"], ascending=[True, True, False, False]).reset_index(drop=True)


def summarize_feature_groups(feature_shift_df: pd.DataFrame) -> pd.DataFrame:
    if feature_shift_df.empty:
        return pd.DataFrame()
    grouped = (
        feature_shift_df.groupby(["comparison", "feature_set", "feature_group"], dropna=False)
        .agg(
            n_features=("feature", "size"),
            mean_abs_smd=("abs_smd", "mean"),
            median_abs_smd=("abs_smd", "median"),
            max_abs_smd=("abs_smd", "max"),
            mean_ks=("ks_stat", "mean"),
            mean_missing_gap=("missing_gap", "mean"),
        )
        .reset_index()
        .sort_values(["comparison", "feature_set", "mean_abs_smd"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return grouped


def _balanced_domain_sample(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    max_rows_per_domain: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        _sample_frame(df_a, n_rows=max_rows_per_domain, random_state=random_state),
        _sample_frame(df_b, n_rows=max_rows_per_domain, random_state=random_state + 1),
    )


def build_domain_classifier_report(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    feature_cols: list[str],
    comparison: str,
    feature_set: str,
    random_state: int,
    max_rows_per_domain: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_a, sample_b = _balanced_domain_sample(
        df_a,
        df_b,
        max_rows_per_domain=max_rows_per_domain,
        random_state=random_state,
    )
    frame = pd.concat([sample_a, sample_b], ignore_index=True)
    X = frame.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    y = np.concatenate([np.zeros(len(sample_a), dtype=int), np.ones(len(sample_b), dtype=int)])
    groups = frame["recording_key"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))

    clf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    auc_scores: list[float] = []
    if n_splits >= 2:
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in splitter.split(X, y, groups=groups):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            clf.fit(X.iloc[train_idx], y[train_idx])
            prob = clf.predict_proba(X.iloc[test_idx])[:, 1]
            auc_scores.append(float(roc_auc_score(y[test_idx], prob)))

    clf.fit(X, y)
    model = clf.named_steps["model"]
    coefs = model.coef_.reshape(-1)
    coef_df = pd.DataFrame(
        {
            "comparison": comparison,
            "feature_set": feature_set,
            "feature": feature_cols,
            "feature_group": [feature_group_for_column(column) for column in feature_cols],
            "coef": coefs,
            "abs_coef": np.abs(coefs),
            "direction": np.where(coefs >= 0, "paired_up", "hybrid_up"),
        }
    ).sort_values(["abs_coef", "feature"], ascending=[False, True]).reset_index(drop=True)

    summary_df = pd.DataFrame(
        [
            {
                "comparison": comparison,
                "feature_set": feature_set,
                "rows_a": int(len(sample_a)),
                "rows_b": int(len(sample_b)),
                "n_features": int(len(feature_cols)),
                "cv_auc_mean": float(np.mean(auc_scores)) if auc_scores else float("nan"),
                "cv_auc_std": float(np.std(auc_scores)) if auc_scores else float("nan"),
            }
        ]
    )
    return summary_df, coef_df


def build_pca_projection(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    feature_cols: list[str],
    comparison: str,
    random_state: int,
    rows_per_domain: int,
    n_components: int = 2,
) -> pd.DataFrame:
    sample_a, sample_b = _balanced_domain_sample(
        df_a,
        df_b,
        max_rows_per_domain=rows_per_domain,
        random_state=random_state,
    )
    sample_a = sample_a.copy()
    sample_b = sample_b.copy()
    sample_a["domain_label"] = "domain_a"
    sample_b["domain_label"] = "domain_b"
    frame = pd.concat([sample_a, sample_b], ignore_index=True)
    X = frame.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    pca = PCA(n_components=int(n_components), random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    projection = {
        "comparison": comparison,
        "row_uid": frame["row_uid"].astype(str).to_numpy(),
        "recording_key": frame["recording_key"].astype(str).to_numpy(),
        "study_set": frame["study_set"].astype(str).to_numpy(),
        "dataset_type": frame["dataset_type"].astype(str).to_numpy(),
        "domain_label": frame["domain_label"].astype(str).to_numpy(),
    }
    for idx in range(coords.shape[1]):
        projection[f"pc{idx + 1}"] = coords[:, idx]
    return pd.DataFrame(projection)


def summarize_study_shift(
    hybrid_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    feature_set: str,
    top_n_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for study_set, study_df in paired_df.groupby("study_set", dropna=False):
        shift = compute_feature_shift_table(
            hybrid_df,
            study_df,
            feature_cols=feature_cols,
            comparison=f"hybrid_vs_{study_set}",
            feature_set=feature_set,
        )
        top_features = shift.sort_values(["abs_smd", "ks_stat"], ascending=[False, False]).head(top_n_features)
        rows.append(
            {
                "study_set": study_set,
                "feature_set": feature_set,
                "rows": int(len(study_df)),
                "recordings": int(study_df["recording_key"].nunique()),
                "matched_rows": int(study_df["matched_gt_unit_id"].notna().sum()),
                "mean_abs_smd": float(shift["abs_smd"].mean()),
                "median_abs_smd": float(shift["abs_smd"].median()),
                "max_abs_smd": float(shift["abs_smd"].max()),
                "top_shifted_features": "|".join(top_features["feature"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_set", "mean_abs_smd"], ascending=[True, False]).reset_index(drop=True)


def _build_readme(
    *,
    config: DomainShiftAuditConfig,
    split_manifest: dict[str, Any],
    classifier_summary_df: pd.DataFrame,
    feature_group_summary_df: pd.DataFrame,
) -> str:
    lines = [
        "# Domain Shift Audit",
        "",
        "## Config",
        f"- parquet_path: `{config.parquet_path}`",
        f"- paired_final_holdout_rows: `{config.paired_final_holdout_rows}`",
        f"- max_domain_rows_for_classifier: `{config.max_domain_rows_for_classifier}`",
        "",
        "## Split",
        f"- paired_test_rows: `{split_manifest['paired_test_rows']}`",
        f"- paired_finetune_rows: `{split_manifest['paired_finetune_rows']}`",
        f"- ssl_pool_rows: `{split_manifest['ssl_pool_rows']}`",
        "",
        "## Domain Separability",
    ]
    if classifier_summary_df.empty:
        lines.append("- No classifier summary produced.")
    else:
        for _, row in classifier_summary_df.sort_values(["comparison", "feature_set"]).iterrows():
            lines.append(
                f"- {row['comparison']} | {row['feature_set']} | CV AUC `{row['cv_auc_mean']:.4f}`"
            )
    lines.extend(["", "## Highest-Shift Feature Groups"])
    if feature_group_summary_df.empty:
        lines.append("- No group summary produced.")
    else:
        for _, row in feature_group_summary_df.sort_values(["comparison", "mean_abs_smd"], ascending=[True, False]).groupby("comparison").head(5).iterrows():
            lines.append(
                f"- {row['comparison']} | {row['feature_set']} | {row['feature_group']} | mean |SMD| `{row['mean_abs_smd']:.4f}`"
            )
    return "\n".join(lines) + "\n"


def run_with_config(config: DomainShiftAuditConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(config, f"Loading parquet from {config.parquet_path}")
    prepared = prepare_feature_table(config.parquet_path)
    split = make_main_split(
        prepared.df,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        random_state=config.random_state,
    )

    hybrid_df = prepared.hybrid.copy()
    paired_df = prepared.paired.copy()
    paired_matched_df = prepared.paired_matched.copy()
    paired_unmatched_df = prepared.paired_unmatched.copy()

    comparisons = [
        ("hybrid_vs_paired_all", hybrid_df, paired_df),
        ("hybrid_vs_paired_matched", hybrid_df, paired_matched_df),
        ("hybrid_vs_paired_unmatched", hybrid_df, paired_unmatched_df),
        ("hybrid_vs_paired_finetune", hybrid_df, split.paired_finetune.copy()),
        ("hybrid_vs_paired_test", hybrid_df, split.paired_test.copy()),
        ("paired_matched_vs_unmatched", paired_matched_df, paired_unmatched_df),
        ("paired_finetune_vs_test", split.paired_finetune.copy(), split.paired_test.copy()),
    ]
    feature_sets = {
        "unit": prepared.unit_feature_cols,
        "shape": prepared.shape_feature_cols,
        "context_only": prepared.context_feature_cols,
        "full_context": prepared.lightgbm_context_cols,
    }

    feature_shift_frames: list[pd.DataFrame] = []
    for comparison, frame_a, frame_b in comparisons:
        _log(config, f"Computing feature drift for {comparison}")
        for feature_set, feature_cols in feature_sets.items():
            feature_shift_frames.append(
                compute_feature_shift_table(
                    frame_a,
                    frame_b,
                    feature_cols=feature_cols,
                    comparison=comparison,
                    feature_set=feature_set,
                )
            )
    feature_shift_df = pd.concat(feature_shift_frames, ignore_index=True)
    feature_group_summary_df = summarize_feature_groups(feature_shift_df)

    classifier_rows: list[pd.DataFrame] = []
    classifier_coef_rows: list[pd.DataFrame] = []
    classifier_targets = [
        ("hybrid_vs_paired_all", hybrid_df, paired_df, "unit"),
        ("hybrid_vs_paired_all", hybrid_df, paired_df, "full_context"),
        ("hybrid_vs_paired_test", hybrid_df, split.paired_test.copy(), "unit"),
        ("hybrid_vs_paired_test", hybrid_df, split.paired_test.copy(), "full_context"),
    ]
    for comparison, frame_a, frame_b, feature_set in classifier_targets:
        _log(config, f"Training domain classifier for {comparison} [{feature_set}]")
        summary_df, coef_df = build_domain_classifier_report(
            frame_a,
            frame_b,
            feature_cols=feature_sets[feature_set],
            comparison=comparison,
            feature_set=feature_set,
            random_state=config.random_state,
            max_rows_per_domain=config.max_domain_rows_for_classifier,
        )
        classifier_rows.append(summary_df)
        classifier_coef_rows.append(coef_df.head(config.top_n_features))
    classifier_summary_df = pd.concat(classifier_rows, ignore_index=True)
    classifier_coef_df = pd.concat(classifier_coef_rows, ignore_index=True)

    pca_df = build_pca_projection(
        hybrid_df,
        paired_df,
        feature_cols=feature_sets["unit"],
        comparison="hybrid_vs_paired_all",
        random_state=config.random_state,
        rows_per_domain=config.pca_rows_per_domain,
        n_components=2,
    )
    pca_3d_df = pd.DataFrame()
    if config.include_pca_3d:
        pca_3d_df = build_pca_projection(
            hybrid_df,
            paired_df,
            feature_cols=feature_sets["unit"],
            comparison="hybrid_vs_paired_all",
            random_state=config.random_state,
            rows_per_domain=config.pca_rows_per_domain,
            n_components=3,
        )

    study_shift_frames = [
        summarize_study_shift(
            hybrid_df,
            paired_df,
            feature_cols=feature_sets["unit"],
            feature_set="unit",
            top_n_features=config.top_n_features,
        ),
        summarize_study_shift(
            hybrid_df,
            paired_df,
            feature_cols=feature_sets["full_context"],
            feature_set="full_context",
            top_n_features=config.top_n_features,
        ),
    ]
    study_shift_df = pd.concat(study_shift_frames, ignore_index=True)

    split_manifest = {
        **split.manifest,
        "hybrid_rows": int(len(hybrid_df)),
        "paired_rows": int(len(paired_df)),
        "paired_matched_rows": int(len(paired_matched_df)),
        "paired_unmatched_rows": int(len(paired_unmatched_df)),
        "unit_feature_count": int(len(prepared.unit_feature_cols)),
        "shape_feature_count": int(len(prepared.shape_feature_cols)),
        "context_feature_count": int(len(prepared.context_feature_cols)),
        "full_context_feature_count": int(len(prepared.lightgbm_context_cols)),
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({"config": config.to_dict(), "main": split_manifest}, f, indent=2, sort_keys=True)
    feature_shift_df.to_csv(output_dir / "feature_shift_summary.csv", index=False)
    feature_group_summary_df.to_csv(output_dir / "feature_group_summary.csv", index=False)
    classifier_summary_df.to_csv(output_dir / "domain_classifier_summary.csv", index=False)
    classifier_coef_df.to_csv(output_dir / "domain_classifier_coefficients.csv", index=False)
    study_shift_df.to_csv(output_dir / "study_shift_summary.csv", index=False)
    pca_df.to_csv(output_dir / "pca_projection.csv", index=False)
    if not pca_3d_df.empty:
        pca_3d_df.to_csv(output_dir / "pca_projection_3d.csv", index=False)
    (output_dir / "README_run.md").write_text(
        _build_readme(
            config=config,
            split_manifest=split_manifest,
            classifier_summary_df=classifier_summary_df,
            feature_group_summary_df=feature_group_summary_df,
        ),
        encoding="utf-8",
    )
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir
