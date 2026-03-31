from __future__ import annotations

import pandas as pd

from .data import build_dataset_bundle
from .features import get_acg_columns, get_feature_view, get_waveform_columns, infer_numeric_feature_columns
from .modeling import (
    build_benchmark_table,
    build_family_summary,
    build_recording_summary,
    get_main_text_recipe_ids,
    get_recipe_groups,
    load_experiment,
)
from .xai import load_fpos_shap_importance


def build_dataset_partition_table() -> pd.DataFrame:
    bundle = build_dataset_bundle(with_context=False)
    rows = [
        {"subset": "full", "rows": len(bundle.full_df), "recordings": bundle.full_df["recording_key"].nunique(), "families": bundle.full_df["study_set"].nunique()},
        {"subset": "hybrid_labeled", "rows": len(bundle.hybrid_df), "recordings": bundle.hybrid_df["recording_key"].nunique(), "families": bundle.hybrid_df["study_set"].nunique()},
        {"subset": "paired_total", "rows": len(bundle.paired_df), "recordings": bundle.paired_df["recording_key"].nunique(), "families": bundle.paired_df["study_set"].nunique()},
        {"subset": "paired_labeled", "rows": len(bundle.paired_labeled_df), "recordings": bundle.paired_labeled_df["recording_key"].nunique(), "families": bundle.paired_labeled_df["study_set"].nunique()},
        {"subset": "paired_unlabeled", "rows": len(bundle.paired_unlabeled_df), "recordings": bundle.paired_unlabeled_df["recording_key"].nunique(), "families": bundle.paired_unlabeled_df["study_set"].nunique()},
    ]
    return pd.DataFrame(rows)


def build_target_missingness_table() -> pd.DataFrame:
    df = build_dataset_bundle(with_context=False).full_df
    rows = []
    for target in ["fpos", "fmiss", "accuracy"]:
        rows.append({
            "target": target,
            "missing_rows": int(df[target].isna().sum()),
            "missing_fraction": float(df[target].isna().mean()),
            "non_missing_rows": int(df[target].notna().sum()),
        })
    return pd.DataFrame(rows)


def build_family_coverage_table() -> pd.DataFrame:
    df = build_dataset_bundle(with_context=False).paired_df
    return (
        df.groupby("study_set", as_index=False)
        .agg(
            rows=("row_uid", "size"),
            recordings=("recording_key", "nunique"),
            sorters=("sorter_name", "nunique"),
            labeled_rows=("is_paired_labeled", "sum"),
        )
        .sort_values(["rows", "labeled_rows"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_feature_inventory_table() -> pd.DataFrame:
    df = build_dataset_bundle(with_context=True).full_df
    numeric = infer_numeric_feature_columns(df)
    waveform = get_waveform_columns(df)
    acg = get_acg_columns(df)
    full_context = get_feature_view(df, "full_context")
    reduced_latent = get_feature_view(df, "reduced_latent")
    waveform_embedding = get_feature_view(df, "waveform_embedding")
    context_only = get_feature_view(df, "contextual")
    rows = [
        {"feature_block": "numeric_total", "feature_count": len(numeric)},
        {"feature_block": "waveform_bins", "feature_count": len(waveform)},
        {"feature_block": "acg_bins", "feature_count": len(acg)},
        {"feature_block": "context_only", "feature_count": len(context_only)},
        {"feature_block": "full_context", "feature_count": len(full_context)},
        {"feature_block": "reduced_latent", "feature_count": len(reduced_latent)},
        {"feature_block": "waveform_embedding_view", "feature_count": len(waveform_embedding)},
    ]
    return pd.DataFrame(rows)


def build_benchmark_progress_table(target: str, protocol: str = "recording_disjoint_main") -> pd.DataFrame:
    recipe_ids = get_main_text_recipe_ids(target)
    table = build_benchmark_table(target, recipe_ids, protocol).copy()
    table["mae"] = pd.to_numeric(table["mae"], errors="coerce")
    table["r2"] = pd.to_numeric(table["r2"], errors="coerce")
    baseline = table.iloc[0]
    simple_transfer = table.iloc[min(4, len(table) - 1)]
    table["delta_r2_vs_floor"] = table["r2"] - float(baseline["r2"])
    table["delta_r2_vs_basic_transfer"] = table["r2"] - float(simple_transfer["r2"])
    table["delta_mae_vs_floor"] = table["mae"] - float(baseline["mae"])
    table["delta_mae_vs_basic_transfer"] = table["mae"] - float(simple_transfer["mae"])
    return table


def build_ablation_winner_table(target: str, protocol: str = "recording_disjoint_main") -> pd.DataFrame:
    rows = []
    for group_name, recipe_ids in get_recipe_groups(target).items():
        table = build_benchmark_table(target, recipe_ids, protocol).copy()
        table["r2"] = pd.to_numeric(table["r2"], errors="coerce")
        table["mae"] = pd.to_numeric(table["mae"], errors="coerce")
        best = table.sort_values(["r2", "mae"], ascending=[False, True]).iloc[0]
        rows.append({
            "ablation_group": group_name,
            "best_label": best["label"],
            "best_recipe_id": best["recipe_id"],
            "best_r2": float(best["r2"]),
            "best_mae": float(best["mae"]),
            "notes": best.get("notes", ""),
        })
    return pd.DataFrame(rows)


def build_main_wave_summary(target: str, recipe_ids: list[str], protocol: str = "recording_disjoint_main") -> dict[str, pd.DataFrame]:
    families = []
    recordings = []
    for recipe_id in recipe_ids:
        fam = build_family_summary(recipe_id, target, protocol)
        rec = build_recording_summary(recipe_id, target, protocol)
        if fam is not None:
            families.append(fam)
        if rec is not None:
            recordings.append(rec)
    return {
        "family": pd.concat(families, ignore_index=True) if families else pd.DataFrame(),
        "recording": pd.concat(recordings, ignore_index=True) if recordings else pd.DataFrame(),
    }


def build_extreme_groups_table(summary_df: pd.DataFrame, *, group_col: str, metric: str = "r2", top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = summary_df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    top = df.sort_values(metric, ascending=False).head(top_n).reset_index(drop=True)
    bottom = df.sort_values(metric, ascending=True).head(top_n).reset_index(drop=True)
    return top, bottom


def build_prediction_diagnostics(recipe_id: str, target: str, protocol: str = "recording_disjoint_main") -> dict[str, float | pd.DataFrame]:
    result = load_experiment(recipe_id, target, protocol)
    preds = result.predictions.copy() if result.predictions is not None else pd.DataFrame()
    if preds.empty:
        return {"bias": float("nan"), "mae": float("nan"), "correlation": float("nan"), "predictions": preds}
    preds["residual"] = preds["pred"] - preds["true"]
    corr = preds[["true", "pred"]].corr().iloc[0, 1]
    return {
        "bias": float(preds["residual"].mean()),
        "mae": float(preds["residual"].abs().mean()),
        "correlation": float(corr),
        "predictions": preds,
    }


def build_shap_group_summary() -> pd.DataFrame:
    shap_df = load_fpos_shap_importance().copy()

    def _group(feature: str) -> str:
        if feature.startswith("wf_"):
            return "waveform"
        if feature.startswith("acg_"):
            return "acg"
        if "_recording_" in feature:
            return "recording_context"
        if feature.startswith("amp_"):
            return "amplitude"
        if feature.startswith("si_") or feature.startswith("isi_"):
            return "spikeinterface_qc"
        if feature == "source_pred":
            return "source_transfer"
        return "other"

    shap_df["feature_group"] = shap_df["feature"].map(_group)
    return (
        shap_df.groupby("feature_group", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), feature_count=("feature", "size"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
