from __future__ import annotations

import pandas as pd

from .modeling import build_family_summary
from .paths import THESIS_ROOT


def load_fpos_shap_importance() -> pd.DataFrame:
    aggregate_path = THESIS_ROOT / "artifacts/aggregates/family_held_out/fpos/fpos_waveform_lofo_shap_aggregate.csv"
    if aggregate_path.exists():
        return pd.read_csv(aggregate_path).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return pd.read_csv(THESIS_ROOT / "data/frozen_inputs/xai/fpos_shap_importance.csv")


def load_fpos_shap_group_summary() -> pd.DataFrame:
    aggregate_path = THESIS_ROOT / "artifacts/aggregates/family_held_out/fpos/fpos_waveform_lofo_shap_group_aggregate.csv"
    if aggregate_path.exists():
        return pd.read_csv(aggregate_path).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df = load_fpos_shap_importance().copy()
    if "feature_group" not in shap_df.columns:
        raise RuntimeError("Feature-level SHAP summary is missing feature_group and no grouped aggregate is available")
    return (
        shap_df.groupby("feature_group", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), feature_count=("feature", "size"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def build_target_asymmetry_table() -> pd.DataFrame:
    fpos = pd.read_csv(THESIS_ROOT / "artifacts/aggregates/family_held_out/fpos/leave_one_family_main5_by_family.csv")
    fmiss = pd.read_csv(THESIS_ROOT / "artifacts/aggregates/family_held_out/fmiss/leave_one_family_main3_by_family.csv")

    fpos = fpos.loc[fpos["recipe_id"] == "fpos_waveform_winner"].copy()
    fmiss = fmiss.loc[fmiss["recipe_id"] == "fmiss_paired_raw"].copy()
    if fpos.empty or fmiss.empty:
        raise RuntimeError("Missing held-out-family summaries for target asymmetry")

    fpos_part = fpos[["held_out_family", "r2", "mae"]].rename(
        columns={"held_out_family": "study_set", "r2": "fpos_r2", "mae": "fpos_mae"}
    )
    fmiss_part = fmiss[["held_out_family", "r2", "mae"]].rename(
        columns={"held_out_family": "study_set", "r2": "fmiss_r2", "mae": "fmiss_mae"}
    )
    return fpos_part.merge(fmiss_part, on="study_set", how="outer").sort_values("study_set").reset_index(drop=True)


def build_target_protocol_summary_table() -> pd.DataFrame:
    fpos_bench = pd.read_csv(THESIS_ROOT / "tables/04_fpos_model_progression/fpos_benchmark_ladder.csv")
    fmiss_bench = pd.read_csv(THESIS_ROOT / "tables/05_fmiss_model_progression/fmiss_benchmark_ladder.csv")
    fpos_lofo = pd.read_csv(THESIS_ROOT / "artifacts/aggregates/family_held_out/fpos/leave_one_family_main5_aggregate.csv")
    fmiss_lofo = pd.read_csv(THESIS_ROOT / "artifacts/aggregates/family_held_out/fmiss/leave_one_family_main3_aggregate.csv")

    best_fpos_bench = fpos_bench.sort_values(["r2", "mae"], ascending=[False, True]).iloc[0]
    best_fmiss_bench = fmiss_bench.sort_values(["r2", "mae"], ascending=[False, True]).iloc[0]
    best_fpos_lofo = fpos_lofo.sort_values(["macro_r2", "macro_mae"], ascending=[False, True]).iloc[0]
    best_fmiss_lofo = fmiss_lofo.sort_values(["macro_r2", "macro_mae"], ascending=[False, True]).iloc[0]

    return pd.DataFrame(
        [
            {
                "target": "fpos",
                "protocol": "recording_disjoint",
                "label": str(best_fpos_bench["label"]),
                "r2": float(best_fpos_bench["r2"]),
                "mae": float(best_fpos_bench["mae"]),
            },
            {
                "target": "fpos",
                "protocol": "family_held_out",
                "label": str(best_fpos_lofo["label"]),
                "r2": float(best_fpos_lofo["macro_r2"]),
                "mae": float(best_fpos_lofo["macro_mae"]),
            },
            {
                "target": "fmiss",
                "protocol": "recording_disjoint",
                "label": str(best_fmiss_bench["label"]),
                "r2": float(best_fmiss_bench["r2"]),
                "mae": float(best_fmiss_bench["mae"]),
            },
            {
                "target": "fmiss",
                "protocol": "family_held_out",
                "label": str(best_fmiss_lofo["label"]),
                "r2": float(best_fmiss_lofo["macro_r2"]),
                "mae": float(best_fmiss_lofo["macro_mae"]),
            },
        ]
    )
