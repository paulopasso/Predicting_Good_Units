from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def feature_group_for_column(column: str) -> str:
    name = str(column)
    if name.startswith("wf_bin_"):
        return "waveform_bins"
    if name.startswith("acg_"):
        return "acg_bins"
    if any(
        suffix in name
        for suffix in (
            "_recording_mean",
            "_recording_std",
            "_recording_median",
            "_recording_iqr",
            "_recording_delta",
        )
    ):
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
    return frame.sort_values(
        ["comparison", "feature_set", "abs_smd", "ks_stat"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


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
