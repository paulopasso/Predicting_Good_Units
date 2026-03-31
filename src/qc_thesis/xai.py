from __future__ import annotations

import pandas as pd

from .modeling import build_family_summary
from .paths import THESIS_ROOT


def load_fpos_shap_importance() -> pd.DataFrame:
    return pd.read_csv(THESIS_ROOT / "data/frozen_inputs/xai/fpos_shap_importance.csv")


def build_target_asymmetry_table() -> pd.DataFrame:
    fpos = build_family_summary("fpos_waveform_winner", "fpos", "recording_disjoint_main")
    fmiss = build_family_summary("fmiss_reduced_latent", "fmiss", "recording_disjoint_main")
    if fpos is None or fmiss is None:
        raise RuntimeError("Missing family summaries for target asymmetry")
    fpos_part = fpos[["study_set", "r2", "mae", "bias"]].rename(columns={"r2": "fpos_r2", "mae": "fpos_mae", "bias": "fpos_bias"})
    fmiss_part = fmiss[["study_set", "r2", "mae", "bias"]].rename(columns={"r2": "fmiss_r2", "mae": "fmiss_mae", "bias": "fmiss_bias"})
    return fpos_part.merge(fmiss_part, on="study_set", how="outer").sort_values("study_set").reset_index(drop=True)
