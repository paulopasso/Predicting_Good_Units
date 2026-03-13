"""RecordingNormalizer — per-recording z-scoring for domain-shift mitigation."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that are recording-dependent and benefit from per-recording normalization.
# Excludes wf_bin_*, acg_* (already normalized by construction), firing_rate_hz,
# isi_*, presence_ratio (already relative).
_DEFAULT_NORMALIZE_COLS = [
    # Amplitude / waveform scalars
    "snr",
    "peak_to_trough_uv",
    "amp_mean",
    "amp_std",
    "amp_p50",
    "template_norm",
    "wf_energy",
    # Spatial
    "spread_um",
    "peak_channel_depth_um",
    "center_of_mass_um",
    # SpikeInterface quality metrics
    "si_snr",
    "si_amplitude_median",
    "si_amplitude_cutoff",
]


class RecordingNormalizer:
    """Per-recording z-scoring that removes domain-specific baselines.

    For each unit, computes ``(feature - recording_mean) / recording_std`` using
    statistics computed from ALL units in that recording (train or test combined).
    This removes recording-level amplitude baselines while preserving relative
    differences between units within the same recording.

    Parameters
    ----------
    cols_to_normalize:
        Columns to z-score. Defaults to ``_DEFAULT_NORMALIZE_COLS`` (all
        amplitude-dependent scalars). Columns not present in the DataFrame are
        silently skipped.
    suffix:
        Suffix appended to the new z-scored columns. Defaults to ``"_z"``.

    Usage
    -----
    Fit on the FULL DataFrame (train+test for a recording) so that test units
    see recording statistics derived from all available data for that recording::

        normalizer = RecordingNormalizer()
        df_full = pd.concat([train_df, test_df])
        df_norm = normalizer.fit_transform(df_full)
        train_norm = df_norm.loc[train_df.index]
        test_norm  = df_norm.loc[test_df.index]
    """

    def __init__(
        self,
        cols_to_normalize: list[str] | None = None,
        suffix: str = "_z",
    ) -> None:
        self.cols_to_normalize = cols_to_normalize
        self.suffix = suffix
        # (recording_key, col) -> (mean, std)
        self._stats: dict[tuple[str, str], tuple[float, float]] = {}

    # ------------------------------------------------------------------
    def _effective_cols(self, df: pd.DataFrame) -> list[str]:
        base = self.cols_to_normalize if self.cols_to_normalize is not None else _DEFAULT_NORMALIZE_COLS
        return [c for c in base if c in df.columns]

    def effective_cols(self, df: pd.DataFrame) -> list[str]:
        """Public helper used by transfer benchmarks to build explicit feature views."""
        return self._effective_cols(df)

    def generated_columns(self, df: pd.DataFrame) -> list[str]:
        return [f"{c}{self.suffix}" for c in self._effective_cols(df)]

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "RecordingNormalizer":
        """Compute per-recording statistics from *df*.

        Parameters
        ----------
        df:
            Must contain a ``recording_key`` column.
        """
        cols = self._effective_cols(df)
        self._stats = {}
        for rk, grp in df.groupby(df["recording_key"].astype(str)):
            for col in cols:
                vals = grp[col].dropna().astype(float)
                mean = float(vals.mean()) if len(vals) > 0 else 0.0
                std = float(vals.std(ddof=1)) if len(vals) > 1 else 1.0
                if not np.isfinite(std) or std == 0.0:
                    std = 1.0
                self._stats[(str(rk), col)] = (mean, std)
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``{col}{suffix}`` columns for each normalizable column.

        Original columns are preserved. Rows whose recording_key was not seen
        during ``fit`` receive ``NaN`` for the z-scored columns.
        """
        cols = self._effective_cols(df)
        out = df.copy()
        for col in cols:
            z_col = pd.Series(np.nan, index=out.index, dtype=float)
            for rk, grp in out.groupby(out["recording_key"].astype(str)):
                stats = self._stats.get((str(rk), col))
                if stats is not None:
                    mean, std = stats
                    z_col.loc[grp.index] = (grp[col].astype(float) - mean) / std
            out[col + self.suffix] = z_col
        return out

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
