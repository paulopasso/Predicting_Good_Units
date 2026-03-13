from __future__ import annotations

from dataclasses import dataclass, field
import re

import pandas as pd

_WF_PREFIX = "wf_bin_"
_ACG_PREFIX = "acg_"
_RELATIVE_SUFFIX = "_recording_pct"
_CONTEXT_MEAN_SUFFIX = "_recording_mean"
_CONTEXT_STD_SUFFIX = "_recording_std"
_CONTEXT_MEDIAN_SUFFIX = "_recording_median"
_CONTEXT_IQR_SUFFIX = "_recording_iqr"
_CONTEXT_CENTERED_SUFFIX = "_recording_delta"
_STUDY_INDICATOR_PREFIX = "study_indicator__"
_WF_RE = re.compile(r"^wf_bin_\d+$")
_ACG_RE = re.compile(r"^acg_\d+$")

_DEFAULT_RECORDING_RELATIVE_COLS = [
    "snr",
    "peak_to_trough_uv",
    "amp_mean",
    "amp_std",
    "amp_p50",
    "template_norm",
    "wf_energy",
    "si_snr",
    "si_amplitude_median",
    "si_amplitude_cutoff",
    "si_sd_ratio",
    "si_drift_ptp",
    "si_drift_std",
    "si_drift_mad",
    "max_cosine",
    "mean_cosine_top3",
    "mean_cosine_top5",
    "median_cosine",
    "std_cosine",
    "cosine_gap_top2",
    "n_confusable",
    "isolation_score",
    "max_waveform_cosine",
    "mean_waveform_cosine_top3",
    "median_waveform_cosine",
    "std_waveform_cosine",
    "n_waveform_confusable",
    "min_neighbor_dist_um",
    "mean_neighbor_dist_um",
]

_DEFAULT_RECORDING_CONTEXT_COLS = [
    "snr",
    "peak_to_trough_uv",
    "amp_mean",
    "amp_std",
    "amp_p50",
    "template_norm",
    "wf_energy",
    "si_snr",
    "si_amplitude_median",
    "si_amplitude_cutoff",
    "si_drift_ptp",
    "si_drift_std",
    "si_drift_mad",
    "firing_rate_hz",
    "isi_violation_rate",
    "max_cosine",
    "mean_cosine_top3",
    "median_cosine",
    "n_confusable",
    "isolation_score",
    "min_neighbor_dist_um",
    "mean_neighbor_dist_um",
]


@dataclass
class FeatureSpec:
    mode: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    excluded_cols: dict[str, str] = field(default_factory=dict)
    soft_excluded_cols: set[str] = field(default_factory=set)

    @property
    def all_features(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols


class FeatureSetBuilder:
    """Detect and classify feature columns after applying hard and soft exclusions."""

    ID_COLUMNS: frozenset[str] = frozenset(
        {
            "row_uid",
            "group_key",
            "recording_key",
            "sorter_recording_key",
            "recording_name",
            "unit_id",
            "matched_gt_unit_id",
            "hungarian_gt_unit_id",
            "gt_key",
        }
    )

    TARGET_DERIVED_COLUMNS: frozenset[str] = frozenset(
        {
            "accuracy",
            "agreement_score",
            "fpos",
            "fmiss",
            "fmiss_extended",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "num_gt",
            "num_tested",
            "match_status",
        }
    )

    FORBIDDEN_CONTEXT_COLUMNS: frozenset[str] = frozenset(
        {
            "study_set",
            "study_name",
            "recording_name",
            "sorter_name",
            "dataset_type",
        }
    )

    SOFT_CONTEXT_EXCLUSIONS: frozenset[str] = frozenset(
        {
            "rec_n_channels",
            "rec_sampling_rate",
            "rec_duration_sec",
        }
    )
    TRANSFER_VIEW_NAMES: frozenset[str] = frozenset(
        {
            "unit_raw",
            "norm_swap",
            "shape_only",
            "recording_relative",
            "recording_context",
            "study_identity",
        }
    )

    def build(
        self,
        df: pd.DataFrame,
        mode: str = "unit_only",
        exclude_soft: bool = False,
        extra_drop_cols: set[str] | None = None,
    ) -> FeatureSpec:
        """Build a FeatureSpec from df columns.

        Parameters
        ----------
        mode:
            "unit_only" excludes rec_* columns.
            "unit_plus_recording" keeps numeric rec_* columns.
        exclude_soft:
            If True, also exclude SOFT_CONTEXT_EXCLUSIONS.
        extra_drop_cols:
            Additional columns to drop (e.g. exact duplicates).
        """
        hard_exclusions = (
            self.ID_COLUMNS | self.TARGET_DERIVED_COLUMNS | self.FORBIDDEN_CONTEXT_COLUMNS
        )
        extra_drop = extra_drop_cols or set()
        soft_excluded: set[str] = set()

        numeric: list[str] = []
        categorical: list[str] = []
        excluded: dict[str, str] = {}

        work = df.copy()

        for col in work.columns:
            if col in hard_exclusions:
                excluded[col] = "hard_exclusion"
                continue
            if col in extra_drop:
                excluded[col] = "exact_duplicate"
                continue
            if mode == "unit_only" and str(col).startswith("rec_"):
                excluded[col] = "recording_context_not_in_unit_only_mode"
                continue
            if exclude_soft and col in self.SOFT_CONTEXT_EXCLUSIONS:
                excluded[col] = "soft_context_exclusion"
                soft_excluded.add(col)
                continue

            if pd.api.types.is_numeric_dtype(work[col]):
                numeric.append(col)
                continue

            maybe_num = pd.to_numeric(work[col], errors="coerce")
            if maybe_num.notna().mean() > 0.95:
                work[col] = maybe_num
                numeric.append(col)
                continue

            if work[col].nunique(dropna=True) <= 20:
                categorical.append(col)
            else:
                excluded[col] = "high_cardinality_string"

        return FeatureSpec(
            mode=mode,
            numeric_cols=numeric,
            categorical_cols=categorical,
            excluded_cols=excluded,
            soft_excluded_cols=soft_excluded,
        )

    @staticmethod
    def find_exact_duplicate_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        pairs: list[tuple[str, str]] = []
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1:]:
                if df[a].equals(df[b]):
                    pairs.append((a, b))
        return pairs

    @staticmethod
    def align_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
        missing = [c for c in spec.all_features if c not in df.columns]
        aligned = df.copy()
        for c in missing:
            aligned[c] = pd.NA
        return aligned[spec.all_features].copy()

    @staticmethod
    def waveform_bin_cols(columns: list[str]) -> list[str]:
        return [c for c in columns if _WF_RE.match(str(c))]

    @staticmethod
    def acg_cols(columns: list[str]) -> list[str]:
        return [c for c in columns if _ACG_RE.match(str(c))]

    @classmethod
    def shape_cols(cls, columns: list[str]) -> list[str]:
        return cls.waveform_bin_cols(columns) + cls.acg_cols(columns)

    @classmethod
    def transfer_feature_columns(
        cls,
        df: pd.DataFrame,
        spec: FeatureSpec,
        view_names: list[str] | tuple[str, ...],
        normalized_raw_cols: list[str] | tuple[str, ...] | None = None,
        normalizer_suffix: str = "_z",
        relative_suffix: str = _RELATIVE_SUFFIX,
        context_suffixes: tuple[str, ...] = (
            _CONTEXT_MEAN_SUFFIX,
            _CONTEXT_STD_SUFFIX,
            _CONTEXT_MEDIAN_SUFFIX,
            _CONTEXT_IQR_SUFFIX,
            _CONTEXT_CENTERED_SUFFIX,
        ),
        study_prefix: str = _STUDY_INDICATOR_PREFIX,
    ) -> list[str]:
        requested = list(view_names)
        unknown = sorted(set(requested) - set(cls.TRANSFER_VIEW_NAMES))
        if unknown:
            raise KeyError(f"Unknown transfer view(s): {unknown}. Available: {sorted(cls.TRANSFER_VIEW_NAMES)}")

        normalized_raw = set(normalized_raw_cols or [])
        base_cols = [c for c in spec.all_features if c in df.columns]
        selected: list[str] = []

        if "unit_raw" in requested:
            selected.extend(base_cols)

        if "norm_swap" in requested:
            selected.extend([c for c in base_cols if c not in normalized_raw])
            selected.extend(
                [f"{c}{normalizer_suffix}" for c in normalized_raw if f"{c}{normalizer_suffix}" in df.columns]
            )

        if "shape_only" in requested:
            selected.extend(cls.shape_cols(base_cols))

        if "recording_relative" in requested:
            relative_cols = [
                c for c in df.columns
                if c.endswith(relative_suffix) and c[: -len(relative_suffix)] in df.columns
            ]
            selected.extend(relative_cols)

        if "recording_context" in requested:
            context_cols = [
                c for c in df.columns
                if any(str(c).endswith(suffix) for suffix in context_suffixes)
            ]
            selected.extend(context_cols)

        if "study_identity" in requested:
            selected.extend([c for c in df.columns if str(c).startswith(study_prefix)])

        seen: set[str] = set()
        deduped: list[str] = []
        for col in selected:
            if col in df.columns and col not in seen:
                deduped.append(col)
                seen.add(col)
        return deduped


class RecordingRelativeFeatureAugmenter:
    """Add within-recording percentile features for transfer benchmarking.

    Each generated feature expresses a unit's position relative to the other
    units in the same recording. This keeps the transformation transductive at
    the recording level without using any labels.
    """

    def __init__(
        self,
        cols_to_rank: list[str] | None = None,
        suffix: str = _RELATIVE_SUFFIX,
    ) -> None:
        self.cols_to_rank = cols_to_rank
        self.suffix = suffix
        self.generated_cols_: list[str] = []

    def _effective_cols(self, df: pd.DataFrame) -> list[str]:
        base = self.cols_to_rank if self.cols_to_rank is not None else _DEFAULT_RECORDING_RELATIVE_COLS
        pattern_cols = [
            c for c in df.columns
            if str(c).startswith(("amp_", "si_amplitude_", "si_drift_"))
            or "cosine" in str(c)
            or "confusable" in str(c)
            or str(c).endswith("_neighbor_dist_um")
        ]
        merged = list(dict.fromkeys(list(base) + pattern_cols))
        return [
            c
            for c in merged
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and not str(c).endswith(self.suffix)
        ]

    def fit(self, df: pd.DataFrame) -> "RecordingRelativeFeatureAugmenter":
        self.generated_cols_ = [f"{c}{self.suffix}" for c in self._effective_cols(df)]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "recording_key" not in df.columns:
            raise KeyError("RecordingRelativeFeatureAugmenter requires a recording_key column")

        cols = self._effective_cols(df)
        out = df.copy()
        groups = out["recording_key"].astype(str)

        for col in cols:
            vals = pd.to_numeric(out[col], errors="coerce")
            ranks = vals.groupby(groups).rank(method="average")
            counts = vals.notna().groupby(groups).transform("sum")
            pct = (ranks - 0.5) / counts.where(counts > 0)
            out[f"{col}{self.suffix}"] = pct.astype(float)

        self.generated_cols_ = [f"{c}{self.suffix}" for c in cols]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class RecordingContextFeatureAugmenter:
    """Add within-recording summary features using only existing parquet columns."""

    def __init__(
        self,
        cols_to_summarize: list[str] | None = None,
        suffixes: tuple[str, str, str, str, str] = (
            _CONTEXT_MEAN_SUFFIX,
            _CONTEXT_STD_SUFFIX,
            _CONTEXT_MEDIAN_SUFFIX,
            _CONTEXT_IQR_SUFFIX,
            _CONTEXT_CENTERED_SUFFIX,
        ),
    ) -> None:
        self.cols_to_summarize = cols_to_summarize
        self.suffixes = suffixes
        self.generated_cols_: list[str] = []

    def _effective_cols(self, df: pd.DataFrame) -> list[str]:
        base = self.cols_to_summarize if self.cols_to_summarize is not None else _DEFAULT_RECORDING_CONTEXT_COLS
        pattern_cols = [
            c for c in df.columns
            if str(c).startswith(("amp_", "si_amplitude_", "si_drift_"))
            or "cosine" in str(c)
            or "confusable" in str(c)
            or str(c).endswith("_neighbor_dist_um")
        ]
        merged = list(dict.fromkeys(list(base) + pattern_cols))
        return [
            c
            for c in merged
            if c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])
            and not any(str(c).endswith(suffix) for suffix in self.suffixes)
        ]

    def fit(self, df: pd.DataFrame) -> "RecordingContextFeatureAugmenter":
        cols = self._effective_cols(df)
        generated: list[str] = []
        for col in cols:
            generated.extend(
                [
                    f"{col}{self.suffixes[0]}",
                    f"{col}{self.suffixes[1]}",
                    f"{col}{self.suffixes[2]}",
                    f"{col}{self.suffixes[3]}",
                    f"{col}{self.suffixes[4]}",
                ]
            )
        self.generated_cols_ = generated
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "recording_key" not in df.columns:
            raise KeyError("RecordingContextFeatureAugmenter requires a recording_key column")

        cols = self._effective_cols(df)
        out = df.copy()
        groups = out["recording_key"].astype(str)
        generated: dict[str, pd.Series] = {}

        for col in cols:
            vals = pd.to_numeric(out[col], errors="coerce")
            mean = vals.groupby(groups).transform("mean")
            std = vals.groupby(groups).transform("std").fillna(0.0)
            median = vals.groupby(groups).transform("median")
            q25 = vals.groupby(groups).transform(lambda s: s.quantile(0.25))
            q75 = vals.groupby(groups).transform(lambda s: s.quantile(0.75))
            iqr = q75 - q25

            generated[f"{col}{self.suffixes[0]}"] = mean.astype(float)
            generated[f"{col}{self.suffixes[1]}"] = std.astype(float)
            generated[f"{col}{self.suffixes[2]}"] = median.astype(float)
            generated[f"{col}{self.suffixes[3]}"] = iqr.astype(float)
            generated[f"{col}{self.suffixes[4]}"] = (vals - median).astype(float)

        generated_df = pd.DataFrame(generated, index=out.index)
        out = pd.concat([out, generated_df], axis=1)
        self.generated_cols_ = list(generated_df.columns)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class StudyIndicatorAugmenter:
    """Add numeric one-hot indicators for low-cardinality study identity."""

    def __init__(
        self,
        source_col: str = "study_set",
        prefix: str = _STUDY_INDICATOR_PREFIX,
    ) -> None:
        self.source_col = source_col
        self.prefix = prefix
        self.levels_: list[str] = []
        self.generated_cols_: list[str] = []

    @staticmethod
    def _sanitize(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
        return cleaned.strip("_").lower() or "unknown"

    def fit(self, df: pd.DataFrame) -> "StudyIndicatorAugmenter":
        if self.source_col not in df.columns:
            self.levels_ = []
            self.generated_cols_ = []
            return self
        levels = (
            df[self.source_col]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .index
            .tolist()
        )
        self.levels_ = levels
        self.generated_cols_ = [f"{self.prefix}{self._sanitize(level)}" for level in levels]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.source_col not in out.columns:
            return out
        values = out[self.source_col].fillna("unknown").astype(str)
        for level, col in zip(self.levels_, self.generated_cols_):
            out[col] = (values == level).astype(float)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
