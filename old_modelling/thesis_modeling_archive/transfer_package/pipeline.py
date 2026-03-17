from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..augmenters import FeatureSetBuilder
from ..backends import JointEvaluator, QCEvaluator, QCModelRegistry, QCTrainer, SHAPAnalyzer, TrainerResult, TransferEvaluator
from .data import QCDataCleaner, QCDataLoader, QCDataSplitter


def _detect_targets(df: pd.DataFrame, preferred: list[str] | None = None) -> list[str]:
    if preferred is None:
        preferred = ["accuracy", "fpos", "fmiss"]
    return [c for c in preferred if c in df.columns]


def _summarize_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": name,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "recordings": int(df["recording_key"].nunique())
                if "recording_key" in df.columns
                else None,
                "sorters": int(df["sorter_name"].nunique())
                if "sorter_name" in df.columns
                else None,
                "duplicate_row_uid": int(df["row_uid"].duplicated().sum())
                if "row_uid" in df.columns
                else None,
            }
        ]
    )


class QCBenchmarkPipeline:
    """End-to-end QC modeling benchmark."""

    def __init__(
        self,
        train_path: str | Path,
        test_path: str | Path | None = None,
        random_state: int = 42,
        grouped_test_size: float = 0.20,
        weighting_strategy: str | None = None,
        exclude_soft_context: bool = False,
        holdout_sorter_min_rows: int = 170,
        min_train_rows: int = 200,
        min_test_rows: int = 40,
        dataset_type_filter: list[str] | None = None,
    ) -> None:
        self.train_path = Path(train_path)
        self.test_path = Path(test_path) if test_path else None
        self.random_state = random_state
        self.grouped_test_size = grouped_test_size
        self.weighting_strategy = weighting_strategy
        self.exclude_soft_context = exclude_soft_context
        self.holdout_sorter_min_rows = holdout_sorter_min_rows
        self.min_train_rows = min_train_rows
        self.min_test_rows = min_test_rows
        self.dataset_type_filter = dataset_type_filter  # e.g. ["hybrid"] or ["paired"]

        # Populated by run()
        self._train_raw: pd.DataFrame | None = None
        self._train_clean: pd.DataFrame | None = None
        self._test_clean: pd.DataFrame | None = None
        self._clean_report: dict | None = None
        self._grouped_train: pd.DataFrame | None = None
        self._grouped_test: pd.DataFrame | None = None
        self._row_random_train: pd.DataFrame | None = None
        self._row_random_test: pd.DataFrame | None = None
        self._specs: list = []
        self._results: list[TrainerResult] = []
        self._artifacts: dict = {}

    # ------------------------------------------------------------------
    @classmethod
    def from_parquet(cls, train_path: str | Path, **kwargs) -> "QCBenchmarkPipeline":
        return cls(train_path=train_path, **kwargs)

    # ------------------------------------------------------------------
    def load_and_clean(self) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
        loader = QCDataLoader()
        train_raw = loader.load(self.train_path)
        test_raw = loader.load(self.test_path) if self.test_path and self.test_path.exists() else None

        if self.dataset_type_filter:
            keep = set(self.dataset_type_filter)
            train_raw = train_raw[train_raw["dataset_type"].isin(keep)].copy()
            if test_raw is not None:
                test_raw = test_raw[test_raw["dataset_type"].isin(keep)].copy()
            print(f"dataset_type_filter={self.dataset_type_filter}: {len(train_raw)} train rows kept")

        cleaner = QCDataCleaner()
        train_clean, clean_report = cleaner.clean(train_raw)

        test_clean = None
        if test_raw is not None:
            test_clean, _ = cleaner.clean(test_raw)

        self._train_raw = train_raw
        self._train_clean = train_clean
        self._test_clean = test_clean
        self._clean_report = clean_report
        return train_clean, test_clean, clean_report

    # ------------------------------------------------------------------
    def build_feature_specs(self, df: pd.DataFrame) -> list:
        builder = FeatureSetBuilder()
        dup_pairs = FeatureSetBuilder.find_exact_duplicate_pairs(df)
        extra_drop = {b for _, b in dup_pairs}

        specs = [
            builder.build(df, mode="unit_only", exclude_soft=False, extra_drop_cols=extra_drop),
            builder.build(df, mode="unit_plus_recording", exclude_soft=False, extra_drop_cols=extra_drop),
        ]
        if self.exclude_soft_context:
            specs.append(
                builder.build(df, mode="unit_only", exclude_soft=True, extra_drop_cols=extra_drop)
            )
        self._specs = specs
        return specs

    # ------------------------------------------------------------------
    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        splitter = QCDataSplitter(
            test_size=self.grouped_test_size, random_state=self.random_state
        )
        grouped_train, grouped_test = splitter.grouped_split(df)
        row_random_train, row_random_test = splitter.row_random_split(df)

        split_report = {
            "grouped": splitter.validate(grouped_train, grouped_test),
            "row_random": splitter.validate(row_random_train, row_random_test),
        }

        self._grouped_train = grouped_train
        self._grouped_test = grouped_test
        self._row_random_train = row_random_train
        self._row_random_test = row_random_test
        return grouped_train, grouped_test, row_random_train, row_random_test, split_report

    # ------------------------------------------------------------------
    def fit_baseline(
        self,
        grouped_train: pd.DataFrame,
        grouped_test: pd.DataFrame,
        row_random_train: pd.DataFrame,
        row_random_test: pd.DataFrame,
        specs: list,
        targets: list[str] | None = None,
    ) -> list[TrainerResult]:
        if targets is None:
            targets = _detect_targets(grouped_train)

        trainer = QCTrainer(
            random_state=self.random_state,
            min_train_rows=self.min_train_rows,
            min_test_rows=self.min_test_rows,
            weighting_strategy=self.weighting_strategy,
        )

        # Row-random reference: unit_only only
        unit_only_spec = next((s for s in specs if s.mode == "unit_only" and not s.soft_excluded_cols), specs[0])
        results = trainer.fit_all(
            train=row_random_train,
            test=row_random_test,
            targets=targets,
            specs=[unit_only_spec],
            split_name="row_random_reference",
        )

        # Grouped: all specs
        results += trainer.fit_all(
            train=grouped_train,
            test=grouped_test,
            targets=targets,
            specs=specs,
            split_name="grouped_recording",
        )

        self._results = results
        return results

    # ------------------------------------------------------------------
    def joint_metrics(self, results: list[TrainerResult]) -> pd.DataFrame:
        return JointEvaluator().score(results)

    # ------------------------------------------------------------------
    def dataset_type_transfer(
        self, df: pd.DataFrame, specs: list, targets: list[str] | None = None
    ) -> pd.DataFrame:
        if targets is None:
            targets = _detect_targets(df)
        trainer = QCTrainer(
            random_state=self.random_state,
            min_train_rows=self.min_train_rows,
            min_test_rows=self.min_test_rows,
        )
        unit_only_spec = next((s for s in specs if s.mode == "unit_only" and not s.soft_excluded_cols), specs[0])
        return TransferEvaluator(trainer).dataset_type_holdout(
            df, targets, [unit_only_spec], min_rows=self.min_test_rows
        )

    # ------------------------------------------------------------------
    def sorter_transfer(
        self, df: pd.DataFrame, specs: list, targets: list[str] | None = None
    ) -> pd.DataFrame:
        if targets is None:
            targets = _detect_targets(df)
        trainer = QCTrainer(
            random_state=self.random_state,
            min_train_rows=self.min_train_rows,
            min_test_rows=self.min_test_rows,
        )
        unit_only_spec = next((s for s in specs if s.mode == "unit_only" and not s.soft_excluded_cols), specs[0])
        return TransferEvaluator(trainer).sorter_holdout(
            df, targets, [unit_only_spec], min_rows=self.holdout_sorter_min_rows
        )

    # ------------------------------------------------------------------
    def shap_analysis(
        self,
        results: list[TrainerResult],
        grouped_train: pd.DataFrame,
        grouped_test: pd.DataFrame,
        specs: list,
        targets: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        if targets is None:
            targets = _detect_targets(grouped_train)
        unit_only_spec = next((s for s in specs if s.mode == "unit_only" and not s.soft_excluded_cols), specs[0])
        analyzer = SHAPAnalyzer()
        shap_results: dict[str, pd.DataFrame] = {}

        for result in results:
            if (
                result.model_name != "lightgbm"
                or result.split_name != "grouped_recording"
                or result.feature_mode != "unit_only"
                or result.target not in targets
            ):
                continue
            X_test = FeatureSetBuilder.align_frame(grouped_test, unit_only_spec)
            shap_df = analyzer.analyze(result.fitted_pipe, X_test, result.target)
            shap_results[result.target] = shap_df

        return shap_results

    # ------------------------------------------------------------------
    def results_table(self, results: list[TrainerResult]) -> pd.DataFrame:
        return QCEvaluator.results_table(results)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Full end-to-end pipeline. Returns dict of result DataFrames."""
        train_clean, test_clean, clean_report = self.load_and_clean()
        specs = self.build_feature_specs(train_clean)
        grouped_train, grouped_test, row_random_train, row_random_test, split_report = self.split(train_clean)
        targets = _detect_targets(train_clean)

        results = self.fit_baseline(
            grouped_train, grouped_test, row_random_train, row_random_test, specs, targets
        )

        artifacts = {
            "clean_report": clean_report,
            "split_report": split_report,
            "results_table": self.results_table(results),
            "joint_metrics": self.joint_metrics(results),
            "dataset_type_transfer": self.dataset_type_transfer(train_clean, specs, targets),
            "sorter_transfer": self.sorter_transfer(train_clean, specs, targets),
            "shap": self.shap_analysis(results, grouped_train, grouped_test, specs, targets),
        }

        if test_clean is not None:
            trainer = QCTrainer(random_state=self.random_state)
            external_results = trainer.fit_all(
                train=train_clean,
                test=test_clean,
                targets=targets,
                specs=specs,
                split_name="external_test",
            )
            artifacts["external_results"] = self.results_table(external_results)

        self._artifacts = artifacts
        return artifacts

    # ------------------------------------------------------------------
    def save_artifacts(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = self._artifacts
        if not artifacts:
            raise RuntimeError("No artifacts to save. Call run() first.")

        for name in ["results_table", "joint_metrics", "dataset_type_transfer", "sorter_transfer"]:
            df = artifacts.get(name)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(output_dir / f"{name}.csv", index=False)

        shap = artifacts.get("shap", {})
        if shap:
            payload = {
                target: df.head(25).set_index("feature")["mean_abs_shap"].to_dict()
                for target, df in shap.items()
                if not df.empty
            }
            (output_dir / "shap_top_features.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )

        leakage_scan = artifacts.get("clean_report", {}).get("leakage_scan")
        if leakage_scan is not None and not leakage_scan.empty:
            leakage_scan.to_csv(output_dir / "leakage_scan.csv", index=False)


class QCDatasetReviewPipeline:
    """Lightweight data review (no modeling)."""

    def __init__(self, train_path: str | Path, test_path: str | Path | None = None) -> None:
        self.train_path = Path(train_path)
        self.test_path = Path(test_path) if test_path else None

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        train = QCDataLoader.load(self.train_path)
        test = QCDataLoader.load(self.test_path) if self.test_path and self.test_path.exists() else None
        return train, test

    def summarize(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        return _summarize_frame(df, name)

    def target_summary(self, df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
        rows = []
        for t in targets:
            s = pd.to_numeric(df[t], errors="coerce")
            rows.append(
                {
                    "target": t,
                    "rows": int(s.notna().sum()),
                    "missing_fraction": float(s.isna().mean()),
                    "mean": float(s.mean()) if s.notna().any() else float("nan"),
                    "std": float(s.std()) if s.notna().any() else float("nan"),
                    "q05": float(s.quantile(0.05)) if s.notna().any() else float("nan"),
                    "q50": float(s.quantile(0.50)) if s.notna().any() else float("nan"),
                    "q95": float(s.quantile(0.95)) if s.notna().any() else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def integrity_report(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaner = QCDataCleaner()
        return pd.DataFrame([cleaner.leakage_summary(df)])

    def detect_feature_families(self, df: pd.DataFrame) -> dict[str, list[str]]:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return {
            "waveform": [
                c for c in numeric_cols
                if c.startswith("wf_") or c in {
                    "snr", "peak_to_trough_uv", "pre_trough_peak_uv",
                    "post_trough_peak_uv", "half_width_ms", "trough_time_ms",
                    "repolarization_slope", "template_norm",
                }
            ],
            "waveform_bins": sorted(
                [c for c in numeric_cols if c.startswith("wf_bin_")],
                key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0,
            ),
            "acg": sorted([c for c in numeric_cols if c.startswith("acg_")]),
            "cosine": [
                c for c in numeric_cols
                if "cosine" in c or c in {
                    "n_confusable", "isolation_score",
                    "min_neighbor_dist_um", "mean_neighbor_dist_um",
                }
            ],
            "spikeinterface": [c for c in numeric_cols if c.startswith("si_")],
            "recording": [c for c in numeric_cols if c.startswith("rec_")],
        }
