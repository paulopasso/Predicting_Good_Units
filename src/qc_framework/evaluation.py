from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .features import FeatureSpec, FeatureSetBuilder
from .models import QCTrainer, TrainerResult, _score_predictions


def _feature_family_from_name(name: str) -> str:
    clean = name.replace("num__", "").replace("cat__", "")
    if "wf_bin_" in clean:
        return "waveform_bins"
    if clean.startswith("wf_") or clean in {
        "snr",
        "peak_to_trough_uv",
        "pre_trough_peak_uv",
        "post_trough_peak_uv",
        "half_width_ms",
        "trough_time_ms",
        "repolarization_slope",
        "template_norm",
    }:
        return "waveform"
    if clean.startswith("acg_"):
        return "acg"
    if clean.startswith("si_"):
        return "spikeinterface"
    if clean.startswith("rec_"):
        return "recording_context"
    if "cosine" in clean or clean in {
        "n_confusable",
        "isolation_score",
        "min_neighbor_dist_um",
        "mean_neighbor_dist_um",
    }:
        return "cosine"
    return "other"


class QCEvaluator:
    """Scoring utilities for TrainerResult lists."""

    @staticmethod
    def score(y_true: np.ndarray, pred: np.ndarray) -> dict:
        return _score_predictions(y_true, pred)

    @staticmethod
    def results_table(results: list[TrainerResult]) -> pd.DataFrame:
        rows = [
            {
                "target": r.target,
                "split": r.split_name,
                "feature_mode": r.feature_mode,
                "model": r.model_name,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "mae": r.mae,
                "rmse": r.rmse,
                "r2": r.r2,
                "bias": r.bias,
            }
            for r in results
        ]
        return pd.DataFrame(rows).sort_values(["split", "feature_mode", "target", "rmse", "mae"])


class JointEvaluator:
    """Compute multi-target distance metrics."""

    def score(
        self,
        results: list[TrainerResult],
        required_targets: tuple[str, ...] = ("accuracy", "fpos", "fmiss"),
    ) -> pd.DataFrame:
        combos: dict[tuple, dict[str, pd.DataFrame]] = {}
        for r in results:
            key = (r.split_name, r.feature_mode, r.model_name)
            combos.setdefault(key, {})[r.target] = r.predictions

        rows = []
        for (split_name, feature_mode, model_name), target_preds in combos.items():
            missing = [t for t in required_targets if t not in target_preds]
            if missing:
                continue
            merged = target_preds[required_targets[0]][
                ["row_uid", "recording_key", "dataset_type", "sorter_name"]
            ].copy()
            for t in required_targets:
                part = target_preds[t][["row_uid", "true", "pred", "abs_error", "sq_error"]].rename(
                    columns={
                        "true": f"true_{t}",
                        "pred": f"pred_{t}",
                        "abs_error": f"abs_error_{t}",
                        "sq_error": f"sq_error_{t}",
                    }
                )
                merged = merged.merge(part, on="row_uid", how="inner")

            if merged.empty:
                continue

            abs_cols = [f"abs_error_{t}" for t in required_targets]
            sq_cols = [f"sq_error_{t}" for t in required_targets]
            merged["average_target_distance"] = merged[abs_cols].mean(axis=1)
            merged["joint_target_rmse"] = np.sqrt(merged[sq_cols].mean(axis=1))

            rows.append(
                {
                    "split": split_name,
                    "feature_mode": feature_mode,
                    "model": model_name,
                    "n_rows": int(len(merged)),
                    "average_target_distance": float(merged["average_target_distance"].mean()),
                    "median_average_target_distance": float(merged["average_target_distance"].median()),
                    "joint_target_rmse": float(merged["joint_target_rmse"].mean()),
                }
            )
        return pd.DataFrame(rows).sort_values("joint_target_rmse") if rows else pd.DataFrame()


class TransferEvaluator:
    """Dataset-type and sorter holdout transfer experiments."""

    def __init__(self, trainer: QCTrainer) -> None:
        self.trainer = trainer

    def dataset_type_holdout(
        self,
        df: pd.DataFrame,
        targets: list[str],
        specs: list[FeatureSpec],
        min_rows: int = 40,
    ) -> pd.DataFrame:
        if "dataset_type" not in df.columns:
            return pd.DataFrame()

        all_results: list[TrainerResult] = []
        observed_types = [
            dt
            for dt, count in df["dataset_type"].value_counts().items()
            if count >= min_rows
        ]

        for held_out in observed_types:
            train_part = df[df["dataset_type"] != held_out].copy()
            test_part = df[df["dataset_type"] == held_out].copy()
            if len(train_part) < self.trainer.min_train_rows or len(test_part) < min_rows:
                continue
            results = self.trainer.fit_all(
                train=train_part,
                test=test_part,
                targets=targets,
                specs=specs,
                split_name=f"holdout_dataset_type:{held_out}",
            )
            all_results.extend(results)

        return QCEvaluator.results_table(all_results) if all_results else pd.DataFrame()

    def sorter_holdout(
        self,
        df: pd.DataFrame,
        targets: list[str],
        specs: list[FeatureSpec],
        min_rows: int = 170,
    ) -> pd.DataFrame:
        if "sorter_name" not in df.columns:
            return pd.DataFrame()

        all_results: list[TrainerResult] = []
        sorter_counts = df["sorter_name"].value_counts()
        held_out_sorters = sorter_counts[sorter_counts >= min_rows].index.tolist()

        for held_out in held_out_sorters:
            train_part = df[df["sorter_name"] != held_out].copy()
            test_part = df[df["sorter_name"] == held_out].copy()
            if len(train_part) < self.trainer.min_train_rows or len(test_part) < min_rows:
                continue
            results = self.trainer.fit_all(
                train=train_part,
                test=test_part,
                targets=targets,
                specs=specs,
                split_name=f"holdout_sorter:{held_out}",
                model_names=["lightgbm", "random_forest"],
            )
            all_results.extend(results)

        return QCEvaluator.results_table(all_results) if all_results else pd.DataFrame()


class SHAPAnalyzer:
    """SHAP-based feature importance for LightGBM models in a Pipeline."""

    def analyze(
        self,
        pipe: Pipeline,
        X: pd.DataFrame,
        target: str,
        sample_n: int = 800,
        random_state: int = 42,
    ) -> pd.DataFrame:
        try:
            import shap
        except ImportError:
            raise ImportError("shap is required for SHAPAnalyzer. pip install shap")

        if len(X) > sample_n:
            X = X.sample(sample_n, random_state=random_state)

        X_trans = pipe.named_steps["pre"].transform(X)
        feature_names = pipe.named_steps["pre"].get_feature_names_out()

        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X_trans)

        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
                "family": [_feature_family_from_name(f) for f in feature_names],
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_df["rank"] = shap_df["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
        return shap_df

    @staticmethod
    def family_summary(shap_df: pd.DataFrame) -> pd.DataFrame:
        return (
            shap_df.groupby("family")["mean_abs_shap"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
