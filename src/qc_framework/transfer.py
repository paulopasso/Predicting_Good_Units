from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from scipy.special import expit, logit

from qc_neural.normalizer import RecordingNormalizer

from .data import QCDataCleaner, QCDataLoader
from .features import (
    FeatureSetBuilder,
    RecordingContextFeatureAugmenter,
    RecordingRelativeFeatureAugmenter,
    StudyIndicatorAugmenter,
)

try:
    from lightgbm import LGBMRegressor

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor

    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBRegressor

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from qc_neural.transfer import SingleTargetDomainAdaptiveRegressor

    _CONTEXT_TRANSFER_AVAILABLE = True
except ImportError:
    _CONTEXT_TRANSFER_AVAILABLE = False

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


def _safe_r2(y_true: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1:
        return float("nan")
    return float(r2_score(y_true, pred))


def _calibration_params(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(pred)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan")
    x = pred[mask]
    y = y_true[mask]
    if len(np.unique(x)) <= 1:
        return 0.0, float(np.mean(y))
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _score_predictions(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    slope, intercept = _calibration_params(y_true, pred)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": _safe_r2(y_true, pred),
        "bias": float(np.mean(pred - y_true)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _clip_pred(pred: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)


def _ensure_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.loc[:, cols].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


@dataclass
class TransferBenchmarkConfig:
    parquet_path: str | Path
    primary_targets: tuple[str, ...] = ("fpos", "fmiss")
    secondary_targets: tuple[str, ...] = ("accuracy",)
    source_fmiss_target: str = "fmiss_extended"
    final_eval_fmiss_target: str = "fmiss"
    paired_final_holdout_rows: int = 100
    allow_unlabeled_paired: bool = True
    feature_views: tuple[str, ...] = (
        "unit_raw",
        "norm_swap",
        "shape_only",
        "recording_relative",
        "recording_context",
        "study_identity",
    )
    model_families: tuple[str, ...] = (
        "dummy",
        "paired_only",
        "hybrid_only",
        "hybrid_calibrated",
        "hybrid_stack",
        "hybrid_plus_paired",
        "context_tabular",
        "context_transfer",
    )
    feature_view_sets: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("shape_only", ("shape_only",)),
        ("unit_raw", ("unit_raw",)),
        (
            "contextual_full",
            ("norm_swap", "recording_relative", "recording_context", "study_identity"),
        ),
    )
    protocol_modes: tuple[str, ...] = ("inductive", "contextual")
    model_selection_splits: int = 5
    model_selection_test_size: float = 0.40
    random_state: int = 42
    lightgbm_estimators: int = 250
    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 31
    xgboost_estimators: int = 250
    xgboost_learning_rate: float = 0.05
    xgboost_max_depth: int = 6
    boosting_backends: tuple[str, ...] = ("lightgbm", "catboost", "xgboost")
    paired_models: tuple[str, ...] = ()
    paired_weight_grid: tuple[float, ...] = (10.0, 25.0, 50.0)
    neural_pretrain_epochs: int = 35
    neural_finetune_epochs: int = 20
    neural_batch_size: int = 128
    neural_lr: float = 1e-3
    neural_domain_loss_weight: float = 0.15
    neural_target_loss_weight: float = 2.0
    target_transform: str = "logit"
    deploy_best_per_target: bool = True
    use_recording_context: bool = True
    include_ceiling_diagnostics: bool = True
    verbose: bool = False
    neural_verbose: bool = False

    def all_targets(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for target in list(self.primary_targets) + list(self.secondary_targets):
            if target not in seen:
                ordered.append(target)
                seen.add(target)
        return tuple(ordered)

    def source_target_for(self, eval_target: str) -> str:
        if eval_target == self.final_eval_fmiss_target:
            return self.source_fmiss_target
        return eval_target

    def effective_backends(self) -> tuple[str, ...]:
        return self.paired_models or self.boosting_backends


@dataclass
class _PreparedTransferData:
    df: pd.DataFrame
    hybrid: pd.DataFrame
    paired: pd.DataFrame
    paired_matched: pd.DataFrame
    spec: Any
    normalized_cols: list[str]
    feature_columns: dict[str, list[str]]
    relative_cols: list[str]
    context_cols: list[str]
    study_indicator_cols: list[str]


class QCTransferBenchmark:
    """Benchmark runner for hybrid→paired few-shot transfer."""

    def __init__(self, config: TransferBenchmarkConfig) -> None:
        self.config = config
        self._prepared: _PreparedTransferData | None = None
        self._selection_results = pd.DataFrame()
        self._selection_summary = pd.DataFrame()
        self._holdout_results = pd.DataFrame()
        self._holdout_summary = pd.DataFrame()
        self._winner_summary = pd.DataFrame()
        self._split_report: dict[str, Any] = {}
        self._leakage_checks = pd.DataFrame()
        self._task_total = 0
        self._task_done = 0
        self._task_started_at: float | None = None

    # ------------------------------------------------------------------
    def prepare_data(self) -> _PreparedTransferData:
        loader = QCDataLoader()
        cleaner = QCDataCleaner()
        raw = loader.load(self.config.parquet_path)
        cleaned, _ = cleaner.clean(raw)

        hybrid_clean = cleaned[cleaned["dataset_type"] == "hybrid"].copy()
        dup_pairs = FeatureSetBuilder.find_exact_duplicate_pairs(hybrid_clean)
        extra_drop = {b for _, b in dup_pairs}
        spec = FeatureSetBuilder().build(
            hybrid_clean,
            mode="unit_only",
            exclude_soft=False,
            extra_drop_cols=extra_drop,
        )

        df = cleaned.copy()
        normalizer = RecordingNormalizer()
        df = normalizer.fit_transform(df)
        relative = RecordingRelativeFeatureAugmenter()
        df = relative.fit_transform(df)
        context = RecordingContextFeatureAugmenter()
        df = context.fit_transform(df)
        study = StudyIndicatorAugmenter()
        df = study.fit_transform(df)

        hybrid = df[df["dataset_type"] == "hybrid"].copy()
        paired = df[df["dataset_type"] == "paired"].copy()
        paired_matched = paired[paired["matched_gt_unit_id"].notna()].copy()

        normalized_cols = normalizer.effective_cols(df)
        allowed_blocks = set(self.config.feature_views)
        feature_columns: dict[str, list[str]] = {}
        for name, blocks in self.config.feature_view_sets:
            if not set(blocks).issubset(allowed_blocks):
                continue
            feature_columns[name] = FeatureSetBuilder.transfer_feature_columns(
                df,
                spec,
                view_names=blocks,
                normalized_raw_cols=normalized_cols,
                normalizer_suffix=normalizer.suffix,
                relative_suffix=relative.suffix,
            )

        self._prepared = _PreparedTransferData(
            df=df,
            hybrid=hybrid,
            paired=paired,
            paired_matched=paired_matched,
            spec=spec,
            normalized_cols=normalized_cols,
            feature_columns=feature_columns,
            relative_cols=list(relative.generated_cols_),
            context_cols=list(context.generated_cols_),
            study_indicator_cols=list(study.generated_cols_),
        )
        self._leakage_checks = self._build_feature_leakage_checks()
        return self._prepared

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        prepared = self._prepared or self.prepare_data()
        self._task_total = self._estimate_total_tasks()
        self._task_done = 0
        self._task_started_at = time.perf_counter()
        self._log(
            f"Transfer benchmark start | paired matched={len(prepared.paired_matched)} | "
            f"hybrid={len(prepared.hybrid)} | paired={len(prepared.paired)} | total tasks={self._task_total}"
        )

        non_holdout_matched, final_holdout, split_report = self._make_final_holdout(prepared.paired_matched)
        holdout_recordings = set(final_holdout["recording_key"].astype(str))
        non_holdout_paired = prepared.paired[~prepared.paired["recording_key"].astype(str).isin(holdout_recordings)].copy()
        selection_frames: list[pd.DataFrame] = []
        holdout_frames: list[pd.DataFrame] = []
        finalists_by_protocol: dict[str, pd.DataFrame] = {}

        for protocol_mode in self.config.protocol_modes:
            paired_unlabeled = (
                non_holdout_paired.copy()
                if protocol_mode == "contextual" and self.config.allow_unlabeled_paired
                else non_holdout_matched.iloc[0:0].copy()
            )
            selection_protocol = self._run_model_selection(
                prepared=prepared,
                paired_matched=non_holdout_matched,
                paired_all_non_holdout=paired_unlabeled,
                protocol_mode=protocol_mode,
            )
            selection_frames.append(selection_protocol)
            selection_summary_protocol = self._summarize_selection(selection_protocol)
            finalists_protocol = self._select_finalists(selection_summary_protocol)
            finalists_by_protocol[protocol_mode] = finalists_protocol
            holdout_protocol = self._run_final_holdout(
                prepared=prepared,
                finalists=finalists_protocol,
                paired_train=non_holdout_matched,
                paired_unlabeled=paired_unlabeled,
                paired_holdout=final_holdout,
                protocol_mode=protocol_mode,
            )
            holdout_frames.append(holdout_protocol)

        selection_results = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
        selection_summary = self._summarize_selection(selection_results)
        holdout_results = pd.concat(holdout_frames, ignore_index=True) if holdout_frames else pd.DataFrame()
        holdout_summary, winner_summary, combined_winner_summary = self._summarize_holdout(
            holdout_results,
            selection_summary,
        )

        self._selection_results = selection_results
        self._selection_summary = selection_summary
        self._holdout_results = holdout_results
        self._holdout_summary = holdout_summary
        self._winner_summary = winner_summary
        self._split_report = split_report
        self._log("Transfer benchmark finished")

        return {
            "config": pd.DataFrame([self._config_dict()]),
            "feature_leakage_checks": self._leakage_checks.copy(),
            "split_report": pd.DataFrame([split_report]),
            "model_selection_results": selection_results.copy(),
            "model_selection_summary": selection_summary.copy(),
            "final_holdout_results": holdout_results.copy(),
            "final_holdout_summary": holdout_summary.copy(),
            "per_target_recommendation": self._per_target_recommendation(holdout_summary).copy(),
            "backend_comparison": self._backend_comparison(holdout_summary).copy(),
            "winner_summary": winner_summary.copy(),
            "combined_winner_summary": combined_winner_summary.copy(),
        }

    def _estimate_total_tasks(self) -> int:
        per_target_families = [
            family
            for family in self.config.model_families
            if family in {
                "dummy",
                "paired_only",
                "hybrid_only",
                "hybrid_calibrated",
                "hybrid_stack",
                "hybrid_plus_paired",
                "context_tabular",
                "context_transfer",
                "neural_transfer",
            }
        ]
        per_split = len(self.config.all_targets()) * len(per_target_families)
        if self.config.include_ceiling_diagnostics:
            per_split += len(self.config.all_targets())
        return per_split * (self.config.model_selection_splits + 1) * len(self.config.protocol_modes)

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message, flush=True)

    def _run_logged_task(self, label: str, fn: Any) -> Any:
        start = time.perf_counter()
        if self.config.verbose:
            print(f"[{self._task_done + 1}/{self._task_total}] {label} ...", flush=True)
        result = fn()
        self._task_done += 1
        if self.config.verbose:
            elapsed = (time.perf_counter() - self._task_started_at) if self._task_started_at is not None else 0.0
            avg = elapsed / max(self._task_done, 1)
            eta = avg * max(self._task_total - self._task_done, 0)
            duration = time.perf_counter() - start
            print(
                f"[{self._task_done}/{self._task_total}] {label} done in {duration:.1f}s | "
                f"elapsed={elapsed/60:.1f}m | eta={eta/60:.1f}m",
                flush=True,
            )
        return result

    # ------------------------------------------------------------------
    def _config_dict(self) -> dict[str, Any]:
        payload = asdict(self.config)
        payload["parquet_path"] = str(payload["parquet_path"])
        return payload

    def _build_feature_leakage_checks(self) -> pd.DataFrame:
        if self._prepared is None:
            return pd.DataFrame()
        rows = []
        for view_name, cols in self._prepared.feature_columns.items():
            raw_overlap = sorted(set(cols) & set(self._prepared.normalized_cols))
            rows.append(
                {
                    "feature_view": view_name,
                    "n_features": len(cols),
                    "raw_normalized_overlap": len(raw_overlap),
                    "passes_norm_swap_check": int(len(raw_overlap) == 0 or view_name != "norm_swap"),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _make_final_holdout(
        self, paired_matched: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        target_rows = min(self.config.paired_final_holdout_rows, len(paired_matched) - 1)
        test_size = min(0.95, max(0.05, target_rows / max(len(paired_matched), 1)))
        splitter = GroupShuffleSplit(
            n_splits=256,
            test_size=test_size,
            random_state=self.config.random_state,
        )

        best_train: pd.DataFrame | None = None
        best_test: pd.DataFrame | None = None
        best_gap: int | None = None

        for train_idx, test_idx in splitter.split(paired_matched, groups=paired_matched["recording_key"]):
            train = paired_matched.iloc[train_idx].copy()
            test = paired_matched.iloc[test_idx].copy()
            gap = abs(len(test) - target_rows)
            if best_gap is None or gap < best_gap:
                best_train = train
                best_test = test
                best_gap = gap
                if gap == 0:
                    break

        assert best_train is not None and best_test is not None
        overlap = set(best_train["recording_key"]) & set(best_test["recording_key"])
        assert not overlap, f"Final holdout recording overlap: {sorted(overlap)[:5]}"

        return best_train, best_test, {
            "paired_matched_rows": int(len(paired_matched)),
            "target_final_holdout_rows": int(target_rows),
            "actual_final_holdout_rows": int(len(best_test)),
            "final_train_rows": int(len(best_train)),
            "final_holdout_recordings": int(best_test["recording_key"].nunique()),
            "final_train_recordings": int(best_train["recording_key"].nunique()),
            "recording_overlap": 0,
        }

    # ------------------------------------------------------------------
    def _make_model_selection_splits(
        self, paired_matched: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
        splits: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []
        for offset in range(self.config.model_selection_splits):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self.config.model_selection_test_size,
                random_state=self.config.random_state + 100 + offset,
            )
            train_idx, test_idx = next(splitter.split(paired_matched, groups=paired_matched["recording_key"]))
            train = paired_matched.iloc[train_idx].copy()
            test = paired_matched.iloc[test_idx].copy()
            overlap = set(train["recording_key"]) & set(test["recording_key"])
            assert not overlap, f"Model-selection recording overlap: {sorted(overlap)[:5]}"
            splits.append((train, test, offset + 1))
        return splits

    # ------------------------------------------------------------------
    def _make_lgbm(self) -> Pipeline:
        if not _LGBM_AVAILABLE:
            raise ImportError("lightgbm is required for QCTransferBenchmark")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=self.config.lightgbm_estimators,
                        learning_rate=self.config.lightgbm_learning_rate,
                        num_leaves=self.config.lightgbm_num_leaves,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.config.random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    def _make_catboost(self) -> Pipeline:
        if not _CATBOOST_AVAILABLE:
            raise ImportError("catboost is required for CatBoost paired-only baseline")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostRegressor(
                        depth=6,
                        learning_rate=0.05,
                        loss_function="RMSE",
                        verbose=False,
                        random_seed=self.config.random_state,
                    ),
                ),
            ]
        )

    def _make_xgboost(self) -> Pipeline:
        if not _XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for XGBoost-backed transfer benchmarks")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=self.config.xgboost_estimators,
                        learning_rate=self.config.xgboost_learning_rate,
                        max_depth=self.config.xgboost_max_depth,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=self.config.random_state,
                        n_jobs=-1,
                        verbosity=0,
                        tree_method="hist",
                    ),
                ),
            ]
        )

    def _make_backend_model(self, backend: str) -> Pipeline:
        if backend == "lightgbm":
            return self._make_lgbm()
        if backend == "catboost":
            return self._make_catboost()
        if backend == "xgboost":
            return self._make_xgboost()
        raise KeyError(f"Unknown boosting backend: {backend}")

    def _available_backends(self) -> list[str]:
        available: list[str] = []
        for backend in self.config.effective_backends():
            if backend == "lightgbm" and _LGBM_AVAILABLE:
                available.append(backend)
            elif backend == "catboost" and _CATBOOST_AVAILABLE:
                available.append(backend)
            elif backend == "xgboost" and _XGBOOST_AVAILABLE:
                available.append(backend)
        return available

    def _transform_target_values(self, values: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if self.config.target_transform == "logit":
            clipped = np.clip(arr, 1e-4, 1.0 - 1e-4)
            return logit(clipped)
        return arr

    def _inverse_target_values(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if self.config.target_transform == "logit":
            return expit(arr)
        return arr

    def _fit_backend_model(
        self,
        model: Pipeline,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | None = None,
        apply_target_transform: bool = True,
    ) -> Pipeline:
        y_fit = self._transform_target_values(y) if apply_target_transform else np.asarray(y, dtype=float)
        if sample_weight is not None:
            model.fit(X, y_fit, model__sample_weight=sample_weight)
        else:
            model.fit(X, y_fit)
        return model

    def _predict_backend_model(self, model: Pipeline, X: pd.DataFrame, apply_target_transform: bool = True) -> np.ndarray:
        pred = model.predict(X)
        if apply_target_transform:
            return _clip_pred(self._inverse_target_values(pred))
        return np.asarray(pred, dtype=float)

    def _feature_views_for_family(self, family: str, protocol_mode: str) -> tuple[str, ...]:
        if protocol_mode == "inductive":
            if family in {"paired_only", "hybrid_only"}:
                return ("shape_only", "unit_raw")
            if family in {"hybrid_calibrated", "hybrid_stack", "hybrid_plus_paired"}:
                return ("unit_raw",)
            return ()
        if family in {"paired_only", "hybrid_only"}:
            return ("shape_only", "contextual_full")
        if family in {"hybrid_calibrated", "hybrid_stack", "hybrid_plus_paired", "context_tabular", "context_transfer"}:
            return ("contextual_full",)
        return ()

    # ------------------------------------------------------------------
    def _target_frames(
        self,
        eval_target: str,
        source_df: pd.DataFrame,
        paired_train: pd.DataFrame,
        paired_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, str]:
        source_target = self.config.source_target_for(eval_target)
        y_source = pd.to_numeric(source_df[source_target], errors="coerce")
        y_train = pd.to_numeric(paired_train[eval_target], errors="coerce")
        y_test = pd.to_numeric(paired_test[eval_target], errors="coerce")
        return (
            source_df.loc[y_source.notna()].copy(),
            y_source.loc[y_source.notna()],
            paired_train.loc[y_train.notna()].copy(),
            y_train.loc[y_train.notna()],
            paired_test.loc[y_test.notna()].copy(),
            y_test.loc[y_test.notna()],
            source_target,
        )

    def _result_row(
        self,
        *,
        phase: str,
        split_id: str | int,
        protocol_mode: str,
        eval_target: str,
        source_target: str,
        feature_view: str,
        model_family: str,
        model_name: str,
        y_true: np.ndarray,
        pred: np.ndarray,
        source_label_rows: int,
        target_label_rows: int,
        target_eval_rows: int,
        target_unlabeled_rows: int,
        used_unlabeled_paired: bool,
        is_ceiling_model: bool = False,
    ) -> dict[str, Any]:
        scores = _score_predictions(y_true, pred)
        candidate_id = f"{protocol_mode}|{model_family}|{model_name}|{feature_view}"
        return {
            "phase": phase,
            "split_id": str(split_id),
            "protocol_mode": protocol_mode,
            "candidate_id": candidate_id,
            "target": eval_target,
            "source_target": source_target,
            "feature_view": feature_view,
            "model_family": model_family,
            "model_name": model_name,
            "is_ceiling_model": bool(is_ceiling_model),
            "source_label_rows": int(source_label_rows),
            "target_label_rows": int(target_label_rows),
            "target_eval_rows": int(target_eval_rows),
            "target_unlabeled_rows": int(target_unlabeled_rows),
            "used_unlabeled_paired": bool(used_unlabeled_paired),
            **scores,
        }

    def _run_recording_ceiling_family(
        self,
        *,
        phase: str,
        split_id: str | int,
        protocol_mode: str,
        eval_target: str,
        source_target: str,
        paired_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> list[dict[str, Any]]:
        pred = (
            paired_test.assign(_target=y_test.values)
            .groupby("recording_key")["_target"]
            .transform("mean")
            .values
        )
        return [
            self._result_row(
                phase=phase,
                split_id=split_id,
                protocol_mode=protocol_mode,
                eval_target=eval_target,
                source_target=source_target,
                feature_view="ceiling",
                model_family="ceiling",
                model_name="recording_oracle_mean",
                y_true=y_test.values,
                pred=pred,
                source_label_rows=0,
                target_label_rows=0,
                target_eval_rows=len(y_test),
                target_unlabeled_rows=0,
                used_unlabeled_paired=False,
                is_ceiling_model=True,
            )
        ]

    # ------------------------------------------------------------------
    def _run_model_selection(
        self,
        *,
        prepared: _PreparedTransferData,
        paired_matched: pd.DataFrame,
        paired_all_non_holdout: pd.DataFrame,
        protocol_mode: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for paired_train, paired_test, split_id in self._make_model_selection_splits(paired_matched):
            rows.extend(
                self._run_candidate_bundle(
                    phase="model_selection",
                    split_id=split_id,
                    protocol_mode=protocol_mode,
                    prepared=prepared,
                    paired_train=paired_train,
                    paired_test=paired_test,
                    paired_unlabeled=paired_all_non_holdout,
                )
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _run_final_holdout(
        self,
        *,
        prepared: _PreparedTransferData,
        finalists: pd.DataFrame,
        paired_train: pd.DataFrame,
        paired_unlabeled: pd.DataFrame,
        paired_holdout: pd.DataFrame,
        protocol_mode: str,
    ) -> pd.DataFrame:
        if finalists.empty:
            return pd.DataFrame()
        all_rows = self._run_candidate_bundle(
            phase="final_holdout",
            split_id="final_holdout",
            protocol_mode=protocol_mode,
            prepared=prepared,
            paired_train=paired_train,
            paired_test=paired_holdout,
            paired_unlabeled=paired_unlabeled,
        )
        if "target" in finalists.columns:
            finalist_keys = {
                (str(candidate_id), str(target))
                for candidate_id, target in finalists[["candidate_id", "target"]].itertuples(index=False, name=None)
            }
            rows = [row for row in all_rows if (row["candidate_id"], row["target"]) in finalist_keys]
        else:
            candidate_ids = set(finalists["candidate_id"].astype(str))
            rows = [row for row in all_rows if row["candidate_id"] in candidate_ids]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _run_candidate_bundle(
        self,
        *,
        phase: str,
        split_id: str | int,
        protocol_mode: str,
        prepared: _PreparedTransferData,
        paired_train: pd.DataFrame,
        paired_test: pd.DataFrame,
        paired_unlabeled: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        unlabeled_rows = len(paired_unlabeled) if self.config.allow_unlabeled_paired else len(paired_train)
        used_unlabeled = protocol_mode == "contextual" and bool(self.config.allow_unlabeled_paired)

        for eval_target in self.config.all_targets():
            (
                source_df,
                y_source,
                paired_train_t,
                y_train,
                paired_test_t,
                y_test,
                source_target,
            ) = self._target_frames(eval_target, prepared.hybrid, paired_train, paired_test)
            if paired_train_t.empty or paired_test_t.empty:
                continue

            if "dummy" in self.config.model_families:
                rows.extend(self._run_logged_task(
                    f"{protocol_mode} {phase} split={split_id} target={eval_target} family=dummy",
                    lambda eval_target=eval_target, source_target=source_target, y_train=y_train, y_test=y_test: self._run_dummy_family(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        eval_target=eval_target,
                        source_target=source_target,
                        y_train=y_train,
                        y_test=y_test,
                    )
                ))
            if self.config.include_ceiling_diagnostics:
                rows.extend(self._run_logged_task(
                    f"{protocol_mode} {phase} split={split_id} target={eval_target} family=ceiling",
                    lambda eval_target=eval_target, source_target=source_target, paired_test_t=paired_test_t, y_test=y_test: self._run_recording_ceiling_family(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        eval_target=eval_target,
                        source_target=source_target,
                        paired_test=paired_test_t,
                        y_test=y_test,
                    )
                ))
            if "paired_only" in self.config.model_families:
                rows.extend(self._run_logged_task(
                    f"{protocol_mode} {phase} split={split_id} target={eval_target} family=paired_only",
                    lambda eval_target=eval_target, source_target=source_target, paired_train_t=paired_train_t, y_train=y_train, paired_test_t=paired_test_t, y_test=y_test: self._run_paired_only_family(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        prepared=prepared,
                        eval_target=eval_target,
                        source_target=source_target,
                        paired_train=paired_train_t,
                        y_train=y_train,
                        paired_test=paired_test_t,
                        y_test=y_test,
                        target_unlabeled_rows=unlabeled_rows,
                        used_unlabeled_paired=used_unlabeled,
                    )
                ))
            if "hybrid_only" in self.config.model_families:
                rows.extend(self._run_logged_task(
                    f"{protocol_mode} {phase} split={split_id} target={eval_target} family=hybrid_only",
                    lambda eval_target=eval_target, source_target=source_target, source_df=source_df, y_source=y_source, paired_test_t=paired_test_t, y_test=y_test: self._run_hybrid_only_family(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        prepared=prepared,
                        eval_target=eval_target,
                        source_target=source_target,
                        source_df=source_df,
                        y_source=y_source,
                        paired_test=paired_test_t,
                        y_test=y_test,
                        target_label_rows=len(y_train),
                        target_unlabeled_rows=unlabeled_rows,
                        used_unlabeled_paired=used_unlabeled,
                    )
                ))
            if "hybrid_calibrated" in self.config.model_families:
                rows.extend(self._run_logged_task(
                    f"{protocol_mode} {phase} split={split_id} target={eval_target} family=hybrid_calibrated",
                    lambda eval_target=eval_target, source_target=source_target, source_df=source_df, y_source=y_source, paired_train_t=paired_train_t, y_train=y_train, paired_test_t=paired_test_t, y_test=y_test: self._run_hybrid_calibration_family(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        prepared=prepared,
                        eval_target=eval_target,
                        source_target=source_target,
                        source_df=source_df,
                        y_source=y_source,
                        paired_train=paired_train_t,
                        y_train=y_train,
                        paired_test=paired_test_t,
                        y_test=y_test,
                        target_unlabeled_rows=unlabeled_rows,
                        used_unlabeled_paired=used_unlabeled,
                    )
                ))
            if "hybrid_stack" in self.config.model_families:
                rows.extend(self._run_logged_task(
                    f"{protocol_mode} {phase} split={split_id} target={eval_target} family=hybrid_stack",
                    lambda eval_target=eval_target, source_target=source_target, source_df=source_df, y_source=y_source, paired_train_t=paired_train_t, y_train=y_train, paired_test_t=paired_test_t, y_test=y_test: self._run_hybrid_stack_family(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        prepared=prepared,
                        eval_target=eval_target,
                        source_target=source_target,
                        source_df=source_df,
                        y_source=y_source,
                        paired_train=paired_train_t,
                        y_train=y_train,
                        paired_test=paired_test_t,
                        y_test=y_test,
                        target_unlabeled_rows=unlabeled_rows,
                        used_unlabeled_paired=used_unlabeled,
                    )
                ))
            if "hybrid_plus_paired" in self.config.model_families:
                rows.extend(self._run_logged_task(
                    f"{protocol_mode} {phase} split={split_id} target={eval_target} family=hybrid_plus_paired",
                    lambda eval_target=eval_target, source_target=source_target, source_df=source_df, y_source=y_source, paired_train_t=paired_train_t, y_train=y_train, paired_test_t=paired_test_t, y_test=y_test: self._run_weighted_hybrid_family(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        prepared=prepared,
                        eval_target=eval_target,
                        source_target=source_target,
                        source_df=source_df,
                        y_source=y_source,
                        paired_train=paired_train_t,
                        y_train=y_train,
                        paired_test=paired_test_t,
                        y_test=y_test,
                        target_unlabeled_rows=unlabeled_rows,
                        used_unlabeled_paired=used_unlabeled,
                    )
                ))
            if "context_tabular" in self.config.model_families:
                rows.extend(self._run_logged_task(
                    f"{protocol_mode} {phase} split={split_id} target={eval_target} family=context_tabular",
                    lambda eval_target=eval_target, source_target=source_target, paired_train_t=paired_train_t, y_train=y_train, paired_test_t=paired_test_t, y_test=y_test: self._run_context_tabular_family(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        prepared=prepared,
                        eval_target=eval_target,
                        source_target=source_target,
                        paired_train=paired_train_t,
                        y_train=y_train,
                        paired_test=paired_test_t,
                        y_test=y_test,
                        target_unlabeled_rows=unlabeled_rows,
                        used_unlabeled_paired=used_unlabeled,
                    )
                ))
            if ("context_transfer" in self.config.model_families or "neural_transfer" in self.config.model_families):
                rows.extend(self._run_logged_task(
                    f"{protocol_mode} {phase} split={split_id} target={eval_target} family=context_transfer",
                    lambda eval_target=eval_target, source_target=source_target, source_df=source_df, y_source=y_source, paired_train_t=paired_train_t, y_train=y_train, paired_test_t=paired_test_t, y_test=y_test: self._run_context_transfer_family(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        prepared=prepared,
                        eval_target=eval_target,
                        source_target=source_target,
                        source_df=source_df,
                        y_source=y_source,
                        paired_train=paired_train_t,
                        y_train=y_train,
                        paired_test=paired_test_t,
                        y_test=y_test,
                        paired_unlabeled=paired_unlabeled if used_unlabeled else paired_train_t.iloc[0:0].copy(),
                    )
                ))
        return rows

    # ------------------------------------------------------------------
    def _run_dummy_family(
        self,
        *,
        phase: str,
        split_id: str | int,
        protocol_mode: str,
        eval_target: str,
        source_target: str,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> list[dict[str, Any]]:
        if "dummy" not in self.config.model_families:
            return []
        model = DummyRegressor(strategy="mean")
        model.fit(np.zeros((len(y_train), 1)), y_train.values)
        pred = model.predict(np.zeros((len(y_test), 1)))
        return [
            self._result_row(
                phase=phase,
                split_id=split_id,
                protocol_mode=protocol_mode,
                eval_target=eval_target,
                source_target=source_target,
                feature_view="none",
                model_family="dummy",
                model_name="paired_mean",
                y_true=y_test.values,
                pred=pred,
                source_label_rows=0,
                target_label_rows=len(y_train),
                target_eval_rows=len(y_test),
                target_unlabeled_rows=0,
                used_unlabeled_paired=False,
            )
        ]

    # ------------------------------------------------------------------
    def _run_paired_only_family(
        self,
        *,
        phase: str,
        split_id: str | int,
        protocol_mode: str,
        prepared: _PreparedTransferData,
        eval_target: str,
        source_target: str,
        paired_train: pd.DataFrame,
        y_train: pd.Series,
        paired_test: pd.DataFrame,
        y_test: pd.Series,
        target_unlabeled_rows: int,
        used_unlabeled_paired: bool,
    ) -> list[dict[str, Any]]:
        if "paired_only" not in self.config.model_families:
            return []
        rows: list[dict[str, Any]] = []
        for feature_view in self._feature_views_for_family("paired_only", protocol_mode):
            cols = prepared.feature_columns.get(feature_view, [])
            if not cols:
                continue
            X_train = _ensure_numeric_frame(paired_train, cols)
            X_test = _ensure_numeric_frame(paired_test, cols)
            for backend in self._available_backends():
                model = self._make_backend_model(backend)
                self._fit_backend_model(model, X_train, y_train.values)
                pred = self._predict_backend_model(model, X_test)
                rows.append(
                    self._result_row(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        eval_target=eval_target,
                        source_target=source_target,
                        feature_view=feature_view,
                        model_family="paired_only",
                        model_name=backend,
                        y_true=y_test.values,
                        pred=pred,
                        source_label_rows=0,
                        target_label_rows=len(y_train),
                        target_eval_rows=len(y_test),
                        target_unlabeled_rows=target_unlabeled_rows,
                        used_unlabeled_paired=used_unlabeled_paired,
                    )
                )
        return rows

    # ------------------------------------------------------------------
    def _run_hybrid_only_family(
        self,
        *,
        phase: str,
        split_id: str | int,
        protocol_mode: str,
        prepared: _PreparedTransferData,
        eval_target: str,
        source_target: str,
        source_df: pd.DataFrame,
        y_source: pd.Series,
        paired_test: pd.DataFrame,
        y_test: pd.Series,
        target_label_rows: int,
        target_unlabeled_rows: int,
        used_unlabeled_paired: bool,
    ) -> list[dict[str, Any]]:
        if "hybrid_only" not in self.config.model_families:
            return []
        rows: list[dict[str, Any]] = []
        for feature_view in self._feature_views_for_family("hybrid_only", protocol_mode):
            cols = prepared.feature_columns.get(feature_view, [])
            if not cols:
                continue
            X_source = _ensure_numeric_frame(source_df, cols)
            X_test = _ensure_numeric_frame(paired_test, cols)
            for backend in self._available_backends():
                model = self._make_backend_model(backend)
                self._fit_backend_model(model, X_source, y_source.values)
                pred = self._predict_backend_model(model, X_test)
                rows.append(
                    self._result_row(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        eval_target=eval_target,
                        source_target=source_target,
                        feature_view=feature_view,
                        model_family="hybrid_only",
                        model_name=backend,
                        y_true=y_test.values,
                        pred=pred,
                        source_label_rows=len(y_source),
                        target_label_rows=target_label_rows,
                        target_eval_rows=len(y_test),
                        target_unlabeled_rows=target_unlabeled_rows,
                        used_unlabeled_paired=used_unlabeled_paired,
                    )
                )
        return rows

    # ------------------------------------------------------------------
    def _run_hybrid_calibration_family(
        self,
        *,
        phase: str,
        split_id: str | int,
        protocol_mode: str,
        prepared: _PreparedTransferData,
        eval_target: str,
        source_target: str,
        source_df: pd.DataFrame,
        y_source: pd.Series,
        paired_train: pd.DataFrame,
        y_train: pd.Series,
        paired_test: pd.DataFrame,
        y_test: pd.Series,
        target_unlabeled_rows: int,
        used_unlabeled_paired: bool,
    ) -> list[dict[str, Any]]:
        if "hybrid_calibrated" not in self.config.model_families:
            return []
        rows: list[dict[str, Any]] = []
        for feature_view in self._feature_views_for_family("hybrid_calibrated", protocol_mode):
            cols = prepared.feature_columns.get(feature_view, [])
            if not cols:
                continue
            X_source = _ensure_numeric_frame(source_df, cols)
            X_train = _ensure_numeric_frame(paired_train, cols)
            X_test = _ensure_numeric_frame(paired_test, cols)

            for backend in self._available_backends():
                base_model = self._make_backend_model(backend)
                self._fit_backend_model(base_model, X_source, y_source.values)
                pred_train = self._predict_backend_model(base_model, X_train)
                pred_test = self._predict_backend_model(base_model, X_test)

                linear = LinearRegression().fit(pred_train.reshape(-1, 1), y_train.values)
                linear_pred = linear.predict(pred_test.reshape(-1, 1))
                rows.append(
                    self._result_row(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        eval_target=eval_target,
                        source_target=source_target,
                        feature_view=feature_view,
                        model_family="hybrid_calibrated",
                        model_name=f"{backend}+linear",
                        y_true=y_test.values,
                        pred=linear_pred,
                        source_label_rows=len(y_source),
                        target_label_rows=len(y_train),
                        target_eval_rows=len(y_test),
                        target_unlabeled_rows=target_unlabeled_rows,
                        used_unlabeled_paired=used_unlabeled_paired,
                    )
                )

                iso = IsotonicRegression(out_of_bounds="clip").fit(pred_train, y_train.values)
                iso_pred = iso.predict(pred_test)
                rows.append(
                    self._result_row(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        eval_target=eval_target,
                        source_target=source_target,
                        feature_view=feature_view,
                        model_family="hybrid_calibrated",
                        model_name=f"{backend}+isotonic",
                        y_true=y_test.values,
                        pred=iso_pred,
                        source_label_rows=len(y_source),
                        target_label_rows=len(y_train),
                        target_eval_rows=len(y_test),
                        target_unlabeled_rows=target_unlabeled_rows,
                        used_unlabeled_paired=used_unlabeled_paired,
                    )
                )
        return rows

    # ------------------------------------------------------------------
    def _run_hybrid_stack_family(
        self,
        *,
        phase: str,
        split_id: str | int,
        protocol_mode: str,
        prepared: _PreparedTransferData,
        eval_target: str,
        source_target: str,
        source_df: pd.DataFrame,
        y_source: pd.Series,
        paired_train: pd.DataFrame,
        y_train: pd.Series,
        paired_test: pd.DataFrame,
        y_test: pd.Series,
        target_unlabeled_rows: int,
        used_unlabeled_paired: bool,
    ) -> list[dict[str, Any]]:
        if "hybrid_stack" not in self.config.model_families:
            return []
        rows: list[dict[str, Any]] = []
        for feature_view in self._feature_views_for_family("hybrid_stack", protocol_mode):
            cols = prepared.feature_columns.get(feature_view, [])
            if not cols:
                continue

            X_source = _ensure_numeric_frame(source_df, cols)
            X_train = _ensure_numeric_frame(paired_train, cols)
            X_test = _ensure_numeric_frame(paired_test, cols)

            for backend in self._available_backends():
                base_model = self._make_backend_model(backend)
                self._fit_backend_model(base_model, X_source, y_source.values)

                pred_train = self._predict_backend_model(base_model, X_train)
                pred_test = self._predict_backend_model(base_model, X_test)

                X_stack_train = X_train.copy()
                X_stack_test = X_test.copy()
                X_stack_train["hybrid_pred"] = pred_train
                X_stack_test["hybrid_pred"] = pred_test

                residual = y_train.values - pred_train
                residual_model = self._make_backend_model(backend)
                self._fit_backend_model(residual_model, X_stack_train, residual, apply_target_transform=False)
                final_pred = pred_test + self._predict_backend_model(
                    residual_model,
                    X_stack_test,
                    apply_target_transform=False,
                )

                rows.append(
                    self._result_row(
                        phase=phase,
                        split_id=split_id,
                        protocol_mode=protocol_mode,
                        eval_target=eval_target,
                        source_target=source_target,
                        feature_view=feature_view,
                        model_family="hybrid_stack",
                        model_name=f"{backend}_residual",
                        y_true=y_test.values,
                        pred=final_pred,
                        source_label_rows=len(y_source),
                        target_label_rows=len(y_train),
                        target_eval_rows=len(y_test),
                        target_unlabeled_rows=target_unlabeled_rows,
                        used_unlabeled_paired=used_unlabeled_paired,
                    )
                )
        return rows

    # ------------------------------------------------------------------
    def _run_weighted_hybrid_family(
        self,
        *,
        phase: str,
        split_id: str | int,
        protocol_mode: str,
        prepared: _PreparedTransferData,
        eval_target: str,
        source_target: str,
        source_df: pd.DataFrame,
        y_source: pd.Series,
        paired_train: pd.DataFrame,
        y_train: pd.Series,
        paired_test: pd.DataFrame,
        y_test: pd.Series,
        target_unlabeled_rows: int,
        used_unlabeled_paired: bool,
    ) -> list[dict[str, Any]]:
        if "hybrid_plus_paired" not in self.config.model_families:
            return []
        rows: list[dict[str, Any]] = []
        for feature_view in self._feature_views_for_family("hybrid_plus_paired", protocol_mode):
            cols = prepared.feature_columns.get(feature_view, [])
            if not cols:
                continue
            X_source = _ensure_numeric_frame(source_df, cols)
            X_train = _ensure_numeric_frame(paired_train, cols)
            X_test = _ensure_numeric_frame(paired_test, cols)

            X_mix = pd.concat([X_source, X_train], axis=0, ignore_index=True)
            y_mix = np.concatenate([y_source.values, y_train.values])
            for backend in self._available_backends():
                for weight in self.config.paired_weight_grid:
                    sample_weight = np.concatenate(
                        [np.ones(len(X_source), dtype=float), np.full(len(X_train), float(weight), dtype=float)]
                    )
                    model = self._make_backend_model(backend)
                    self._fit_backend_model(model, X_mix, y_mix, sample_weight=sample_weight)
                    pred = self._predict_backend_model(model, X_test)
                    rows.append(
                        self._result_row(
                            phase=phase,
                            split_id=split_id,
                            protocol_mode=protocol_mode,
                            eval_target=eval_target,
                            source_target=source_target,
                            feature_view=feature_view,
                            model_family="hybrid_plus_paired",
                            model_name=f"{backend}_w{weight:g}",
                            y_true=y_test.values,
                            pred=pred,
                            source_label_rows=len(y_source),
                            target_label_rows=len(y_train),
                            target_eval_rows=len(y_test),
                            target_unlabeled_rows=target_unlabeled_rows,
                            used_unlabeled_paired=used_unlabeled_paired,
                        )
                    )
        return rows

    # ------------------------------------------------------------------
    def _run_context_tabular_family(
        self,
        *,
        phase: str,
        split_id: str | int,
        protocol_mode: str,
        prepared: _PreparedTransferData,
        eval_target: str,
        source_target: str,
        paired_train: pd.DataFrame,
        y_train: pd.Series,
        paired_test: pd.DataFrame,
        y_test: pd.Series,
        target_unlabeled_rows: int,
        used_unlabeled_paired: bool,
    ) -> list[dict[str, Any]]:
        if protocol_mode != "contextual":
            return []
        feature_view = "contextual_full"
        cols = prepared.feature_columns.get(feature_view, [])
        if not cols:
            return []
        X_train = _ensure_numeric_frame(paired_train, cols)
        X_test = _ensure_numeric_frame(paired_test, cols)
        rows: list[dict[str, Any]] = []
        for backend in self._available_backends():
            model = self._make_backend_model(backend)
            self._fit_backend_model(model, X_train, y_train.values)
            pred = self._predict_backend_model(model, X_test)
            rows.append(
                self._result_row(
                    phase=phase,
                    split_id=split_id,
                    protocol_mode=protocol_mode,
                    eval_target=eval_target,
                    source_target=source_target,
                    feature_view=feature_view,
                    model_family="context_tabular",
                    model_name=backend,
                    y_true=y_test.values,
                    pred=pred,
                    source_label_rows=0,
                    target_label_rows=len(y_train),
                    target_eval_rows=len(y_test),
                    target_unlabeled_rows=target_unlabeled_rows,
                    used_unlabeled_paired=used_unlabeled_paired,
                )
            )
        return rows

    def _run_context_transfer_family(
        self,
        *,
        phase: str,
        split_id: str | int,
        protocol_mode: str,
        prepared: _PreparedTransferData,
        eval_target: str,
        source_target: str,
        source_df: pd.DataFrame,
        y_source: pd.Series,
        paired_train: pd.DataFrame,
        y_train: pd.Series,
        paired_test: pd.DataFrame,
        y_test: pd.Series,
        paired_unlabeled: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        if protocol_mode != "contextual" or not _CONTEXT_TRANSFER_AVAILABLE:
            return []
        feature_view = "contextual_full"
        cols = prepared.feature_columns.get(feature_view, [])
        if not cols:
            return []

        source_fit = source_df.copy()
        source_fit[eval_target] = y_source.values
        target_fit = paired_train.copy()
        target_fit[eval_target] = y_train.values

        model = SingleTargetDomainAdaptiveRegressor(
            feature_columns=cols,
            target_name=eval_target,
            source_target_name=source_target,
            pretrain_epochs=self.config.neural_pretrain_epochs,
            finetune_epochs=self.config.neural_finetune_epochs,
            batch_size=self.config.neural_batch_size,
            lr=self.config.neural_lr,
            domain_loss_weight=self.config.neural_domain_loss_weight,
            target_loss_weight=self.config.neural_target_loss_weight,
            target_transform=self.config.target_transform,
            random_state=self.config.random_state,
            verbose=self.config.neural_verbose,
            progress_prefix=f"{protocol_mode} {phase} split={split_id} context_transfer target={eval_target}",
        )
        model.fit(
            source_df=source_fit,
            target_labeled_df=target_fit,
            target_unlabeled_df=paired_unlabeled.copy(),
        )
        pred = model.predict(paired_test)
        return [
            self._result_row(
                phase=phase,
                split_id=split_id,
                protocol_mode=protocol_mode,
                eval_target=eval_target,
                source_target=source_target,
                feature_view=feature_view,
                model_family="context_transfer",
                model_name=f"single_target_coral_{eval_target}",
                y_true=y_test.values,
                pred=pred,
                source_label_rows=len(y_source),
                target_label_rows=len(y_train),
                target_eval_rows=len(y_test),
                target_unlabeled_rows=len(paired_unlabeled),
                used_unlabeled_paired=bool(len(paired_unlabeled)),
            )
        ]

    # ------------------------------------------------------------------
    def _summarize_selection(self, results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return pd.DataFrame()

        summary = (
            results.groupby(
                ["protocol_mode", "candidate_id", "model_family", "model_name", "feature_view", "target", "is_ceiling_model"],
                as_index=False,
            )[["mae", "rmse", "r2", "bias", "calibration_slope", "calibration_intercept"]]
            .mean()
        )

        scored = summary[~summary["is_ceiling_model"]].copy()
        primary = scored[scored["target"].isin(self.config.primary_targets)].copy()
        primary_avg = (
            primary.groupby(["protocol_mode", "candidate_id", "model_family", "model_name", "feature_view"], as_index=False)["mae"]
            .mean()
            .rename(columns={"mae": "primary_avg_mae"})
        )
        summary = summary.merge(
            primary_avg,
            on=["protocol_mode", "candidate_id", "model_family", "model_name", "feature_view"],
            how="left",
        )

        dummy_by_target = (
            primary[primary["model_family"] == "dummy"][["protocol_mode", "target", "mae"]]
            .rename(columns={"mae": "dummy_mae"})
        )
        hybrid_best_by_target = (
            primary[primary["model_family"] == "hybrid_only"]
            .groupby(["protocol_mode", "target"], as_index=False)["mae"]
            .min()
            .rename(columns={"mae": "best_hybrid_only_mae"})
        )
        summary = summary.merge(dummy_by_target, on=["protocol_mode", "target"], how="left")
        summary = summary.merge(hybrid_best_by_target, on=["protocol_mode", "target"], how="left")
        summary["beats_dummy"] = summary["mae"] < summary["dummy_mae"]
        summary["beats_best_hybrid_only"] = summary["mae"] < summary["best_hybrid_only_mae"]

        gate = (
            summary[(summary["target"].isin(self.config.primary_targets)) & (~summary["is_ceiling_model"])]
            .groupby(["protocol_mode", "candidate_id"], as_index=False)[["beats_dummy", "beats_best_hybrid_only"]]
            .all()
            .rename(
                columns={
                    "beats_dummy": "beats_dummy_all_primary_targets",
                    "beats_best_hybrid_only": "beats_hybrid_only_all_primary_targets",
                }
            )
        )
        summary = summary.merge(gate, on=["protocol_mode", "candidate_id"], how="left")
        summary["advances_target"] = summary["beats_dummy"].fillna(False) & summary["beats_best_hybrid_only"].fillna(False)
        beats_dummy_all = summary["beats_dummy_all_primary_targets"].astype("boolean").fillna(False).astype(bool)
        beats_hybrid_all = summary["beats_hybrid_only_all_primary_targets"].astype("boolean").fillna(False).astype(bool)
        summary["advances_all_primary_targets"] = beats_dummy_all & beats_hybrid_all
        summary["advances"] = (
            summary["advances_target"]
            if self.config.deploy_best_per_target
            else summary["advances_all_primary_targets"]
        )
        return summary.sort_values(["protocol_mode", "primary_avg_mae", "candidate_id", "target"])

    # ------------------------------------------------------------------
    def _select_finalists(self, selection_summary: pd.DataFrame) -> pd.DataFrame:
        if selection_summary.empty:
            return pd.DataFrame()

        finalists: list[pd.DataFrame] = []
        family_order = (
            "dummy",
            "hybrid_only",
            "paired_only",
            "hybrid_calibrated",
            "hybrid_stack",
            "hybrid_plus_paired",
            "context_tabular",
            "context_transfer",
        )
        if self.config.deploy_best_per_target:
            scored = selection_summary[
                selection_summary["target"].isin(self.config.all_targets()) & ~selection_summary["is_ceiling_model"]
            ].copy()
            for family in family_order:
                family_rows = scored[scored["model_family"] == family].copy()
                if family_rows.empty:
                    continue
                family_rows["_advances_sort"] = family_rows["advances"].astype("boolean").fillna(False).astype(bool)
                finalists.append(
                    family_rows.sort_values(
                        ["protocol_mode", "target", "_advances_sort", "mae", "candidate_id"],
                        ascending=[True, True, False, True, True],
                    )
                    .groupby(["protocol_mode", "target"], as_index=False)
                    .head(1)
                    .drop(columns=["_advances_sort"], errors="ignore")
                )
        else:
            primary = selection_summary[
                selection_summary["target"].isin(self.config.primary_targets) & ~selection_summary["is_ceiling_model"]
            ].copy()
            candidate_rows = (
                primary.groupby(
                    ["protocol_mode", "candidate_id", "model_family", "model_name", "feature_view", "primary_avg_mae"],
                    as_index=False,
                )[["advances", "beats_dummy_all_primary_targets", "beats_hybrid_only_all_primary_targets"]]
                .first()
            )
            for family in family_order:
                family_rows = candidate_rows[candidate_rows["model_family"] == family].sort_values("primary_avg_mae")
                if family_rows.empty:
                    continue
                finalists.append(family_rows.head(1))
        if not finalists:
            return pd.DataFrame()
        return pd.concat(finalists, ignore_index=True)

    # ------------------------------------------------------------------
    def _summarize_holdout(
        self, holdout_results: pd.DataFrame, selection_summary: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if holdout_results.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        holdout_summary = (
            holdout_results.groupby(
                ["protocol_mode", "candidate_id", "model_family", "model_name", "feature_view", "target", "is_ceiling_model"],
                as_index=False,
            )[["mae", "rmse", "r2", "bias", "calibration_slope", "calibration_intercept"]]
            .mean()
        )

        primary_avg = (
            holdout_summary[
                holdout_summary["target"].isin(self.config.primary_targets) & ~holdout_summary["is_ceiling_model"]
            ]
            .groupby(["protocol_mode", "candidate_id", "model_family", "model_name", "feature_view"], as_index=False)["mae"]
            .mean()
            .rename(columns={"mae": "primary_avg_mae"})
        )
        holdout_summary = holdout_summary.merge(
            primary_avg,
            on=["protocol_mode", "candidate_id", "model_family", "model_name", "feature_view"],
            how="left",
        )

        selection_gate = selection_summary[
            [
                "protocol_mode",
                "candidate_id",
                "target",
                "advances",
                "advances_target",
                "advances_all_primary_targets",
                "beats_dummy_all_primary_targets",
                "beats_hybrid_only_all_primary_targets",
            ]
        ].drop_duplicates()
        holdout_summary = holdout_summary.merge(
            selection_gate,
            on=["protocol_mode", "candidate_id", "target"],
            how="left",
        )

        best_paired_only_by_protocol = (
            primary_avg[primary_avg["model_family"] == "paired_only"]
            .groupby("protocol_mode", as_index=False)["primary_avg_mae"]
            .min()
            .rename(columns={"primary_avg_mae": "best_paired_only_primary_avg_mae"})
        )
        holdout_summary = holdout_summary.merge(best_paired_only_by_protocol, on="protocol_mode", how="left")
        holdout_summary["beats_best_paired_only"] = (
            holdout_summary["primary_avg_mae"] < holdout_summary["best_paired_only_primary_avg_mae"]
        )
        best_paired_only_by_target = (
            holdout_summary[holdout_summary["model_family"] == "paired_only"]
            .groupby(["protocol_mode", "target"], as_index=False)["mae"]
            .min()
            .rename(columns={"mae": "best_paired_only_target_mae"})
        )
        holdout_summary = holdout_summary.merge(
            best_paired_only_by_target,
            on=["protocol_mode", "target"],
            how="left",
        )
        holdout_summary["beats_best_paired_only_target"] = (
            holdout_summary["mae"] < holdout_summary["best_paired_only_target_mae"]
        )
        target_advances = holdout_summary["advances_target"].astype("boolean").fillna(False).astype(bool)
        holdout_summary["eligible_target_default"] = (
            (holdout_summary["model_family"] == "paired_only")
            | (holdout_summary["beats_best_paired_only_target"] & target_advances)
        )

        combined_default_row = (
            primary_avg[primary_avg["protocol_mode"] == "inductive"].sort_values("primary_avg_mae")
            .merge(
                holdout_summary[
                    ["protocol_mode", "candidate_id", "beats_best_paired_only", "advances_all_primary_targets"]
                ].drop_duplicates(subset=["protocol_mode", "candidate_id"]),
                on=["protocol_mode", "candidate_id"],
                how="left",
            )
            .assign(
                eligible_default=lambda df: (
                    (df["model_family"] == "paired_only")
                    | (df["beats_best_paired_only"] & df["advances_all_primary_targets"].fillna(False))
                ),
            )
            .sort_values(["eligible_default", "primary_avg_mae"], ascending=[False, True])
            .head(1)
        )

        combined_winner_summary = combined_default_row.copy()
        combined_winner_summary = combined_winner_summary.merge(
            best_paired_only_by_protocol.rename(columns={"protocol_mode": "winner_protocol_mode"}),
            left_on="protocol_mode",
            right_on="winner_protocol_mode",
            how="left",
        ).drop(columns=["winner_protocol_mode"])
        if self.config.deploy_best_per_target:
            winner_summary = self._per_target_recommendation(
                holdout_summary,
                protocol_mode="inductive",
                eligible_only=True,
            ).copy()
        else:
            winner_summary = combined_winner_summary.copy()
        return (
            holdout_summary.sort_values(["protocol_mode", "target", "mae", "candidate_id"]),
            winner_summary,
            combined_winner_summary,
        )

    def _per_target_recommendation(
        self,
        holdout_summary: pd.DataFrame,
        *,
        protocol_mode: str | None = None,
        eligible_only: bool = False,
    ) -> pd.DataFrame:
        if holdout_summary.empty:
            return pd.DataFrame()
        scored = holdout_summary[~holdout_summary["is_ceiling_model"]].copy()
        scored = scored[scored["target"].isin(self.config.primary_targets)].copy()
        if protocol_mode is not None:
            scored = scored[scored["protocol_mode"] == protocol_mode].copy()
        if scored.empty:
            return pd.DataFrame()
        if eligible_only and "eligible_target_default" in scored.columns:
            eligible = scored[scored["eligible_target_default"].astype("boolean").fillna(False).astype(bool)].copy()
            if not eligible.empty:
                scored = eligible
        if "eligible_target_default" in scored.columns:
            scored["_eligible_sort"] = scored["eligible_target_default"].astype("boolean").fillna(False).astype(bool)
            sort_cols = ["protocol_mode", "target", "_eligible_sort", "mae", "candidate_id"]
            ascending = [True, True, False, True, True]
        else:
            sort_cols = ["protocol_mode", "target", "mae", "candidate_id"]
            ascending = [True, True, True, True]
        best = (
            scored.sort_values(sort_cols, ascending=ascending)
            .groupby(["protocol_mode", "target"], as_index=False)
            .head(1)
            .drop(columns=["_eligible_sort"], errors="ignore")
        )
        return best.reset_index(drop=True)

    def _backend_comparison(self, holdout_summary: pd.DataFrame) -> pd.DataFrame:
        if holdout_summary.empty:
            return pd.DataFrame()
        scored = holdout_summary[
            ~holdout_summary["is_ceiling_model"] & holdout_summary["target"].isin(self.config.primary_targets)
        ].copy()
        if scored.empty:
            return pd.DataFrame()
        backend = scored["model_name"].str.extract(r"^(lightgbm|catboost|xgboost)", expand=False)
        scored = scored.assign(backend=backend.fillna("other"))
        return (
            scored.groupby(["protocol_mode", "backend"], as_index=False)[["mae", "r2"]]
            .mean()
            .sort_values(["protocol_mode", "mae", "backend"])
        )
