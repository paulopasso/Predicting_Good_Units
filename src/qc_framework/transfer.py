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

from qc_neural.normalizer import RecordingNormalizer

from .data import QCDataCleaner, QCDataLoader
from .features import FeatureSetBuilder, RecordingRelativeFeatureAugmenter

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
    from qc_neural.transfer import MultiTaskDomainAdaptiveRegressor

    _NEURAL_TRANSFER_AVAILABLE = True
except ImportError:
    _NEURAL_TRANSFER_AVAILABLE = False

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
    feature_views: tuple[str, ...] = ("norm_swap", "shape_only", "recording_relative")
    model_families: tuple[str, ...] = (
        "dummy",
        "paired_only",
        "hybrid_only",
        "hybrid_calibrated",
        "hybrid_stack",
        "hybrid_plus_paired",
        "neural_transfer",
    )
    feature_view_sets: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("shape_only", ("shape_only",)),
        ("norm_swap", ("norm_swap",)),
        ("norm_swap+recording_relative", ("norm_swap", "recording_relative")),
    )
    model_selection_splits: int = 5
    model_selection_test_size: float = 0.40
    random_state: int = 42
    lightgbm_estimators: int = 250
    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 31
    paired_models: tuple[str, ...] = ("lightgbm", "catboost")
    paired_weight_grid: tuple[float, ...] = (10.0, 25.0, 50.0)
    neural_pretrain_epochs: int = 35
    neural_finetune_epochs: int = 20
    neural_batch_size: int = 128
    neural_lr: float = 1e-3
    neural_domain_loss_weight: float = 0.15
    neural_target_loss_weight: float = 2.0
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


@dataclass
class _PreparedTransferData:
    df: pd.DataFrame
    hybrid: pd.DataFrame
    paired: pd.DataFrame
    paired_matched: pd.DataFrame
    spec: Any
    normalized_cols: list[str]
    feature_columns: dict[str, list[str]]


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
        df, _ = cleaner.clean(raw)

        normalizer = RecordingNormalizer()
        df = normalizer.fit_transform(df)
        relative = RecordingRelativeFeatureAugmenter()
        df = relative.fit_transform(df)

        hybrid = df[df["dataset_type"] == "hybrid"].copy()
        paired = df[df["dataset_type"] == "paired"].copy()
        paired_matched = paired[paired["matched_gt_unit_id"].notna()].copy()

        dup_pairs = FeatureSetBuilder.find_exact_duplicate_pairs(hybrid)
        extra_drop = {b for _, b in dup_pairs}
        spec = FeatureSetBuilder().build(
            hybrid,
            mode="unit_only",
            exclude_soft=False,
            extra_drop_cols=extra_drop,
        )

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

        selection_results = self._run_model_selection(
            prepared=prepared,
            paired_matched=non_holdout_matched,
            paired_all_non_holdout=non_holdout_paired,
        )
        selection_summary = self._summarize_selection(selection_results)
        finalists = self._select_finalists(selection_summary)
        holdout_results = self._run_final_holdout(
            prepared=prepared,
            finalists=finalists,
            paired_train=non_holdout_matched,
            paired_unlabeled=non_holdout_paired,
            paired_holdout=final_holdout,
        )
        holdout_summary, winner_summary = self._summarize_holdout(holdout_results, selection_summary)

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
            "winner_summary": winner_summary.copy(),
        }

    def _estimate_total_tasks(self) -> int:
        non_neural_families = [
            family
            for family in self.config.model_families
            if family in {
                "dummy",
                "paired_only",
                "hybrid_only",
                "hybrid_calibrated",
                "hybrid_stack",
                "hybrid_plus_paired",
            }
        ]
        per_split = (len(self.config.all_targets()) * len(non_neural_families)) + int(
            "neural_transfer" in self.config.model_families and _NEURAL_TRANSFER_AVAILABLE
        )
        return per_split * (self.config.model_selection_splits + 1)

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
    ) -> dict[str, Any]:
        scores = _score_predictions(y_true, pred)
        candidate_id = f"{model_family}|{model_name}|{feature_view}"
        return {
            "phase": phase,
            "split_id": str(split_id),
            "candidate_id": candidate_id,
            "target": eval_target,
            "source_target": source_target,
            "feature_view": feature_view,
            "model_family": model_family,
            "model_name": model_name,
            "source_label_rows": int(source_label_rows),
            "target_label_rows": int(target_label_rows),
            "target_eval_rows": int(target_eval_rows),
            "target_unlabeled_rows": int(target_unlabeled_rows),
            "used_unlabeled_paired": bool(used_unlabeled_paired),
            **scores,
        }

    # ------------------------------------------------------------------
    def _run_model_selection(
        self,
        *,
        prepared: _PreparedTransferData,
        paired_matched: pd.DataFrame,
        paired_all_non_holdout: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for paired_train, paired_test, split_id in self._make_model_selection_splits(paired_matched):
            rows.extend(
                self._run_candidate_bundle(
                    phase="model_selection",
                    split_id=split_id,
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
    ) -> pd.DataFrame:
        if finalists.empty:
            return pd.DataFrame()
        candidate_ids = set(finalists["candidate_id"].astype(str))
        all_rows = self._run_candidate_bundle(
            phase="final_holdout",
            split_id="final_holdout",
            prepared=prepared,
            paired_train=paired_train,
            paired_test=paired_holdout,
            paired_unlabeled=paired_unlabeled,
        )
        rows = [row for row in all_rows if row["candidate_id"] in candidate_ids]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _run_candidate_bundle(
        self,
        *,
        phase: str,
        split_id: str | int,
        prepared: _PreparedTransferData,
        paired_train: pd.DataFrame,
        paired_test: pd.DataFrame,
        paired_unlabeled: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        unlabeled_rows = len(paired_unlabeled) if self.config.allow_unlabeled_paired else len(paired_train)
        used_unlabeled = bool(self.config.allow_unlabeled_paired)
        target_context: dict[str, dict[str, Any]] = {}

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
            target_context[eval_target] = {
                "source_df": source_df,
                "y_source": y_source,
                "paired_train": paired_train_t,
                "y_train": y_train,
                "paired_test": paired_test_t,
                "y_test": y_test,
                "source_target": source_target,
            }

            if "dummy" in self.config.model_families:
                rows.extend(self._run_logged_task(
                    f"{phase} split={split_id} target={eval_target} family=dummy",
                    lambda eval_target=eval_target, source_target=source_target, y_train=y_train, y_test=y_test: self._run_dummy_family(
                        phase=phase,
                        split_id=split_id,
                        eval_target=eval_target,
                        source_target=source_target,
                        y_train=y_train,
                        y_test=y_test,
                    )
                ))
            if "paired_only" in self.config.model_families:
                rows.extend(self._run_logged_task(
                    f"{phase} split={split_id} target={eval_target} family=paired_only",
                    lambda eval_target=eval_target, source_target=source_target, paired_train_t=paired_train_t, y_train=y_train, paired_test_t=paired_test_t, y_test=y_test: self._run_paired_only_family(
                        phase=phase,
                        split_id=split_id,
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
                    f"{phase} split={split_id} target={eval_target} family=hybrid_only",
                    lambda eval_target=eval_target, source_target=source_target, source_df=source_df, y_source=y_source, paired_test_t=paired_test_t, y_test=y_test: self._run_hybrid_only_family(
                        phase=phase,
                        split_id=split_id,
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
                    f"{phase} split={split_id} target={eval_target} family=hybrid_calibrated",
                    lambda eval_target=eval_target, source_target=source_target, source_df=source_df, y_source=y_source, paired_train_t=paired_train_t, y_train=y_train, paired_test_t=paired_test_t, y_test=y_test: self._run_hybrid_calibration_family(
                        phase=phase,
                        split_id=split_id,
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
                    f"{phase} split={split_id} target={eval_target} family=hybrid_stack",
                    lambda eval_target=eval_target, source_target=source_target, source_df=source_df, y_source=y_source, paired_train_t=paired_train_t, y_train=y_train, paired_test_t=paired_test_t, y_test=y_test: self._run_hybrid_stack_family(
                        phase=phase,
                        split_id=split_id,
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
                    f"{phase} split={split_id} target={eval_target} family=hybrid_plus_paired",
                    lambda eval_target=eval_target, source_target=source_target, source_df=source_df, y_source=y_source, paired_train_t=paired_train_t, y_train=y_train, paired_test_t=paired_test_t, y_test=y_test: self._run_weighted_hybrid_family(
                        phase=phase,
                        split_id=split_id,
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

        if target_context and "neural_transfer" in self.config.model_families and _NEURAL_TRANSFER_AVAILABLE:
            rows.extend(self._run_logged_task(
                f"{phase} split={split_id} family=neural_transfer",
                lambda: self._run_neural_transfer_bundle(
                    phase=phase,
                    split_id=split_id,
                    prepared=prepared,
                    paired_train=paired_train,
                    paired_test=paired_test,
                    paired_unlabeled=paired_unlabeled if self.config.allow_unlabeled_paired else paired_train,
                    target_context=target_context,
                )
            ))
        return rows

    # ------------------------------------------------------------------
    def _run_dummy_family(
        self,
        *,
        phase: str,
        split_id: str | int,
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
        for feature_view in ("shape_only", "norm_swap+recording_relative"):
            cols = prepared.feature_columns.get(feature_view, [])
            if not cols:
                continue
            X_train = _ensure_numeric_frame(paired_train, cols)
            X_test = _ensure_numeric_frame(paired_test, cols)
            estimators: list[tuple[str, Pipeline]] = []
            if "lightgbm" in self.config.paired_models:
                estimators.append(("lightgbm", self._make_lgbm()))
            if "catboost" in self.config.paired_models and _CATBOOST_AVAILABLE:
                estimators.append(("catboost", self._make_catboost()))
            for model_name, model in estimators:
                model.fit(X_train, y_train.values)
                pred = model.predict(X_test)
                rows.append(
                    self._result_row(
                        phase=phase,
                        split_id=split_id,
                        eval_target=eval_target,
                        source_target=source_target,
                        feature_view=feature_view,
                        model_family="paired_only",
                        model_name=model_name,
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
        for feature_view in ("shape_only", "norm_swap+recording_relative"):
            cols = prepared.feature_columns.get(feature_view, [])
            if not cols:
                continue
            X_source = _ensure_numeric_frame(source_df, cols)
            X_test = _ensure_numeric_frame(paired_test, cols)
            model = self._make_lgbm()
            model.fit(X_source, y_source.values)
            pred = model.predict(X_test)
            rows.append(
                self._result_row(
                    phase=phase,
                    split_id=split_id,
                    eval_target=eval_target,
                    source_target=source_target,
                    feature_view=feature_view,
                    model_family="hybrid_only",
                    model_name="lightgbm",
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
        feature_view = "norm_swap+recording_relative"
        cols = prepared.feature_columns.get(feature_view, [])
        if not cols:
            return []
        X_source = _ensure_numeric_frame(source_df, cols)
        X_train = _ensure_numeric_frame(paired_train, cols)
        X_test = _ensure_numeric_frame(paired_test, cols)

        base_model = self._make_lgbm()
        base_model.fit(X_source, y_source.values)
        pred_train = _clip_pred(base_model.predict(X_train))
        pred_test = _clip_pred(base_model.predict(X_test))

        rows: list[dict[str, Any]] = []
        linear = LinearRegression().fit(pred_train.reshape(-1, 1), y_train.values)
        linear_pred = linear.predict(pred_test.reshape(-1, 1))
        rows.append(
            self._result_row(
                phase=phase,
                split_id=split_id,
                eval_target=eval_target,
                source_target=source_target,
                feature_view=feature_view,
                model_family="hybrid_calibrated",
                model_name="lightgbm+linear",
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
                eval_target=eval_target,
                source_target=source_target,
                feature_view=feature_view,
                model_family="hybrid_calibrated",
                model_name="lightgbm+isotonic",
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
        feature_view = "norm_swap+recording_relative"
        cols = prepared.feature_columns.get(feature_view, [])
        if not cols:
            return []

        base_model = self._make_lgbm()
        X_source = _ensure_numeric_frame(source_df, cols)
        X_train = _ensure_numeric_frame(paired_train, cols)
        X_test = _ensure_numeric_frame(paired_test, cols)
        base_model.fit(X_source, y_source.values)

        pred_train = _clip_pred(base_model.predict(X_train))
        pred_test = _clip_pred(base_model.predict(X_test))

        X_stack_train = X_train.copy()
        X_stack_test = X_test.copy()
        X_stack_train["hybrid_pred"] = pred_train
        X_stack_test["hybrid_pred"] = pred_test

        residual = y_train.values - pred_train
        residual_model = self._make_lgbm()
        residual_model.fit(X_stack_train, residual)
        final_pred = pred_test + residual_model.predict(X_stack_test)

        return [
            self._result_row(
                phase=phase,
                split_id=split_id,
                eval_target=eval_target,
                source_target=source_target,
                feature_view=feature_view,
                model_family="hybrid_stack",
                model_name="lightgbm_residual",
                y_true=y_test.values,
                pred=final_pred,
                source_label_rows=len(y_source),
                target_label_rows=len(y_train),
                target_eval_rows=len(y_test),
                target_unlabeled_rows=target_unlabeled_rows,
                used_unlabeled_paired=used_unlabeled_paired,
            )
        ]

    # ------------------------------------------------------------------
    def _run_weighted_hybrid_family(
        self,
        *,
        phase: str,
        split_id: str | int,
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
        feature_view = "norm_swap+recording_relative"
        cols = prepared.feature_columns.get(feature_view, [])
        if not cols:
            return []
        X_source = _ensure_numeric_frame(source_df, cols)
        X_train = _ensure_numeric_frame(paired_train, cols)
        X_test = _ensure_numeric_frame(paired_test, cols)

        X_mix = pd.concat([X_source, X_train], axis=0, ignore_index=True)
        y_mix = np.concatenate([y_source.values, y_train.values])
        rows: list[dict[str, Any]] = []
        for weight in self.config.paired_weight_grid:
            sample_weight = np.concatenate(
                [np.ones(len(X_source), dtype=float), np.full(len(X_train), float(weight), dtype=float)]
            )
            model = self._make_lgbm()
            model.fit(X_mix, y_mix, model__sample_weight=sample_weight)
            pred = model.predict(X_test)
            rows.append(
                self._result_row(
                    phase=phase,
                    split_id=split_id,
                    eval_target=eval_target,
                    source_target=source_target,
                    feature_view=feature_view,
                    model_family="hybrid_plus_paired",
                    model_name=f"lightgbm_w{weight:g}",
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
    def _run_neural_transfer_bundle(
        self,
        *,
        phase: str,
        split_id: str | int,
        prepared: _PreparedTransferData,
        paired_train: pd.DataFrame,
        paired_test: pd.DataFrame,
        paired_unlabeled: pd.DataFrame,
        target_context: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not _NEURAL_TRANSFER_AVAILABLE:
            return []
        feature_view = "norm_swap+recording_relative"
        cols = prepared.feature_columns.get(feature_view, [])
        if not cols:
            return []

        model = MultiTaskDomainAdaptiveRegressor(
            feature_columns=cols,
            pretrain_epochs=self.config.neural_pretrain_epochs,
            finetune_epochs=self.config.neural_finetune_epochs,
            batch_size=self.config.neural_batch_size,
            lr=self.config.neural_lr,
            domain_loss_weight=self.config.neural_domain_loss_weight,
            target_loss_weight=self.config.neural_target_loss_weight,
            random_state=self.config.random_state,
            verbose=self.config.neural_verbose,
            progress_prefix=f"{phase} split={split_id} neural_transfer",
        )
        model.fit(
            source_df=prepared.hybrid,
            target_labeled_df=paired_train,
            target_unlabeled_df=paired_unlabeled,
            source_target_map={
                "accuracy": "accuracy",
                "fpos": "fpos",
                "fmiss": self.config.source_fmiss_target,
            },
        )
        pred_df = model.predict_df(paired_test)
        rows: list[dict[str, Any]] = []
        for eval_target, ctx in target_context.items():
            if eval_target not in pred_df.columns:
                continue
            eval_index = ctx["paired_test"].index
            pred = pred_df.loc[eval_index, eval_target].values
            rows.append(
                self._result_row(
                    phase=phase,
                    split_id=split_id,
                    eval_target=eval_target,
                    source_target=ctx["source_target"],
                    feature_view=feature_view,
                    model_family="neural_transfer",
                    model_name="multitask_coral",
                    y_true=ctx["y_test"].values,
                    pred=pred,
                    source_label_rows=int(prepared.hybrid[ctx["source_target"]].notna().sum()),
                    target_label_rows=int(paired_train[eval_target].notna().sum()),
                    target_eval_rows=len(ctx["y_test"]),
                    target_unlabeled_rows=len(paired_unlabeled),
                    used_unlabeled_paired=True,
                )
            )
        return rows

    # ------------------------------------------------------------------
    def _summarize_selection(self, results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return pd.DataFrame()

        summary = (
            results.groupby(
                ["candidate_id", "model_family", "model_name", "feature_view", "target"],
                as_index=False,
            )[["mae", "rmse", "r2", "bias", "calibration_slope", "calibration_intercept"]]
            .mean()
        )

        primary = summary[summary["target"].isin(self.config.primary_targets)].copy()
        primary_avg = (
            primary.groupby(["candidate_id", "model_family", "model_name", "feature_view"], as_index=False)["mae"]
            .mean()
            .rename(columns={"mae": "primary_avg_mae"})
        )
        summary = summary.merge(
            primary_avg,
            on=["candidate_id", "model_family", "model_name", "feature_view"],
            how="left",
        )

        dummy_by_target = (
            primary[primary["model_family"] == "dummy"][["target", "mae"]]
            .rename(columns={"mae": "dummy_mae"})
        )
        hybrid_best_by_target = (
            primary[primary["model_family"] == "hybrid_only"]
            .groupby("target", as_index=False)["mae"]
            .min()
            .rename(columns={"mae": "best_hybrid_only_mae"})
        )
        summary = summary.merge(dummy_by_target, on="target", how="left")
        summary = summary.merge(hybrid_best_by_target, on="target", how="left")
        summary["beats_dummy"] = summary["mae"] < summary["dummy_mae"]
        summary["beats_best_hybrid_only"] = summary["mae"] < summary["best_hybrid_only_mae"]

        gate = (
            summary[summary["target"].isin(self.config.primary_targets)]
            .groupby("candidate_id", as_index=False)[["beats_dummy", "beats_best_hybrid_only"]]
            .all()
            .rename(
                columns={
                    "beats_dummy": "beats_dummy_all_primary_targets",
                    "beats_best_hybrid_only": "beats_hybrid_only_all_primary_targets",
                }
            )
        )
        summary = summary.merge(gate, on="candidate_id", how="left")
        summary["advances"] = (
            summary["beats_dummy_all_primary_targets"].fillna(False)
            & summary["beats_hybrid_only_all_primary_targets"].fillna(False)
        )
        return summary.sort_values(["primary_avg_mae", "candidate_id", "target"])

    # ------------------------------------------------------------------
    def _select_finalists(self, selection_summary: pd.DataFrame) -> pd.DataFrame:
        if selection_summary.empty:
            return pd.DataFrame()

        primary = selection_summary[selection_summary["target"].isin(self.config.primary_targets)].copy()
        candidate_rows = (
            primary.groupby(
                ["candidate_id", "model_family", "model_name", "feature_view", "primary_avg_mae"],
                as_index=False,
            )[["advances", "beats_dummy_all_primary_targets", "beats_hybrid_only_all_primary_targets"]]
            .first()
        )

        finalists: list[pd.DataFrame] = []
        for family in ("dummy", "hybrid_only", "paired_only", "hybrid_calibrated", "hybrid_stack", "hybrid_plus_paired", "neural_transfer"):
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
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if holdout_results.empty:
            return pd.DataFrame(), pd.DataFrame()

        holdout_summary = (
            holdout_results.groupby(
                ["candidate_id", "model_family", "model_name", "feature_view", "target"],
                as_index=False,
            )[["mae", "rmse", "r2", "bias", "calibration_slope", "calibration_intercept"]]
            .mean()
        )

        primary_avg = (
            holdout_summary[holdout_summary["target"].isin(self.config.primary_targets)]
            .groupby(["candidate_id", "model_family", "model_name", "feature_view"], as_index=False)["mae"]
            .mean()
            .rename(columns={"mae": "primary_avg_mae"})
        )
        holdout_summary = holdout_summary.merge(
            primary_avg,
            on=["candidate_id", "model_family", "model_name", "feature_view"],
            how="left",
        )

        selection_gate = selection_summary[
            ["candidate_id", "advances", "beats_dummy_all_primary_targets", "beats_hybrid_only_all_primary_targets"]
        ].drop_duplicates()
        holdout_summary = holdout_summary.merge(selection_gate, on="candidate_id", how="left")

        paired_only_rows = primary_avg[primary_avg["model_family"] == "paired_only"]
        has_paired_only = not paired_only_rows.empty
        best_paired_only = float(paired_only_rows["primary_avg_mae"].min()) if has_paired_only else float("nan")
        holdout_summary["beats_best_paired_only"] = (
            holdout_summary["primary_avg_mae"] < best_paired_only if has_paired_only else False
        )

        default_row = (
            primary_avg.sort_values("primary_avg_mae")
            .merge(
                holdout_summary[
                    ["candidate_id", "beats_best_paired_only"]
                ].drop_duplicates(),
                on="candidate_id",
                how="left",
            )
            .assign(
                advances=lambda df: df["candidate_id"].map(
                    selection_gate.set_index("candidate_id")["advances"].to_dict()
                ).fillna(False),
                eligible_default=lambda df: (
                    (df["model_family"] == "paired_only")
                    | (df["beats_best_paired_only"] & df["advances"])
                ),
            )
            .sort_values(["eligible_default", "primary_avg_mae"], ascending=[False, True])
            .head(1)
        )

        winner_summary = default_row.copy()
        winner_summary["best_paired_only_primary_avg_mae"] = best_paired_only
        return holdout_summary.sort_values(["primary_avg_mae", "candidate_id", "target"]), winner_summary
