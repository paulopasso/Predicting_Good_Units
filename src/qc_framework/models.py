from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import FeatureSpec, FeatureSetBuilder

try:
    from lightgbm import LGBMRegressor as _LGBMRegressor

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False


def _safe_r2(y_true: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1:
        return float("nan")
    return float(r2_score(y_true, pred))


@dataclass
class TrainerResult:
    target: str
    split_name: str
    feature_mode: str
    model_name: str
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r2: float
    bias: float
    fitted_pipe: Pipeline
    predictions: pd.DataFrame  # row_uid, recording_key, dataset_type, true, pred, abs_error


class QCModelRegistry:
    """Registry of model factories."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[int], BaseEstimator]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("dummy", lambda rs: DummyRegressor(strategy="mean"))
        self.register(
            "random_forest",
            lambda rs: RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=3,
                random_state=rs,
                n_jobs=-1,
            ),
        )
        if _LGBM_AVAILABLE:
            self.register(
                "lightgbm",
                lambda rs: _LGBMRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=rs,
                    n_jobs=-1,
                    verbose=-1,
                ),
            )

    def register(self, name: str, factory: Callable[[int], BaseEstimator]) -> None:
        self._factories[name] = factory

    def make(self, name: str, random_state: int = 42) -> BaseEstimator:
        if name not in self._factories:
            raise KeyError(f"Unknown model: {name!r}. Available: {self.names()}")
        return self._factories[name](random_state)

    def names(self) -> list[str]:
        return list(self._factories.keys())


class QCPreprocessor:
    """Build a ColumnTransformer for a FeatureSpec."""

    @staticmethod
    def build(spec: FeatureSpec) -> ColumnTransformer:
        transformers: list[Any] = []
        if spec.numeric_cols:
            transformers.append(
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    spec.numeric_cols,
                )
            )
        if spec.categorical_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    spec.categorical_cols,
                )
            )
        return ColumnTransformer(transformers=transformers, remainder="drop")


class QCTrainer:
    """Fit models across targets x feature specs."""

    def __init__(
        self,
        registry: QCModelRegistry | None = None,
        random_state: int = 42,
        min_train_rows: int = 200,
        min_test_rows: int = 40,
        weighting_strategy: str | None = None,
    ) -> None:
        self.registry = registry or QCModelRegistry()
        self.random_state = random_state
        self.min_train_rows = min_train_rows
        self.min_test_rows = min_test_rows
        self.weighting_strategy = weighting_strategy

    # ------------------------------------------------------------------
    def fit_all(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        targets: list[str],
        specs: list[FeatureSpec],
        split_name: str,
        model_names: list[str] | None = None,
    ) -> list[TrainerResult]:
        if model_names is None:
            model_names = self.registry.names()

        results: list[TrainerResult] = []
        for spec in specs:
            for target in targets:
                results.extend(
                    self._fit_one_target(
                        train, test, target, spec, split_name, model_names
                    )
                )
        return results

    # ------------------------------------------------------------------
    def _fit_one_target(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        spec: FeatureSpec,
        split_name: str,
        model_names: list[str],
    ) -> list[TrainerResult]:
        y_train = pd.to_numeric(train[target], errors="coerce")
        y_test = pd.to_numeric(test[target], errors="coerce")

        train_mask = y_train.notna()
        test_mask = y_test.notna()

        if int(train_mask.sum()) < self.min_train_rows or int(test_mask.sum()) < self.min_test_rows:
            return []

        X_train = FeatureSetBuilder.align_frame(train.loc[train_mask], spec)
        X_test = FeatureSetBuilder.align_frame(test.loc[test_mask], spec)
        y_train_vals = y_train.loc[train_mask]
        y_test_vals = y_test.loc[test_mask]

        sample_weight = self._compute_weights(train.loc[train_mask]) if self.weighting_strategy else None

        results: list[TrainerResult] = []
        for model_name in model_names:
            model = self.registry.make(model_name, random_state=self.random_state)
            pre = QCPreprocessor.build(spec)
            pipe = Pipeline([("pre", pre), ("model", model)])

            if sample_weight is not None:
                pipe.fit(X_train, y_train_vals, model__sample_weight=sample_weight)
            else:
                pipe.fit(X_train, y_train_vals)

            pred = np.clip(pipe.predict(X_test), 0.0, 1.0)
            scores = _score_predictions(y_test_vals.values, pred)

            pred_df = pd.DataFrame(
                {
                    "row_uid": (
                        test.loc[test_mask, "row_uid"].astype(str).values
                        if "row_uid" in test.columns
                        else test.index[test_mask.values].astype(str)
                    ),
                    "recording_key": test.loc[test_mask, "recording_key"].astype(str).values,
                    "dataset_type": test.loc[test_mask, "dataset_type"].astype(str).values
                    if "dataset_type" in test.columns
                    else "unknown",
                    "sorter_name": (
                        test.loc[test_mask, "sorter_name"].astype(str).values
                        if "sorter_name" in test.columns
                        else "unknown"
                    ),
                    "target": target,
                    "true": y_test_vals.values,
                    "pred": pred,
                    "abs_error": np.abs(pred - y_test_vals.values),
                    "sq_error": (pred - y_test_vals.values) ** 2,
                }
            )

            results.append(
                TrainerResult(
                    target=target,
                    split_name=split_name,
                    feature_mode=spec.mode,
                    model_name=model_name,
                    n_train=int(len(X_train)),
                    n_test=int(len(X_test)),
                    mae=scores["mae"],
                    rmse=scores["rmse"],
                    r2=scores["r2"],
                    bias=scores["bias"],
                    fitted_pipe=pipe,
                    predictions=pred_df,
                )
            )
        return results

    # ------------------------------------------------------------------
    def _compute_weights(self, train: pd.DataFrame) -> np.ndarray | None:
        if self.weighting_strategy == "inverse_type":
            if "dataset_type" not in train.columns:
                return None
            counts = train["dataset_type"].value_counts()
            n_total = len(train)
            return train["dataset_type"].map(lambda t: n_total / counts.get(t, 1)).values
        if self.weighting_strategy == "inverse_recording":
            if "recording_key" not in train.columns:
                return None
            counts = train["recording_key"].value_counts()
            n_total = len(train)
            return train["recording_key"].map(lambda k: n_total / counts.get(k, 1)).values
        if self.weighting_strategy == "paired_3x":
            if "dataset_type" not in train.columns:
                return None
            return np.where(train["dataset_type"] == "paired", 3.0, 1.0)
        return None


def _score_predictions(y_true: np.ndarray, pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": _safe_r2(y_true, pred),
        "bias": float(np.mean(pred - y_true)),
    }
