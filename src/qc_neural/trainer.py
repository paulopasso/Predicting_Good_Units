"""NeuralTrainer — QCTrainer subclass that bypasses sklearn preprocessing for neural models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from qc_framework.features import FeatureSetBuilder, FeatureSpec
from qc_framework.models import QCModelRegistry, QCTrainer, TrainerResult

from .model import QCNeuralWrapper

try:
    from qc_framework.models import _LGBM_AVAILABLE  # noqa: F401
except ImportError:
    pass

try:
    _TORCH_AVAILABLE = True
    import torch as _torch  # noqa: F401
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_r2(y_true: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1:
        return float("nan")
    return float(r2_score(y_true, pred))


def _score(y_true: np.ndarray, pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": _safe_r2(y_true, pred),
        "bias": float(np.mean(pred - y_true)),
    }


# ---------------------------------------------------------------------------
# Registry factory helpers
# ---------------------------------------------------------------------------

def _make_neural_registry() -> QCModelRegistry:
    """Return a QCModelRegistry pre-populated with neural models (if torch is available)."""
    reg = QCModelRegistry()
    if _TORCH_AVAILABLE:
        for target in ("accuracy", "fpos", "fmiss_extended"):
            # Capture target in default arg to avoid late-binding closure issues
            reg.register(
                f"neural_{target}",
                lambda rs, t=target: QCNeuralWrapper(target=t, random_state=rs),
            )
    return reg


# ---------------------------------------------------------------------------
# NeuralTrainer
# ---------------------------------------------------------------------------

_NEURAL_PREFIX = "neural_"


class NeuralTrainer(QCTrainer):
    """QCTrainer that routes neural model names through a dedicated fitting path.

    Neural models (names starting with ``"neural_"``) bypass the sklearn
    ``QCPreprocessor`` pipeline and receive the raw feature DataFrame directly.
    This allows :class:`~qc_neural.model.QCNeuralWrapper` to detect wf_bin_* and
    acg_* columns by name and route them through their respective CNN encoders.

    All non-neural model names are handled by the parent
    :class:`~qc_framework.models.QCTrainer` without modification.

    Parameters
    ----------
    registry:
        Defaults to a registry populated with dummy, random_forest, lightgbm
        (if available), and neural_* models (if torch is available).
    random_state, min_train_rows, min_test_rows, weighting_strategy:
        Forwarded to :class:`~qc_framework.models.QCTrainer`.
    """

    def __init__(
        self,
        registry: QCModelRegistry | None = None,
        random_state: int = 42,
        min_train_rows: int = 200,
        min_test_rows: int = 40,
        weighting_strategy: str | None = None,
    ) -> None:
        if registry is None:
            registry = _make_neural_registry()
        super().__init__(
            registry=registry,
            random_state=random_state,
            min_train_rows=min_train_rows,
            min_test_rows=min_test_rows,
            weighting_strategy=weighting_strategy,
        )

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
        neural = [n for n in model_names if n.startswith(_NEURAL_PREFIX)]
        others = [n for n in model_names if not n.startswith(_NEURAL_PREFIX)]

        results: list[TrainerResult] = []
        if others:
            results.extend(super()._fit_one_target(train, test, target, spec, split_name, others))
        if neural:
            results.extend(self._fit_neural_target(train, test, target, spec, split_name, neural))
        return results

    # ------------------------------------------------------------------
    def _fit_neural_target(
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
        y_tr = y_train.loc[train_mask].values
        y_te = y_test.loc[test_mask].values

        results: list[TrainerResult] = []
        for model_name in model_names:
            wrapper: QCNeuralWrapper = self.registry.make(model_name, random_state=self.random_state)
            wrapper.fit(X_train, y_tr)
            pred = np.clip(wrapper.predict(X_test), 0.0, 1.0)
            scores = _score(y_te, pred)

            pred_df = pd.DataFrame(
                {
                    "row_uid": (
                        test.loc[test_mask, "row_uid"].astype(str).values
                        if "row_uid" in test.columns
                        else test.index[test_mask.values].astype(str)
                    ),
                    "recording_key": test.loc[test_mask, "recording_key"].astype(str).values,
                    "dataset_type": (
                        test.loc[test_mask, "dataset_type"].astype(str).values
                        if "dataset_type" in test.columns
                        else "unknown"
                    ),
                    "sorter_name": (
                        test.loc[test_mask, "sorter_name"].astype(str).values
                        if "sorter_name" in test.columns
                        else "unknown"
                    ),
                    "target": target,
                    "true": y_te,
                    "pred": pred,
                    "abs_error": np.abs(pred - y_te),
                    "sq_error": (pred - y_te) ** 2,
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
                    fitted_pipe=wrapper,  # type: ignore[arg-type]
                    predictions=pred_df,
                )
            )
        return results
