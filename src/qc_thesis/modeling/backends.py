from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import warnings

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

from .shared import FeatureSetBuilder
from .specs import FeatureSpec

try:
    from lightgbm import LGBMRegressor as _LGBMRegressor

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor as _XGBRegressor

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _safe_r2(y_true: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1:
        return float("nan")
    return float(r2_score(y_true, pred))


def _score_predictions(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": _safe_r2(y_true, pred),
        "bias": float(np.mean(pred - y_true)),
    }


def target_transform(values: np.ndarray, mode: str = "logit") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if mode == "logit":
        clipped = np.clip(arr, 1e-4, 1.0 - 1e-4)
        return np.log(clipped / (1.0 - clipped)).astype(np.float32)
    return arr


def target_inverse(values: np.ndarray, mode: str = "logit") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if mode == "logit":
        return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)
    return arr


@dataclass
class TorchBatches:
    wf: "torch.Tensor"
    acg: "torch.Tensor"
    scalars: "torch.Tensor"
    wf_mask: "torch.Tensor"
    acg_mask: "torch.Tensor"
    scalar_mask: "torch.Tensor"
    context: "torch.Tensor | None" = None
    context_mask: "torch.Tensor | None" = None


if _TORCH_AVAILABLE:

    class _BranchEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, out_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


    class UnitEncoder(nn.Module):
        def __init__(
            self,
            n_wf_features: int,
            n_acg_features: int,
            n_scalar_features: int,
            *,
            hidden_dim: int,
            latent_dim: int,
        ) -> None:
            super().__init__()
            self.wf_encoder = _BranchEncoder(max(n_wf_features, 1), hidden_dim, 32)
            self.acg_encoder = _BranchEncoder(max(n_acg_features, 1), hidden_dim, 32)
            self.scalar_encoder = _BranchEncoder(max(n_scalar_features, 1), hidden_dim, 32)
            self.fusion = nn.Sequential(
                nn.Linear(96, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, latent_dim),
                nn.ReLU(),
            )

        def forward(self, wf: "torch.Tensor", acg: "torch.Tensor", scalars: "torch.Tensor") -> "torch.Tensor":
            return self.fusion(torch.cat([self.wf_encoder(wf), self.acg_encoder(acg), self.scalar_encoder(scalars)], dim=1))


    class UnitAutoencoder(nn.Module):
        def __init__(
            self,
            n_wf_features: int,
            n_acg_features: int,
            n_scalar_features: int,
            *,
            hidden_dim: int,
            latent_dim: int,
        ) -> None:
            super().__init__()
            self.encoder = UnitEncoder(
                n_wf_features=n_wf_features,
                n_acg_features=n_acg_features,
                n_scalar_features=n_scalar_features,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
            )
            self.wf_decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, max(n_wf_features, 1)))
            self.acg_decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, max(n_acg_features, 1)))
            self.scalar_decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, max(n_scalar_features, 1)))

        def forward(
            self,
            wf: "torch.Tensor",
            acg: "torch.Tensor",
            scalars: "torch.Tensor",
        ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            latent = self.encoder(wf, acg, scalars)
            return self.wf_decoder(latent), self.acg_decoder(latent), self.scalar_decoder(latent), latent


    class UnitOnlyRegressor(nn.Module):
        def __init__(self, encoder: UnitEncoder, *, latent_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            wf: "torch.Tensor",
            acg: "torch.Tensor",
            scalars: "torch.Tensor",
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            latent = self.encoder(wf, acg, scalars)
            return self.head(latent).squeeze(1), latent


    class ContextualRegressor(nn.Module):
        def __init__(
            self,
            encoder: UnitEncoder,
            *,
            n_context_features: int,
            latent_dim: int,
            hidden_dim: int,
            context_hidden_dim: int,
        ) -> None:
            super().__init__()
            self.encoder = encoder
            self.context_encoder = nn.Sequential(
                nn.Linear(max(n_context_features, 1), context_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(context_hidden_dim, 32),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(latent_dim + 32, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            wf: "torch.Tensor",
            acg: "torch.Tensor",
            scalars: "torch.Tensor",
            context: "torch.Tensor",
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            latent = self.encoder(wf, acg, scalars)
            return self.head(torch.cat([latent, self.context_encoder(context)], dim=1)).squeeze(1), latent

else:

    class UnitEncoder:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required for the neural modeling components")

    class UnitAutoencoder:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required for the neural modeling components")

    class UnitOnlyRegressor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required for the neural modeling components")

    class ContextualRegressor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required for the neural modeling components")


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
    predictions: pd.DataFrame


class QCModelRegistry:
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
        if _XGBOOST_AVAILABLE:
            self.register(
                "xgboost",
                lambda rs: _XGBRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=rs,
                    n_jobs=-1,
                    verbosity=0,
                    tree_method="hist",
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
    @staticmethod
    def build(spec: FeatureSpec) -> ColumnTransformer:
        transformers: list[Any] = []
        if spec.numeric_cols:
            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), spec.numeric_cols))
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
                results.extend(self._fit_one_target(train, test, target, spec, split_name, model_names))
        return results

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
            pipe = Pipeline([("pre", QCPreprocessor.build(spec)), ("model", model)])
            if sample_weight is not None:
                pipe.fit(X_train, y_train_vals, model__sample_weight=sample_weight)
            else:
                pipe.fit(X_train, y_train_vals)
            pred = np.clip(pipe.predict(X_test), 0.0, 1.0)
            scores = _score_predictions(y_test_vals.values, pred)
            pred_df = pd.DataFrame(
                {
                    "row_uid": test.loc[test_mask, "row_uid"].astype(str).values if "row_uid" in test.columns else test.index[test_mask.values].astype(str),
                    "recording_key": test.loc[test_mask, "recording_key"].astype(str).values,
                    "dataset_type": test.loc[test_mask, "dataset_type"].astype(str).values if "dataset_type" in test.columns else "unknown",
                    "sorter_name": test.loc[test_mask, "sorter_name"].astype(str).values if "sorter_name" in test.columns else "unknown",
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

    def _compute_weights(self, train: pd.DataFrame) -> np.ndarray | None:
        if self.weighting_strategy == "inverse_type":
            if "dataset_type" not in train.columns:
                return None
            counts = train["dataset_type"].value_counts()
            n_total = len(train)
            return train["dataset_type"].map(lambda value: n_total / counts.get(value, 1)).values
        if self.weighting_strategy == "inverse_recording":
            if "recording_key" not in train.columns:
                return None
            counts = train["recording_key"].value_counts()
            n_total = len(train)
            return train["recording_key"].map(lambda value: n_total / counts.get(value, 1)).values
        if self.weighting_strategy == "paired_3x":
            if "dataset_type" not in train.columns:
                return None
            return np.where(train["dataset_type"] == "paired", 3.0, 1.0)
        return None


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
    if "cosine" in clean or clean in {"n_confusable", "isolation_score", "min_neighbor_dist_um", "mean_neighbor_dist_um"}:
        return "cosine"
    return "other"


class QCEvaluator:
    @staticmethod
    def score(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        return _score_predictions(y_true, pred)

    @staticmethod
    def results_table(results: list[TrainerResult]) -> pd.DataFrame:
        rows = [
            {
                "target": result.target,
                "split": result.split_name,
                "feature_mode": result.feature_mode,
                "model": result.model_name,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "mae": result.mae,
                "rmse": result.rmse,
                "r2": result.r2,
                "bias": result.bias,
            }
            for result in results
        ]
        return pd.DataFrame(rows).sort_values(["split", "feature_mode", "target", "rmse", "mae"])


class JointEvaluator:
    def score(
        self,
        results: list[TrainerResult],
        required_targets: tuple[str, ...] = ("accuracy", "fpos", "fmiss"),
    ) -> pd.DataFrame:
        combos: dict[tuple, dict[str, pd.DataFrame]] = {}
        for result in results:
            key = (result.split_name, result.feature_mode, result.model_name)
            combos.setdefault(key, {})[result.target] = result.predictions

        rows = []
        for (split_name, feature_mode, model_name), target_preds in combos.items():
            missing = [target for target in required_targets if target not in target_preds]
            if missing:
                continue
            merged = target_preds[required_targets[0]][["row_uid", "recording_key", "dataset_type", "sorter_name"]].copy()
            for target in required_targets:
                part = target_preds[target][["row_uid", "true", "pred", "abs_error", "sq_error"]].rename(
                    columns={
                        "true": f"true_{target}",
                        "pred": f"pred_{target}",
                        "abs_error": f"abs_error_{target}",
                        "sq_error": f"sq_error_{target}",
                    }
                )
                merged = merged.merge(part, on="row_uid", how="inner")
            if merged.empty:
                continue
            abs_cols = [f"abs_error_{target}" for target in required_targets]
            sq_cols = [f"sq_error_{target}" for target in required_targets]
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
        observed_types = [dataset_type for dataset_type, count in df["dataset_type"].value_counts().items() if count >= min_rows]
        for held_out in observed_types:
            train_part = df[df["dataset_type"] != held_out].copy()
            test_part = df[df["dataset_type"] == held_out].copy()
            if len(train_part) < self.trainer.min_train_rows or len(test_part) < min_rows:
                continue
            all_results.extend(
                self.trainer.fit_all(
                    train=train_part,
                    test=test_part,
                    targets=targets,
                    specs=specs,
                    split_name=f"holdout_dataset_type:{held_out}",
                )
            )
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
            all_results.extend(
                self.trainer.fit_all(
                    train=train_part,
                    test=test_part,
                    targets=targets,
                    specs=specs,
                    split_name=f"holdout_sorter:{held_out}",
                    model_names=["lightgbm", "random_forest"],
                )
            )
        return QCEvaluator.results_table(all_results) if all_results else pd.DataFrame()


class SHAPAnalyzer:
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
                "family": [_feature_family_from_name(feature) for feature in feature_names],
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_df["rank"] = shap_df["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
        return shap_df

    @staticmethod
    def family_summary(shap_df: pd.DataFrame) -> pd.DataFrame:
        return shap_df.groupby("family")["mean_abs_shap"].sum().sort_values(ascending=False).reset_index()


def ensure_torch_available() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for the neural modeling components")


def batch_to_device(
    *,
    wf: np.ndarray,
    acg: np.ndarray,
    scalars: np.ndarray,
    wf_mask: np.ndarray,
    acg_mask: np.ndarray,
    scalar_mask: np.ndarray,
    context: np.ndarray | None,
    context_mask: np.ndarray | None,
    device: "torch.device",
) -> TorchBatches:
    ensure_torch_available()
    return TorchBatches(
        wf=torch.tensor(wf, dtype=torch.float32, device=device),
        acg=torch.tensor(acg, dtype=torch.float32, device=device),
        scalars=torch.tensor(scalars, dtype=torch.float32, device=device),
        wf_mask=torch.tensor(wf_mask, dtype=torch.bool, device=device),
        acg_mask=torch.tensor(acg_mask, dtype=torch.bool, device=device),
        scalar_mask=torch.tensor(scalar_mask, dtype=torch.bool, device=device),
        context=None if context is None else torch.tensor(context, dtype=torch.float32, device=device),
        context_mask=None if context_mask is None else torch.tensor(context_mask, dtype=torch.bool, device=device),
    )


def clone_encoder_state(encoder: UnitEncoder) -> dict[str, Any]:
    ensure_torch_available()
    return {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}


def make_unit_only_regressor(
    *,
    n_wf_features: int,
    n_acg_features: int,
    n_scalar_features: int,
    hidden_dim: int,
    latent_dim: int,
    encoder_state: dict[str, Any] | None,
    device: "torch.device",
) -> "torch.nn.Module":
    encoder = UnitEncoder(
        n_wf_features=n_wf_features,
        n_acg_features=n_acg_features,
        n_scalar_features=n_scalar_features,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    if encoder_state is not None:
        encoder.load_state_dict(encoder_state)
    model = UnitOnlyRegressor(encoder=encoder, latent_dim=latent_dim, hidden_dim=hidden_dim)
    return model.to(device)


def _pairwise_sq_dists(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    sq = x_norm + y_norm - (2.0 * x.mm(y.transpose(0, 1)))
    return torch.clamp(sq, min=0.0)


def _median_bandwidth(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
    combined = torch.cat([x, y], dim=0)
    dists = _pairwise_sq_dists(combined, combined)
    positive = dists[dists > 0]
    if positive.numel() == 0:
        return combined.new_tensor(1.0)
    return positive.median()


def rbf_multi_scale_mmd(
    source: "torch.Tensor",
    target: "torch.Tensor",
    *,
    bandwidth_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> "torch.Tensor":
    if source.shape[0] == 0 or target.shape[0] == 0:
        return source.new_tensor(0.0)
    base_bw = torch.clamp(_median_bandwidth(source, target), min=1e-6)
    d_xx = _pairwise_sq_dists(source, source)
    d_yy = _pairwise_sq_dists(target, target)
    d_xy = _pairwise_sq_dists(source, target)
    loss = source.new_tensor(0.0)
    for mult in bandwidth_multipliers:
        bw = torch.clamp(base_bw * float(mult), min=1e-6)
        gamma = 1.0 / (2.0 * bw)
        loss = loss + torch.exp(-gamma * d_xx).mean() + torch.exp(-gamma * d_yy).mean() - (2.0 * torch.exp(-gamma * d_xy).mean())
    return torch.clamp(loss / max(len(bandwidth_multipliers), 1), min=0.0)


__all__ = [
    "JointEvaluator",
    "ContextualRegressor",
    "QCEvaluator",
    "QCModelRegistry",
    "QCPreprocessor",
    "QCTrainer",
    "SHAPAnalyzer",
    "TorchBatches",
    "TrainerResult",
    "TransferEvaluator",
    "UnitAutoencoder",
    "UnitEncoder",
    "UnitOnlyRegressor",
    "_score_predictions",
    "batch_to_device",
    "clone_encoder_state",
    "ensure_torch_available",
    "make_unit_only_regressor",
    "rbf_multi_scale_mmd",
    "target_inverse",
    "target_transform",
]
