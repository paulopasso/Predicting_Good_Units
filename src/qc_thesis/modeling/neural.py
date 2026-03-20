from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from qc_framework.data import QCDataCleaner, QCDataLoader
    from qc_framework.features import (
        FeatureSetBuilder,
        FeatureSpec,
        RecordingContextFeatureAugmenter,
        RecordingRelativeFeatureAugmenter,
    )
except ImportError:  # qc_framework only available in Colab/training environment
    from .shared import (
        FeatureSetBuilder,
        QCDataCleaner,
        QCDataLoader,
        RecordingContextFeatureAugmenter,
        RecordingRelativeFeatureAugmenter,
    )
    from .specs import FeatureSpec
from lightgbm import LGBMRegressor

from qc_thesis.modeling.backends import (
    ContextualRegressor,
    UnitAutoencoder,
    UnitEncoder,
    UnitOnlyRegressor,
    batch_to_device,
    ensure_torch_available,
    make_unit_only_regressor,
    rbf_multi_scale_mmd,
    target_inverse,
    target_transform,
)
from qc_thesis.modeling.shared import (
    MainExperimentSplit,
    StressStudySplit,
    build_stress_splits,
    make_grouped_validation_splits,
    make_main_split,
)
from qc_thesis.modeling.specs import MMDDomainAdaptationConfig, SSLDomainAdaptationConfig

import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD


@dataclass
class PreparedFeatureTable:
    df: pd.DataFrame
    spec: FeatureSpec
    hybrid: pd.DataFrame
    paired: pd.DataFrame
    paired_matched: pd.DataFrame
    paired_unmatched: pd.DataFrame
    unit_feature_cols: list[str]
    context_feature_cols: list[str]
    context_feature_cols_by_target: dict[str, list[str]]
    lightgbm_context_cols: list[str]
    shape_feature_cols: list[str]
    wf_feature_cols: list[str]
    acg_feature_cols: list[str]
    scalar_feature_cols: list[str]


@dataclass
class NaNStandardScaler:
    columns: list[str]
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "NaNStandardScaler":
        values = df.loc[:, self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(values)
        counts = valid.sum(axis=0).astype(float)
        safe_counts = np.where(counts > 0.0, counts, 1.0)
        filled = np.where(valid, values, 0.0)
        mean = filled.sum(axis=0) / safe_counts
        centered = np.where(valid, values - mean, 0.0)
        scale = np.sqrt((centered**2).sum(axis=0) / safe_counts)
        mean = np.where(counts > 0.0, mean, 0.0)
        scale = np.where(counts > 0.0, scale, 1.0)
        scale[scale == 0.0] = 1.0
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("NaNStandardScaler must be fit before transform")
        values = df.loc[:, self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return (values - self.mean_) / self.scale_


@dataclass
class PreparedModelInput:
    row_uid: np.ndarray
    recording_key: np.ndarray
    study_set: np.ndarray
    wf: np.ndarray
    acg: np.ndarray
    scalars: np.ndarray
    wf_mask: np.ndarray
    acg_mask: np.ndarray
    scalar_mask: np.ndarray
    context: np.ndarray | None
    context_mask: np.ndarray | None

    def __len__(self) -> int:
        return int(len(self.row_uid))

    def subset(self, index: np.ndarray | list[int]) -> "PreparedModelInput":
        idx = np.asarray(index)
        return PreparedModelInput(
            row_uid=self.row_uid[idx],
            recording_key=self.recording_key[idx],
            study_set=self.study_set[idx],
            wf=self.wf[idx],
            acg=self.acg[idx],
            scalars=self.scalars[idx],
            wf_mask=self.wf_mask[idx],
            acg_mask=self.acg_mask[idx],
            scalar_mask=self.scalar_mask[idx],
            context=None if self.context is None else self.context[idx],
            context_mask=None if self.context_mask is None else self.context_mask[idx],
        )


_CURATED_CONTEXT_BASES: dict[str, tuple[str, ...]] = {
    "fpos": (
        "snr",
        "peak_to_trough_uv",
        "amp_mean",
        "amp_p50",
        "template_norm",
        "wf_energy",
        "si_snr",
        "max_cosine",
        "mean_cosine_top3",
        "isolation_score",
        "min_neighbor_dist_um",
        "mean_neighbor_dist_um",
    ),
    "fmiss": (
        "si_amplitude_cutoff",
        "si_drift_ptp",
        "si_drift_std",
        "si_drift_mad",
        "firing_rate_hz",
        "isi_violation_rate",
        "max_cosine",
        "mean_cosine_top3",
        "n_confusable",
        "isolation_score",
        "max_waveform_cosine",
        "n_waveform_confusable",
        "amp_cv",
        "amp_drift_slope",
        "amp_drift_r2",
        "amp_outlier_pct",
    ),
}

_CURATED_CONTEXT_SUFFIXES: tuple[str, ...] = (
    "recording_pct",
    "recording_mean",
    "recording_std",
    "recording_delta",
)


def curated_context_columns(all_context_cols: list[str], *, target: str) -> list[str]:
    bases = _CURATED_CONTEXT_BASES.get(target, ())
    selected: list[str] = []
    for base in bases:
        for suffix in _CURATED_CONTEXT_SUFFIXES:
            name = f"{base}_{suffix}"
            if name in all_context_cols:
                selected.append(name)
    if selected:
        return selected
    fallback = [
        col
        for col in all_context_cols
        if col.endswith("_recording_pct") or col.endswith("_recording_mean") or col.endswith("_recording_delta")
    ]
    if fallback:
        return fallback[: min(len(fallback), 24)]
    return selected


def prepare_feature_table_from_frame(frame: pd.DataFrame) -> PreparedFeatureTable:
    cleaner = QCDataCleaner()
    cleaned, _ = cleaner.clean(frame)

    hybrid_clean = cleaned[cleaned["dataset_type"] == "hybrid"].copy()
    dup_pairs = FeatureSetBuilder.find_exact_duplicate_pairs(hybrid_clean)
    extra_drop = {b for _, b in dup_pairs}
    spec = FeatureSetBuilder().build(
        hybrid_clean,
        mode="unit_only",
        exclude_soft=False,
        extra_drop_cols=extra_drop,
    )

    relative = RecordingRelativeFeatureAugmenter()
    df = relative.fit_transform(cleaned)
    context = RecordingContextFeatureAugmenter()
    df = context.fit_transform(df)

    unit_feature_cols = [c for c in spec.numeric_cols if c in df.columns]
    shape_feature_cols = FeatureSetBuilder.transfer_feature_columns(
        df,
        spec,
        view_names=["shape_only"],
    )
    context_feature_cols = FeatureSetBuilder.transfer_feature_columns(
        df,
        spec,
        view_names=["recording_relative", "recording_context"],
    )
    context_feature_cols_by_target = {
        target: curated_context_columns(context_feature_cols, target=target)
        for target in ("fpos", "fmiss")
    }
    lightgbm_context_cols = FeatureSetBuilder.transfer_feature_columns(
        df,
        spec,
        view_names=["unit_raw", "recording_relative", "recording_context"],
    )
    wf_feature_cols = FeatureSetBuilder.waveform_bin_cols(unit_feature_cols)
    acg_feature_cols = FeatureSetBuilder.acg_cols(unit_feature_cols)
    scalar_feature_cols = [c for c in unit_feature_cols if c not in set(wf_feature_cols + acg_feature_cols)]

    paired = df[df["dataset_type"] == "paired"].copy()

    return PreparedFeatureTable(
        df=df,
        spec=spec,
        hybrid=df[df["dataset_type"] == "hybrid"].copy(),
        paired=paired,
        paired_matched=paired[paired["matched_gt_unit_id"].notna()].copy(),
        paired_unmatched=paired[paired["matched_gt_unit_id"].isna()].copy(),
        unit_feature_cols=unit_feature_cols,
        context_feature_cols=context_feature_cols,
        context_feature_cols_by_target=context_feature_cols_by_target,
        lightgbm_context_cols=lightgbm_context_cols,
        shape_feature_cols=shape_feature_cols,
        wf_feature_cols=wf_feature_cols,
        acg_feature_cols=acg_feature_cols,
        scalar_feature_cols=scalar_feature_cols,
    )


def prepare_feature_table(parquet_path: str | Path) -> PreparedFeatureTable:
    loader = QCDataLoader()
    frame = loader.load(parquet_path)
    return prepare_feature_table_from_frame(frame)


def fit_training_side_scalers(
    train_side_df: pd.DataFrame,
    prepared: PreparedFeatureTable,
    *,
    context_columns: list[str] | None = None,
) -> tuple[NaNStandardScaler, NaNStandardScaler | None]:
    unit_scaler = NaNStandardScaler(prepared.unit_feature_cols).fit(train_side_df)
    context_scaler = None
    selected_context_cols = prepared.context_feature_cols if context_columns is None else context_columns
    if selected_context_cols:
        context_scaler = NaNStandardScaler(selected_context_cols).fit(train_side_df)
    return unit_scaler, context_scaler


def prepare_model_input(
    df: pd.DataFrame,
    prepared: PreparedFeatureTable,
    unit_scaler: NaNStandardScaler,
    context_scaler: NaNStandardScaler | None,
    *,
    use_context: bool,
    context_columns: list[str] | None = None,
) -> PreparedModelInput:
    unit_values = unit_scaler.transform(df)
    unit_mask = np.isfinite(unit_values)
    unit_filled = np.nan_to_num(unit_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    wf_idx = [prepared.unit_feature_cols.index(c) for c in prepared.wf_feature_cols]
    acg_idx = [prepared.unit_feature_cols.index(c) for c in prepared.acg_feature_cols]
    scalar_idx = [prepared.unit_feature_cols.index(c) for c in prepared.scalar_feature_cols]

    context_values: np.ndarray | None = None
    context_mask: np.ndarray | None = None
    if use_context:
        if context_scaler is None:
            raise RuntimeError("Context scaler is required when use_context=True")
        context_values = context_scaler.transform(df)
        context_mask = np.isfinite(context_values)
        context_values = np.nan_to_num(context_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return PreparedModelInput(
        row_uid=df["row_uid"].astype(str).to_numpy(),
        recording_key=df["recording_key"].astype(str).to_numpy(),
        study_set=df["study_set"].astype(str).to_numpy(),
        wf=unit_filled[:, wf_idx].astype(np.float32),
        acg=unit_filled[:, acg_idx].astype(np.float32),
        scalars=unit_filled[:, scalar_idx].astype(np.float32) if scalar_idx else np.zeros((len(df), 1), dtype=np.float32),
        wf_mask=unit_mask[:, wf_idx],
        acg_mask=unit_mask[:, acg_idx],
        scalar_mask=unit_mask[:, scalar_idx] if scalar_idx else np.ones((len(df), 1), dtype=bool),
        context=context_values,
        context_mask=context_mask,
    )


def numeric_target(df: pd.DataFrame, target: str) -> np.ndarray:
    return pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)


def target_name_for_source(eval_target: str, source_fmiss_target: str) -> str:
    return source_fmiss_target if eval_target == "fmiss" else eval_target


@dataclass
class SSLPretrainResult:
    encoder_state: dict[str, Any]
    history: list[dict[str, Any]]
    checkpoint_path: str | None


@dataclass
class SupervisedTrainResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    history: list[dict[str, Any]]
    checkpoint_path: str | None
    selected_finetune_epochs: int
    source_checkpoint_path: str | None


def set_torch_runtime(random_state: int) -> None:
    ensure_torch_available()
    try:
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(random_state)
    np.random.seed(random_state)


def get_device() -> "torch.device":
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _vlog(config: Any, message: str) -> None:
    if config is not None and config.verbose:
        print(message, flush=True)


def _clone_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _calibration_params(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) < 2 or len(np.unique(pred)) <= 1:
        return float("nan"), float(np.nanmean(y_true))
    slope, intercept = np.polyfit(pred, y_true, 1)
    return float(slope), float(intercept)


def score_predictions(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=float)
    slope, intercept = _calibration_params(y_true, pred)
    r2 = float("nan") if len(np.unique(y_true)) <= 1 else float(r2_score(y_true, pred))
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": r2,
        "bias": float(np.mean(pred - y_true)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _batch_indices(n_rows: int, batch_size: int, rng: np.random.RandomState) -> list[np.ndarray]:
    if n_rows <= batch_size:
        return [np.arange(n_rows)]
    shuffled = rng.permutation(n_rows)
    return [shuffled[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def _make_optimizer(
    params: Any,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> "torch.optim.Optimizer":
    name = optimizer_name.lower()
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise KeyError(f"Unknown optimizer: {optimizer_name}")


def _regression_loss(pred: torch.Tensor, target: torch.Tensor, *, loss_name: str) -> torch.Tensor:
    name = loss_name.lower()
    if name == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    if name == "mse":
        return torch.mean((pred - target) ** 2)
    raise KeyError(f"Unknown supervised loss: {loss_name}")


def _mask_and_noise(x: torch.Tensor, observed_mask: torch.Tensor, *, mask_prob: float, noise_std: float) -> torch.Tensor:
    corruption_mask = (torch.rand_like(x) < mask_prob) & observed_mask
    noisy = torch.where(observed_mask, x + (torch.randn_like(x) * noise_std), x)
    return torch.where(corruption_mask, torch.zeros_like(noisy), noisy)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return pred.new_tensor(0.0)
    diff = pred[mask] - target[mask]
    return torch.mean(diff * diff)


def _feature_dims(data: PreparedModelInput) -> tuple[int, int, int, int]:
    return (
        int(data.wf.shape[1]),
        int(data.acg.shape[1]),
        int(data.scalars.shape[1]),
        0 if data.context is None else int(data.context.shape[1]),
    )


def _make_regressor(
    *,
    arm: str,
    data: PreparedModelInput,
    config: SSLDomainAdaptationConfig,
    encoder_state: dict[str, Any] | None,
    device: torch.device,
) -> torch.nn.Module:
    n_wf, n_acg, n_scalar, n_context = _feature_dims(data)
    encoder = UnitEncoder(
        n_wf_features=n_wf,
        n_acg_features=n_acg,
        n_scalar_features=n_scalar,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    )
    if encoder_state is not None:
        encoder.load_state_dict(encoder_state)
    if arm == "unit_only":
        model = UnitOnlyRegressor(
            encoder=encoder,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
        )
    elif arm == "contextual":
        model = ContextualRegressor(
            encoder=encoder,
            n_context_features=max(n_context, 1),
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            context_hidden_dim=config.context_hidden_dim,
        )
    else:
        raise KeyError(f"Unknown arm: {arm}")
    return model.to(device)


def _predict_regressor(model: torch.nn.Module, data: PreparedModelInput, *, arm: str, device: torch.device) -> np.ndarray:
    model.eval()
    batch = batch_to_device(
        wf=data.wf,
        acg=data.acg,
        scalars=data.scalars,
        wf_mask=data.wf_mask,
        acg_mask=data.acg_mask,
        scalar_mask=data.scalar_mask,
        context=data.context,
        context_mask=data.context_mask,
        device=device,
    )
    with torch.no_grad():
        if arm == "contextual":
            assert batch.context is not None
            pred, _ = model(batch.wf, batch.acg, batch.scalars, batch.context)
        else:
            pred, _ = model(batch.wf, batch.acg, batch.scalars)
    return np.clip(target_inverse(pred.detach().cpu().numpy()), 0.0, 1.0)


def _fit_regressor(
    *,
    model: torch.nn.Module,
    arm: str,
    train_data: PreparedModelInput,
    train_target: np.ndarray,
    val_data: PreparedModelInput | None,
    val_target: np.ndarray | None,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    freeze_encoder_epochs: int,
    random_state: int,
    device: torch.device,
    optimizer_name: str,
    weight_decay: float,
    loss_name: str,
    verbose: bool = False,
    log_prefix: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    rng = np.random.RandomState(random_state)
    optimizer = _make_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
    )
    train_batch = batch_to_device(
        wf=train_data.wf,
        acg=train_data.acg,
        scalars=train_data.scalars,
        wf_mask=train_data.wf_mask,
        acg_mask=train_data.acg_mask,
        scalar_mask=train_data.scalar_mask,
        context=train_data.context,
        context_mask=train_data.context_mask,
        device=device,
    )
    train_target_t = torch.tensor(target_transform(train_target), dtype=torch.float32, device=device)

    if val_data is not None and val_target is not None:
        val_batch = batch_to_device(
            wf=val_data.wf,
            acg=val_data.acg,
            scalars=val_data.scalars,
            wf_mask=val_data.wf_mask,
            acg_mask=val_data.acg_mask,
            scalar_mask=val_data.scalar_mask,
            context=val_data.context,
            context_mask=val_data.context_mask,
            device=device,
        )
    else:
        val_batch = None

    best_state = _clone_state_dict(model)
    best_epoch = 1
    best_score = float("inf")
    wait = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        freeze_encoder = epoch <= freeze_encoder_epochs
        for param in model.encoder.parameters():
            param.requires_grad = not freeze_encoder

        model.train()
        losses: list[float] = []
        for batch_idx in _batch_indices(len(train_data), batch_size, rng):
            idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
            if arm == "contextual":
                assert train_batch.context is not None
                pred, _ = model(
                    train_batch.wf[idx],
                    train_batch.acg[idx],
                    train_batch.scalars[idx],
                    train_batch.context[idx],
                )
            else:
                pred, _ = model(
                    train_batch.wf[idx],
                    train_batch.acg[idx],
                    train_batch.scalars[idx],
                )
            loss = _regression_loss(pred, train_target_t[idx], loss_name=loss_name)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
        }

        if val_batch is not None and val_target is not None:
            model.eval()
            with torch.no_grad():
                if arm == "contextual":
                    assert val_batch.context is not None
                    val_pred_t, _ = model(
                        val_batch.wf,
                        val_batch.acg,
                        val_batch.scalars,
                        val_batch.context,
                    )
                else:
                    val_pred_t, _ = model(val_batch.wf, val_batch.acg, val_batch.scalars)
            val_pred = np.clip(target_inverse(val_pred_t.detach().cpu().numpy()), 0.0, 1.0)
            val_score = score_predictions(val_target, val_pred)["mae"]
            row["val_mae"] = float(val_score)
            if val_score < best_score:
                best_score = float(val_score)
                best_epoch = epoch
                best_state = _clone_state_dict(model)
                wait = 0
            else:
                wait += 1
            if verbose:
                freeze_label = "frozen" if freeze_encoder else "trainable"
                print(
                    f"{log_prefix} epoch={epoch}/{epochs} encoder={freeze_label} "
                    f"train_loss={row['train_loss']:.6f} val_mae={val_score:.6f} "
                    f"best_epoch={best_epoch}",
                    flush=True,
                )
            history.append(row)
            if wait >= patience:
                break
        else:
            best_state = _clone_state_dict(model)
            best_epoch = epoch
            if verbose:
                freeze_label = "frozen" if freeze_encoder else "trainable"
                print(
                    f"{log_prefix} epoch={epoch}/{epochs} encoder={freeze_label} "
                    f"train_loss={row['train_loss']:.6f}",
                    flush=True,
                )
            history.append(row)

    model.load_state_dict(best_state)
    return best_state, history, best_epoch


def pretrain_ssl_autoencoder(
    *,
    train_data: PreparedModelInput,
    val_data: PreparedModelInput,
    config: SSLDomainAdaptationConfig,
    checkpoint_path: Path | None = None,
) -> SSLPretrainResult:
    set_torch_runtime(config.random_state)
    device = get_device()
    n_wf, n_acg, n_scalar, _ = _feature_dims(train_data)
    model = UnitAutoencoder(
        n_wf_features=n_wf,
        n_acg_features=n_acg,
        n_scalar_features=n_scalar,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    ).to(device)
    optimizer = _make_optimizer(
        model.parameters(),
        optimizer_name=config.ssl_optimizer,
        lr=config.ssl_lr,
        weight_decay=config.weight_decay,
    )
    rng = np.random.RandomState(config.random_state)

    train_batch = batch_to_device(
        wf=train_data.wf,
        acg=train_data.acg,
        scalars=train_data.scalars,
        wf_mask=train_data.wf_mask,
        acg_mask=train_data.acg_mask,
        scalar_mask=train_data.scalar_mask,
        context=None,
        context_mask=None,
        device=device,
    )
    val_batch = batch_to_device(
        wf=val_data.wf,
        acg=val_data.acg,
        scalars=val_data.scalars,
        wf_mask=val_data.wf_mask,
        acg_mask=val_data.acg_mask,
        scalar_mask=val_data.scalar_mask,
        context=None,
        context_mask=None,
        device=device,
    )

    best_state = _clone_state_dict(model.encoder)
    best_loss = float("inf")
    wait = 0
    history: list[dict[str, Any]] = []

    _vlog(
        config,
        f"ssl_pretrain: epochs={config.ssl_pretrain_epochs} train_rows={len(train_data)} val_rows={len(val_data)}",
    )
    for epoch in range(1, config.ssl_pretrain_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_idx in _batch_indices(len(train_data), config.batch_size, rng):
            idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
            wf_in = _mask_and_noise(
                train_batch.wf[idx],
                train_batch.wf_mask[idx],
                mask_prob=config.ssl_mask_prob,
                noise_std=config.ssl_noise_std,
            )
            acg_in = _mask_and_noise(
                train_batch.acg[idx],
                train_batch.acg_mask[idx],
                mask_prob=config.ssl_mask_prob,
                noise_std=config.ssl_noise_std,
            )
            scalar_in = _mask_and_noise(
                train_batch.scalars[idx],
                train_batch.scalar_mask[idx],
                mask_prob=config.ssl_mask_prob,
                noise_std=config.ssl_noise_std,
            )
            wf_pred, acg_pred, scalar_pred, _ = model(wf_in, acg_in, scalar_in)
            loss = (
                _masked_mse(wf_pred, train_batch.wf[idx], train_batch.wf_mask[idx])
                + _masked_mse(acg_pred, train_batch.acg[idx], train_batch.acg_mask[idx])
                + _masked_mse(scalar_pred, train_batch.scalars[idx], train_batch.scalar_mask[idx])
            ) / 3.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            wf_in = _mask_and_noise(
                val_batch.wf,
                val_batch.wf_mask,
                mask_prob=config.ssl_mask_prob,
                noise_std=config.ssl_noise_std,
            )
            acg_in = _mask_and_noise(
                val_batch.acg,
                val_batch.acg_mask,
                mask_prob=config.ssl_mask_prob,
                noise_std=config.ssl_noise_std,
            )
            scalar_in = _mask_and_noise(
                val_batch.scalars,
                val_batch.scalar_mask,
                mask_prob=config.ssl_mask_prob,
                noise_std=config.ssl_noise_std,
            )
            wf_pred, acg_pred, scalar_pred, _ = model(wf_in, acg_in, scalar_in)
            val_loss = float(
                (
                    _masked_mse(wf_pred, val_batch.wf, val_batch.wf_mask)
                    + _masked_mse(acg_pred, val_batch.acg, val_batch.acg_mask)
                    + _masked_mse(scalar_pred, val_batch.scalars, val_batch.scalar_mask)
                ).item()
                / 3.0
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_loss": val_loss,
            }
        )
        _vlog(
            config,
            f"ssl_pretrain epoch={epoch}/{config.ssl_pretrain_epochs} "
            f"train_loss={history[-1]['train_loss']:.6f} val_loss={val_loss:.6f}",
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = _clone_state_dict(model.encoder)
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                _vlog(config, f"ssl_pretrain early_stop epoch={epoch} patience={config.patience}")
                break

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)

    return SSLPretrainResult(
        encoder_state=best_state,
        history=history,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
    )


def run_supervised_experiment(
    *,
    arm: str,
    target_name: str,
    source_target_name: str,
    source_train_data: PreparedModelInput,
    source_train_target: np.ndarray,
    source_val_data: PreparedModelInput,
    source_val_target: np.ndarray,
    paired_train_splits: list[tuple[PreparedModelInput, np.ndarray, PreparedModelInput, np.ndarray]],
    paired_full_train_data: PreparedModelInput,
    paired_full_train_target: np.ndarray,
    paired_test_data: PreparedModelInput,
    paired_test_target: np.ndarray,
    output_checkpoint_dir: Path,
    model_stem: str,
    config: SSLDomainAdaptationConfig,
    pretrained_encoder_state: dict[str, Any] | None,
) -> SupervisedTrainResult:
    set_torch_runtime(config.random_state)
    device = get_device()

    model = _make_regressor(
        arm=arm,
        data=source_train_data,
        config=config,
        encoder_state=pretrained_encoder_state,
        device=device,
    )
    _vlog(
        config,
        f"{model_stem}: source_train arm={arm} epochs={config.source_train_epochs} "
        f"train_rows={len(source_train_data)} val_rows={len(source_val_data)}",
    )
    source_state, source_history, _ = _fit_regressor(
        model=model,
        arm=arm,
        train_data=source_train_data,
        train_target=source_train_target,
        val_data=source_val_data,
        val_target=source_val_target,
        epochs=config.source_train_epochs,
        lr=config.target_source_lr(target_name),
        batch_size=config.batch_size,
        patience=config.patience,
        freeze_encoder_epochs=config.source_warmup_epochs,
        random_state=config.random_state,
        device=device,
        optimizer_name=config.supervised_optimizer,
        weight_decay=config.weight_decay,
        loss_name=config.supervised_loss,
        verbose=config.verbose,
        log_prefix=f"{model_stem}:source",
    )

    source_checkpoint_path = output_checkpoint_dir / f"{model_stem}_source.pt"
    if config.save_checkpoints:
        source_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(source_state, source_checkpoint_path)

    selected_epochs: list[int] = []
    cv_history: list[dict[str, Any]] = []
    for split_idx, (train_data, train_target, val_data, val_target) in enumerate(paired_train_splits, start=1):
        _vlog(
            config,
            f"{model_stem}: paired_cv split={split_idx}/{len(paired_train_splits)} "
            f"epochs={config.finetune_epochs} train_rows={len(train_data)} val_rows={len(val_data)}",
        )
        cv_model = _make_regressor(
            arm=arm,
            data=train_data,
            config=config,
            encoder_state=None,
            device=device,
        )
        cv_model.load_state_dict(source_state)
        _, fold_history, best_epoch = _fit_regressor(
            model=cv_model,
            arm=arm,
            train_data=train_data,
            train_target=train_target,
            val_data=val_data,
            val_target=val_target,
            epochs=config.finetune_epochs,
            lr=config.target_finetune_lr(target_name),
            batch_size=config.batch_size,
            patience=config.patience,
            freeze_encoder_epochs=0,
            random_state=config.random_state + split_idx,
            device=device,
            optimizer_name=config.supervised_optimizer,
            weight_decay=config.weight_decay,
            loss_name=config.supervised_loss,
            verbose=config.verbose,
            log_prefix=f"{model_stem}:paired_cv[{split_idx}]",
        )
        selected_epochs.append(best_epoch)
        cv_history.extend(
            {
                "stage": "paired_cv",
                "split_id": split_idx,
                **row,
            }
            for row in fold_history
        )

    selected_finetune_epochs = max(1, int(round(float(np.median(selected_epochs))))) if selected_epochs else 1

    final_model = _make_regressor(
        arm=arm,
        data=paired_full_train_data,
        config=config,
        encoder_state=None,
        device=device,
    )
    final_model.load_state_dict(source_state)
    _vlog(
        config,
        f"{model_stem}: paired_final epochs={selected_finetune_epochs} train_rows={len(paired_full_train_data)}",
    )
    final_state, final_history, _ = _fit_regressor(
        model=final_model,
        arm=arm,
        train_data=paired_full_train_data,
        train_target=paired_full_train_target,
        val_data=None,
        val_target=None,
        epochs=selected_finetune_epochs,
        lr=config.target_finetune_lr(target_name),
        batch_size=config.batch_size,
        patience=config.patience,
        freeze_encoder_epochs=0,
        random_state=config.random_state + 500,
        device=device,
        optimizer_name=config.supervised_optimizer,
        weight_decay=config.weight_decay,
        loss_name=config.supervised_loss,
        verbose=config.verbose,
        log_prefix=f"{model_stem}:paired_final",
    )
    pred = _predict_regressor(final_model, paired_test_data, arm=arm, device=device)
    metrics = {
        "target": target_name,
        "arm": arm,
        "selected_finetune_epochs": int(selected_finetune_epochs),
        **score_predictions(paired_test_target, pred),
    }

    final_checkpoint_path = output_checkpoint_dir / f"{model_stem}_final.pt"
    if config.save_checkpoints:
        final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_state, final_checkpoint_path)

    predictions = pd.DataFrame(
        {
            "row_uid": paired_test_data.row_uid,
            "recording_key": paired_test_data.recording_key,
            "study_set": paired_test_data.study_set,
            "target": target_name,
            "arm": arm,
            "model_id": model_stem,
            "true": paired_test_target,
            "pred": pred,
            "abs_error": np.abs(pred - paired_test_target),
        }
    )
    history = [{"stage": "source", **row} for row in source_history] + cv_history + [
        {"stage": "paired_final", **row} for row in final_history
    ]
    return SupervisedTrainResult(
        metrics=metrics,
        predictions=predictions,
        history=history,
        checkpoint_path=None if not config.save_checkpoints else str(final_checkpoint_path),
        selected_finetune_epochs=selected_finetune_epochs,
        source_checkpoint_path=None if not config.save_checkpoints else str(source_checkpoint_path),
    )


@dataclass
class StageTrainResult:
    model_state: dict[str, Any]
    history: list[dict[str, Any]]
    checkpoint_path: str | None
    best_epoch: int


def _predict_model(
    model: torch.nn.Module,
    data: PreparedModelInput,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    batch = batch_to_device(
        wf=data.wf,
        acg=data.acg,
        scalars=data.scalars,
        wf_mask=data.wf_mask,
        acg_mask=data.acg_mask,
        scalar_mask=data.scalar_mask,
        context=None,
        context_mask=None,
        device=device,
    )
    with torch.no_grad():
        pred, _ = model(batch.wf, batch.acg, batch.scalars)
    return np.clip(target_inverse(pred.detach().cpu().numpy()), 0.0, 1.0)


def train_source_with_mmd(
    *,
    source_train_data: PreparedModelInput,
    source_train_target: np.ndarray,
    source_val_data: PreparedModelInput,
    source_val_target: np.ndarray,
    target_unlabeled_data: PreparedModelInput,
    mmd_lambda: float,
    config: MMDDomainAdaptationConfig,
    target_name: str,
    checkpoint_path: Path | None = None,
    progress_prefix: str = "mmd_source",
) -> StageTrainResult:
    set_torch_runtime(config.random_state)
    device = get_device()
    rng = np.random.RandomState(config.random_state)
    n_wf, n_acg, n_scalar, _ = _feature_dims(source_train_data)
    model = make_unit_only_regressor(
        n_wf_features=n_wf,
        n_acg_features=n_acg,
        n_scalar_features=n_scalar,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        encoder_state=None,
        device=device,
    )
    optimizer = _make_optimizer(
        model.parameters(),
        optimizer_name=config.optimizer_name,
        lr=config.source_lr,
        weight_decay=config.weight_decay,
    )
    source_train_batch = batch_to_device(
        wf=source_train_data.wf,
        acg=source_train_data.acg,
        scalars=source_train_data.scalars,
        wf_mask=source_train_data.wf_mask,
        acg_mask=source_train_data.acg_mask,
        scalar_mask=source_train_data.scalar_mask,
        context=None,
        context_mask=None,
        device=device,
    )
    source_val_batch = batch_to_device(
        wf=source_val_data.wf,
        acg=source_val_data.acg,
        scalars=source_val_data.scalars,
        wf_mask=source_val_data.wf_mask,
        acg_mask=source_val_data.acg_mask,
        scalar_mask=source_val_data.scalar_mask,
        context=None,
        context_mask=None,
        device=device,
    )
    target_unlabeled_batch = batch_to_device(
        wf=target_unlabeled_data.wf,
        acg=target_unlabeled_data.acg,
        scalars=target_unlabeled_data.scalars,
        wf_mask=target_unlabeled_data.wf_mask,
        acg_mask=target_unlabeled_data.acg_mask,
        scalar_mask=target_unlabeled_data.scalar_mask,
        context=None,
        context_mask=None,
        device=device,
    )
    source_train_target_t = torch.tensor(
        target_transform(source_train_target),
        dtype=torch.float32,
        device=device,
    )

    best_state = _clone_state_dict(model)
    best_epoch = 1
    best_val = float("inf")
    wait = 0
    history: list[dict[str, Any]] = []

    _vlog(
        config,
        f"{progress_prefix}: lambda={mmd_lambda:.4f} source_rows={len(source_train_data)} "
        f"target_unlabeled_rows={len(target_unlabeled_data)} epochs={config.source_train_epochs}",
    )
    for epoch in range(1, config.source_train_epochs + 1):
        model.train()
        task_losses: list[float] = []
        mmd_losses: list[float] = []
        total_losses: list[float] = []

        for batch_idx in _batch_indices(len(source_train_data), config.batch_size, rng):
            src_idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
            src_pred, src_latent = model(
                source_train_batch.wf[src_idx],
                source_train_batch.acg[src_idx],
                source_train_batch.scalars[src_idx],
            )
            task_loss = _regression_loss(
                src_pred,
                source_train_target_t[src_idx],
                loss_name=config.supervised_loss,
            )

            if len(target_unlabeled_data):
                tgt_idx_np = rng.randint(0, len(target_unlabeled_data), size=min(len(batch_idx), len(target_unlabeled_data)))
                tgt_idx = torch.tensor(tgt_idx_np, dtype=torch.long, device=device)
                _, tgt_latent = model(
                    target_unlabeled_batch.wf[tgt_idx],
                    target_unlabeled_batch.acg[tgt_idx],
                    target_unlabeled_batch.scalars[tgt_idx],
                )
                mmd_loss = rbf_multi_scale_mmd(
                    src_latent,
                    tgt_latent,
                    bandwidth_multipliers=config.mmd_bandwidth_multipliers,
                )
            else:
                mmd_loss = task_loss.new_tensor(0.0)

            loss = task_loss + (float(mmd_lambda) * mmd_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            task_losses.append(float(task_loss.item()))
            mmd_losses.append(float(mmd_loss.item()))
            total_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred_t, _ = model(
                source_val_batch.wf,
                source_val_batch.acg,
                source_val_batch.scalars,
            )
        val_pred = np.clip(target_inverse(val_pred_t.detach().cpu().numpy()), 0.0, 1.0)
        val_mae = score_predictions(source_val_target, val_pred)["mae"]

        row = {
            "epoch": epoch,
            "train_task_loss": float(np.mean(task_losses)) if task_losses else float("nan"),
            "train_mmd_loss": float(np.mean(mmd_losses)) if mmd_losses else float("nan"),
            "train_total_loss": float(np.mean(total_losses)) if total_losses else float("nan"),
            "val_mae": float(val_mae),
            "lambda": float(mmd_lambda),
        }
        history.append(row)
        _vlog(
            config,
            f"{progress_prefix} epoch={epoch}/{config.source_train_epochs} "
            f"task={row['train_task_loss']:.6f} mmd={row['train_mmd_loss']:.6f} "
            f"total={row['train_total_loss']:.6f} val_mae={val_mae:.6f}",
        )

        if val_mae < best_val:
            best_val = float(val_mae)
            best_epoch = epoch
            best_state = _clone_state_dict(model)
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)

    return StageTrainResult(
        model_state=best_state,
        history=history,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        best_epoch=best_epoch,
    )


def finetune_labeled_target(
    *,
    source_state: dict[str, Any],
    train_data: PreparedModelInput,
    train_target: np.ndarray,
    config: MMDDomainAdaptationConfig,
    target_name: str,
    random_state: int,
    checkpoint_path: Path | None = None,
    progress_prefix: str = "mmd_finetune",
) -> StageTrainResult:
    set_torch_runtime(random_state)
    device = get_device()
    rng = np.random.RandomState(random_state)
    n_wf, n_acg, n_scalar, _ = _feature_dims(train_data)
    model = make_unit_only_regressor(
        n_wf_features=n_wf,
        n_acg_features=n_acg,
        n_scalar_features=n_scalar,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        encoder_state=None,
        device=device,
    )
    model.load_state_dict(source_state)
    optimizer = _make_optimizer(
        model.parameters(),
        optimizer_name=config.optimizer_name,
        lr=config.finetune_lr,
        weight_decay=config.weight_decay,
    )
    train_batch = batch_to_device(
        wf=train_data.wf,
        acg=train_data.acg,
        scalars=train_data.scalars,
        wf_mask=train_data.wf_mask,
        acg_mask=train_data.acg_mask,
        scalar_mask=train_data.scalar_mask,
        context=None,
        context_mask=None,
        device=device,
    )
    train_target_t = torch.tensor(
        target_transform(train_target),
        dtype=torch.float32,
        device=device,
    )

    history: list[dict[str, Any]] = []
    state = _clone_state_dict(model)
    _vlog(
        config,
        f"{progress_prefix}: train_rows={len(train_data)} epochs={config.finetune_epochs}",
    )
    for epoch in range(1, config.finetune_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_idx in _batch_indices(len(train_data), config.batch_size, rng):
            idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
            pred, _ = model(
                train_batch.wf[idx],
                train_batch.acg[idx],
                train_batch.scalars[idx],
            )
            loss = _regression_loss(pred, train_target_t[idx], loss_name=config.supervised_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        state = _clone_state_dict(model)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
            }
        )
        _vlog(
            config,
            f"{progress_prefix} epoch={epoch}/{config.finetune_epochs} train_loss={history[-1]['train_loss']:.6f}",
        )

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint_path)

    return StageTrainResult(
        model_state=state,
        history=history,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        best_epoch=config.finetune_epochs,
    )


def evaluate_state(
    *,
    model_state: dict[str, Any],
    eval_data: PreparedModelInput,
    eval_target: np.ndarray,
    config: MMDDomainAdaptationConfig,
) -> tuple[dict[str, float], np.ndarray]:
    set_torch_runtime(config.random_state)
    device = get_device()
    n_wf, n_acg, n_scalar, _ = _feature_dims(eval_data)
    model = make_unit_only_regressor(
        n_wf_features=n_wf,
        n_acg_features=n_acg,
        n_scalar_features=n_scalar,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        encoder_state=None,
        device=device,
    )
    model.load_state_dict(model_state)
    pred = _predict_model(model, eval_data, device=device)
    return score_predictions(eval_target, pred), pred


def _filter_target_rows(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    return df.loc[mask].copy(), y.loc[mask].to_numpy(dtype=float)


def _frame_with_columns(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns + [c for c in frame.columns if c not in columns]]


_SSL_METRIC_COLUMNS = [
    "protocol",
    "family",
    "model_id",
    "source_target",
    "target",
    "arm",
    "selected_finetune_epochs",
    "mae",
    "rmse",
    "r2",
    "bias",
    "calibration_slope",
    "calibration_intercept",
]
_SSL_PREDICTION_COLUMNS = [
    "protocol",
    "family",
    "row_uid",
    "recording_key",
    "study_set",
    "model_id",
    "target",
    "true",
    "pred",
    "abs_error",
]
_SSL_STRESS_COLUMNS = ["held_out_study"] + _SSL_METRIC_COLUMNS
_SSL_HISTORY_COLUMNS = ["protocol", "model_id", "stage", "split_id", "epoch", "train_loss", "val_loss", "val_mae"]


def _ssl_make_lightgbm(config: SSLDomainAdaptationConfig) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    n_estimators=config.lightgbm_estimators,
                    learning_rate=config.lightgbm_learning_rate,
                    num_leaves=config.lightgbm_num_leaves,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=config.random_state,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )


def _ssl_prepare_source_train_val(
    df: pd.DataFrame,
    *,
    target: str,
    use_context: bool,
    prepared: PreparedFeatureTable,
    unit_scaler: Any,
    context_scaler: Any,
    config: SSLDomainAdaptationConfig,
) -> tuple[PreparedModelInput, np.ndarray, PreparedModelInput, np.ndarray]:
    target_df, _ = _filter_target_rows(df, target)
    splits = make_grouped_validation_splits(
        target_df,
        n_splits=1,
        test_size=config.source_val_fraction,
        random_state=config.random_state,
    )
    if not splits:
        raise RuntimeError(f"Unable to make grouped source validation split for target={target}")
    train_df, val_df = splits[0]
    train_y = pd.to_numeric(train_df[target], errors="coerce").to_numpy(dtype=float)
    val_y = pd.to_numeric(val_df[target], errors="coerce").to_numpy(dtype=float)
    return (
        prepare_model_input(train_df, prepared, unit_scaler, context_scaler, use_context=use_context),
        train_y,
        prepare_model_input(val_df, prepared, unit_scaler, context_scaler, use_context=use_context),
        val_y,
    )


def _ssl_prepare_paired_cv_splits(
    df: pd.DataFrame,
    *,
    target: str,
    use_context: bool,
    prepared: PreparedFeatureTable,
    unit_scaler: Any,
    context_scaler: Any,
    config: SSLDomainAdaptationConfig,
) -> list[tuple[PreparedModelInput, np.ndarray, PreparedModelInput, np.ndarray]]:
    target_df, _ = _filter_target_rows(df, target)
    folds = make_grouped_validation_splits(
        target_df,
        n_splits=config.paired_finetune_val_splits,
        test_size=config.paired_finetune_val_fraction,
        random_state=config.random_state + 200,
    )
    prepared_folds: list[tuple[PreparedModelInput, np.ndarray, PreparedModelInput, np.ndarray]] = []
    for train_df, val_df in folds:
        train_y = pd.to_numeric(train_df[target], errors="coerce").to_numpy(dtype=float)
        val_y = pd.to_numeric(val_df[target], errors="coerce").to_numpy(dtype=float)
        prepared_folds.append(
            (
                prepare_model_input(train_df, prepared, unit_scaler, context_scaler, use_context=use_context),
                train_y,
                prepare_model_input(val_df, prepared, unit_scaler, context_scaler, use_context=use_context),
                val_y,
            )
        )
    return prepared_folds


def _ssl_run_lightgbm_baseline(
    *,
    model_id: str,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    protocol_name: str,
    config: SSLDomainAdaptationConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_t, y_train = _filter_target_rows(train_df, target)
    test_t, y_test = _filter_target_rows(test_df, target)
    model = _ssl_make_lightgbm(config)
    model.fit(train_t[feature_cols], y_train)
    pred = np.clip(model.predict(test_t[feature_cols]), 0.0, 1.0)
    metrics = {
        "protocol": protocol_name,
        "family": "baseline",
        "model_id": model_id,
        "target": target,
        **score_predictions(y_test, pred),
    }
    predictions = pd.DataFrame(
        {
            "protocol": protocol_name,
            "row_uid": test_t["row_uid"].astype(str).to_numpy(),
            "recording_key": test_t["recording_key"].astype(str).to_numpy(),
            "study_set": test_t["study_set"].astype(str).to_numpy(),
            "model_id": model_id,
            "target": target,
            "true": y_test,
            "pred": pred,
            "abs_error": np.abs(pred - y_test),
        }
    )
    return metrics, predictions


def _run_ssl_single_protocol(
    *,
    protocol_name: str,
    split: MainExperimentSplit | StressStudySplit,
    prepared: PreparedFeatureTable,
    config: SSLDomainAdaptationConfig,
    output_dir: Path,
    include_baselines: bool,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]], dict[str, Any]]:
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []

    train_side_df = pd.concat([split.hybrid_source, split.paired_non_test_all], ignore_index=True)
    unit_scaler, _ = fit_training_side_scalers(train_side_df, prepared)
    ssl_splits = make_grouped_validation_splits(
        split.ssl_pool,
        n_splits=1,
        test_size=config.ssl_val_recording_frac,
        random_state=config.random_state + 100,
    )
    if not ssl_splits:
        raise RuntimeError(f"Unable to build SSL train/validation split for protocol={protocol_name}")
    ssl_train_df, ssl_val_df = ssl_splits[0]
    ssl_train_input = prepare_model_input(ssl_train_df, prepared, unit_scaler, None, use_context=False)
    ssl_val_input = prepare_model_input(ssl_val_df, prepared, unit_scaler, None, use_context=False)

    ssl_checkpoint = output_dir / "checkpoints" / f"{protocol_name}_ssl_encoder.pt"
    ssl_result = pretrain_ssl_autoencoder(
        train_data=ssl_train_input,
        val_data=ssl_val_input,
        config=config,
        checkpoint_path=ssl_checkpoint if config.save_checkpoints else None,
    )
    history_rows.extend({"protocol": protocol_name, "model_id": "ssl_pretrain", **row} for row in ssl_result.history)

    for target in config.targets:
        source_target = target_name_for_source(target, config.source_fmiss_target)
        target_context_cols = config.target_context_columns(target, prepared.context_feature_cols_by_target)
        _, context_scaler = fit_training_side_scalers(train_side_df, prepared, context_columns=target_context_cols)

        source_train_unit, y_source_train, source_val_unit, y_source_val = _ssl_prepare_source_train_val(
            split.hybrid_source,
            target=source_target,
            use_context=False,
            prepared=prepared,
            unit_scaler=unit_scaler,
            context_scaler=context_scaler,
            config=config,
        )
        paired_full_df, paired_full_target = _filter_target_rows(split.paired_finetune, target)
        paired_test_df, paired_test_target = _filter_target_rows(split.paired_test, target)
        paired_full_unit = prepare_model_input(paired_full_df, prepared, unit_scaler, context_scaler, use_context=False)
        paired_test_unit = prepare_model_input(paired_test_df, prepared, unit_scaler, context_scaler, use_context=False)
        paired_cv_unit = _ssl_prepare_paired_cv_splits(
            split.paired_finetune,
            target=target,
            use_context=False,
            prepared=prepared,
            unit_scaler=unit_scaler,
            context_scaler=context_scaler,
            config=config,
        )

        if config.run_unit_only_arm:
            ssl_unit = run_supervised_experiment(
                arm="unit_only",
                target_name=target,
                source_target_name=source_target,
                source_train_data=source_train_unit,
                source_train_target=y_source_train,
                source_val_data=source_val_unit,
                source_val_target=y_source_val,
                paired_train_splits=paired_cv_unit,
                paired_full_train_data=paired_full_unit,
                paired_full_train_target=paired_full_target,
                paired_test_data=paired_test_unit,
                paired_test_target=paired_test_target,
                output_checkpoint_dir=output_dir / "checkpoints",
                model_stem=f"{protocol_name}_ssl_{target}_unit_only",
                config=config,
                pretrained_encoder_state=ssl_result.encoder_state,
            )
            metrics_rows.append({"protocol": protocol_name, "family": "ssl", "model_id": f"ssl_{target}_unit_only", "source_target": source_target, **ssl_unit.metrics})
            prediction_frames.append(ssl_unit.predictions.assign(protocol=protocol_name, family="ssl"))
            history_rows.extend({"protocol": protocol_name, "model_id": f"ssl_{target}_unit_only", **row} for row in ssl_unit.history)

            if include_baselines:
                no_ssl_unit = run_supervised_experiment(
                    arm="unit_only",
                    target_name=target,
                    source_target_name=source_target,
                    source_train_data=source_train_unit,
                    source_train_target=y_source_train,
                    source_val_data=source_val_unit,
                    source_val_target=y_source_val,
                    paired_train_splits=paired_cv_unit,
                    paired_full_train_data=paired_full_unit,
                    paired_full_train_target=paired_full_target,
                    paired_test_data=paired_test_unit,
                    paired_test_target=paired_test_target,
                    output_checkpoint_dir=output_dir / "checkpoints",
                    model_stem=f"{protocol_name}_neural_no_ssl_{target}_unit_only",
                    config=config,
                    pretrained_encoder_state=None,
                )
                metrics_rows.append({"protocol": protocol_name, "family": "baseline", "model_id": f"neural_no_ssl_{target}_unit_only", "source_target": source_target, **no_ssl_unit.metrics})
                prediction_frames.append(no_ssl_unit.predictions.assign(protocol=protocol_name, family="baseline"))
                history_rows.extend({"protocol": protocol_name, "model_id": f"neural_no_ssl_{target}_unit_only", **row} for row in no_ssl_unit.history)

        if config.run_contextual_arm:
            source_train_ctx, _, source_val_ctx, _ = _ssl_prepare_source_train_val(
                split.hybrid_source,
                target=source_target,
                use_context=True,
                prepared=prepared,
                unit_scaler=unit_scaler,
                context_scaler=context_scaler,
                config=config,
            )
            paired_full_ctx = prepare_model_input(paired_full_df, prepared, unit_scaler, context_scaler, use_context=True)
            paired_test_ctx = prepare_model_input(paired_test_df, prepared, unit_scaler, context_scaler, use_context=True)
            paired_cv_ctx = _ssl_prepare_paired_cv_splits(
                split.paired_finetune,
                target=target,
                use_context=True,
                prepared=prepared,
                unit_scaler=unit_scaler,
                context_scaler=context_scaler,
                config=config,
            )
            ssl_contextual = run_supervised_experiment(
                arm="contextual",
                target_name=target,
                source_target_name=source_target,
                source_train_data=source_train_ctx,
                source_train_target=y_source_train,
                source_val_data=source_val_ctx,
                source_val_target=y_source_val,
                paired_train_splits=paired_cv_ctx,
                paired_full_train_data=paired_full_ctx,
                paired_full_train_target=paired_full_target,
                paired_test_data=paired_test_ctx,
                paired_test_target=paired_test_target,
                output_checkpoint_dir=output_dir / "checkpoints",
                model_stem=f"{protocol_name}_ssl_{target}_contextual",
                config=config,
                pretrained_encoder_state=ssl_result.encoder_state,
            )
            metrics_rows.append({"protocol": protocol_name, "family": "ssl", "model_id": f"ssl_{target}_contextual", "source_target": source_target, **ssl_contextual.metrics})
            prediction_frames.append(ssl_contextual.predictions.assign(protocol=protocol_name, family="ssl"))
            history_rows.extend({"protocol": protocol_name, "model_id": f"ssl_{target}_contextual", **row} for row in ssl_contextual.history)

            if include_baselines:
                no_ssl_contextual = run_supervised_experiment(
                    arm="contextual",
                    target_name=target,
                    source_target_name=source_target,
                    source_train_data=source_train_ctx,
                    source_train_target=y_source_train,
                    source_val_data=source_val_ctx,
                    source_val_target=y_source_val,
                    paired_train_splits=paired_cv_ctx,
                    paired_full_train_data=paired_full_ctx,
                    paired_full_train_target=paired_full_target,
                    paired_test_data=paired_test_ctx,
                    paired_test_target=paired_test_target,
                    output_checkpoint_dir=output_dir / "checkpoints",
                    model_stem=f"{protocol_name}_neural_no_ssl_{target}_contextual",
                    config=config,
                    pretrained_encoder_state=None,
                )
                metrics_rows.append({"protocol": protocol_name, "family": "baseline", "model_id": f"neural_no_ssl_{target}_contextual", "source_target": source_target, **no_ssl_contextual.metrics})
                prediction_frames.append(no_ssl_contextual.predictions.assign(protocol=protocol_name, family="baseline"))
                history_rows.extend({"protocol": protocol_name, "model_id": f"neural_no_ssl_{target}_contextual", **row} for row in no_ssl_contextual.history)

        if include_baselines:
            paired_shape_metrics, paired_shape_pred = _ssl_run_lightgbm_baseline(
                model_id="paired_only_lightgbm_shape",
                feature_cols=prepared.shape_feature_cols,
                train_df=split.paired_finetune,
                test_df=split.paired_test,
                target=target,
                protocol_name=protocol_name,
                config=config,
            )
            metrics_rows.append(paired_shape_metrics)
            prediction_frames.append(paired_shape_pred.assign(family="baseline"))

            contextual_lgbm_metrics, contextual_lgbm_pred = _ssl_run_lightgbm_baseline(
                model_id="contextual_lightgbm_no_study_id",
                feature_cols=prepared.lightgbm_context_cols,
                train_df=split.paired_finetune,
                test_df=split.paired_test,
                target=target,
                protocol_name=protocol_name,
                config=config,
            )
            metrics_rows.append(contextual_lgbm_metrics)
            prediction_frames.append(contextual_lgbm_pred.assign(family="baseline"))

    split_manifest = {
        **split.manifest,
        "training_side_scaler_rows": int(len(train_side_df)),
        "context_feature_count": int(len(prepared.context_feature_cols)),
        "context_feature_count_by_target": {target: int(len(prepared.context_feature_cols_by_target.get(target, ()))) for target in config.targets},
        "unit_feature_count": int(len(prepared.unit_feature_cols)),
        "shape_feature_count": int(len(prepared.shape_feature_cols)),
    }
    return metrics_rows, prediction_frames, history_rows, split_manifest


def _build_ssl_readme(
    *,
    config: SSLDomainAdaptationConfig,
    main_manifest: dict[str, Any],
    metrics_df: pd.DataFrame,
    stress_df: pd.DataFrame,
) -> str:
    lines = [
        "# SSL + Context Domain Adaptation Run",
        "",
        "## Config",
        f"- parquet_path: `{config.parquet_path}`",
        f"- main_protocol: `{config.main_protocol}`",
        f"- stress_protocol: `{config.stress_protocol}`",
        f"- targets: `{', '.join(config.targets)}`",
        f"- ssl_objective: `{config.ssl_objective}`",
        f"- allow_test_time_unlabeled_context: `{config.allow_test_time_unlabeled_context}`",
        "",
        "## Main Split",
        f"- paired_test_rows: `{main_manifest['paired_test_rows']}`",
        f"- paired_finetune_rows: `{main_manifest['paired_finetune_rows']}`",
        f"- ssl_pool_rows: `{main_manifest['ssl_pool_rows']}`",
        f"- paired_test_recordings: `{main_manifest['paired_test_recordings']}`",
        "",
        "## Main Results",
    ]
    if metrics_df.empty:
        lines.append("- No metrics were produced.")
    else:
        for _, row in metrics_df.sort_values(["target", "mae", "model_id"]).iterrows():
            lines.append(f"- {row['target']} | {row['model_id']} | MAE `{row['mae']:.4f}` | R2 `{row['r2']:.4f}`")
    lines.extend(["", "## Stress Test"])
    if stress_df.empty:
        lines.append("- Stress-test skipped or produced no results.")
    else:
        for _, row in stress_df.sort_values(["held_out_study", "target", "mae"]).iterrows():
            lines.append(f"- {row['held_out_study']} | {row['target']} | {row['model_id']} | MAE `{row['mae']:.4f}` | R2 `{row['r2']:.4f}`")
    return "\n".join(lines) + "\n"


def run_ssl_with_config(config: SSLDomainAdaptationConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_feature_table(config.parquet_path)
    main_split = make_main_split(prepared.df, paired_final_holdout_rows=config.paired_final_holdout_rows, random_state=config.random_state)
    metrics_rows, prediction_frames, history_rows, main_manifest = _run_ssl_single_protocol(
        protocol_name="main",
        split=main_split,
        prepared=prepared,
        config=config,
        output_dir=output_dir,
        include_baselines=True,
    )

    stress_rows: list[dict[str, Any]] = []
    stress_manifests: list[dict[str, Any]] = []
    if config.run_stress_test:
        for stress_split in build_stress_splits(prepared.df, max_stress_studies=config.max_stress_studies):
            split_metrics, split_predictions, split_history, split_manifest = _run_ssl_single_protocol(
                protocol_name=f"stress:{stress_split.held_out_study}",
                split=stress_split,
                prepared=prepared,
                config=config,
                output_dir=output_dir / "stress" / stress_split.held_out_study.lower(),
                include_baselines=False,
            )
            prediction_frames.extend(split_predictions)
            history_rows.extend(split_history)
            stress_manifests.append(split_manifest)
            for row in split_metrics:
                stress_rows.append({"held_out_study": stress_split.held_out_study, **row})

    metrics_df = _frame_with_columns(metrics_rows, _SSL_METRIC_COLUMNS)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["target", "mae", "model_id"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(["protocol", "target", "model_id", "row_uid"]) if prediction_frames else pd.DataFrame(columns=_SSL_PREDICTION_COLUMNS)
    stress_df = _frame_with_columns(stress_rows, _SSL_STRESS_COLUMNS)
    if not stress_df.empty:
        stress_df = stress_df.sort_values(["held_out_study", "target", "mae", "model_id"]).reset_index(drop=True)
    baseline_df = metrics_df[metrics_df["family"] == "baseline"].copy() if "family" in metrics_df.columns else pd.DataFrame(columns=_SSL_METRIC_COLUMNS)
    history_df = _frame_with_columns(history_rows, _SSL_HISTORY_COLUMNS)
    split_manifest = {"config": config.to_dict(), "main": main_manifest, "stress": stress_manifests}

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, sort_keys=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    baseline_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    stress_df.to_csv(output_dir / "stress_test_results.csv", index=False)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "README_run.md").write_text(
        _build_ssl_readme(config=config, main_manifest=main_manifest, metrics_df=metrics_df, stress_df=stress_df),
        encoding="utf-8",
    )
    return output_dir


_MMD_METRIC_COLUMNS = [
    "protocol",
    "target",
    "variant_id",
    "family",
    "model_id",
    "source_target",
    "eval_stage",
    "lambda",
    "selected_lambda",
    "mae",
    "rmse",
    "r2",
    "bias",
    "calibration_slope",
    "calibration_intercept",
]
_MMD_PREDICTION_COLUMNS = [
    "protocol",
    "target",
    "variant_id",
    "family",
    "eval_stage",
    "lambda",
    "row_uid",
    "recording_key",
    "study_set",
    "true",
    "pred",
    "abs_error",
]
_MMD_LAMBDA_SWEEP_COLUMNS = [
    "target",
    "eval_stage",
    "lambda",
    "cv_mae_mean",
    "cv_rmse_mean",
    "cv_r2_mean",
    "cv_bias_mean",
    "cv_calibration_slope_mean",
    "cv_calibration_intercept_mean",
    "n_splits",
]
_MMD_HISTORY_COLUMNS = [
    "protocol",
    "target",
    "variant_id",
    "eval_stage",
    "lambda",
    "stage",
    "split_id",
    "epoch",
    "train_task_loss",
    "train_mmd_loss",
    "train_total_loss",
    "train_loss",
    "val_mae",
]


def _mmd_make_lightgbm(config: MMDDomainAdaptationConfig) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    n_estimators=config.lightgbm_estimators,
                    learning_rate=config.lightgbm_learning_rate,
                    num_leaves=config.lightgbm_num_leaves,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=config.random_state,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )


def _mmd_prepare_source_train_val_frames(
    hybrid_df: pd.DataFrame,
    *,
    source_target: str,
    random_state: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_df, _ = _filter_target_rows(hybrid_df, source_target)
    splits = make_grouped_validation_splits(target_df, n_splits=1, test_size=val_fraction, random_state=random_state)
    if not splits:
        raise RuntimeError(f"Unable to make grouped source validation split for target={source_target}")
    return splits[0]


def _mmd_prepare_unit_input(df: pd.DataFrame, prepared: PreparedFeatureTable, unit_scaler: Any) -> PreparedModelInput:
    return prepare_model_input(df, prepared, unit_scaler, None, use_context=False)


def _mmd_run_lightgbm_baseline(
    *,
    model_id: str,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    config: MMDDomainAdaptationConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_t, y_train = _filter_target_rows(train_df, target)
    test_t, y_test = _filter_target_rows(test_df, target)
    model = _mmd_make_lightgbm(config)
    model.fit(train_t[feature_cols], y_train)
    pred = np.clip(model.predict(test_t[feature_cols]), 0.0, 1.0)
    metrics = {
        "protocol": "main",
        "target": target,
        "variant_id": model_id,
        "family": "baseline",
        "model_id": model_id,
        "source_target": np.nan,
        "eval_stage": "holdout",
        "lambda": np.nan,
        "selected_lambda": np.nan,
        **score_predictions(y_test, pred),
    }
    predictions = pd.DataFrame(
        {
            "protocol": "main",
            "target": target,
            "variant_id": model_id,
            "family": "baseline",
            "eval_stage": "holdout",
            "lambda": np.nan,
            "row_uid": test_t["row_uid"].astype(str).to_numpy(),
            "recording_key": test_t["recording_key"].astype(str).to_numpy(),
            "study_set": test_t["study_set"].astype(str).to_numpy(),
            "true": y_test,
            "pred": pred,
            "abs_error": np.abs(pred - y_test),
        }
    )
    return metrics, predictions


def _summarize_cv_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    return {
        "cv_mae_mean": float(frame["mae"].mean()),
        "cv_rmse_mean": float(frame["rmse"].mean()),
        "cv_r2_mean": float(frame["r2"].mean()),
        "cv_bias_mean": float(frame["bias"].mean()),
        "cv_calibration_slope_mean": float(frame["calibration_slope"].mean()),
        "cv_calibration_intercept_mean": float(frame["calibration_intercept"].mean()),
        "n_splits": int(len(frame)),
    }


def _select_best_positive_lambda(lambda_rows: list[dict[str, Any]]) -> float:
    frame = pd.DataFrame(lambda_rows)
    positive = frame[frame["lambda"] > 0.0].copy()
    if positive.empty:
        return 0.0
    positive = positive.sort_values(["cv_mae_mean", "lambda"]).reset_index(drop=True)
    return float(positive.iloc[0]["lambda"])


def _append_prediction_frame(
    pred_rows: list[pd.DataFrame],
    *,
    variant_id: str,
    family: str,
    target: str,
    eval_stage: str,
    selected_lambda: float | None,
    prepared_df: pd.DataFrame,
    y_true: np.ndarray,
    pred: np.ndarray,
) -> None:
    pred_rows.append(
        pd.DataFrame(
            {
                "protocol": "main",
                "target": target,
                "variant_id": variant_id,
                "family": family,
                "eval_stage": eval_stage,
                "lambda": selected_lambda,
                "row_uid": prepared_df["row_uid"].astype(str).to_numpy(),
                "recording_key": prepared_df["recording_key"].astype(str).to_numpy(),
                "study_set": prepared_df["study_set"].astype(str).to_numpy(),
                "true": y_true,
                "pred": pred,
                "abs_error": np.abs(pred - y_true),
            }
        )
    )


def _run_mmd_target(
    *,
    target: str,
    split: MainExperimentSplit,
    prepared: PreparedFeatureTable,
    config: MMDDomainAdaptationConfig,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    source_target = target_name_for_source(target, config.source_fmiss_target)
    target_train_df, target_train_y = _filter_target_rows(split.paired_finetune, target)
    target_test_df, target_test_y = _filter_target_rows(split.paired_test, target)
    if target_train_df.empty or target_test_df.empty:
        raise RuntimeError(f"Target {target} does not have enough paired labeled rows")

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []

    source_train_df, source_val_df = _mmd_prepare_source_train_val_frames(
        split.hybrid_source,
        source_target=source_target,
        random_state=config.random_state,
        val_fraction=config.source_val_fraction,
    )
    paired_cv_splits = make_grouped_validation_splits(
        target_train_df,
        n_splits=config.paired_finetune_val_splits,
        test_size=config.paired_finetune_val_fraction,
        random_state=config.random_state + 200,
    )
    if not paired_cv_splits:
        raise RuntimeError(f"Unable to build paired validation splits for target={target}")

    zero_shot_lambda_rows: list[dict[str, Any]] = []
    finetune_lambda_rows: list[dict[str, Any]] = []
    for mmd_lambda in config.lambda_grid:
        zero_metrics: list[dict[str, Any]] = []
        finetune_metrics: list[dict[str, Any]] = []
        for split_id, (fold_train_df, fold_val_df) in enumerate(paired_cv_splits, start=1):
            held_out_recordings = set(fold_val_df["recording_key"].astype(str))
            paired_non_val_all = split.paired_non_test_all[~split.paired_non_test_all["recording_key"].astype(str).isin(held_out_recordings)].copy()
            unlabeled_df = paired_non_val_all[paired_non_val_all["matched_gt_unit_id"].isna()].copy()
            if unlabeled_df.empty:
                unlabeled_df = split.ssl_pool.copy()

            scaler_fit_df = pd.concat([split.hybrid_source, paired_non_val_all], ignore_index=True)
            unit_scaler, _ = fit_training_side_scalers(scaler_fit_df, prepared)
            source_train_input = _mmd_prepare_unit_input(source_train_df, prepared, unit_scaler)
            source_val_input = _mmd_prepare_unit_input(source_val_df, prepared, unit_scaler)
            unlabeled_input = _mmd_prepare_unit_input(unlabeled_df, prepared, unit_scaler)
            fold_val_input = _mmd_prepare_unit_input(fold_val_df, prepared, unit_scaler)
            fold_train_input = _mmd_prepare_unit_input(fold_train_df, prepared, unit_scaler)
            source_train_target = pd.to_numeric(source_train_df[source_target], errors="coerce").to_numpy(dtype=float)
            source_val_target = pd.to_numeric(source_val_df[source_target], errors="coerce").to_numpy(dtype=float)
            fold_train_target = pd.to_numeric(fold_train_df[target], errors="coerce").to_numpy(dtype=float)
            fold_val_target = pd.to_numeric(fold_val_df[target], errors="coerce").to_numpy(dtype=float)

            source_result = train_source_with_mmd(
                source_train_data=source_train_input,
                source_train_target=source_train_target,
                source_val_data=source_val_input,
                source_val_target=source_val_target,
                target_unlabeled_data=unlabeled_input,
                mmd_lambda=float(mmd_lambda),
                config=config,
                target_name=target,
                checkpoint_path=None,
                progress_prefix=f"{target}:lambda={mmd_lambda:.2f}:cv{split_id}:source",
            )
            history_rows.extend({"protocol": "main", "target": target, "variant_id": "lambda_sweep", "eval_stage": "selection_zero_shot", "lambda": float(mmd_lambda), "stage": "source", "split_id": split_id, **row} for row in source_result.history)
            zero_metrics.append(evaluate_state(model_state=source_result.model_state, eval_data=fold_val_input, eval_target=fold_val_target, config=config)[0])

            if config.run_finetune_stage:
                finetune_result = finetune_labeled_target(
                    source_state=source_result.model_state,
                    train_data=fold_train_input,
                    train_target=fold_train_target,
                    config=config,
                    target_name=target,
                    random_state=config.random_state + split_id,
                    checkpoint_path=None,
                    progress_prefix=f"{target}:lambda={mmd_lambda:.2f}:cv{split_id}:finetune",
                )
                history_rows.extend({"protocol": "main", "target": target, "variant_id": "lambda_sweep", "eval_stage": "selection_finetuned", "lambda": float(mmd_lambda), "stage": "finetune", "split_id": split_id, **row} for row in finetune_result.history)
                finetune_metrics.append(evaluate_state(model_state=finetune_result.model_state, eval_data=fold_val_input, eval_target=fold_val_target, config=config)[0])

        zero_row = {"target": target, "eval_stage": "zero_shot", "lambda": float(mmd_lambda), **_summarize_cv_metrics(zero_metrics)}
        lambda_rows.append(zero_row)
        zero_shot_lambda_rows.append(zero_row)
        if config.run_finetune_stage:
            finetune_row = {"target": target, "eval_stage": "finetuned", "lambda": float(mmd_lambda), **_summarize_cv_metrics(finetune_metrics)}
            lambda_rows.append(finetune_row)
            finetune_lambda_rows.append(finetune_row)

    selected_zero_lambda = _select_best_positive_lambda(zero_shot_lambda_rows)
    selected_finetune_lambda = _select_best_positive_lambda(finetune_lambda_rows) if config.run_finetune_stage else 0.0

    train_side_df = pd.concat([split.hybrid_source, split.paired_non_test_all], ignore_index=True)
    unit_scaler, _ = fit_training_side_scalers(train_side_df, prepared)
    source_train_input = _mmd_prepare_unit_input(source_train_df, prepared, unit_scaler)
    source_val_input = _mmd_prepare_unit_input(source_val_df, prepared, unit_scaler)
    unlabeled_input = _mmd_prepare_unit_input(split.ssl_pool, prepared, unit_scaler)
    paired_train_input = _mmd_prepare_unit_input(target_train_df, prepared, unit_scaler)
    paired_test_input = _mmd_prepare_unit_input(target_test_df, prepared, unit_scaler)
    source_train_target = pd.to_numeric(source_train_df[source_target], errors="coerce").to_numpy(dtype=float)
    source_val_target = pd.to_numeric(source_val_df[source_target], errors="coerce").to_numpy(dtype=float)

    final_variants = [("neural_no_mmd_zero_shot", 0.0, "zero_shot"), ("neural_mmd_zero_shot", selected_zero_lambda, "zero_shot")]
    if config.run_finetune_stage:
        final_variants.extend([("neural_no_mmd_finetuned", 0.0, "finetuned"), ("neural_mmd_finetuned", selected_finetune_lambda, "finetuned")])

    for variant_id, mmd_lambda, eval_stage in final_variants:
        source_result = train_source_with_mmd(
            source_train_data=source_train_input,
            source_train_target=source_train_target,
            source_val_data=source_val_input,
            source_val_target=source_val_target,
            target_unlabeled_data=unlabeled_input,
            mmd_lambda=float(mmd_lambda),
            config=config,
            target_name=target,
            checkpoint_path=(output_dir / "checkpoints" / f"{variant_id}_{target}_source.pt") if config.save_checkpoints else None,
            progress_prefix=f"{variant_id}:{target}:source",
        )
        history_rows.extend({"protocol": "main", "target": target, "variant_id": variant_id, "eval_stage": eval_stage, "lambda": float(mmd_lambda), "stage": "source", "split_id": pd.NA, **row} for row in source_result.history)
        final_state = source_result.model_state

        if eval_stage == "finetuned":
            finetune_result = finetune_labeled_target(
                source_state=source_result.model_state,
                train_data=paired_train_input,
                train_target=target_train_y,
                config=config,
                target_name=target,
                random_state=config.random_state + 500,
                checkpoint_path=(output_dir / "checkpoints" / f"{variant_id}_{target}_final.pt") if config.save_checkpoints else None,
                progress_prefix=f"{variant_id}:{target}:finetune",
            )
            history_rows.extend({"protocol": "main", "target": target, "variant_id": variant_id, "eval_stage": eval_stage, "lambda": float(mmd_lambda), "stage": "finetune", "split_id": pd.NA, **row} for row in finetune_result.history)
            final_state = finetune_result.model_state

        metrics, pred = evaluate_state(model_state=final_state, eval_data=paired_test_input, eval_target=target_test_y, config=config)
        metric_rows.append(
            {
                "protocol": "main",
                "target": target,
                "variant_id": variant_id,
                "family": "neural_mmd" if "mmd" in variant_id and "no_mmd" not in variant_id else "neural_no_mmd",
                "model_id": variant_id,
                "source_target": source_target,
                "eval_stage": "holdout",
                "lambda": float(mmd_lambda),
                "selected_lambda": float(mmd_lambda),
                **metrics,
            }
        )
        _append_prediction_frame(prediction_frames, variant_id=variant_id, family="neural", target=target, eval_stage="holdout", selected_lambda=float(mmd_lambda), prepared_df=target_test_df, y_true=target_test_y, pred=pred)

    paired_shape_metrics, paired_shape_pred = _mmd_run_lightgbm_baseline(
        model_id="paired_only_lightgbm_shape",
        feature_cols=prepared.shape_feature_cols,
        train_df=target_train_df,
        test_df=target_test_df,
        target=target,
        config=config,
    )
    metric_rows.append(paired_shape_metrics)
    prediction_frames.append(paired_shape_pred)

    contextual_metrics, contextual_pred = _mmd_run_lightgbm_baseline(
        model_id="contextual_lightgbm_no_study_id",
        feature_cols=prepared.lightgbm_context_cols,
        train_df=target_train_df,
        test_df=target_test_df,
        target=target,
        config=config,
    )
    metric_rows.append(contextual_metrics)
    prediction_frames.append(contextual_pred)
    return metric_rows, prediction_frames, history_rows, lambda_rows


def _build_mmd_readme(
    *,
    config: MMDDomainAdaptationConfig,
    split_manifest: dict[str, Any],
    metrics_df: pd.DataFrame,
    lambda_df: pd.DataFrame,
) -> str:
    lines = [
        "# MMD Domain Adaptation Run",
        "",
        "## Config",
        f"- parquet_path: `{config.parquet_path}`",
        f"- targets: `{', '.join(config.targets)}`",
        f"- lambda_grid: `{', '.join(str(x) for x in config.lambda_grid)}`",
        f"- run_finetune_stage: `{config.run_finetune_stage}`",
        "",
        "## Main Split",
        f"- paired_test_rows: `{split_manifest['paired_test_rows']}`",
        f"- paired_finetune_rows: `{split_manifest['paired_finetune_rows']}`",
        f"- ssl_pool_rows: `{split_manifest['ssl_pool_rows']}`",
        f"- paired_test_recordings: `{split_manifest['paired_test_recordings']}`",
        "",
        "## Holdout Results",
    ]
    if metrics_df.empty:
        lines.append("- No metrics were produced.")
    else:
        for _, row in metrics_df.sort_values(["target", "mae", "variant_id"]).iterrows():
            lines.append(f"- {row['target']} | {row['variant_id']} | MAE `{row['mae']:.4f}` | R2 `{row['r2']:.4f}` | lambda `{row['selected_lambda']}`")
    lines.extend(["", "## Lambda Sweep"])
    if lambda_df.empty:
        lines.append("- No lambda sweep rows were produced.")
    else:
        for _, row in lambda_df.sort_values(["target", "eval_stage", "cv_mae_mean", "lambda"]).iterrows():
            lines.append(f"- {row['target']} | {row['eval_stage']} | lambda `{row['lambda']}` | CV MAE `{row['cv_mae_mean']:.4f}`")
    return "\n".join(lines) + "\n"


def run_mmd_with_config(config: MMDDomainAdaptationConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_feature_table(config.parquet_path)
    split = make_main_split(prepared.df, paired_final_holdout_rows=config.paired_final_holdout_rows, random_state=config.random_state)

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    for target in config.targets:
        target_metrics, target_preds, target_history, target_lambdas = _run_mmd_target(
            target=target,
            split=split,
            prepared=prepared,
            config=config,
            output_dir=output_dir,
        )
        metrics_rows.extend(target_metrics)
        prediction_frames.extend(target_preds)
        history_rows.extend(target_history)
        lambda_rows.extend(target_lambdas)

    split_manifest = {
        **split.manifest,
        "training_side_scaler_rows": int(len(pd.concat([split.hybrid_source, split.paired_non_test_all], ignore_index=True))),
        "unit_feature_count": int(len(prepared.unit_feature_cols)),
        "shape_feature_count": int(len(prepared.shape_feature_cols)),
        "context_baseline_feature_count": int(len(prepared.lightgbm_context_cols)),
    }
    metrics_df = _frame_with_columns(metrics_rows, _MMD_METRIC_COLUMNS).sort_values(["target", "mae", "variant_id"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(["target", "variant_id", "row_uid"]).reset_index(drop=True) if prediction_frames else pd.DataFrame(columns=_MMD_PREDICTION_COLUMNS)
    lambda_df = _frame_with_columns(lambda_rows, _MMD_LAMBDA_SWEEP_COLUMNS).sort_values(["target", "eval_stage", "cv_mae_mean", "lambda"]).reset_index(drop=True)
    history_df = _frame_with_columns(history_rows, _MMD_HISTORY_COLUMNS)
    baseline_df = metrics_df.copy()

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": config.to_dict(), "main": split_manifest}, handle, indent=2, sort_keys=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    baseline_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    lambda_df.to_csv(output_dir / "lambda_sweep.csv", index=False)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "README_run.md").write_text(
        _build_mmd_readme(config=config, split_manifest=split_manifest, metrics_df=metrics_df, lambda_df=lambda_df),
        encoding="utf-8",
    )
    return output_dir



def run_ssl_cli(args=None) -> int:
    # from ssl/run_experiment.py main() logic
    import argparse
    parser = argparse.ArgumentParser(description="Run the SSL + context paired QC experiment")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/neural/ssl")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--max-stress-studies", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parsed = parser.parse_args(args)
    REPO_ROOT = Path(__file__).resolve().parents[3]
    config = SSLDomainAdaptationConfig(
        parquet_path=REPO_ROOT / parsed.parquet,
        output_root=REPO_ROOT / parsed.output_root,
        run_name=parsed.run_name,
        run_stress_test=not parsed.skip_stress,
        max_stress_studies=parsed.max_stress_studies,
        verbose=not parsed.quiet,
    )
    if parsed.smoke:
        config.ssl_pretrain_epochs = 1
        config.source_warmup_epochs = 1
        config.source_train_epochs = 1
        config.finetune_epochs = 1
        config.paired_finetune_val_splits = 2
        config.max_stress_studies = 1 if config.max_stress_studies is None else config.max_stress_studies
    run_ssl_with_config(config)
    return 0


def run_mmd_cli(args=None) -> int:
    # from mmd/run_experiment.py main() logic
    parser = argparse.ArgumentParser(description="Run the MMD domain adaptation experiment")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/neural/mmd")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parsed = parser.parse_args(args)
    REPO_ROOT = Path(__file__).resolve().parents[3]
    config = MMDDomainAdaptationConfig(
        parquet_path=REPO_ROOT / parsed.parquet,
        output_root=REPO_ROOT / parsed.output_root,
        run_name=parsed.run_name,
        run_finetune_stage=not parsed.skip_finetune,
        verbose=not parsed.quiet,
    )
    if parsed.smoke:
        config.source_train_epochs = 1
        config.finetune_epochs = 1
        config.paired_finetune_val_splits = 2
        config.patience = 1
    run_mmd_with_config(config)
    return 0


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(description="Neural domain adaptation runner")
    _parser.add_argument("--mode", choices=["ssl", "mmd"], required=True)
    _known, _remaining = _parser.parse_known_args()
    if _known.mode == "ssl":
        raise SystemExit(run_ssl_cli(_remaining))
    else:
        raise SystemExit(run_mmd_cli(_remaining))
