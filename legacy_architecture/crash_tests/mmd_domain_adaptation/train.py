from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MMDDomainAdaptationConfig
from .dataset import PreparedModelInput
from .model import (
    batch_to_device,
    ensure_torch_available,
    make_unit_only_regressor,
    rbf_multi_scale_mmd,
    target_inverse,
    target_transform,
)

import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD


@dataclass
class StageTrainResult:
    model_state: dict[str, Any]
    history: list[dict[str, Any]]
    checkpoint_path: str | None
    best_epoch: int


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


def _vlog(config: MMDDomainAdaptationConfig | None, message: str) -> None:
    if config is not None and config.verbose:
        print(message, flush=True)


def _clone_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


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


def _feature_dims(data: PreparedModelInput) -> tuple[int, int, int]:
    return int(data.wf.shape[1]), int(data.acg.shape[1]), int(data.scalars.shape[1])


def _batch_indices(n_rows: int, batch_size: int, rng: np.random.RandomState) -> list[np.ndarray]:
    if n_rows <= batch_size:
        return [np.arange(n_rows)]
    shuffled = rng.permutation(n_rows)
    return [shuffled[start : start + batch_size] for start in range(0, n_rows, batch_size)]


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
    n_wf, n_acg, n_scalar = _feature_dims(source_train_data)
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
    n_wf, n_acg, n_scalar = _feature_dims(train_data)
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
    n_wf, n_acg, n_scalar = _feature_dims(eval_data)
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
