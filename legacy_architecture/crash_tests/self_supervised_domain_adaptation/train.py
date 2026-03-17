from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import SSLDomainAdaptationConfig
from .dataset import PreparedModelInput
from .model import (
    ContextualRegressor,
    UnitAutoencoder,
    UnitEncoder,
    UnitOnlyRegressor,
    batch_to_device,
    ensure_torch_available,
    target_inverse,
    target_transform,
)

import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD


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


def _vlog(config: SSLDomainAdaptationConfig | None, message: str) -> None:
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
