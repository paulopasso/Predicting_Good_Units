from __future__ import annotations

from typing import Any

import numpy as np

from crash_tests.self_supervised_domain_adaptation.model import (
    TorchBatches,
    UnitEncoder,
    UnitOnlyRegressor,
    batch_to_device,
    ensure_torch_available,
    target_inverse,
    target_transform,
)

import torch


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
    model = UnitOnlyRegressor(
        encoder=encoder,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )
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
        k_xx = torch.exp(-gamma * d_xx)
        k_yy = torch.exp(-gamma * d_yy)
        k_xy = torch.exp(-gamma * d_xy)
        loss = loss + k_xx.mean() + k_yy.mean() - (2.0 * k_xy.mean())
    return torch.clamp(loss / max(len(bandwidth_multipliers), 1), min=0.0)


__all__ = [
    "TorchBatches",
    "UnitEncoder",
    "UnitOnlyRegressor",
    "batch_to_device",
    "ensure_torch_available",
    "make_unit_only_regressor",
    "rbf_multi_scale_mmd",
    "target_inverse",
    "target_transform",
]
