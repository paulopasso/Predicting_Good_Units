from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


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

        def forward(
            self,
            wf: "torch.Tensor",
            acg: "torch.Tensor",
            scalars: "torch.Tensor",
        ) -> "torch.Tensor":
            parts = [
                self.wf_encoder(wf),
                self.acg_encoder(acg),
                self.scalar_encoder(scalars),
            ]
            return self.fusion(torch.cat(parts, dim=1))


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
            self.wf_decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max(n_wf_features, 1)),
            )
            self.acg_decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max(n_acg_features, 1)),
            )
            self.scalar_decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, max(n_scalar_features, 1)),
            )

        def forward(
            self,
            wf: "torch.Tensor",
            acg: "torch.Tensor",
            scalars: "torch.Tensor",
        ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            latent = self.encoder(wf, acg, scalars)
            return (
                self.wf_decoder(latent),
                self.acg_decoder(latent),
                self.scalar_decoder(latent),
                latent,
            )


    class UnitOnlyRegressor(nn.Module):
        def __init__(
            self,
            encoder: UnitEncoder,
            *,
            latent_dim: int,
            hidden_dim: int,
        ) -> None:
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
            ctx = self.context_encoder(context)
            fused = torch.cat([latent, ctx], dim=1)
            return self.head(fused).squeeze(1), latent

else:

    class UnitEncoder:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required for the SSL domain adaptation prototype")

    class UnitAutoencoder:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required for the SSL domain adaptation prototype")

    class UnitOnlyRegressor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required for the SSL domain adaptation prototype")

    class ContextualRegressor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required for the SSL domain adaptation prototype")


def ensure_torch_available() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for the SSL domain adaptation prototype")


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
    return {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
