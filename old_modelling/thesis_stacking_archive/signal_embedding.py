from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from qc_thesis.modeling.core import FrameAugmenter

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class SignalEmbeddingConfig(Protocol):
    conv_embed_dim: int
    conv_hidden_channels: int
    ae_epochs: int
    batch_size: int
    learning_rate: float
    random_state: int


def _ensure_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for the signal embedding stack benchmark")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


if _TORCH_AVAILABLE:

    class SignalConvAutoencoder(nn.Module):
        def __init__(self, *, input_len: int, hidden_channels: int, embed_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(hidden_channels * input_len, embed_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, hidden_channels * input_len),
                nn.ReLU(),
                nn.Unflatten(1, (hidden_channels, input_len)),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1),
            )

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.encoder(x)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            emb = self.encode(x)
            recon = self.decoder(emb)
            return recon, emb


@dataclass(frozen=True)
class SignalEmbeddingBundle:
    train: pd.DataFrame
    unlabeled: pd.DataFrame
    test: pd.DataFrame
    history: dict[str, float]

    def as_augmenter(self) -> FrameAugmenter:
        return FrameAugmenter(
            train_features=self.train,
            unlabeled_features=self.unlabeled,
            test_features=self.test,
        )


def _fit_signal_autoencoder(
    *,
    values: np.ndarray,
    input_len: int,
    config: SignalEmbeddingConfig,
    device: "torch.device",
) -> tuple[SignalConvAutoencoder, dict[str, float]]:
    _ensure_torch()
    model = SignalConvAutoencoder(
        input_len=input_len,
        hidden_channels=config.conv_hidden_channels,
        embed_dim=config.conv_embed_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    all_values = torch.tensor(values[:, None, :], dtype=torch.float32, device=device)
    model.train()
    final_loss = float("nan")
    rng = np.random.default_rng(config.random_state)
    for _epoch in range(int(config.ae_epochs)):
        order = rng.permutation(len(values))
        losses: list[float] = []
        for start in range(0, len(order), int(config.batch_size)):
            batch_idx = order[start : start + int(config.batch_size)]
            batch = all_values[batch_idx]
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else float("nan")
    return model.eval(), {"train_recon_loss": final_loss}


def _encode_signal(model: SignalConvAutoencoder, values: np.ndarray, device: "torch.device") -> np.ndarray:
    _ensure_torch()
    with torch.no_grad():
        tensor = torch.tensor(values[:, None, :], dtype=torch.float32, device=device)
        emb = model.encode(tensor).detach().cpu().numpy()
    return emb.astype(np.float32)


def fit_signal_embeddings(
    *,
    train_signal: pd.DataFrame,
    train_universe_signal: pd.DataFrame,
    unlabeled_signal: pd.DataFrame,
    test_signal: pd.DataFrame,
    prefix: str,
    config: SignalEmbeddingConfig,
) -> SignalEmbeddingBundle:
    _ensure_torch()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    universe = scaler.fit_transform(imputer.fit_transform(train_universe_signal))
    train_values = scaler.transform(imputer.transform(train_signal))
    unlabeled_values = scaler.transform(imputer.transform(unlabeled_signal))
    test_values = scaler.transform(imputer.transform(test_signal))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = _fit_signal_autoencoder(
        values=universe.astype(np.float32),
        input_len=universe.shape[1],
        config=config,
        device=device,
    )
    train_emb = _encode_signal(model, train_values.astype(np.float32), device=device)
    unlabeled_emb = _encode_signal(model, unlabeled_values.astype(np.float32), device=device)
    test_emb = _encode_signal(model, test_values.astype(np.float32), device=device)
    cols = [f"{prefix}_embed_{i:02d}" for i in range(train_emb.shape[1])]
    return SignalEmbeddingBundle(
        train=pd.DataFrame(train_emb, columns=cols),
        unlabeled=pd.DataFrame(unlabeled_emb, columns=cols),
        test=pd.DataFrame(test_emb, columns=cols),
        history=history,
    )
