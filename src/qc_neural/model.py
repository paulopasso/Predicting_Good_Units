"""QCNeuralNet and QCNeuralWrapper — 1D CNN architecture for cross-domain QC transfer."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

_WF_RE = re.compile(r"^wf_bin_\d+$")
_ACG_RE = re.compile(r"^acg_\d+$")


def _split_column_indices(columns: list[str]) -> tuple[list[int], list[int], list[int]]:
    """Return (wf_indices, acg_indices, scalar_indices) by column name."""
    wf, acg, scalar = [], [], []
    for i, col in enumerate(columns):
        if _WF_RE.match(col):
            wf.append(i)
        elif _ACG_RE.match(col):
            acg.append(i)
        else:
            scalar.append(i)
    return wf, acg, scalar


# ---------------------------------------------------------------------------
# Sub-modules (only defined when torch is available)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _WaveformCNN(nn.Module):
        """Encodes 30-bin peak-channel waveform into a 32-d embedding."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool1d(5),  # 30/5=6 ✓  (MPS requires divisible sizes)
            )
            self.head = nn.Sequential(
                nn.Linear(160, 32),       # 32 channels × 5 pool positions
                nn.LayerNorm(32),
                nn.ReLU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # [B, 30]
            x = x.unsqueeze(1)          # [B, 1, 30]
            x = self.conv(x)            # [B, 32, 5]
            x = x.flatten(1)            # [B, 160]
            return self.head(x)         # [B, 32]

    class _ACGCNN(nn.Module):
        """Encodes 40-bin autocorrelogram into a 32-d embedding."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=5, padding=2),
                nn.AdaptiveAvgPool1d(5),  # 40/5=8 ✓  (MPS requires divisible sizes)
            )
            self.head = nn.Sequential(
                nn.Linear(160, 32),       # 32 channels × 5 pool positions
                nn.LayerNorm(32),
                nn.ReLU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # [B, 40]
            x = x.unsqueeze(1)          # [B, 1, 40]
            x = self.conv(x)            # [B, 32, 5]
            x = x.flatten(1)            # [B, 160]
            return self.head(x)         # [B, 32]

    class _ScalarMLP(nn.Module):
        """Encodes scalar features into a 32-d embedding."""

        def __init__(self, n_features: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class QCNeuralNet(nn.Module):
        """Three-stream 1D CNN with multi-task regression heads.

        Streams
        -------
        * **WaveformCNN**: processes the 30 wf_bin_* columns as a 1-D sequence,
          learning shift-invariant spike-shape patterns.
        * **ACGCNN**: processes the 40 acg_* columns, learning refractory and
          burst patterns.
        * **ScalarMLP**: processes all remaining scalar features.

        The three embeddings (each 32-d) are concatenated and passed through a
        fusion MLP.  Three sigmoid-output heads predict *accuracy*, *fpos*, and
        *fmiss* independently (multi-task).

        Parameters
        ----------
        n_scalar_features:
            Number of scalar features (everything that is not wf_bin_* or acg_*).
        dropout:
            Dropout probability applied in the ScalarMLP and the fusion layer.
        """

        def __init__(self, n_scalar_features: int, dropout: float = 0.2) -> None:
            super().__init__()
            self.wf_cnn = _WaveformCNN()
            self.acg_cnn = _ACGCNN()
            self.scalar_mlp = _ScalarMLP(n_scalar_features, dropout)
            self.fusion = nn.Sequential(
                nn.Linear(96, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.accuracy_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
            self.fpos_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
            self.fmiss_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

        def forward(
            self,
            wf_bins: "torch.Tensor",   # [B, 30]
            acg_bins: "torch.Tensor",  # [B, 40]
            scalars: "torch.Tensor",   # [B, N]
        ) -> "dict[str, torch.Tensor]":
            wf_emb = self.wf_cnn(wf_bins)
            acg_emb = self.acg_cnn(acg_bins)
            scalar_emb = self.scalar_mlp(scalars)

            fused = torch.cat([wf_emb, acg_emb, scalar_emb], dim=1)  # [B, 96]
            fused = self.fusion(fused)                                  # [B, 32]

            return {
                "accuracy": self.accuracy_head(fused).squeeze(1),
                "fpos": self.fpos_head(fused).squeeze(1),
                "fmiss": self.fmiss_head(fused).squeeze(1),
            }

else:
    # Stubs so the module is importable without torch
    class QCNeuralNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required for QCNeuralNet")


# ---------------------------------------------------------------------------
# sklearn-compatible wrapper
# ---------------------------------------------------------------------------

# Maps the user-facing target name to the model's output key
_HEAD_MAP: dict[str, str] = {
    "accuracy": "accuracy",
    "fpos": "fpos",
    "fmiss": "fmiss",
    "fmiss_extended": "fmiss",
}


class QCNeuralWrapper(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper for :class:`QCNeuralNet`.

    Handles
    -------
    * Splitting the feature DataFrame into *wf_bins* / *acg_bins* / *scalars*
      by column name prefix (``wf_bin_\\d+``, ``acg_\\d+``, everything else).
    * Training loop with AdamW + ReduceLROnPlateau + early stopping.
    * Single-target prediction (predict one target at a time, compatible with
      :class:`~qc_framework.models.QCTrainer`).
    * GPU / MPS / CPU device auto-detection.

    This wrapper **must** be used with :class:`~qc_neural.trainer.NeuralTrainer`
    (not the standard :class:`~qc_framework.models.QCTrainer`), because it
    receives the raw feature DataFrame instead of a preprocessed numpy array.

    Parameters
    ----------
    target:
        One of ``"accuracy"``, ``"fpos"``, ``"fmiss"``, ``"fmiss_extended"``.
    n_epochs:
        Maximum training epochs.
    lr:
        Initial learning rate for AdamW.
    batch_size:
        Mini-batch size.
    patience:
        Early-stopping patience (epochs without validation improvement).
    dropout:
        Dropout probability in the network.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        target: str = "accuracy",
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 20,
        dropout: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.target = target
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout
        self.random_state = random_state
        self.verbose = verbose

        # Set during fit
        self._model: "QCNeuralNet | None" = None
        self._wf_idx: list[int] = []
        self._acg_idx: list[int] = []
        self._scalar_idx: list[int] = []
        self._device: "torch.device | None" = None

    # ------------------------------------------------------------------
    def _get_device(self) -> "torch.device":
        # MPS dispatch overhead makes it slower than CPU for small models/batches.
        # Only use CUDA (large GPU) or CPU; skip MPS.
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _array_to_tensors(
        self, vals: np.ndarray
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """Convert a pre-cleaned float32 numpy array to (wf, acg, scalar) tensors."""
        dev = self._device
        wf = torch.tensor(vals[:, self._wf_idx], dtype=torch.float32, device=dev)
        acg = torch.tensor(vals[:, self._acg_idx], dtype=torch.float32, device=dev)
        if self._scalar_idx:
            scalars = torch.tensor(vals[:, self._scalar_idx], dtype=torch.float32, device=dev)
        else:
            scalars = torch.zeros((len(vals), 1), dtype=torch.float32, device=dev)
        return wf, acg, scalars

    def _to_tensors(
        self, X: pd.DataFrame
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        vals = np.nan_to_num(X.values.astype(np.float32), nan=0.0)
        return self._array_to_tensors(vals)

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "QCNeuralWrapper":
        """Train the network.

        Parameters
        ----------
        X:
            Feature DataFrame produced by ``FeatureSetBuilder.align_frame()``.
            Must contain ``wf_bin_*`` and ``acg_*`` columns.
        y:
            Target values, shape ``(n_samples,)``.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for QCNeuralWrapper. Install with: pip install torch")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        columns = list(X.columns)
        self._wf_idx, self._acg_idx, self._scalar_idx = _split_column_indices(columns)

        if not self._wf_idx:
            raise ValueError(
                f"No wf_bin_* columns found in X. Columns: {columns[:10]}..."
            )
        if not self._acg_idx:
            raise ValueError(
                f"No acg_* columns found in X. Columns: {columns[:10]}..."
            )

        self._device = self._get_device()
        n_scalar = len(self._scalar_idx) if self._scalar_idx else 1
        self._model = QCNeuralNet(n_scalar_features=n_scalar, dropout=self.dropout).to(self._device)

        y_arr = np.asarray(y, dtype=np.float32)
        n = len(X)
        n_val = max(1, int(n * 0.1))

        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        val_pos, train_pos = perm[:n_val], perm[n_val:]

        # Pre-convert to numpy once — avoids repeated pandas iloc in the inner loop
        X_np = np.nan_to_num(X.values.astype(np.float32), nan=0.0)
        X_tr_np, y_tr = X_np[train_pos], y_arr[train_pos]
        X_val_np, y_val = X_np[val_pos], y_arr[val_pos]
        n_tr = len(X_tr_np)

        optimizer = AdamW(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        loss_fn = nn.MSELoss()
        head_key = _HEAD_MAP.get(self.target, "accuracy")

        best_val_loss = float("inf")
        patience_count = 0
        best_state: dict | None = None
        stopped_epoch = self.n_epochs

        if self.verbose:
            import sys
            print(f"[{self.target}]  device={self._device}  "
                  f"train={n_tr:,}  val={len(X_val_np):,}  "
                  f"max_epochs={self.n_epochs}  patience={self.patience}")

        for epoch in range(self.n_epochs):
            self._model.train()
            batch_perm = rng.permutation(n_tr)
            train_loss = 0.0
            n_batches = 0
            for start in range(0, n_tr, self.batch_size):
                idx = batch_perm[start : start + self.batch_size]
                wf, acg, sc = self._array_to_tensors(X_tr_np[idx])
                y_b = torch.tensor(y_tr[idx], dtype=torch.float32, device=self._device)
                out = self._model(wf, acg, sc)
                loss = loss_fn(out[head_key], y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= n_batches

            # Validation
            self._model.eval()
            with torch.no_grad():
                wf_v, acg_v, sc_v = self._array_to_tensors(X_val_np)
                y_v = torch.tensor(y_val, dtype=torch.float32, device=self._device)
                val_loss = loss_fn(self._model(wf_v, acg_v, sc_v)[head_key], y_v).item()

            scheduler.step(val_loss)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if self.verbose:
                lr_now = optimizer.param_groups[0]["lr"]
                marker = " *" if is_best else f"  (patience {patience_count}/{self.patience})"
                print(f"  epoch {epoch+1:>4}/{self.n_epochs}  "
                      f"train={train_loss:.4f}  val={val_loss:.4f}  "
                      f"best={best_val_loss:.4f}  lr={lr_now:.2e}{marker}",
                      flush=True)

            if patience_count >= self.patience:
                stopped_epoch = epoch + 1
                break

        if self.verbose:
            print(f"[{self.target}]  stopped at epoch {stopped_epoch}  "
                  f"best_val_loss={best_val_loss:.4f}")

        if best_state is not None:
            self._model.load_state_dict(best_state)

        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        head_key = _HEAD_MAP.get(self.target, "accuracy")
        self._model.eval()
        with torch.no_grad():
            wf, acg, sc = self._to_tensors(X)
            pred = self._model(wf, acg, sc)[head_key].cpu().numpy()
        return np.clip(pred, 0.0, 1.0)

    # ------------------------------------------------------------------
    # sklearn compatibility
    def get_params(self, deep: bool = True) -> dict:
        return {
            "target": self.target,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "dropout": self.dropout,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> "QCNeuralWrapper":
        for k, v in params.items():
            setattr(self, k, v)
        return self
