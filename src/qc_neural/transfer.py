"""Multi-task neural transfer model for hybrid→paired adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import re
import time
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.optim import SGD

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


_WF_PREFIX = "wf_bin_"
_ACG_PREFIX = "acg_"
_HEADS = ("accuracy", "fpos", "fmiss")
_WF_RE = re.compile(r"^wf_bin_\d+$")
_ACG_RE = re.compile(r"^acg_\d+$")


def _split_column_indices(columns: list[str]) -> tuple[list[int], list[int], list[int]]:
    wf, acg, scalar = [], [], []
    for idx, col in enumerate(columns):
        if _WF_RE.match(str(col)):
            wf.append(idx)
        elif _ACG_RE.match(str(col)):
            acg.append(idx)
        else:
            scalar.append(idx)
    return wf, acg, scalar


if _TORCH_AVAILABLE:

    class _WaveformEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(30, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


    class _ACGEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(40, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


    class _ScalarEncoder(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


    class _SharedTransferNet(nn.Module):
        def __init__(self, n_scalar_features: int) -> None:
            super().__init__()
            self.wf = _WaveformEncoder()
            self.acg = _ACGEncoder()
            self.scalar = _ScalarEncoder(max(1, n_scalar_features))
            self.fusion = nn.Sequential(
                nn.Linear(96, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 48),
                nn.ReLU(),
            )
            self.heads = nn.ModuleDict(
                {
                    "accuracy": nn.Sequential(nn.Linear(48, 1), nn.Sigmoid()),
                    "fpos": nn.Sequential(nn.Linear(48, 1), nn.Sigmoid()),
                    "fmiss": nn.Sequential(nn.Linear(48, 1), nn.Sigmoid()),
                }
            )

        def encode(
            self,
            wf: "torch.Tensor",
            acg: "torch.Tensor",
            scalars: "torch.Tensor",
        ) -> "torch.Tensor":
            parts = [self.wf(wf), self.acg(acg), self.scalar(scalars)]
            return self.fusion(torch.cat(parts, dim=1))

        def forward(
            self,
            wf: "torch.Tensor",
            acg: "torch.Tensor",
            scalars: "torch.Tensor",
        ) -> "tuple[dict[str, torch.Tensor], torch.Tensor]":
            latent = self.encode(wf, acg, scalars)
            outputs = {name: head(latent).squeeze(1) for name, head in self.heads.items()}
            return outputs, latent

else:

    class _SharedTransferNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch is required for MultiTaskDomainAdaptiveRegressor")


def _coral_loss(source: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.new_tensor(0.0)

    def _cov(x: "torch.Tensor") -> "torch.Tensor":
        centered = x - x.mean(dim=0, keepdim=True)
        return centered.t().mm(centered) / (x.shape[0] - 1)

    cov_s = _cov(source)
    cov_t = _cov(target)
    d = source.shape[1]
    return ((cov_s - cov_t) ** 2).mean() / (4.0 * d * d)


@dataclass
class _TensorBatch:
    wf: "torch.Tensor"
    acg: "torch.Tensor"
    scalars: "torch.Tensor"


class MultiTaskDomainAdaptiveRegressor:
    """Shared-encoder neural regressor with CORAL-style target alignment."""

    def __init__(
        self,
        feature_columns: list[str],
        pretrain_epochs: int = 35,
        finetune_epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        domain_loss_weight: float = 0.15,
        target_loss_weight: float = 2.0,
        random_state: int = 42,
        verbose: bool = False,
        progress_prefix: str = "neural_transfer",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for MultiTaskDomainAdaptiveRegressor")
        self.feature_columns = list(feature_columns)
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.domain_loss_weight = domain_loss_weight
        self.target_loss_weight = target_loss_weight
        self.random_state = random_state
        self.verbose = verbose
        self.progress_prefix = progress_prefix
        self._wf_idx: list[int] = []
        self._acg_idx: list[int] = []
        self._scalar_idx: list[int] = []
        self._device: "torch.device | None" = None
        self._model: "_SharedTransferNet | None" = None
        self._scalar_mean: np.ndarray | None = None
        self._scalar_std: np.ndarray | None = None

    def _get_device(self) -> "torch.device":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _align_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.feature_columns:
            if col not in out.columns:
                out[col] = pd.NA
        aligned = out[self.feature_columns].copy()
        for col in aligned.columns:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
        return aligned

    def _tensor_batch(self, frame: pd.DataFrame) -> _TensorBatch:
        values = np.nan_to_num(self._align_frame(frame).values.astype(np.float32), nan=0.0)
        wf = torch.tensor(values[:, self._wf_idx], dtype=torch.float32, device=self._device)
        acg = torch.tensor(values[:, self._acg_idx], dtype=torch.float32, device=self._device)
        if self._scalar_idx:
            scalar_vals = values[:, self._scalar_idx]
            if self._scalar_mean is not None and self._scalar_std is not None:
                scalar_vals = (scalar_vals - self._scalar_mean) / self._scalar_std
            scalar_vals = np.clip(np.nan_to_num(scalar_vals, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
            scalars = torch.tensor(scalar_vals, dtype=torch.float32, device=self._device)
        else:
            scalars = torch.zeros((len(values), 1), dtype=torch.float32, device=self._device)
        return _TensorBatch(wf=wf, acg=acg, scalars=scalars)

    def _supervised_loss(
        self,
        outputs: dict[str, "torch.Tensor"],
        targets: dict[str, np.ndarray],
        batch_idx: np.ndarray,
    ) -> "torch.Tensor":
        total = outputs["accuracy"].new_tensor(0.0)
        n_terms = 0
        for head in _HEADS:
            vals = targets.get(head)
            if vals is None:
                continue
            batch_vals = vals[batch_idx]
            mask = np.isfinite(batch_vals)
            if not np.any(mask):
                continue
            pred = outputs[head][torch.tensor(mask, dtype=torch.bool, device=self._device)]
            target = torch.tensor(batch_vals[mask], dtype=torch.float32, device=self._device)
            total = total + torch.mean((pred - target) ** 2)
            n_terms += 1
        if n_terms == 0:
            return total
        return total / n_terms

    def _predict_outputs(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        batch = self._tensor_batch(df)
        self._model.eval()
        with torch.no_grad():
            outputs, _ = self._model(batch.wf, batch.acg, batch.scalars)
        return {name: outputs[name].cpu().numpy() for name in _HEADS}

    def fit(
        self,
        *,
        source_df: pd.DataFrame,
        target_labeled_df: pd.DataFrame,
        target_unlabeled_df: pd.DataFrame,
        source_target_map: dict[str, str] | None = None,
    ) -> "MultiTaskDomainAdaptiveRegressor":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        source_target_map = source_target_map or {
            "accuracy": "accuracy",
            "fpos": "fpos",
            "fmiss": "fmiss_extended",
        }
        source_targets = {
            head: pd.to_numeric(source_df[source_target_map[head]], errors="coerce").values.astype(np.float32)
            for head in _HEADS
        }
        target_targets = {
            head: pd.to_numeric(target_labeled_df[head], errors="coerce").values.astype(np.float32)
            for head in _HEADS
            if head in target_labeled_df.columns
        }

        self._wf_idx, self._acg_idx, self._scalar_idx = _split_column_indices(self.feature_columns)
        if not self._wf_idx:
            raise ValueError("MultiTaskDomainAdaptiveRegressor requires wf_bin_* feature columns")
        if not self._acg_idx:
            raise ValueError("MultiTaskDomainAdaptiveRegressor requires acg_* feature columns")

        if self._scalar_idx:
            aligned_frames = [
                self._align_frame(source_df),
                self._align_frame(target_labeled_df),
                self._align_frame(target_unlabeled_df),
            ]
            scalar_arrays = [
                frame.iloc[:, self._scalar_idx].to_numpy(dtype=np.float32, na_value=np.nan)
                for frame in aligned_frames
                if len(frame)
            ]
            scalar_values = np.concatenate(scalar_arrays, axis=0)
            self._scalar_mean = np.nanmean(scalar_values, axis=0)
            self._scalar_std = np.nanstd(scalar_values, axis=0)
            self._scalar_mean = np.nan_to_num(self._scalar_mean, nan=0.0).astype(np.float32)
            self._scalar_std = np.nan_to_num(self._scalar_std, nan=1.0).astype(np.float32)
            self._scalar_std[self._scalar_std == 0.0] = 1.0
        else:
            self._scalar_mean = None
            self._scalar_std = None

        self._device = self._get_device()
        self._model = _SharedTransferNet(n_scalar_features=len(self._scalar_idx)).to(self._device)
        optimizer = SGD(self._model.parameters(), lr=self.lr, momentum=0.9)

        n_source = len(source_df)
        n_target_lab = len(target_labeled_df)
        n_target_unlab = len(target_unlabeled_df)
        rng = np.random.RandomState(self.random_state)

        if n_source == 0 or n_target_lab == 0:
            raise ValueError("MultiTaskDomainAdaptiveRegressor requires labeled source and labeled target rows")
        if n_target_unlab == 0:
            target_unlabeled_df = target_labeled_df.copy()
            n_target_unlab = len(target_unlabeled_df)

        if self.verbose:
            print(
                f"{self.progress_prefix} | source={n_source} labeled_target={n_target_lab} "
                f"unlabeled_target={n_target_unlab} pretrain_epochs={self.pretrain_epochs} "
                f"finetune_epochs={self.finetune_epochs}",
                flush=True,
            )

        pretrain_started = time.perf_counter()
        for _epoch in range(self.pretrain_epochs):
            self._model.train()
            n_batches = max(1, ceil(max(n_source, n_target_unlab, n_target_lab) / self.batch_size))
            epoch_total = 0.0
            epoch_src = 0.0
            epoch_tgt = 0.0
            epoch_dom = 0.0
            for _ in range(n_batches):
                src_idx = rng.randint(0, n_source, size=min(self.batch_size, n_source))
                tgt_lab_idx = rng.randint(0, n_target_lab, size=min(self.batch_size, n_target_lab))
                tgt_unlab_idx = rng.randint(0, n_target_unlab, size=min(self.batch_size, n_target_unlab))

                src_batch = self._tensor_batch(source_df.iloc[src_idx])
                tgt_lab_batch = self._tensor_batch(target_labeled_df.iloc[tgt_lab_idx])
                tgt_unlab_batch = self._tensor_batch(target_unlabeled_df.iloc[tgt_unlab_idx])

                src_outputs, src_latent = self._model(src_batch.wf, src_batch.acg, src_batch.scalars)
                tgt_lab_outputs, tgt_lab_latent = self._model(
                    tgt_lab_batch.wf, tgt_lab_batch.acg, tgt_lab_batch.scalars
                )
                _, tgt_unlab_latent = self._model(
                    tgt_unlab_batch.wf, tgt_unlab_batch.acg, tgt_unlab_batch.scalars
                )

                src_loss = self._supervised_loss(src_outputs, source_targets, src_idx)
                tgt_loss = self._supervised_loss(tgt_lab_outputs, target_targets, tgt_lab_idx)
                domain_loss = _coral_loss(src_latent, torch.cat([tgt_lab_latent, tgt_unlab_latent], dim=0))
                loss = src_loss + (self.target_loss_weight * tgt_loss) + (self.domain_loss_weight * domain_loss)
                epoch_total += float(loss.item())
                epoch_src += float(src_loss.item())
                epoch_tgt += float(tgt_loss.item())
                epoch_dom += float(domain_loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                elapsed = time.perf_counter() - pretrain_started
                avg_epoch = elapsed / max(_epoch + 1, 1)
                eta = avg_epoch * max(self.pretrain_epochs - (_epoch + 1), 0)
                print(
                    f"{self.progress_prefix} | pretrain {_epoch + 1}/{self.pretrain_epochs} "
                    f"loss={epoch_total / n_batches:.4f} src={epoch_src / n_batches:.4f} "
                    f"tgt={epoch_tgt / n_batches:.4f} domain={epoch_dom / n_batches:.4f} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

        finetune_started = time.perf_counter()
        for _epoch in range(self.finetune_epochs):
            self._model.train()
            n_batches = max(1, ceil(n_target_lab / self.batch_size))
            epoch_loss = 0.0
            for _ in range(n_batches):
                tgt_idx = rng.randint(0, n_target_lab, size=min(self.batch_size, n_target_lab))
                tgt_batch = self._tensor_batch(target_labeled_df.iloc[tgt_idx])
                tgt_outputs, _ = self._model(tgt_batch.wf, tgt_batch.acg, tgt_batch.scalars)
                loss = self._supervised_loss(tgt_outputs, target_targets, tgt_idx)
                epoch_loss += float(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                elapsed = time.perf_counter() - finetune_started
                avg_epoch = elapsed / max(_epoch + 1, 1)
                eta = avg_epoch * max(self.finetune_epochs - (_epoch + 1), 0)
                print(
                    f"{self.progress_prefix} | finetune {_epoch + 1}/{self.finetune_epochs} "
                    f"loss={epoch_loss / n_batches:.4f} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

        return self

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        outputs = self._predict_outputs(df)
        return pd.DataFrame({head: np.clip(outputs[head], 0.0, 1.0) for head in _HEADS}, index=df.index)
