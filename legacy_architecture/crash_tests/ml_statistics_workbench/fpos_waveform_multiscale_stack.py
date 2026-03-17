from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.ml_statistics_workbench.fpos_signal_embedding_stack import (
    FPosSignalEmbeddingStackConfig,
    _cv_and_holdout,
    _ensure_torch,
    _prepare_base_frames,
    _signal_embeddings,
)
from crash_tests.ml_statistics_workbench.methods import make_metric_row, make_prediction_frame, score_predictions

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class FPosWaveformMultiscaleStackConfig(FPosSignalEmbeddingStackConfig):
    multiscale_embed_dim: int = 12
    multiscale_hidden_channels: int = 16
    denoise_std: float = 0.025
    derivative_loss_weight: float = 0.30

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_waveform_multiscale_stack_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fpos multiscale waveform stack benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosWaveformMultiscaleStackConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


if _TORCH_AVAILABLE:

    class MultiscaleSignalAutoencoder(nn.Module):
        def __init__(self, *, input_len: int, hidden_channels: int, embed_dim: int) -> None:
            super().__init__()
            self.conv3 = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(1, hidden_channels, kernel_size=5, padding=2)
            self.conv7 = nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3)
            self.mix = nn.Sequential(
                nn.Conv1d(hidden_channels * 3, hidden_channels * 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.encoder_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(hidden_channels * 2 * input_len, hidden_channels * 6),
                nn.ReLU(),
                nn.Linear(hidden_channels * 6, embed_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(embed_dim, hidden_channels * 6),
                nn.ReLU(),
                nn.Linear(hidden_channels * 6, hidden_channels * 2 * input_len),
                nn.ReLU(),
                nn.Unflatten(1, (hidden_channels * 2, input_len)),
                nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1),
            )

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            multiscale = torch.cat(
                [torch.relu(self.conv3(x)), torch.relu(self.conv5(x)), torch.relu(self.conv7(x))],
                dim=1,
            )
            mixed = self.mix(multiscale)
            return self.encoder_head(mixed)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            emb = self.encode(x)
            recon = self.decoder(emb)
            return recon, emb


def _fit_multiscale_autoencoder(
    *,
    values: np.ndarray,
    config: FPosWaveformMultiscaleStackConfig,
    device: "torch.device",
) -> tuple[MultiscaleSignalAutoencoder, dict[str, float]]:
    _ensure_torch()
    input_len = values.shape[1]
    model = MultiscaleSignalAutoencoder(
        input_len=input_len,
        hidden_channels=config.multiscale_hidden_channels,
        embed_dim=config.multiscale_embed_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    all_values = torch.tensor(values[:, None, :], dtype=torch.float32, device=device)
    rng = np.random.default_rng(config.random_state)
    model.train()
    final_loss = float("nan")
    final_recon = float("nan")
    final_deriv = float("nan")
    for _epoch in range(int(config.ae_epochs)):
        order = rng.permutation(len(values))
        losses: list[float] = []
        recon_losses: list[float] = []
        deriv_losses: list[float] = []
        for start in range(0, len(order), int(config.batch_size)):
            batch_idx = order[start : start + int(config.batch_size)]
            batch = all_values[batch_idx]
            noise = torch.randn_like(batch) * float(config.denoise_std)
            noisy = batch + noise
            recon, _ = model(noisy)
            recon_loss = loss_fn(recon, batch)
            recon_dx = recon[:, :, 1:] - recon[:, :, :-1]
            batch_dx = batch[:, :, 1:] - batch[:, :, :-1]
            deriv_loss = loss_fn(recon_dx, batch_dx)
            loss = recon_loss + float(config.derivative_loss_weight) * deriv_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            recon_losses.append(float(recon_loss.detach().cpu()))
            deriv_losses.append(float(deriv_loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else float("nan")
        final_recon = float(np.mean(recon_losses)) if recon_losses else float("nan")
        final_deriv = float(np.mean(deriv_losses)) if deriv_losses else float("nan")
    return model.eval(), {
        "train_total_loss": final_loss,
        "train_recon_loss": final_recon,
        "train_deriv_loss": final_deriv,
    }


def _encode_multiscale(model: MultiscaleSignalAutoencoder, values: np.ndarray, device: "torch.device") -> np.ndarray:
    _ensure_torch()
    with torch.no_grad():
        tensor = torch.tensor(values[:, None, :], dtype=torch.float32, device=device)
        emb = model.encode(tensor).detach().cpu().numpy()
    return emb.astype(np.float32)


def _multiscale_waveform_embeddings(
    *,
    train_signal: pd.DataFrame,
    train_universe_signal: pd.DataFrame,
    unlabeled_signal: pd.DataFrame,
    test_signal: pd.DataFrame,
    config: FPosWaveformMultiscaleStackConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    _ensure_torch()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    universe = scaler.fit_transform(imputer.fit_transform(train_universe_signal))
    train_values = scaler.transform(imputer.transform(train_signal))
    unlabeled_values = scaler.transform(imputer.transform(unlabeled_signal))
    test_values = scaler.transform(imputer.transform(test_signal))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = _fit_multiscale_autoencoder(
        values=universe.astype(np.float32),
        config=config,
        device=device,
    )
    train_emb = _encode_multiscale(model, train_values.astype(np.float32), device=device)
    unlabeled_emb = _encode_multiscale(model, unlabeled_values.astype(np.float32), device=device)
    test_emb = _encode_multiscale(model, test_values.astype(np.float32), device=device)
    cols = [f"wf_multi_embed_{i:02d}" for i in range(train_emb.shape[1])]
    return (
        pd.DataFrame(train_emb, columns=cols),
        pd.DataFrame(unlabeled_emb, columns=cols),
        pd.DataFrame(test_emb, columns=cols),
        history,
    )


def _run_with_config(config: FPosWaveformMultiscaleStackConfig) -> Path:
    _ensure_torch()
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _prepare_base_frames(config)
    base = state["base"]
    prepared = state["prepared"]
    target_train_df = state["target_train_df"]
    target_test_df = state["target_test_df"]
    unlabeled_df = state["unlabeled_df"]
    y_train = state["y_train"]
    y_test = state["y_test"]
    trust_selected = state["trust_selected"]
    train_frame = state["train_frame"]
    unlabeled_frame = state["unlabeled_frame"]
    test_frame = state["test_frame"]

    universe_df = pd.concat(
        [
            state["split"].hybrid_source.reset_index(drop=True),
            state["split"].paired_non_test_all.reset_index(drop=True),
            state["split"].ssl_pool.reset_index(drop=True),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["row_uid"]).reset_index(drop=True)
    if config.universe_max_rows is not None and len(universe_df) > int(config.universe_max_rows):
        universe_df = universe_df.sample(
            n=int(config.universe_max_rows),
            random_state=config.random_state,
        ).reset_index(drop=True)

    _log(config, "[embed] training baseline waveform autoencoder")
    wf_train_emb, wf_unlabeled_emb, wf_test_emb, wf_hist = _signal_embeddings(
        train_signal=target_train_df.loc[:, prepared.wf_feature_cols],
        train_universe_signal=universe_df.loc[:, prepared.wf_feature_cols],
        unlabeled_signal=unlabeled_df.loc[:, prepared.wf_feature_cols],
        test_signal=target_test_df.loc[:, prepared.wf_feature_cols],
        prefix="wf",
        config=config,
    )
    _log(config, "[embed] training multiscale waveform autoencoder")
    wf_multi_train, wf_multi_unlabeled, wf_multi_test, wf_multi_hist = _multiscale_waveform_embeddings(
        train_signal=target_train_df.loc[:, prepared.wf_feature_cols],
        train_universe_signal=universe_df.loc[:, prepared.wf_feature_cols],
        unlabeled_signal=unlabeled_df.loc[:, prepared.wf_feature_cols],
        test_signal=target_test_df.loc[:, prepared.wf_feature_cols],
        config=config,
    )

    variants = [
        (
            "wf_embed_anchor_stack_xgboost",
            pd.concat([train_frame.reset_index(drop=True), wf_train_emb], axis=1),
            pd.concat([unlabeled_frame.reset_index(drop=True), wf_unlabeled_emb], axis=1),
            pd.concat([test_frame.reset_index(drop=True), wf_test_emb], axis=1),
        ),
        (
            "wf_multi_embed_anchor_stack_xgboost",
            pd.concat([train_frame.reset_index(drop=True), wf_multi_train], axis=1),
            pd.concat([unlabeled_frame.reset_index(drop=True), wf_multi_unlabeled], axis=1),
            pd.concat([test_frame.reset_index(drop=True), wf_multi_test], axis=1),
        ),
        (
            "wf_embed_plus_multi_anchor_stack_xgboost",
            pd.concat([train_frame.reset_index(drop=True), wf_train_emb, wf_multi_train], axis=1),
            pd.concat([unlabeled_frame.reset_index(drop=True), wf_unlabeled_emb, wf_multi_unlabeled], axis=1),
            pd.concat([test_frame.reset_index(drop=True), wf_test_emb, wf_multi_test], axis=1),
        ),
    ]

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []

    for idx, (variant_id, train_variant, unlabeled_variant, test_variant) in enumerate(variants, start=1):
        _log(config, f"[variant] {variant_id}")
        cv_metrics, holdout_pred = _cv_and_holdout(
            labeled_meta_df=target_train_df,
            labeled_frame=train_variant,
            labeled_y=y_train,
            pseudo_selected=trust_selected,
            unlabeled_frame=unlabeled_variant,
            test_frame=test_variant,
            base=base,
            backend_id="xgboost_default",
            family_alpha=1.0,
            random_state_offset=4200 + (100 * idx),
        )
        holdout_metrics = score_predictions(y_test, holdout_pred)
        metric_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family="waveform_multiscale_stack",
                backend="xgboost_default",
                source_target="fpos",
                metrics=holdout_metrics,
                notes="multiscale waveform representation variant",
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family="waveform_multiscale_stack",
                backend="xgboost_default",
                df=target_test_df,
                y_true=y_test,
                pred=holdout_pred,
            )
        )
        cv_rows.append(
            {
                "target": "fpos",
                "variant_id": variant_id,
                "cv_mae": float(cv_metrics["mae"]),
                "cv_r2": float(cv_metrics["r2"]),
                "selected_rows": int(len(trust_selected)),
                "mean_selected_weight": float(trust_selected["pseudo_weight"].mean()) if not trust_selected.empty else 0.0,
                "n_features": int(train_variant.shape[1]),
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows).sort_values(["cv_r2", "cv_mae"], ascending=[False, True]).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    pd.DataFrame([{"wf_recon_loss": wf_hist["train_recon_loss"], **wf_multi_hist}]).to_csv(
        output_dir / "embedding_history.csv", index=False
    )
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(state["split"].manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosWaveformMultiscaleStackConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosWaveformMultiscaleStackConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.max_pseudo_multiplier = 1.0
        cli_config.max_pseudo_per_recording = 24
        cli_config.max_pseudo_per_cluster = 64
        cli_config.ae_epochs = 5
        cli_config.conv_embed_dim = 6
        cli_config.multiscale_embed_dim = 8
        cli_config.universe_max_rows = 4000
    _run_with_config(cli_config)
