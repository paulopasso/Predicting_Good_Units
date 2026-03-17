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

from crash_tests.ml_statistics_workbench.config import MLStatisticsWorkbenchConfig
from crash_tests.ml_statistics_workbench.fpos_anchor_stack import _fit_anchor_scores
from crash_tests.ml_statistics_workbench.fpos_backend_sweep import _fit_backend_predict, _grouped_oof_splits
from crash_tests.ml_statistics_workbench.fpos_cluster_trust_pseudo import (
    FPosClusterTrustPseudoConfig,
    _apply_trust_selection,
    _trust_from_group_stats,
)
from crash_tests.ml_statistics_workbench.fpos_family_robustness import _family_weights
from crash_tests.ml_statistics_workbench.methods import (
    build_stack_frame,
    make_metric_row,
    make_prediction_frame,
    mask_feature_columns,
    score_predictions,
)
from crash_tests.ml_statistics_workbench.pseudo_label_manifold import (
    PseudoLabelManifoldConfig,
    _build_filtered_drop_cols,
    _build_support_artifacts,
    _manifold_support_table,
    _select_pseudo_labels,
    _teacher_frames_for_target,
    _teacher_predictions,
)
from crash_tests.mmd_domain_adaptation.dataset import prepare_feature_table
from crash_tests.self_supervised_domain_adaptation.split_utils import make_main_split

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class FPosSignalEmbeddingStackConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("crash_tests/ml_statistics_workbench/outputs")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    max_pseudo_multiplier: float = 1.5
    max_pseudo_per_recording: int = 48
    max_pseudo_per_cluster: int = 128
    max_pseudo_per_labeled_ratio: float = 1.5
    cluster_trust_power: float = 1.35
    study_trust_weight: float = 0.30
    min_trust: float = 0.25
    conv_embed_dim: int = 8
    conv_hidden_channels: int = 12
    ae_epochs: int = 28
    batch_size: int = 256
    learning_rate: float = 2e-3
    universe_max_rows: int | None = 18000
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_signal_embedding_stack_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fpos signal embedding stack benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosSignalEmbeddingStackConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


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


def _fit_signal_autoencoder(
    *,
    values: np.ndarray,
    input_len: int,
    config: FPosSignalEmbeddingStackConfig,
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


def _signal_embeddings(
    *,
    train_signal: pd.DataFrame,
    train_universe_signal: pd.DataFrame,
    unlabeled_signal: pd.DataFrame,
    test_signal: pd.DataFrame,
    prefix: str,
    config: FPosSignalEmbeddingStackConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
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
    return (
        pd.DataFrame(train_emb, columns=cols),
        pd.DataFrame(unlabeled_emb, columns=cols),
        pd.DataFrame(test_emb, columns=cols),
        history,
    )


def _assemble_training_data(
    *,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    labeled_meta_df: pd.DataFrame,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    family_alpha: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    labeled_weight = _family_weights(labeled_meta_df["study_set"], alpha=family_alpha)
    if pseudo_selected.empty:
        return labeled_frame.reset_index(drop=True), np.asarray(labeled_y, dtype=float), labeled_weight
    pseudo_idx = pseudo_selected["unlabeled_row_idx"].to_numpy(dtype=int)
    pseudo_frame = unlabeled_frame.iloc[pseudo_idx].reset_index(drop=True)
    full_frame = pd.concat([labeled_frame.reset_index(drop=True), pseudo_frame], ignore_index=True)
    full_y = np.concatenate([np.asarray(labeled_y, dtype=float), pseudo_selected["pseudo_label"].to_numpy(dtype=float)])
    full_weight = np.concatenate([labeled_weight, pseudo_selected["pseudo_weight"].to_numpy(dtype=np.float32)])
    return full_frame, full_y, full_weight


def _prepare_base_frames(config: FPosSignalEmbeddingStackConfig):
    base = MLStatisticsWorkbenchConfig(
        parquet_path=config.parquet_path,
        output_root=config.output_root,
        run_name=config.run_name,
        random_state=config.random_state,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        verbose=config.verbose,
    )
    pseudo_base = PseudoLabelManifoldConfig(
        parquet_path=config.parquet_path,
        output_root=config.output_root,
        run_name=config.run_name,
        random_state=config.random_state,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        max_pseudo_multiplier=config.max_pseudo_multiplier,
        max_pseudo_per_recording=config.max_pseudo_per_recording,
        max_pseudo_per_cluster=config.max_pseudo_per_cluster,
        max_pseudo_per_labeled_ratio=config.max_pseudo_per_labeled_ratio,
        verbose=config.verbose,
    )
    trust_base = FPosClusterTrustPseudoConfig(
        parquet_path=config.parquet_path,
        output_root=config.output_root,
        run_name=config.run_name,
        random_state=config.random_state,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        max_pseudo_multiplier=config.max_pseudo_multiplier,
        max_pseudo_per_recording=config.max_pseudo_per_recording,
        max_pseudo_per_cluster=config.max_pseudo_per_cluster,
        max_pseudo_per_labeled_ratio=config.max_pseudo_per_labeled_ratio,
        cluster_trust_power=config.cluster_trust_power,
        study_trust_weight=config.study_trust_weight,
        min_trust=config.min_trust,
        verbose=config.verbose,
    )

    prepared = prepare_feature_table(config.parquet_path)
    split = make_main_split(prepared.df, paired_final_holdout_rows=config.paired_final_holdout_rows, random_state=config.random_state)
    unit_drop_cols, context_drop_cols, _ = _build_filtered_drop_cols(split=split, prepared=prepared, base_config=base)
    (
        target_train_df,
        y_train,
        target_test_df,
        y_test,
        unlabeled_df,
        teachers,
        _student_templates,
        _transport_manifest,
    ) = _teacher_frames_for_target(
        target="fpos",
        split=split,
        prepared=prepared,
        base=base,
        config=pseudo_base,
        unit_drop_cols=unit_drop_cols,
        context_drop_cols=context_drop_cols,
    )
    teacher_train_df, teacher_unlabeled_df, teacher_test_df, _ = _teacher_predictions(
        teachers=teachers,
        labeled_df=target_train_df,
        y_train=y_train,
        base=base,
    )
    support_artifacts = _build_support_artifacts(
        paired_non_test_all=split.paired_non_test_all,
        labeled_train_df=target_train_df,
        unit_cols=prepared.unit_feature_cols,
        config=pseudo_base,
    )
    support_df = _manifold_support_table(unlabeled_df=unlabeled_df, labeled_train_df=target_train_df, support_artifacts=support_artifacts)
    support_train_df = _manifold_support_table(unlabeled_df=target_train_df, labeled_train_df=target_train_df, support_artifacts=support_artifacts)
    train_trust_df = pd.concat(
        [
            target_train_df.loc[:, ["row_uid", "recording_key", "study_set"]].reset_index(drop=True),
            support_train_df.loc[:, ["cluster_id", "support_confidence"]].reset_index(drop=True),
        ],
        axis=1,
    )
    train_trust_df["teacher_abs_error"] = np.abs(y_train - teacher_train_df["teacher_mean"].to_numpy(dtype=float))
    cluster_trust_df = _trust_from_group_stats(train_trust_df, group_col="cluster_id")
    study_trust_df = _trust_from_group_stats(train_trust_df, group_col="study_set")
    candidate_df, _ = _select_pseudo_labels(
        unlabeled_df=unlabeled_df,
        unlabeled_teacher_df=teacher_unlabeled_df,
        support_df=support_df,
        config=pseudo_base,
        max_rows=int(round(len(target_train_df) * float(config.max_pseudo_multiplier))),
    )
    trust_selected = _apply_trust_selection(
        candidate_df=candidate_df,
        cluster_trust_df=cluster_trust_df,
        study_trust_df=study_trust_df,
        max_rows=int(round(len(target_train_df) * float(config.max_pseudo_multiplier))),
        config=trust_base,
    )

    filtered_drop_cols = set(unit_drop_cols) | set(context_drop_cols)
    filtered_context_cols = [col for col in prepared.lightgbm_context_cols if col not in filtered_drop_cols]
    filtered_unit_cols = [col for col in prepared.unit_feature_cols if col not in set(unit_drop_cols)]
    target_train_masked = mask_feature_columns(target_train_df, filtered_drop_cols)
    target_test_masked = mask_feature_columns(target_test_df, filtered_drop_cols)
    unlabeled_masked = mask_feature_columns(unlabeled_df, filtered_drop_cols)
    anchor_feature_cols = sorted(set(filtered_context_cols + filtered_unit_cols))
    anchor_oof, anchor_unlabeled, anchor_test = _fit_anchor_scores(
        matched_train_df=target_train_masked,
        matched_test_df=target_test_masked,
        unmatched_df=unlabeled_masked,
        feature_cols=anchor_feature_cols,
        n_splits=base.model_selection_splits,
        config=type(
            "AnchorCfg",
            (),
            {"anchor_learning_rate": 0.06, "anchor_max_depth": 4, "anchor_max_iter": 250, "random_state": config.random_state},
        )(),
    )
    matched_support_conf_train = support_train_df["support_confidence"].to_numpy(dtype=float)
    matched_support_conf_unlabeled = support_df["support_confidence"].to_numpy(dtype=float)
    matched_support_conf_test = _manifold_support_table(
        unlabeled_df=target_test_df,
        labeled_train_df=target_train_df,
        support_artifacts=support_artifacts,
    )["support_confidence"].to_numpy(dtype=float)

    train_frame = build_stack_frame(
        target_train_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_train_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_oof,
            "matched_support_conf": matched_support_conf_train,
            "anchor_support_margin": anchor_oof - matched_support_conf_train,
        },
    )
    unlabeled_frame = build_stack_frame(
        unlabeled_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_unlabeled_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_unlabeled,
            "matched_support_conf": matched_support_conf_unlabeled,
            "anchor_support_margin": anchor_unlabeled - matched_support_conf_unlabeled,
        },
    )
    test_frame = build_stack_frame(
        target_test_masked,
        context_cols=filtered_context_cols,
        extra_features={
            "source_pred": teacher_test_df[teachers[1].variant_id].to_numpy(dtype=float),
            "anchor_fp_score": anchor_test,
            "matched_support_conf": matched_support_conf_test,
            "anchor_support_margin": anchor_test - matched_support_conf_test,
        },
    )
    return {
        "base": base,
        "prepared": prepared,
        "split": split,
        "target_train_df": target_train_df,
        "target_test_df": target_test_df,
        "unlabeled_df": unlabeled_df,
        "y_train": y_train,
        "y_test": y_test,
        "trust_selected": trust_selected,
        "train_frame": train_frame,
        "unlabeled_frame": unlabeled_frame,
        "test_frame": test_frame,
        "filtered_context_cols": filtered_context_cols,
    }


def _cv_and_holdout(
    *,
    labeled_meta_df: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    base: MLStatisticsWorkbenchConfig,
    backend_id: str,
    family_alpha: float,
    random_state_offset: int,
) -> tuple[dict[str, float], np.ndarray]:
    splits = _grouped_oof_splits(labeled_meta_df, base.model_selection_splits)
    oof = np.full(len(labeled_meta_df), np.nan, dtype=float)
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        val_recordings = set(labeled_meta_df.iloc[val_idx]["recording_key"].astype(str))
        fold_pseudo = pseudo_selected[
            ~pseudo_selected["recording_key"].astype(str).isin(val_recordings)
        ].reset_index(drop=True)
        train_fold, train_y, train_weight = _assemble_training_data(
            labeled_frame=labeled_frame.iloc[train_idx].reset_index(drop=True),
            labeled_y=labeled_y[train_idx],
            labeled_meta_df=labeled_meta_df.iloc[train_idx].reset_index(drop=True),
            pseudo_selected=fold_pseudo,
            unlabeled_frame=unlabeled_frame,
            family_alpha=family_alpha,
        )
        oof[val_idx] = _fit_backend_predict(
            X_train=train_fold,
            y_train=train_y,
            X_eval=labeled_frame.iloc[val_idx].reset_index(drop=True),
            sample_weight=train_weight,
            target_transform_mode=base.resolved_target_transform(),
            backend_id=backend_id,
            random_state=base.random_state + random_state_offset + split_id,
        )

    full_train, full_y, full_weight = _assemble_training_data(
        labeled_frame=labeled_frame,
        labeled_y=labeled_y,
        labeled_meta_df=labeled_meta_df,
        pseudo_selected=pseudo_selected,
        unlabeled_frame=unlabeled_frame,
        family_alpha=family_alpha,
    )
    holdout_pred = _fit_backend_predict(
        X_train=full_train,
        y_train=full_y,
        X_eval=test_frame.reset_index(drop=True),
        sample_weight=full_weight,
        target_transform_mode=base.resolved_target_transform(),
        backend_id=backend_id,
        random_state=base.random_state + random_state_offset + 100,
    )
    return score_predictions(labeled_y, oof), holdout_pred


def _run_with_config(config: FPosSignalEmbeddingStackConfig) -> Path:
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

    _log(config, "[embed] training waveform autoencoder")
    wf_train_emb, wf_unlabeled_emb, wf_test_emb, wf_hist = _signal_embeddings(
        train_signal=target_train_df.loc[:, prepared.wf_feature_cols],
        train_universe_signal=universe_df.loc[:, prepared.wf_feature_cols],
        unlabeled_signal=unlabeled_df.loc[:, prepared.wf_feature_cols],
        test_signal=target_test_df.loc[:, prepared.wf_feature_cols],
        prefix="wf",
        config=config,
    )
    _log(config, "[embed] training acg autoencoder")
    acg_train_emb, acg_unlabeled_emb, acg_test_emb, acg_hist = _signal_embeddings(
        train_signal=target_train_df.loc[:, prepared.acg_feature_cols],
        train_universe_signal=universe_df.loc[:, prepared.acg_feature_cols],
        unlabeled_signal=unlabeled_df.loc[:, prepared.acg_feature_cols],
        test_signal=target_test_df.loc[:, prepared.acg_feature_cols],
        prefix="acg",
        config=config,
    )

    variants = [
        ("reference_anchor_stack_xgboost", train_frame, unlabeled_frame, test_frame),
        (
            "wf_embed_anchor_stack_xgboost",
            pd.concat([train_frame.reset_index(drop=True), wf_train_emb], axis=1),
            pd.concat([unlabeled_frame.reset_index(drop=True), wf_unlabeled_emb], axis=1),
            pd.concat([test_frame.reset_index(drop=True), wf_test_emb], axis=1),
        ),
        (
            "acg_embed_anchor_stack_xgboost",
            pd.concat([train_frame.reset_index(drop=True), acg_train_emb], axis=1),
            pd.concat([unlabeled_frame.reset_index(drop=True), acg_unlabeled_emb], axis=1),
            pd.concat([test_frame.reset_index(drop=True), acg_test_emb], axis=1),
        ),
        (
            "wf_acg_embed_anchor_stack_xgboost",
            pd.concat([train_frame.reset_index(drop=True), wf_train_emb, acg_train_emb], axis=1),
            pd.concat([unlabeled_frame.reset_index(drop=True), wf_unlabeled_emb, acg_unlabeled_emb], axis=1),
            pd.concat([test_frame.reset_index(drop=True), wf_test_emb, acg_test_emb], axis=1),
        ),
    ]

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []
    histories = {"wf": wf_hist, "acg": acg_hist}

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
            random_state_offset=2200 + (100 * idx),
        )
        holdout_metrics = score_predictions(y_test, holdout_pred)
        metric_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family="signal_embedding_stack",
                backend="xgboost_default",
                source_target="fpos",
                metrics=holdout_metrics,
                notes="conv embedding stack variant",
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family="signal_embedding_stack",
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
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows).sort_values(["cv_r2", "cv_mae"], ascending=[False, True]).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    pd.DataFrame(histories).to_csv(output_dir / "embedding_history.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(state["split"].manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosSignalEmbeddingStackConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosSignalEmbeddingStackConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.max_pseudo_multiplier = 1.0
        cli_config.max_pseudo_per_recording = 24
        cli_config.max_pseudo_per_cluster = 64
        cli_config.ae_epochs = 4
        cli_config.conv_embed_dim = 6
        cli_config.universe_max_rows = 4000
    _run_with_config(cli_config)
