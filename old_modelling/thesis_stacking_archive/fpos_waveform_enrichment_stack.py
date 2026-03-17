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

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qc_thesis.modeling.stacking.fpos_signal_embedding_stack import (
    FPosSignalEmbeddingStackConfig,
    _cv_and_holdout,
    _prepare_base_frames,
    _signal_embeddings,
)
from qc_thesis.modeling.stacking.methods import make_metric_row, make_prediction_frame, score_predictions


@dataclass
class FPosWaveformEnrichmentStackConfig(FPosSignalEmbeddingStackConfig):
    dct_components: int = 8

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_waveform_enrichment_stack_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fpos waveform enrichment stack benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosWaveformEnrichmentStackConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _waveform_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)


def _normalize_waveforms(values: np.ndarray) -> np.ndarray:
    centered = values - np.nanmean(values, axis=1, keepdims=True)
    scale = np.linalg.norm(centered, axis=1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return centered / scale


def _dct_frame(values: np.ndarray, *, prefix: str, n_components: int) -> pd.DataFrame:
    x = _normalize_waveforms(values)
    n = x.shape[1]
    basis = []
    grid = np.arange(n, dtype=np.float32) + 0.5
    for k in range(1, n_components + 1):
        vec = np.cos(np.pi * k * grid / n).astype(np.float32)
        vec = vec / max(np.linalg.norm(vec), 1e-6)
        basis.append(vec)
    mat = np.stack(basis, axis=1)
    coeffs = x @ mat
    cols = [f"{prefix}_dct_{i:02d}" for i in range(coeffs.shape[1])]
    return pd.DataFrame(coeffs, columns=cols)


def _segment_pool_frame(values: np.ndarray, *, prefix: str) -> pd.DataFrame:
    x = _normalize_waveforms(values)
    feature_dict: dict[str, np.ndarray] = {}
    for segments in (2, 3, 5, 6):
        splits = np.array_split(np.arange(x.shape[1]), segments)
        for idx, seg in enumerate(splits):
            block = x[:, seg]
            feature_dict[f"{prefix}_seg{segments}_{idx:02d}_mean"] = block.mean(axis=1)
            feature_dict[f"{prefix}_seg{segments}_{idx:02d}_max"] = block.max(axis=1)
            feature_dict[f"{prefix}_seg{segments}_{idx:02d}_min"] = block.min(axis=1)
    return pd.DataFrame(feature_dict)


def _derivative_shape_frame(values: np.ndarray, *, prefix: str) -> pd.DataFrame:
    x = _normalize_waveforms(values)
    dx = np.diff(x, axis=1)
    ddx = np.diff(dx, axis=1)
    peak_idx = np.argmax(x, axis=1).astype(np.float32)
    trough_idx = np.argmin(x, axis=1).astype(np.float32)
    peak_val = np.max(x, axis=1)
    trough_val = np.min(x, axis=1)
    zero_cross = (np.diff(np.signbit(dx), axis=1) != 0).sum(axis=1).astype(np.float32)
    area_pre = x[:, : x.shape[1] // 2].sum(axis=1)
    area_post = x[:, x.shape[1] // 2 :].sum(axis=1)
    frame = pd.DataFrame(
        {
            f"{prefix}_diff_mean_abs": np.mean(np.abs(dx), axis=1),
            f"{prefix}_diff_max_abs": np.max(np.abs(dx), axis=1),
            f"{prefix}_diff_std": np.std(dx, axis=1),
            f"{prefix}_curv_mean_abs": np.mean(np.abs(ddx), axis=1),
            f"{prefix}_curv_max_abs": np.max(np.abs(ddx), axis=1),
            f"{prefix}_peak_idx": peak_idx,
            f"{prefix}_trough_idx": trough_idx,
            f"{prefix}_peak_minus_trough": peak_val - trough_val,
            f"{prefix}_peak_plus_trough": peak_val + trough_val,
            f"{prefix}_peak_trough_gap": np.abs(peak_idx - trough_idx),
            f"{prefix}_slope_zero_crossings": zero_cross,
            f"{prefix}_area_pre": area_pre,
            f"{prefix}_area_post": area_post,
            f"{prefix}_area_balance": area_post - area_pre,
        }
    )
    return frame


def _waveform_enrichment_frames(
    *,
    train_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    test_df: pd.DataFrame,
    wf_cols: list[str],
    config: FPosWaveformEnrichmentStackConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_values = _waveform_array(train_df, wf_cols)
    unlabeled_values = _waveform_array(unlabeled_df, wf_cols)
    test_values = _waveform_array(test_df, wf_cols)
    train_features = pd.concat(
        [
            _dct_frame(train_values, prefix="wf", n_components=config.dct_components),
            _segment_pool_frame(train_values, prefix="wf"),
            _derivative_shape_frame(train_values, prefix="wf"),
        ],
        axis=1,
    )
    unlabeled_features = pd.concat(
        [
            _dct_frame(unlabeled_values, prefix="wf", n_components=config.dct_components),
            _segment_pool_frame(unlabeled_values, prefix="wf"),
            _derivative_shape_frame(unlabeled_values, prefix="wf"),
        ],
        axis=1,
    )
    test_features = pd.concat(
        [
            _dct_frame(test_values, prefix="wf", n_components=config.dct_components),
            _segment_pool_frame(test_values, prefix="wf"),
            _derivative_shape_frame(test_values, prefix="wf"),
        ],
        axis=1,
    )
    return train_features, unlabeled_features, test_features


def _run_with_config(config: FPosWaveformEnrichmentStackConfig) -> Path:
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
    _log(config, "[shape] building waveform enrichment features")
    wf_train_rich, wf_unlabeled_rich, wf_test_rich = _waveform_enrichment_frames(
        train_df=target_train_df,
        unlabeled_df=unlabeled_df,
        test_df=target_test_df,
        wf_cols=prepared.wf_feature_cols,
        config=config,
    )

    variants = [
        ("wf_embed_anchor_stack_xgboost", train_frame.reset_index(drop=True), unlabeled_frame.reset_index(drop=True), test_frame.reset_index(drop=True)),
        (
            "wf_rich_anchor_stack_xgboost",
            pd.concat([train_frame.reset_index(drop=True), wf_train_rich], axis=1),
            pd.concat([unlabeled_frame.reset_index(drop=True), wf_unlabeled_rich], axis=1),
            pd.concat([test_frame.reset_index(drop=True), wf_test_rich], axis=1),
        ),
        (
            "wf_embed_rich_anchor_stack_xgboost",
            pd.concat([train_frame.reset_index(drop=True), wf_train_emb, wf_train_rich], axis=1),
            pd.concat([unlabeled_frame.reset_index(drop=True), wf_unlabeled_emb, wf_unlabeled_rich], axis=1),
            pd.concat([test_frame.reset_index(drop=True), wf_test_emb, wf_test_rich], axis=1),
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
            random_state_offset=3200 + (100 * idx),
        )
        holdout_metrics = score_predictions(y_test, holdout_pred)
        metric_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family="waveform_enrichment_stack",
                backend="xgboost_default",
                source_target="fpos",
                metrics=holdout_metrics,
                notes="waveform-enriched stack variant",
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family="waveform_enrichment_stack",
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
    pd.DataFrame([wf_hist]).to_csv(output_dir / "embedding_history.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(state["split"].manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosWaveformEnrichmentStackConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosWaveformEnrichmentStackConfig(
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
