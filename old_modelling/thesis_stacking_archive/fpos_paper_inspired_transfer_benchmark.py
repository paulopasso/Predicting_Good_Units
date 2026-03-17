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
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qc_thesis.modeling.stacking.fpos_backend_sweep import _fit_backend_predict, _grouped_oof_splits
from qc_thesis.modeling.stacking.fpos_signal_embedding_stack import (
    FPosSignalEmbeddingStackConfig,
    _assemble_training_data,
    _prepare_base_frames,
    _signal_embeddings,
)
from qc_thesis.modeling.stacking.methods import make_metric_row, make_prediction_frame, score_predictions


@dataclass
class FPosPaperInspiredTransferConfig(FPosSignalEmbeddingStackConfig):
    knn_k: int = 6
    local_support_k: int = 4

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_paper_inspired_transfer_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-inspired fpos transfer benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosPaperInspiredTransferConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _ridge_adapter_score(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_eval: np.ndarray,
    *,
    sample_weight: np.ndarray,
    random_state: int,
) -> np.ndarray:
    scaler = StandardScaler()
    Zt = scaler.fit_transform(Z_train)
    Ze = scaler.transform(Z_eval)
    model = Ridge(alpha=1.0, random_state=random_state)
    model.fit(Zt, y_train, sample_weight=sample_weight)
    return np.clip(np.asarray(model.predict(Ze), dtype=float), 0.0, 1.0)


def _fewshot_family_score(
    train_meta: pd.DataFrame,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    eval_meta: pd.DataFrame,
    Z_eval: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    pred = np.zeros(len(Z_eval), dtype=float)
    global_nn = NearestNeighbors(n_neighbors=min(max(1, k), len(Z_train)))
    global_nn.fit(Z_train)
    for i, (study, z) in enumerate(zip(eval_meta["study_set"].astype(str), Z_eval, strict=False)):
        mask = train_meta["study_set"].astype(str).to_numpy() == str(study)
        if int(mask.sum()) >= max(2, k):
            Z_local = Z_train[mask]
            y_local = y_train[mask]
        else:
            Z_local = Z_train
            y_local = y_train
        nn = NearestNeighbors(n_neighbors=min(max(1, k), len(Z_local)))
        nn.fit(Z_local)
        distances, idx = nn.kneighbors(z.reshape(1, -1), return_distance=True)
        w = 1.0 / np.clip(distances[0], 1e-4, None)
        pred[i] = float(np.average(y_local[idx[0]], weights=w))
    return np.clip(pred, 0.0, 1.0)


def _labeled_neighborhood_features(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_eval: np.ndarray,
    *,
    k: int,
    prefix: str,
) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=min(max(1, k), len(Z_train)))
    nn.fit(Z_train)
    distances, idx = nn.kneighbors(Z_eval, return_distance=True)
    neighbor_y = y_train[idx]
    weights = 1.0 / np.clip(distances, 1e-4, None)
    weighted_mean = np.sum(neighbor_y * weights, axis=1) / np.sum(weights, axis=1)
    weighted_std = np.sqrt(np.sum(weights * np.square(neighbor_y - weighted_mean[:, None]), axis=1) / np.sum(weights, axis=1))
    return pd.DataFrame(
        {
            f"{prefix}_nbr_mean": weighted_mean.astype(np.float32),
            f"{prefix}_nbr_std": weighted_std.astype(np.float32),
            f"{prefix}_nbr_min_dist": distances[:, 0].astype(np.float32),
            f"{prefix}_nbr_density": (1.0 / np.clip(distances.mean(axis=1), 1e-4, None)).astype(np.float32),
        }
    )


def _unlabeled_manifold_features(
    Z_unlabeled: np.ndarray,
    unlabeled_frame: pd.DataFrame,
    Z_eval: np.ndarray,
    *,
    k: int,
    prefix: str,
) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=min(max(1, k), len(Z_unlabeled)))
    nn.fit(Z_unlabeled)
    distances, idx = nn.kneighbors(Z_eval, return_distance=True)
    source_pred = unlabeled_frame["source_pred"].to_numpy(dtype=float)[idx]
    anchor_score = unlabeled_frame["anchor_fp_score"].to_numpy(dtype=float)[idx]
    support_conf = unlabeled_frame["matched_support_conf"].to_numpy(dtype=float)[idx]
    return pd.DataFrame(
        {
            f"{prefix}_u_dist": distances.mean(axis=1).astype(np.float32),
            f"{prefix}_u_source_mean": source_pred.mean(axis=1).astype(np.float32),
            f"{prefix}_u_anchor_mean": anchor_score.mean(axis=1).astype(np.float32),
            f"{prefix}_u_support_mean": support_conf.mean(axis=1).astype(np.float32),
        }
    )


def _compute_fold_features(
    *,
    train_meta: pd.DataFrame,
    train_frame: pd.DataFrame,
    train_y: np.ndarray,
    unlabeled_meta: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    eval_meta: pd.DataFrame,
    eval_frame: pd.DataFrame,
    wf_train_emb: pd.DataFrame,
    wf_unlabeled_emb: pd.DataFrame,
    wf_eval_emb: pd.DataFrame,
    config: FPosPaperInspiredTransferConfig,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    Z_train = wf_train_emb.to_numpy(dtype=float)
    Z_eval = wf_eval_emb.to_numpy(dtype=float)
    Z_unlabeled = wf_unlabeled_emb.to_numpy(dtype=float)
    sample_weight = np.ones(len(train_y), dtype=np.float32)

    adapter_score_eval = _ridge_adapter_score(
        Z_train,
        train_y,
        Z_eval,
        sample_weight=sample_weight,
        random_state=random_state,
    )
    adapter_score_unlabeled = _ridge_adapter_score(
        Z_train,
        train_y,
        Z_unlabeled,
        sample_weight=sample_weight,
        random_state=random_state,
    )
    fewshot_eval = _fewshot_family_score(
        train_meta,
        Z_train,
        train_y,
        eval_meta,
        Z_eval,
        k=config.local_support_k,
    )
    fewshot_unlabeled = _fewshot_family_score(
        train_meta,
        Z_train,
        train_y,
        unlabeled_meta.loc[:, ["study_set"]].reset_index(drop=True),
        Z_unlabeled,
        k=config.local_support_k,
    )
    labeled_eval = _labeled_neighborhood_features(
        Z_train,
        train_y,
        Z_eval,
        k=config.knn_k,
        prefix="lab",
    )
    labeled_unlabeled = _labeled_neighborhood_features(
        Z_train,
        train_y,
        Z_unlabeled,
        k=config.knn_k,
        prefix="lab",
    )
    unlabeled_eval = _unlabeled_manifold_features(
        Z_unlabeled,
        unlabeled_frame,
        Z_eval,
        k=config.knn_k,
        prefix="unlab",
    )
    unlabeled_unlabeled = _unlabeled_manifold_features(
        Z_unlabeled,
        unlabeled_frame,
        Z_unlabeled,
        k=config.knn_k,
        prefix="unlab",
    )

    def plus(base: pd.DataFrame, *frames: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([base.reset_index(drop=True), *[f.reset_index(drop=True) for f in frames]], axis=1)

    eval_frames = {
        "baseline": plus(eval_frame, wf_eval_emb),
        "adapter": plus(eval_frame, wf_eval_emb, pd.DataFrame({"adapter_score": adapter_score_eval})),
        "fewshot": plus(eval_frame, wf_eval_emb, pd.DataFrame({"fewshot_family_score": fewshot_eval})),
        "labeled_knn": plus(eval_frame, wf_eval_emb, labeled_eval),
        "unlabeled_manifold": plus(eval_frame, wf_eval_emb, unlabeled_eval),
        "combined": plus(
            eval_frame,
            wf_eval_emb,
            pd.DataFrame({"adapter_score": adapter_score_eval, "fewshot_family_score": fewshot_eval}),
            labeled_eval,
            unlabeled_eval,
        ),
    }
    unlabeled_frames = {
        "baseline": plus(unlabeled_frame, wf_unlabeled_emb),
        "adapter": plus(unlabeled_frame, wf_unlabeled_emb, pd.DataFrame({"adapter_score": adapter_score_unlabeled})),
        "fewshot": plus(unlabeled_frame, wf_unlabeled_emb, pd.DataFrame({"fewshot_family_score": fewshot_unlabeled})),
        "labeled_knn": plus(unlabeled_frame, wf_unlabeled_emb, labeled_unlabeled),
        "unlabeled_manifold": plus(unlabeled_frame, wf_unlabeled_emb, unlabeled_unlabeled),
        "combined": plus(
            unlabeled_frame,
            wf_unlabeled_emb,
            pd.DataFrame({"adapter_score": adapter_score_unlabeled, "fewshot_family_score": fewshot_unlabeled}),
            labeled_unlabeled,
            unlabeled_unlabeled,
        ),
    }
    train_frames = {
        "baseline": plus(train_frame, wf_train_emb),
        "adapter": plus(train_frame, wf_train_emb, pd.DataFrame({"adapter_score": _ridge_adapter_score(Z_train, train_y, Z_train, sample_weight=sample_weight, random_state=random_state)})),
        "fewshot": plus(train_frame, wf_train_emb, pd.DataFrame({"fewshot_family_score": _fewshot_family_score(train_meta, Z_train, train_y, train_meta, Z_train, k=config.local_support_k)})),
        "labeled_knn": plus(train_frame, wf_train_emb, _labeled_neighborhood_features(Z_train, train_y, Z_train, k=config.knn_k, prefix="lab")),
        "unlabeled_manifold": plus(train_frame, wf_train_emb, _unlabeled_manifold_features(Z_unlabeled, unlabeled_frame, Z_train, k=config.knn_k, prefix="unlab")),
        "combined": plus(
            train_frame,
            wf_train_emb,
            pd.DataFrame(
                {
                    "adapter_score": _ridge_adapter_score(Z_train, train_y, Z_train, sample_weight=sample_weight, random_state=random_state),
                    "fewshot_family_score": _fewshot_family_score(train_meta, Z_train, train_y, train_meta, Z_train, k=config.local_support_k),
                }
            ),
            _labeled_neighborhood_features(Z_train, train_y, Z_train, k=config.knn_k, prefix="lab"),
            _unlabeled_manifold_features(Z_unlabeled, unlabeled_frame, Z_train, k=config.knn_k, prefix="unlab"),
        ),
    }
    return {"train": train_frames, "eval": eval_frames, "unlabeled": unlabeled_frames}


def _run_with_config(config: FPosPaperInspiredTransferConfig) -> Path:
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
        universe_df = universe_df.sample(n=int(config.universe_max_rows), random_state=config.random_state).reset_index(drop=True)

    _log(config, "[embed] training waveform autoencoder")
    wf_train_emb, wf_unlabeled_emb, wf_test_emb, wf_hist = _signal_embeddings(
        train_signal=target_train_df.loc[:, prepared.wf_feature_cols],
        train_universe_signal=universe_df.loc[:, prepared.wf_feature_cols],
        unlabeled_signal=unlabeled_df.loc[:, prepared.wf_feature_cols],
        test_signal=target_test_df.loc[:, prepared.wf_feature_cols],
        prefix="wf",
        config=config,
    )

    variant_ids = ["baseline", "adapter", "fewshot", "labeled_knn", "unlabeled_manifold", "combined"]
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []

    splits = _grouped_oof_splits(target_train_df, base.model_selection_splits)
    oof_preds = {variant: np.full(len(target_train_df), np.nan, dtype=float) for variant in variant_ids}

    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        _log(config, f"[cv] fold {split_id}")
        val_recordings = set(target_train_df.iloc[val_idx]["recording_key"].astype(str))
        fold_pseudo = trust_selected[
            ~trust_selected["recording_key"].astype(str).isin(val_recordings)
        ].reset_index(drop=True)
        feat = _compute_fold_features(
            train_meta=target_train_df.iloc[train_idx].reset_index(drop=True),
            train_frame=train_frame.iloc[train_idx].reset_index(drop=True),
            train_y=y_train[train_idx],
            unlabeled_meta=unlabeled_df.reset_index(drop=True),
            unlabeled_frame=unlabeled_frame.reset_index(drop=True),
            eval_meta=target_train_df.iloc[val_idx].reset_index(drop=True),
            eval_frame=train_frame.iloc[val_idx].reset_index(drop=True),
            wf_train_emb=wf_train_emb.iloc[train_idx].reset_index(drop=True),
            wf_unlabeled_emb=wf_unlabeled_emb.reset_index(drop=True),
            wf_eval_emb=wf_train_emb.iloc[val_idx].reset_index(drop=True),
            config=config,
            random_state=config.random_state + split_id,
        )
        for variant in variant_ids:
            train_aug, train_y_aug, train_weight_aug = _assemble_training_data(
                labeled_frame=feat["train"][variant],
                labeled_y=y_train[train_idx],
                labeled_meta_df=target_train_df.iloc[train_idx].reset_index(drop=True),
                pseudo_selected=fold_pseudo,
                unlabeled_frame=feat["unlabeled"][variant],
                family_alpha=1.0,
            )
            pred = _fit_backend_predict(
                X_train=train_aug,
                y_train=train_y_aug,
                X_eval=feat["eval"][variant],
                sample_weight=train_weight_aug,
                target_transform_mode=base.resolved_target_transform(),
                backend_id="xgboost_default",
                random_state=base.random_state + 5000 + split_id,
            )
            oof_preds[variant][val_idx] = pred

    full_feat = _compute_fold_features(
        train_meta=target_train_df.reset_index(drop=True),
        train_frame=train_frame.reset_index(drop=True),
        train_y=y_train,
        unlabeled_meta=unlabeled_df.reset_index(drop=True),
        unlabeled_frame=unlabeled_frame.reset_index(drop=True),
        eval_meta=target_test_df.reset_index(drop=True),
        eval_frame=test_frame.reset_index(drop=True),
        wf_train_emb=wf_train_emb.reset_index(drop=True),
        wf_unlabeled_emb=wf_unlabeled_emb.reset_index(drop=True),
        wf_eval_emb=wf_test_emb.reset_index(drop=True),
        config=config,
        random_state=config.random_state + 1000,
    )

    for variant in variant_ids:
        _log(config, f"[variant] {variant}")
        train_aug, train_y_aug, train_weight_aug = _assemble_training_data(
            labeled_frame=full_feat["train"][variant],
            labeled_y=y_train,
            labeled_meta_df=target_train_df.reset_index(drop=True),
            pseudo_selected=trust_selected,
            unlabeled_frame=full_feat["unlabeled"][variant],
            family_alpha=1.0,
        )
        holdout_pred = _fit_backend_predict(
            X_train=train_aug,
            y_train=train_y_aug,
            X_eval=full_feat["eval"][variant],
            sample_weight=train_weight_aug,
            target_transform_mode=base.resolved_target_transform(),
            backend_id="xgboost_default",
            random_state=base.random_state + 9000,
        )
        cv_metrics = score_predictions(y_train, oof_preds[variant])
        holdout_metrics = score_predictions(y_test, holdout_pred)
        variant_id = f"wf_embed_{variant}_xgboost"
        metric_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family="paper_inspired_transfer",
                backend="xgboost_default",
                source_target="fpos",
                metrics=holdout_metrics,
                notes=variant,
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family="paper_inspired_transfer",
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
    pd.DataFrame([{"wf_train_recon_loss": wf_hist["train_recon_loss"]}]).to_csv(output_dir / "feature_learning_history.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config) | {"parquet_path": str(config.parquet_path), "output_root": str(config.output_root)}, f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(state["split"].manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosPaperInspiredTransferConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosPaperInspiredTransferConfig(
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
