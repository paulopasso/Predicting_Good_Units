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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qc_thesis.modeling.stacking.fpos_signal_embedding_stack import (
    FPosSignalEmbeddingStackConfig,
    _cv_and_holdout,
    _prepare_base_frames,
    _assemble_training_data as _signal_assemble_training_data,
    _signal_embeddings,
)
from qc_thesis.modeling.stacking.fpos_backend_sweep import _fit_backend_predict, _grouped_oof_splits
from qc_thesis.modeling.stacking.fpos_family_robustness import _family_weights
from qc_thesis.modeling.stacking.methods import make_metric_row, make_prediction_frame, score_predictions


@dataclass
class FPosTreeInteractionStackConfig(FPosSignalEmbeddingStackConfig):
    interaction_trees: int = 24
    interaction_learning_rate: float = 0.05
    interaction_max_depth: int = 3
    interaction_min_samples_leaf: int = 8

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_tree_interaction_stack_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fpos waveform tree-interaction stack benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosTreeInteractionStackConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _interaction_feature_subset(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "source_pred",
        "anchor_fp_score",
        "matched_support_conf",
        "anchor_support_margin",
        "amp_bimodality",
        "isi_violation_rate",
        "acg_26",
        "n_waveform_confusable_recording_pct_recording_iqr",
        "amp_mean_recording_pct",
        "si_amplitude_cv_range",
        "wf_embed_00",
        "wf_embed_01",
        "wf_embed_02",
        "wf_embed_03",
    ]
    return [col for col in preferred if col in frame.columns]


def _fit_leaf_encoder(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    *,
    config: FPosTreeInteractionStackConfig,
    random_state: int,
) -> tuple[GradientBoostingRegressor, SimpleImputer, OneHotEncoder]:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    model = GradientBoostingRegressor(
        n_estimators=config.interaction_trees,
        learning_rate=config.interaction_learning_rate,
        max_depth=config.interaction_max_depth,
        min_samples_leaf=config.interaction_min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train_imp, y_train, sample_weight=sample_weight)
    leaves = model.apply(X_train_imp)
    leaves_2d = np.asarray(leaves).reshape(len(X_train_imp), -1)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(leaves_2d.astype(str))
    return model, imputer, encoder


def _transform_leaf_features(
    model: GradientBoostingRegressor,
    imputer: SimpleImputer,
    encoder: OneHotEncoder,
    X_eval: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    X_eval_imp = imputer.transform(X_eval)
    leaves = model.apply(X_eval_imp)
    leaves_2d = np.asarray(leaves).reshape(len(X_eval_imp), -1)
    encoded = encoder.transform(leaves_2d.astype(str))
    cols = [f"{prefix}_leaf_{i:03d}" for i in range(encoded.shape[1])]
    return pd.DataFrame(encoded.astype(np.float32), columns=cols)


def _augment_with_interactions(
    *,
    train_frame: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    config: FPosTreeInteractionStackConfig,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    interaction_cols = _interaction_feature_subset(train_frame)
    model, imputer, encoder = _fit_leaf_encoder(
        train_frame.loc[:, interaction_cols],
        y_train,
        sample_weight,
        config=config,
        random_state=random_state,
    )
    train_leaf = _transform_leaf_features(model, imputer, encoder, train_frame.loc[:, interaction_cols], prefix="tree")
    unlabeled_leaf = _transform_leaf_features(model, imputer, encoder, unlabeled_frame.loc[:, interaction_cols], prefix="tree")
    test_leaf = _transform_leaf_features(model, imputer, encoder, test_frame.loc[:, interaction_cols], prefix="tree")
    return (
        pd.concat([train_frame.reset_index(drop=True), train_leaf], axis=1),
        pd.concat([unlabeled_frame.reset_index(drop=True), unlabeled_leaf], axis=1),
        pd.concat([test_frame.reset_index(drop=True), test_leaf], axis=1),
        int(train_leaf.shape[1]),
    )


def _cv_and_holdout_tree_interaction(
    *,
    labeled_meta_df: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    base: Any,
    config: FPosTreeInteractionStackConfig,
    family_alpha: float,
    random_state_offset: int,
) -> tuple[dict[str, float], np.ndarray, int]:
    splits = _grouped_oof_splits(labeled_meta_df, base.model_selection_splits)
    oof = np.full(len(labeled_meta_df), np.nan, dtype=float)
    leaf_dim = 0
    interaction_cols = _interaction_feature_subset(labeled_frame)
    for split_id, (train_idx, val_idx) in enumerate(splits, start=1):
        val_recordings = set(labeled_meta_df.iloc[val_idx]["recording_key"].astype(str))
        fold_pseudo = pseudo_selected[
            ~pseudo_selected["recording_key"].astype(str).isin(val_recordings)
        ].reset_index(drop=True)
        train_labeled = labeled_frame.iloc[train_idx].reset_index(drop=True)
        train_meta = labeled_meta_df.iloc[train_idx].reset_index(drop=True)
        train_weight = _family_weights(train_meta["study_set"], alpha=family_alpha)
        aug_train_labeled, aug_unlabeled_fold, aug_val_fold, leaf_dim = _augment_with_interactions(
            train_frame=train_labeled,
            unlabeled_frame=unlabeled_frame,
            test_frame=labeled_frame.iloc[val_idx].reset_index(drop=True),
            y_train=labeled_y[train_idx],
            sample_weight=train_weight,
            config=config,
            random_state=config.random_state + random_state_offset + split_id,
        )
        train_full, train_y, train_sample_weight = _signal_assemble_training_data(
            labeled_frame=aug_train_labeled,
            labeled_y=labeled_y[train_idx],
            labeled_meta_df=train_meta,
            pseudo_selected=fold_pseudo,
            unlabeled_frame=aug_unlabeled_fold,
            family_alpha=family_alpha,
        )
        oof[val_idx] = _fit_backend_predict(
            X_train=train_full,
            y_train=train_y,
            X_eval=aug_val_fold.reset_index(drop=True),
            sample_weight=train_sample_weight,
            target_transform_mode=base.resolved_target_transform(),
            backend_id="xgboost_default",
            random_state=base.random_state + random_state_offset + split_id,
        )

    full_weight = _family_weights(labeled_meta_df["study_set"], alpha=family_alpha)
    aug_train_full, aug_unlabeled_full, aug_test_full, leaf_dim = _augment_with_interactions(
        train_frame=labeled_frame.reset_index(drop=True),
        unlabeled_frame=unlabeled_frame,
        test_frame=test_frame,
        y_train=labeled_y,
        sample_weight=full_weight,
        config=config,
        random_state=config.random_state + random_state_offset + 100,
    )
    train_full, train_y, train_sample_weight = _signal_assemble_training_data(
        labeled_frame=aug_train_full,
        labeled_y=labeled_y,
        labeled_meta_df=labeled_meta_df,
        pseudo_selected=pseudo_selected,
        unlabeled_frame=aug_unlabeled_full,
        family_alpha=family_alpha,
    )
    holdout_pred = _fit_backend_predict(
        X_train=train_full,
        y_train=train_y,
        X_eval=aug_test_full.reset_index(drop=True),
        sample_weight=train_sample_weight,
        target_transform_mode=base.resolved_target_transform(),
        backend_id="xgboost_default",
        random_state=base.random_state + random_state_offset + 100,
    )
    return score_predictions(labeled_y, oof), holdout_pred, int(leaf_dim)


def _run_with_config(config: FPosTreeInteractionStackConfig) -> Path:
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

    wf_train_frame = pd.concat([train_frame.reset_index(drop=True), wf_train_emb], axis=1)
    wf_unlabeled_frame = pd.concat([unlabeled_frame.reset_index(drop=True), wf_unlabeled_emb], axis=1)
    wf_test_frame = pd.concat([test_frame.reset_index(drop=True), wf_test_emb], axis=1)

    variants = [
        ("wf_embed_anchor_stack_xgboost", "base", wf_train_frame, wf_unlabeled_frame, wf_test_frame),
        ("wf_embed_tree_interact_xgboost", "tree_interact", wf_train_frame, wf_unlabeled_frame, wf_test_frame),
    ]

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_rows: list[dict[str, Any]] = []

    for idx, (variant_id, variant_kind, train_variant, unlabeled_variant, test_variant) in enumerate(variants, start=1):
        _log(config, f"[variant] {variant_id}")
        if variant_kind == "base":
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
            leaf_dim = 0
        else:
            cv_metrics, holdout_pred, leaf_dim = _cv_and_holdout_tree_interaction(
                labeled_meta_df=target_train_df,
                labeled_frame=train_variant,
                labeled_y=y_train,
                pseudo_selected=trust_selected,
                unlabeled_frame=unlabeled_variant,
                test_frame=test_variant,
                base=base,
                config=config,
                family_alpha=1.0,
                random_state_offset=3600 + (100 * idx),
            )
        holdout_metrics = score_predictions(y_test, holdout_pred)
        metric_rows.append(
            make_metric_row(
                target="fpos",
                variant_id=variant_id,
                family="tree_interaction_stack",
                backend="xgboost_default",
                source_target="fpos",
                metrics=holdout_metrics,
                notes=f"depth3 interaction features leaf_dim={leaf_dim}",
            )
        )
        prediction_frames.append(
            make_prediction_frame(
                target="fpos",
                variant_id=variant_id,
                family="tree_interaction_stack",
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
                "leaf_dim": int(leaf_dim),
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows).sort_values(["cv_r2", "cv_mae"], ascending=[False, True]).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    cv_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    pd.DataFrame([{"wf_train_recon_loss": wf_hist["train_recon_loss"]}]).to_csv(
        output_dir / "interaction_history.csv",
        index=False,
    )
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(state["split"].manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosTreeInteractionStackConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosTreeInteractionStackConfig(
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
        cli_config.interaction_trees = 12
    _run_with_config(cli_config)
