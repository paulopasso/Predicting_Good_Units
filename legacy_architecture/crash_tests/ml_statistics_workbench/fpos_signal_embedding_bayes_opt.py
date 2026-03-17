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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.ml_statistics_workbench.fpos_signal_embedding_stack import (
    FPosSignalEmbeddingStackConfig,
    _assemble_training_data,
    _prepare_base_frames,
    _signal_embeddings,
)
from crash_tests.ml_statistics_workbench.fpos_backend_sweep import _grouped_oof_splits
from crash_tests.ml_statistics_workbench.methods import make_metric_row, make_prediction_frame, score_predictions
from crash_tests.mmd_domain_adaptation.model import target_inverse, target_transform

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore[assignment]
    _OPTUNA_AVAILABLE = False


@dataclass
class FPosSignalEmbeddingBayesOptConfig(FPosSignalEmbeddingStackConfig):
    n_trials: int = 12
    timeout_seconds: int | None = None

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_signal_embedding_bayes_opt_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian optimization for the fpos signal-embedding stack")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="crash_tests/ml_statistics_workbench/outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosSignalEmbeddingBayesOptConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _fit_custom_xgboost_predict(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    sample_weight: np.ndarray,
    random_state: int,
    params: dict[str, float | int],
) -> np.ndarray:
    from xgboost import XGBRegressor

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_eval_imp = imputer.transform(X_eval)
    y_train_t = target_transform(y_train, mode="logit_clip")
    model = XGBRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        gamma=float(params["gamma"]),
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train_imp, y_train_t, sample_weight=sample_weight)
    pred_t = np.asarray(model.predict(X_eval_imp), dtype=float)
    return np.clip(target_inverse(pred_t, mode="logit_clip"), 0.0, 1.0)


def _cv_and_holdout_custom(
    *,
    labeled_meta_df: pd.DataFrame,
    labeled_frame: pd.DataFrame,
    labeled_y: np.ndarray,
    pseudo_selected: pd.DataFrame,
    unlabeled_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    family_alpha: float,
    random_state: int,
    params: dict[str, float | int],
) -> tuple[dict[str, float], np.ndarray]:
    splits = _grouped_oof_splits(labeled_meta_df, 3)
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
        oof[val_idx] = _fit_custom_xgboost_predict(
            X_train=train_fold,
            y_train=train_y,
            X_eval=labeled_frame.iloc[val_idx].reset_index(drop=True),
            sample_weight=train_weight,
            random_state=random_state + split_id,
            params=params,
        )

    full_train, full_y, full_weight = _assemble_training_data(
        labeled_frame=labeled_frame,
        labeled_y=labeled_y,
        labeled_meta_df=labeled_meta_df,
        pseudo_selected=pseudo_selected,
        unlabeled_frame=unlabeled_frame,
        family_alpha=family_alpha,
    )
    holdout_pred = _fit_custom_xgboost_predict(
        X_train=full_train,
        y_train=full_y,
        X_eval=test_frame.reset_index(drop=True),
        sample_weight=full_weight,
        random_state=random_state + 100,
        params=params,
    )
    return score_predictions(labeled_y, oof), holdout_pred


def _sample_trial_config(trial: "optuna.Trial", base_config: FPosSignalEmbeddingBayesOptConfig) -> tuple[FPosSignalEmbeddingStackConfig, dict[str, float | int], float]:
    config = FPosSignalEmbeddingStackConfig(
        parquet_path=base_config.parquet_path,
        output_root=base_config.output_root,
        run_name=base_config.run_name,
        random_state=base_config.random_state,
        paired_final_holdout_rows=base_config.paired_final_holdout_rows,
        max_pseudo_multiplier=trial.suggest_float("max_pseudo_multiplier", 1.0, 1.8),
        max_pseudo_per_recording=trial.suggest_int("max_pseudo_per_recording", 24, 64, step=8),
        max_pseudo_per_cluster=trial.suggest_int("max_pseudo_per_cluster", 64, 160, step=16),
        max_pseudo_per_labeled_ratio=trial.suggest_float("max_pseudo_per_labeled_ratio", 1.0, 1.8),
        cluster_trust_power=trial.suggest_float("cluster_trust_power", 1.0, 1.8),
        study_trust_weight=trial.suggest_float("study_trust_weight", 0.1, 0.5),
        min_trust=trial.suggest_float("min_trust", 0.10, 0.35),
        conv_embed_dim=trial.suggest_int("conv_embed_dim", 6, 16, step=2),
        conv_hidden_channels=trial.suggest_int("conv_hidden_channels", 8, 24, step=4),
        ae_epochs=trial.suggest_int("ae_epochs", 16, 40, step=4),
        batch_size=base_config.batch_size,
        learning_rate=trial.suggest_float("embed_learning_rate", 1e-3, 5e-3, log=True),
        universe_max_rows=base_config.universe_max_rows,
        verbose=False,
    )
    model_params: dict[str, float | int] = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 220, 700, step=40),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.02, 0.10, log=True),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 5),
        "min_child_weight": trial.suggest_float("xgb_min_child_weight", 2.0, 8.0),
        "subsample": trial.suggest_float("xgb_subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.5, 4.0),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.0, 1.0),
        "gamma": trial.suggest_float("xgb_gamma", 0.0, 0.5),
    }
    family_alpha = trial.suggest_float("family_alpha", 0.7, 1.4)
    return config, model_params, family_alpha


def _run_with_config(config: FPosSignalEmbeddingBayesOptConfig) -> Path:
    if not _OPTUNA_AVAILABLE:
        raise ImportError("optuna is required for Bayesian optimization runs")

    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(config, "[baseline] preparing base state")
    baseline_state = _prepare_base_frames(config)
    prepared = baseline_state["prepared"]
    target_train_df = baseline_state["target_train_df"]
    target_test_df = baseline_state["target_test_df"]
    unlabeled_df = baseline_state["unlabeled_df"]
    y_train = baseline_state["y_train"]
    y_test = baseline_state["y_test"]
    train_frame = baseline_state["train_frame"]
    unlabeled_frame = baseline_state["unlabeled_frame"]
    test_frame = baseline_state["test_frame"]
    trust_selected = baseline_state["trust_selected"]
    universe_df = pd.concat(
        [
            baseline_state["split"].hybrid_source.reset_index(drop=True),
            baseline_state["split"].paired_non_test_all.reset_index(drop=True),
            baseline_state["split"].ssl_pool.reset_index(drop=True),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["row_uid"]).reset_index(drop=True)
    if config.universe_max_rows is not None and len(universe_df) > int(config.universe_max_rows):
        universe_df = universe_df.sample(
            n=int(config.universe_max_rows),
            random_state=config.random_state,
        ).reset_index(drop=True)
    wf_train_emb, wf_unlabeled_emb, wf_test_emb, _ = _signal_embeddings(
        train_signal=target_train_df.loc[:, prepared.wf_feature_cols],
        train_universe_signal=universe_df.loc[:, prepared.wf_feature_cols],
        unlabeled_signal=unlabeled_df.loc[:, prepared.wf_feature_cols],
        test_signal=target_test_df.loc[:, prepared.wf_feature_cols],
        prefix="wf",
        config=config,
    )
    baseline_train_variant = pd.concat([train_frame.reset_index(drop=True), wf_train_emb], axis=1)
    baseline_unlabeled_variant = pd.concat([unlabeled_frame.reset_index(drop=True), wf_unlabeled_emb], axis=1)
    baseline_test_variant = pd.concat([test_frame.reset_index(drop=True), wf_test_emb], axis=1)

    baseline_params = {
        "n_estimators": 350,
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 4.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
    }
    baseline_cv, baseline_holdout = _cv_and_holdout_custom(
        labeled_meta_df=target_train_df,
        labeled_frame=baseline_train_variant,
        labeled_y=y_train,
        pseudo_selected=trust_selected,
        unlabeled_frame=baseline_unlabeled_variant,
        test_frame=baseline_test_variant,
        family_alpha=1.0,
        random_state=config.random_state + 10,
        params=baseline_params,
    )
    baseline_holdout_metrics = score_predictions(y_test, baseline_holdout)

    trial_rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] = {}

    def objective(trial: "optuna.Trial") -> float:
        trial_config, model_params, family_alpha = _sample_trial_config(trial, config)
        state = _prepare_base_frames(trial_config)
        universe_df = pd.concat(
            [
                state["split"].hybrid_source.reset_index(drop=True),
                state["split"].paired_non_test_all.reset_index(drop=True),
                state["split"].ssl_pool.reset_index(drop=True),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["row_uid"]).reset_index(drop=True)
        if trial_config.universe_max_rows is not None and len(universe_df) > int(trial_config.universe_max_rows):
            universe_df = universe_df.sample(
                n=int(trial_config.universe_max_rows),
                random_state=trial_config.random_state,
            ).reset_index(drop=True)
        wf_train_emb, wf_unlabeled_emb, wf_test_emb, wf_hist = _signal_embeddings(
            train_signal=state["target_train_df"].loc[:, prepared.wf_feature_cols],
            train_universe_signal=universe_df.loc[:, prepared.wf_feature_cols],
            unlabeled_signal=state["unlabeled_df"].loc[:, prepared.wf_feature_cols],
            test_signal=state["target_test_df"].loc[:, prepared.wf_feature_cols],
            prefix="wf",
            config=trial_config,
        )
        train_variant = pd.concat([state["train_frame"].reset_index(drop=True), wf_train_emb], axis=1)
        unlabeled_variant = pd.concat([state["unlabeled_frame"].reset_index(drop=True), wf_unlabeled_emb], axis=1)
        test_variant = pd.concat([state["test_frame"].reset_index(drop=True), wf_test_emb], axis=1)
        cv_metrics, holdout_pred = _cv_and_holdout_custom(
            labeled_meta_df=state["target_train_df"],
            labeled_frame=train_variant,
            labeled_y=state["y_train"],
            pseudo_selected=state["trust_selected"],
            unlabeled_frame=unlabeled_variant,
            test_frame=test_variant,
            family_alpha=family_alpha,
            random_state=trial_config.random_state + 1000 + trial.number,
            params=model_params,
        )
        holdout_metrics = score_predictions(state["y_test"], holdout_pred)
        row = {
            "trial": int(trial.number),
            "cv_mae": float(cv_metrics["mae"]),
            "cv_r2": float(cv_metrics["r2"]),
            "holdout_mae": float(holdout_metrics["mae"]),
            "holdout_r2": float(holdout_metrics["r2"]),
            "family_alpha": float(family_alpha),
            "selected_rows": int(len(state["trust_selected"])),
            "embed_recon_loss": float(wf_hist["train_recon_loss"]),
        }
        row.update({k: (float(v) if isinstance(v, float) else int(v)) for k, v in asdict(trial_config).items() if k in {
            "max_pseudo_multiplier", "max_pseudo_per_recording", "max_pseudo_per_cluster",
            "max_pseudo_per_labeled_ratio", "cluster_trust_power", "study_trust_weight", "min_trust",
            "conv_embed_dim", "conv_hidden_channels", "ae_epochs", "learning_rate"
        }})
        row.update({k: float(v) if isinstance(v, float) else int(v) for k, v in model_params.items()})
        trial_rows.append(row)
        trial.set_user_attr("holdout_r2", float(holdout_metrics["r2"]))
        trial.set_user_attr("holdout_mae", float(holdout_metrics["mae"]))
        trial.set_user_attr("selected_rows", int(len(state["trust_selected"])))
        nonlocal best_payload
        if not best_payload or float(cv_metrics["r2"]) > float(best_payload["cv_metrics"]["r2"]):
            best_payload = {
                "trial_config": trial_config,
                "model_params": model_params,
                "family_alpha": family_alpha,
                "state": state,
                "train_variant": train_variant,
                "unlabeled_variant": unlabeled_variant,
                "test_variant": test_variant,
                "cv_metrics": cv_metrics,
                "holdout_pred": holdout_pred,
                "holdout_metrics": holdout_metrics,
            }
        return float(cv_metrics["r2"])

    sampler = optuna.samplers.TPESampler(seed=config.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(config.n_trials), timeout=config.timeout_seconds)

    if not best_payload:
        raise RuntimeError("Bayesian optimization produced no trials")

    metrics_df = pd.DataFrame(
        [
            make_metric_row(
                target="fpos",
                variant_id="baseline_wf_embed_anchor_stack_xgboost",
                family="signal_embedding_bayes_opt",
                backend="xgboost_default",
                source_target="fpos",
                metrics=baseline_holdout_metrics,
                notes="baseline signal embedding stack before optimization",
            ),
            make_metric_row(
                target="fpos",
                variant_id="bayes_opt_wf_embed_anchor_stack_xgboost",
                family="signal_embedding_bayes_opt",
                backend="xgboost_tuned",
                source_target="fpos",
                metrics=best_payload["holdout_metrics"],
                notes="best Bayesian-optimized signal embedding stack",
            ),
        ]
    ).sort_values(["r2", "mae"], ascending=[False, True]).reset_index(drop=True)
    predictions_df = pd.concat(
        [
            make_prediction_frame(
                target="fpos",
                variant_id="baseline_wf_embed_anchor_stack_xgboost",
                family="signal_embedding_bayes_opt",
                backend="xgboost_default",
                df=target_test_df,
                y_true=y_test,
                pred=baseline_holdout,
            ),
            make_prediction_frame(
                target="fpos",
                variant_id="bayes_opt_wf_embed_anchor_stack_xgboost",
                family="signal_embedding_bayes_opt",
                backend="xgboost_tuned",
                df=best_payload["state"]["target_test_df"],
                y_true=best_payload["state"]["y_test"],
                pred=best_payload["holdout_pred"],
            ),
        ],
        ignore_index=True,
    )
    trials_df = pd.DataFrame(trial_rows).sort_values(["cv_r2", "cv_mae"], ascending=[False, True]).reset_index(drop=True)
    cv_summary_df = pd.DataFrame(
        [
            {
                "variant_id": "baseline_wf_embed_anchor_stack_xgboost",
                "cv_mae": float(baseline_cv["mae"]),
                "cv_r2": float(baseline_cv["r2"]),
            },
            {
                "variant_id": "bayes_opt_wf_embed_anchor_stack_xgboost",
                "cv_mae": float(best_payload["cv_metrics"]["mae"]),
                "cv_r2": float(best_payload["cv_metrics"]["r2"]),
            },
        ]
    )

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    trials_df.to_csv(output_dir / "trial_history.csv", index=False)
    cv_summary_df.to_csv(output_dir / "variant_cv_summary.csv", index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config) | {"parquet_path": str(config.parquet_path), "output_root": str(config.output_root)}, f, indent=2)
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(
            {
                "family_alpha": best_payload["family_alpha"],
                "trial_config": {
                    "max_pseudo_multiplier": best_payload["trial_config"].max_pseudo_multiplier,
                    "max_pseudo_per_recording": best_payload["trial_config"].max_pseudo_per_recording,
                    "max_pseudo_per_cluster": best_payload["trial_config"].max_pseudo_per_cluster,
                    "max_pseudo_per_labeled_ratio": best_payload["trial_config"].max_pseudo_per_labeled_ratio,
                    "cluster_trust_power": best_payload["trial_config"].cluster_trust_power,
                    "study_trust_weight": best_payload["trial_config"].study_trust_weight,
                    "min_trust": best_payload["trial_config"].min_trust,
                    "conv_embed_dim": best_payload["trial_config"].conv_embed_dim,
                    "conv_hidden_channels": best_payload["trial_config"].conv_hidden_channels,
                    "ae_epochs": best_payload["trial_config"].ae_epochs,
                    "learning_rate": best_payload["trial_config"].learning_rate,
                },
                "model_params": best_payload["model_params"],
                "cv_metrics": best_payload["cv_metrics"],
                "holdout_metrics": best_payload["holdout_metrics"],
            },
            f,
            indent=2,
        )
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(best_payload["state"]["split"].manifest, f, indent=2)
    _log(config, f"Saved outputs to {output_dir}")
    return output_dir


def run_with_config(config: FPosSignalEmbeddingBayesOptConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosSignalEmbeddingBayesOptConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        n_trials=args.trials,
        timeout_seconds=args.timeout,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.n_trials = min(cli_config.n_trials, 4)
        cli_config.max_pseudo_multiplier = 1.0
        cli_config.max_pseudo_per_recording = 24
        cli_config.max_pseudo_per_cluster = 64
        cli_config.ae_epochs = 12
        cli_config.universe_max_rows = 4000
    _run_with_config(cli_config)
