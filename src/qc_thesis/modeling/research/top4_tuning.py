from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

from qc_thesis.modeling.backends import target_inverse, target_transform
from qc_thesis.modeling.neural import prepare_feature_table, score_predictions
from qc_thesis.modeling.shared import FrameAugmenter, StackSet, make_main_split
from qc_thesis.modeling.stacking.augmenters import fit_signal_embeddings
from qc_thesis.modeling.stacking.base_stack_runner import (
    assemble_training_data,
    build_base_stack_state,
    build_embedding_universe,
)
from qc_thesis.modeling.stacking.backends import _grouped_oof_splits
from qc_thesis.modeling.stacking.recipes import (
    FPosBackendSweepConfig,
    FPosSignalEmbeddingStackConfig,
    _context_columns,
    _subset_stack,
)
from qc_thesis.paths import PRE_WAVEFORM_LABEL, THESIS_ROOT, WAVEFORM_AUGMENTED_LABEL


BASELINE_RECIPE_MAP = {
    "paired_raw": "fpos_paired_raw",
    "paired_context": "fpos_paired_context",
    "pre_waveform": "fpos_pre_waveform_unlabeled",
    "waveform_augmented": "fpos_waveform_winner",
}

FAMILY_LABELS = {
    "paired_raw": "Paired-only raw",
    "paired_context": "Paired-only context",
    "pre_waveform": PRE_WAVEFORM_LABEL,
    "waveform_augmented": WAVEFORM_AUGMENTED_LABEL,
}


@dataclass(frozen=True)
class BackendParams:
    backend: str
    target_transform_mode: str = "logit"
    lightgbm_estimators: int = 250
    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 31
    xgboost_estimators: int = 350
    xgboost_learning_rate: float = 0.04
    xgboost_max_depth: int = 4
    xgboost_min_child_weight: float = 4.0
    xgboost_subsample: float = 0.8
    xgboost_colsample_bytree: float = 0.8
    catboost_iterations: int = 400
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 5
    catboost_l2_leaf_reg: float = 3.0
    histgb_max_iter: int = 350
    histgb_learning_rate: float = 0.05
    histgb_max_depth: int = 4
    histgb_min_samples_leaf: int = 8
    histgb_l2_regularization: float = 0.05


@dataclass(frozen=True)
class Top4Candidate:
    family_id: str
    candidate_id: str
    backend: BackendParams
    feature_mode: str = "default"
    filtered_shift_top_n_full_context: int = 24
    filtered_shift_min_abs_smd: float = 1.0
    max_pseudo_multiplier: float = 1.5
    max_pseudo_per_recording: int = 48
    max_pseudo_per_cluster: int = 128
    max_pseudo_per_labeled_ratio: float = 1.5
    cluster_trust_power: float = 1.35
    study_trust_weight: float = 0.30
    min_trust: float = 0.25
    signal_mode: str = "wf"
    conv_embed_dim: int = 8
    conv_hidden_channels: int = 12
    ae_epochs: int = 28
    ae_learning_rate: float = 2e-3
    batch_size: int = 256
    universe_max_rows: int | None = 18000


@dataclass(frozen=True)
class Top4TuningConfig:
    parquet_path: Path = THESIS_ROOT / "data/raw/33000_ROWS.parquet"
    target: str = "fpos"
    search_seed: int = 42
    shortlist_seeds: tuple[int, ...] = (43, 44, 45, 46, 47)
    report_seeds: tuple[int, ...] = tuple(range(48, 62))
    paired_final_holdout_rows: int = 100
    table_root: Path = THESIS_ROOT / "tables" / "10_top4_tuning"
    figure_root: Path = THESIS_ROOT / "figures" / "10_top4_tuning"
    artifact_root: Path = THESIS_ROOT / "artifacts" / "research" / "top4_tuning"
    shortlist_topk_cv: int = 3
    shortlist_topk_holdout: int = 3


def _make_backend_candidates() -> dict[str, list[BackendParams]]:
    return {
        "paired_raw": [
            BackendParams("lightgbm", "logit", 250, 0.05, 31),
            BackendParams("lightgbm", "logit", 500, 0.03, 63),
            BackendParams("lightgbm", "identity", 350, 0.04, 31),
            BackendParams("xgboost", "logit", xgboost_estimators=250, xgboost_learning_rate=0.05, xgboost_max_depth=6, xgboost_min_child_weight=1.0),
            BackendParams("xgboost", "logit", xgboost_estimators=500, xgboost_learning_rate=0.03, xgboost_max_depth=4, xgboost_min_child_weight=4.0),
            BackendParams("xgboost", "identity", xgboost_estimators=350, xgboost_learning_rate=0.04, xgboost_max_depth=3, xgboost_min_child_weight=6.0),
            BackendParams("catboost", "logit", catboost_iterations=400, catboost_learning_rate=0.05, catboost_depth=5),
            BackendParams("catboost", "logit", catboost_iterations=700, catboost_learning_rate=0.03, catboost_depth=6, catboost_l2_leaf_reg=8.0),
            BackendParams("histgb", "logit", histgb_max_iter=400, histgb_learning_rate=0.04, histgb_max_depth=4),
        ],
        "paired_context": [
            BackendParams("lightgbm", "logit", 250, 0.05, 31),
            BackendParams("lightgbm", "logit", 450, 0.03, 63),
            BackendParams("lightgbm", "identity", 300, 0.05, 15),
            BackendParams("xgboost", "logit", xgboost_estimators=350, xgboost_learning_rate=0.04, xgboost_max_depth=4, xgboost_min_child_weight=4.0),
            BackendParams("xgboost", "logit", xgboost_estimators=500, xgboost_learning_rate=0.03, xgboost_max_depth=3, xgboost_min_child_weight=6.0),
            BackendParams("catboost", "logit", catboost_iterations=500, catboost_learning_rate=0.04, catboost_depth=6, catboost_l2_leaf_reg=8.0),
            BackendParams("histgb", "logit", histgb_max_iter=500, histgb_learning_rate=0.04, histgb_max_depth=4, histgb_min_samples_leaf=12),
        ],
        "pre_waveform": [
            BackendParams("xgboost", "logit", xgboost_estimators=350, xgboost_learning_rate=0.04, xgboost_max_depth=4, xgboost_min_child_weight=4.0),
            BackendParams("xgboost", "logit", xgboost_estimators=500, xgboost_learning_rate=0.03, xgboost_max_depth=3, xgboost_min_child_weight=6.0),
            BackendParams("xgboost", "identity", xgboost_estimators=350, xgboost_learning_rate=0.04, xgboost_max_depth=4, xgboost_min_child_weight=4.0),
            BackendParams("lightgbm", "logit", 300, 0.05, 31),
            BackendParams("lightgbm", "logit", 500, 0.03, 63),
            BackendParams("catboost", "logit", catboost_iterations=500, catboost_learning_rate=0.04, catboost_depth=6, catboost_l2_leaf_reg=8.0),
        ],
        "waveform_augmented": [
            BackendParams("xgboost", "logit", xgboost_estimators=350, xgboost_learning_rate=0.04, xgboost_max_depth=4, xgboost_min_child_weight=4.0),
            BackendParams("xgboost", "logit", xgboost_estimators=500, xgboost_learning_rate=0.03, xgboost_max_depth=3, xgboost_min_child_weight=6.0),
            BackendParams("lightgbm", "logit", 300, 0.05, 31),
            BackendParams("lightgbm", "logit", 500, 0.03, 63),
            BackendParams("catboost", "logit", catboost_iterations=500, catboost_learning_rate=0.04, catboost_depth=6, catboost_l2_leaf_reg=8.0),
        ],
    }


def _candidate_spaces() -> dict[str, list[Top4Candidate]]:
    backends = _make_backend_candidates()
    spaces: dict[str, list[Top4Candidate]] = {"paired_raw": [], "paired_context": [], "pre_waveform": [], "waveform_augmented": []}

    for idx, backend in enumerate(backends["paired_raw"], start=1):
        spaces["paired_raw"].append(
            Top4Candidate(
                family_id="paired_raw",
                candidate_id=f"paired_raw_{backend.backend}_{idx:02d}",
                backend=backend,
            )
        )

    feature_modes = [
        ("filtered_full", 24, 1.0),
        ("filtered_full", 36, 0.8),
        ("full_unfiltered", 24, 1.0),
        ("curated_context", 24, 1.0),
    ]
    idx = 1
    for feature_mode, top_n, smd in feature_modes:
        backend_pool = backends["paired_context"]
        if feature_mode == "curated_context":
            backend_pool = backend_pool[:2]
        elif feature_mode == "full_unfiltered":
            backend_pool = backend_pool[:4]
        else:
            backend_pool = backend_pool[:5]
        for backend in backend_pool:
            spaces["paired_context"].append(
                Top4Candidate(
                    family_id="paired_context",
                    candidate_id=f"paired_context_{feature_mode}_{backend.backend}_{idx:02d}",
                    backend=backend,
                    feature_mode=feature_mode,
                    filtered_shift_top_n_full_context=top_n,
                    filtered_shift_min_abs_smd=smd,
                )
            )
            idx += 1

    pre_structures = [
        ("baseline", 1.5, 48, 128, 1.5, 1.35, 0.30, 0.25, 24, 1.0),
        ("more_pseudo", 2.0, 64, 192, 2.0, 1.20, 0.25, 0.20, 24, 1.0),
        ("stricter_trust", 1.2, 48, 128, 1.2, 1.60, 0.35, 0.35, 24, 1.0),
    ]
    idx = 1
    for name, mult, per_rec, per_cl, per_lab, cluster_power, study_w, min_trust, top_n, smd in pre_structures:
        for backend in backends["pre_waveform"][:4]:
            spaces["pre_waveform"].append(
                Top4Candidate(
                    family_id="pre_waveform",
                    candidate_id=f"pre_waveform_{name}_{backend.backend}_{idx:02d}",
                    backend=backend,
                    max_pseudo_multiplier=mult,
                    max_pseudo_per_recording=per_rec,
                    max_pseudo_per_cluster=per_cl,
                    max_pseudo_per_labeled_ratio=per_lab,
                    cluster_trust_power=cluster_power,
                    study_trust_weight=study_w,
                    min_trust=min_trust,
                    filtered_shift_top_n_full_context=top_n,
                    filtered_shift_min_abs_smd=smd,
                )
            )
            idx += 1

    waveform_structures = [
        ("wf_baseline", "wf", 8, 12, 28, 2e-3, 18000, 1.5, 48, 128, 1.5, 1.35, 0.30, 0.25),
        ("wf_big", "wf", 12, 16, 40, 1e-3, 24000, 1.5, 48, 128, 1.5, 1.35, 0.30, 0.25),
        ("wf_acg_big", "wf_acg", 12, 16, 40, 1e-3, 24000, 1.5, 48, 128, 1.5, 1.35, 0.30, 0.25),
        ("wf_acg_more_pseudo", "wf_acg", 12, 16, 40, 2e-3, 18000, 2.0, 64, 192, 2.0, 1.20, 0.25, 0.20),
    ]
    idx = 1
    for name, signal_mode, embed_dim, hidden_ch, epochs, ae_lr, universe_max, mult, per_rec, per_cl, per_lab, cluster_power, study_w, min_trust in waveform_structures:
        for backend in backends["waveform_augmented"][:4]:
            spaces["waveform_augmented"].append(
                Top4Candidate(
                    family_id="waveform_augmented",
                    candidate_id=f"waveform_{name}_{backend.backend}_{idx:02d}",
                    backend=backend,
                    signal_mode=signal_mode,
                    conv_embed_dim=embed_dim,
                    conv_hidden_channels=hidden_ch,
                    ae_epochs=epochs,
                    ae_learning_rate=ae_lr,
                    universe_max_rows=universe_max,
                    max_pseudo_multiplier=mult,
                    max_pseudo_per_recording=per_rec,
                    max_pseudo_per_cluster=per_cl,
                    max_pseudo_per_labeled_ratio=per_lab,
                    cluster_trust_power=cluster_power,
                    study_trust_weight=study_w,
                    min_trust=min_trust,
                )
            )
            idx += 1
    return spaces


def _backend_label(params: BackendParams) -> str:
    return f"{params.backend}_{params.target_transform_mode}"


def _make_model(params: BackendParams, random_state: int) -> Any:
    backend = params.backend
    if backend == "lightgbm":
        return LGBMRegressor(
            n_estimators=params.lightgbm_estimators,
            learning_rate=params.lightgbm_learning_rate,
            num_leaves=params.lightgbm_num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    if backend == "xgboost":
        return XGBRegressor(
            n_estimators=params.xgboost_estimators,
            learning_rate=params.xgboost_learning_rate,
            max_depth=params.xgboost_max_depth,
            min_child_weight=params.xgboost_min_child_weight,
            subsample=params.xgboost_subsample,
            colsample_bytree=params.xgboost_colsample_bytree,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    if backend == "catboost":
        return CatBoostRegressor(
            iterations=params.catboost_iterations,
            learning_rate=params.catboost_learning_rate,
            depth=params.catboost_depth,
            l2_leaf_reg=params.catboost_l2_leaf_reg,
            loss_function="RMSE",
            verbose=False,
            random_seed=random_state,
        )
    if backend == "histgb":
        return HistGradientBoostingRegressor(
            learning_rate=params.histgb_learning_rate,
            max_depth=params.histgb_max_depth,
            max_iter=params.histgb_max_iter,
            min_samples_leaf=params.histgb_min_samples_leaf,
            l2_regularization=params.histgb_l2_regularization,
            random_state=random_state,
        )
    raise KeyError(backend)


def _fit_predict_backend(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    sample_weight: np.ndarray | None,
    params: BackendParams,
    random_state: int,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    train_imp = imputer.fit_transform(X_train)
    eval_imp = imputer.transform(X_eval)
    y_train_t = target_transform(y_train, mode=params.target_transform_mode)
    model = _make_model(params, random_state=random_state)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(train_imp, y_train_t, **fit_kwargs)
    pred_t = np.asarray(model.predict(eval_imp), dtype=np.float32)
    return np.clip(target_inverse(pred_t, mode=params.target_transform_mode), 0.0, 1.0)


def _raw_split(seed: int, config: Top4TuningConfig) -> tuple[object, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    prepared = prepare_feature_table(config.parquet_path)
    split = make_main_split(prepared.df, paired_final_holdout_rows=config.paired_final_holdout_rows, random_state=seed)
    train_df = split.paired_finetune.reset_index(drop=True)
    test_df = split.paired_test.reset_index(drop=True)
    y_train = pd.to_numeric(train_df[config.target], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(test_df[config.target], errors="coerce").to_numpy(dtype=float)
    return prepared, train_df, test_df, y_train, y_test


def _evaluate_grouped_frame(
    train_meta: pd.DataFrame,
    train_frame: pd.DataFrame,
    y_train: np.ndarray,
    test_frame: pd.DataFrame,
    y_test: np.ndarray,
    candidate: Top4Candidate,
    *,
    pseudo_selected: pd.DataFrame | None = None,
    unlabeled_frame: pd.DataFrame | None = None,
    family_alpha: float = 1.0,
) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
    if pseudo_selected is None:
        pseudo_selected = pd.DataFrame()
    if unlabeled_frame is None:
        unlabeled_frame = pd.DataFrame(index=range(0))
    splits = _grouped_oof_splits(train_meta, 3)
    oof = np.full(len(train_meta), np.nan, dtype=np.float32)
    empty_like = pseudo_selected.head(0).copy()
    for fold_id, (fit_idx, val_idx) in enumerate(splits, start=1):
        val_recordings = set(train_meta.iloc[val_idx]["recording_key"].astype(str))
        fold_pseudo = empty_like
        if not pseudo_selected.empty:
            fold_pseudo = pseudo_selected[
                ~pseudo_selected["recording_key"].astype(str).isin(val_recordings)
            ].reset_index(drop=True)
        fold_train, fold_y, fold_weight = assemble_training_data(
            labeled_frame=train_frame.iloc[fit_idx].reset_index(drop=True),
            labeled_y=y_train[fit_idx],
            labeled_meta_df=train_meta.iloc[fit_idx].reset_index(drop=True),
            pseudo_selected=fold_pseudo,
            unlabeled_frame=unlabeled_frame,
            family_alpha=family_alpha,
        )
        oof[val_idx] = _fit_predict_backend(
            fold_train,
            fold_y,
            train_frame.iloc[val_idx].reset_index(drop=True),
            fold_weight,
            candidate.backend,
            random_state=1000 + fold_id,
        )
    full_train, full_y, full_weight = assemble_training_data(
        labeled_frame=train_frame.reset_index(drop=True),
        labeled_y=y_train,
        labeled_meta_df=train_meta.reset_index(drop=True),
        pseudo_selected=pseudo_selected.reset_index(drop=True),
        unlabeled_frame=unlabeled_frame.reset_index(drop=True),
        family_alpha=family_alpha,
    )
    holdout_pred = _fit_predict_backend(
        full_train,
        full_y,
        test_frame.reset_index(drop=True),
        full_weight,
        candidate.backend,
        random_state=9999,
    )
    return score_predictions(y_train, oof), score_predictions(y_test, holdout_pred), holdout_pred


_STATE_CACHE: dict[tuple[Any, ...], Any] = {}
_WAVE_CACHE: dict[tuple[Any, ...], StackSet] = {}


def _state_key(candidate: Top4Candidate, seed: int) -> tuple[Any, ...]:
    return (
        candidate.family_id,
        seed,
        candidate.backend.target_transform_mode,
        candidate.filtered_shift_top_n_full_context,
        candidate.filtered_shift_min_abs_smd,
        candidate.max_pseudo_multiplier,
        candidate.max_pseudo_per_recording,
        candidate.max_pseudo_per_cluster,
        candidate.max_pseudo_per_labeled_ratio,
        candidate.cluster_trust_power,
        candidate.study_trust_weight,
        candidate.min_trust,
    )


def _build_state(candidate: Top4Candidate, seed: int, config: Top4TuningConfig) -> Any:
    key = _state_key(candidate, seed)
    if key in _STATE_CACHE:
        return _STATE_CACHE[key]
    config_cls = FPosSignalEmbeddingStackConfig if candidate.family_id == "waveform_augmented" else FPosBackendSweepConfig
    stack_cfg = config_cls(
        parquet_path=config.parquet_path,
        output_root=config.artifact_root,
        run_name=f"cache_seed_{seed}",
        random_state=seed,
        paired_final_holdout_rows=config.paired_final_holdout_rows,
        verbose=False,
    )
    stack_cfg.target_transform = candidate.backend.target_transform_mode
    stack_cfg.filtered_shift_top_n_full_context = int(candidate.filtered_shift_top_n_full_context)
    stack_cfg.filtered_shift_min_abs_smd = float(candidate.filtered_shift_min_abs_smd)
    stack_cfg.max_pseudo_multiplier = float(candidate.max_pseudo_multiplier)
    stack_cfg.max_pseudo_per_recording = int(candidate.max_pseudo_per_recording)
    stack_cfg.max_pseudo_per_cluster = int(candidate.max_pseudo_per_cluster)
    stack_cfg.max_pseudo_per_labeled_ratio = float(candidate.max_pseudo_per_labeled_ratio)
    stack_cfg.cluster_trust_power = float(candidate.cluster_trust_power)
    stack_cfg.study_trust_weight = float(candidate.study_trust_weight)
    stack_cfg.min_trust = float(candidate.min_trust)
    if isinstance(stack_cfg, FPosSignalEmbeddingStackConfig):
        stack_cfg.conv_embed_dim = int(candidate.conv_embed_dim)
        stack_cfg.conv_hidden_channels = int(candidate.conv_hidden_channels)
        stack_cfg.ae_epochs = int(candidate.ae_epochs)
        stack_cfg.batch_size = int(candidate.batch_size)
        stack_cfg.learning_rate = float(candidate.ae_learning_rate)
        stack_cfg.universe_max_rows = candidate.universe_max_rows
    state = build_base_stack_state(target=config.target, config=stack_cfg)
    _STATE_CACHE[key] = state
    return state


def _waveform_stack(candidate: Top4Candidate, seed: int, config: Top4TuningConfig) -> tuple[Any, StackSet]:
    key = (
        candidate.candidate_id,
        seed,
        candidate.signal_mode,
        candidate.conv_embed_dim,
        candidate.conv_hidden_channels,
        candidate.ae_epochs,
        candidate.ae_learning_rate,
        candidate.batch_size,
        candidate.universe_max_rows,
    )
    state = _build_state(candidate, seed, config)
    if key in _WAVE_CACHE:
        return state, _WAVE_CACHE[key]
    universe_df = build_embedding_universe(
        split=state.split,
        max_rows=candidate.universe_max_rows,
        random_state=seed,
    )
    bundles: list[Any] = []
    if candidate.signal_mode in {"wf", "wf_acg"}:
        wf_bundle = fit_signal_embeddings(
            train_signal=state.target_train_df.loc[:, state.prepared.wf_feature_cols],
            train_universe_signal=universe_df.loc[:, state.prepared.wf_feature_cols],
            unlabeled_signal=state.unlabeled_df.loc[:, state.prepared.wf_feature_cols],
            test_signal=state.target_test_df.loc[:, state.prepared.wf_feature_cols],
            prefix="wf",
            config=state.base,
        )
        bundles.append((wf_bundle.train, wf_bundle.unlabeled, wf_bundle.test))
    if candidate.signal_mode in {"acg", "wf_acg"}:
        acg_bundle = fit_signal_embeddings(
            train_signal=state.target_train_df.loc[:, state.prepared.acg_feature_cols],
            train_universe_signal=universe_df.loc[:, state.prepared.acg_feature_cols],
            unlabeled_signal=state.unlabeled_df.loc[:, state.prepared.acg_feature_cols],
            test_signal=state.target_test_df.loc[:, state.prepared.acg_feature_cols],
            prefix="acg",
            config=state.base,
        )
        bundles.append((acg_bundle.train, acg_bundle.unlabeled, acg_bundle.test))
    stack = state.base_stack
    for train_frame, unlabeled_frame, test_frame in bundles:
        stack = stack.apply(
            FrameAugmenter(
                train_features=train_frame,
                unlabeled_features=unlabeled_frame,
                test_features=test_frame,
            )
        )
    _WAVE_CACHE[key] = stack
    return state, stack


def _context_stack_from_state(candidate: Top4Candidate, state: Any) -> StackSet:
    if candidate.feature_mode == "filtered_full":
        return _subset_stack(state.base_stack, _context_columns(state.base_stack))
    if candidate.feature_mode == "full_unfiltered":
        cols = list(state.prepared.lightgbm_context_cols)
        return StackSet.from_context_frames(
            train_df=state.target_train_df,
            unlabeled_df=state.unlabeled_df,
            test_df=state.target_test_df,
            context_cols=cols,
        )
    if candidate.feature_mode == "curated_context":
        cols = list(state.prepared.context_feature_cols_by_target["fpos"])
        return StackSet.from_context_frames(
            train_df=state.target_train_df,
            unlabeled_df=state.unlabeled_df,
            test_df=state.target_test_df,
            context_cols=cols,
        )
    raise KeyError(candidate.feature_mode)


def _evaluate_candidate(candidate: Top4Candidate, seed: int, config: Top4TuningConfig) -> dict[str, Any]:
    if candidate.family_id == "paired_raw":
        prepared, train_df, test_df, y_train, y_test = _raw_split(seed, config)
        train_frame = train_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        test_frame = test_df.loc[:, prepared.unit_feature_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        cv_metrics, holdout_metrics, holdout_pred = _evaluate_grouped_frame(
            train_df,
            train_frame,
            y_train,
            test_frame,
            y_test,
            candidate,
            pseudo_selected=None,
            unlabeled_frame=None,
            family_alpha=1.0,
        )
        family_summary = (
            pd.DataFrame({"study_set": test_df["study_set"].astype(str), "y_true": y_test, "y_pred": holdout_pred})
            .groupby("study_set", dropna=False)
            .apply(lambda g: pd.Series(score_predictions(g["y_true"].to_numpy(dtype=float), g["y_pred"].to_numpy(dtype=float))))
            .reset_index()
        )
        return {
            "seed": seed,
            "candidate_id": candidate.candidate_id,
            "family_id": candidate.family_id,
            "cv_r2": float(cv_metrics["r2"]),
            "cv_mae": float(cv_metrics["mae"]),
            "holdout_r2": float(holdout_metrics["r2"]),
            "holdout_mae": float(holdout_metrics["mae"]),
            "holdout_rmse": float(holdout_metrics["rmse"]),
            "predictions": pd.DataFrame({"row_uid": test_df["row_uid"].astype(str), "study_set": test_df["study_set"].astype(str), "y_true": y_test, "y_pred": holdout_pred}),
            "family_summary": family_summary,
        }

    if candidate.family_id == "paired_context":
        state = _build_state(candidate, seed, config)
        stack = _context_stack_from_state(candidate, state)
        cv_metrics, holdout_metrics, holdout_pred = _evaluate_grouped_frame(
            state.target_train_df,
            stack.train,
            state.y_train,
            stack.test,
            state.y_test,
            candidate,
            pseudo_selected=state.trust_selected.head(0).copy(),
            unlabeled_frame=stack.unlabeled.head(0).copy(),
            family_alpha=1.0,
        )
        family_summary = (
            pd.DataFrame({"study_set": state.target_test_df["study_set"].astype(str), "y_true": state.y_test, "y_pred": holdout_pred})
            .groupby("study_set", dropna=False)
            .apply(lambda g: pd.Series(score_predictions(g["y_true"].to_numpy(dtype=float), g["y_pred"].to_numpy(dtype=float))))
            .reset_index()
        )
        return {
            "seed": seed,
            "candidate_id": candidate.candidate_id,
            "family_id": candidate.family_id,
            "cv_r2": float(cv_metrics["r2"]),
            "cv_mae": float(cv_metrics["mae"]),
            "holdout_r2": float(holdout_metrics["r2"]),
            "holdout_mae": float(holdout_metrics["mae"]),
            "holdout_rmse": float(holdout_metrics["rmse"]),
            "predictions": pd.DataFrame({"row_uid": state.target_test_df["row_uid"].astype(str), "study_set": state.target_test_df["study_set"].astype(str), "y_true": state.y_test, "y_pred": holdout_pred}),
            "family_summary": family_summary,
        }

    if candidate.family_id == "pre_waveform":
        state = _build_state(candidate, seed, config)
        cv_metrics, holdout_metrics, holdout_pred = _evaluate_grouped_frame(
            state.target_train_df,
            state.base_stack.train,
            state.y_train,
            state.base_stack.test,
            state.y_test,
            candidate,
            pseudo_selected=state.trust_selected,
            unlabeled_frame=state.base_stack.unlabeled,
            family_alpha=1.0,
        )
        family_summary = (
            pd.DataFrame({"study_set": state.target_test_df["study_set"].astype(str), "y_true": state.y_test, "y_pred": holdout_pred})
            .groupby("study_set", dropna=False)
            .apply(lambda g: pd.Series(score_predictions(g["y_true"].to_numpy(dtype=float), g["y_pred"].to_numpy(dtype=float))))
            .reset_index()
        )
        return {
            "seed": seed,
            "candidate_id": candidate.candidate_id,
            "family_id": candidate.family_id,
            "cv_r2": float(cv_metrics["r2"]),
            "cv_mae": float(cv_metrics["mae"]),
            "holdout_r2": float(holdout_metrics["r2"]),
            "holdout_mae": float(holdout_metrics["mae"]),
            "holdout_rmse": float(holdout_metrics["rmse"]),
            "predictions": pd.DataFrame({"row_uid": state.target_test_df["row_uid"].astype(str), "study_set": state.target_test_df["study_set"].astype(str), "y_true": state.y_test, "y_pred": holdout_pred}),
            "family_summary": family_summary,
        }

    if candidate.family_id == "waveform_augmented":
        state, stack = _waveform_stack(candidate, seed, config)
        cv_metrics, holdout_metrics, holdout_pred = _evaluate_grouped_frame(
            state.target_train_df,
            stack.train,
            state.y_train,
            stack.test,
            state.y_test,
            candidate,
            pseudo_selected=state.trust_selected,
            unlabeled_frame=stack.unlabeled,
            family_alpha=1.0,
        )
        family_summary = (
            pd.DataFrame({"study_set": state.target_test_df["study_set"].astype(str), "y_true": state.y_test, "y_pred": holdout_pred})
            .groupby("study_set", dropna=False)
            .apply(lambda g: pd.Series(score_predictions(g["y_true"].to_numpy(dtype=float), g["y_pred"].to_numpy(dtype=float))))
            .reset_index()
        )
        return {
            "seed": seed,
            "candidate_id": candidate.candidate_id,
            "family_id": candidate.family_id,
            "cv_r2": float(cv_metrics["r2"]),
            "cv_mae": float(cv_metrics["mae"]),
            "holdout_r2": float(holdout_metrics["r2"]),
            "holdout_mae": float(holdout_metrics["mae"]),
            "holdout_rmse": float(holdout_metrics["rmse"]),
            "predictions": pd.DataFrame({"row_uid": state.target_test_df["row_uid"].astype(str), "study_set": state.target_test_df["study_set"].astype(str), "y_true": state.y_test, "y_pred": holdout_pred}),
            "family_summary": family_summary,
        }

    raise KeyError(candidate.family_id)


def _write_selected_seed_outputs(config: Top4TuningConfig, family_id: str, seed: int, result: dict[str, Any]) -> None:
    output_dir = config.artifact_root / family_id / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(output_dir / "predictions.csv", index=False)
    result["family_summary"].to_csv(output_dir / "per_family_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed": seed,
                "cv_r2": result["cv_r2"],
                "cv_mae": result["cv_mae"],
                "holdout_r2": result["holdout_r2"],
                "holdout_mae": result["holdout_mae"],
                "holdout_rmse": result["holdout_rmse"],
            }
        ]
    ).to_csv(output_dir / "metrics.csv", index=False)


def _baseline_report_summary(report_seeds: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id, recipe_id in BASELINE_RECIPE_MAP.items():
        path = THESIS_ROOT / "artifacts" / "runs_aggregate" / "recording_disjoint_main" / "fpos" / recipe_id / "seed_metrics.csv"
        frame = pd.read_csv(path)
        subset = frame[frame["seed"].isin(report_seeds)].copy()
        rows.append(
            {
                "family_id": family_id,
                "comparison_label": FAMILY_LABELS[family_id],
                "mean_r2": float(subset["r2"].mean()),
                "median_r2": float(subset["r2"].median()),
                "std_r2": float(subset["r2"].std()),
                "worst_r2": float(subset["r2"].min()),
                "mean_mae": float(subset["mae"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _make_figures(config: Top4TuningConfig, report_summary: pd.DataFrame, seed_metrics: pd.DataFrame) -> None:
    config.figure_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(12, 6))
    order = report_summary.sort_values("tuned_mean_r2", ascending=False)["label"].tolist()
    plot_df = report_summary.melt(
        id_vars=["family_id", "label"],
        value_vars=["baseline_mean_r2", "tuned_mean_r2"],
        var_name="variant",
        value_name="mean_r2",
    )
    plot_df["variant"] = plot_df["variant"].map({"baseline_mean_r2": "Current baseline", "tuned_mean_r2": "Tuned"})
    sns.barplot(data=plot_df, x="mean_r2", y="label", hue="variant", order=order, palette=["#888888", "#228833"], ax=ax)
    ax.set_xlabel("Mean R² on untouched report seeds (48-61)")
    ax.set_ylabel("")
    ax.set_title("Top-4 model tuning on the report panel")
    fig.tight_layout()
    fig.savefig(config.figure_root / "top4_tuned_vs_baseline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    seed_plot = seed_metrics.copy()
    seed_plot["label"] = seed_plot["family_id"].map(FAMILY_LABELS)
    sns.lineplot(data=seed_plot, x="seed", y="holdout_r2", hue="label", marker="o", ax=ax)
    ax.set_xlabel("Report seed")
    ax.set_ylabel("Holdout R²")
    ax.set_title("Selected tuned models across report seeds")
    fig.tight_layout()
    fig.savefig(config.figure_root / "top4_selected_seed_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_top4_tuning(*, force: bool = False) -> dict[str, Path]:
    del force  # current runner always recomputes
    config = Top4TuningConfig()
    config.table_root.mkdir(parents=True, exist_ok=True)
    config.figure_root.mkdir(parents=True, exist_ok=True)
    spaces = _candidate_spaces()

    stage1_rows: list[dict[str, Any]] = []
    shortlist_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for family_id, candidates in spaces.items():
        print(f"[tune] {family_id}: stage1 candidates={len(candidates)}", flush=True)
        family_stage1: list[dict[str, Any]] = []
        for candidate in candidates:
            result = _evaluate_candidate(candidate, config.search_seed, config)
            row = {
                "family_id": family_id,
                "candidate_id": candidate.candidate_id,
                "backend": _backend_label(candidate.backend),
                "feature_mode": candidate.feature_mode,
                "signal_mode": candidate.signal_mode,
                "cv_r2": result["cv_r2"],
                "cv_mae": result["cv_mae"],
                "holdout_r2": result["holdout_r2"],
                "holdout_mae": result["holdout_mae"],
            }
            family_stage1.append(row)
            stage1_rows.append(row)
        family_stage1_df = pd.DataFrame(family_stage1).sort_values(["cv_r2", "holdout_r2"], ascending=[False, False]).reset_index(drop=True)
        print(f"[tune] {family_id}: shortlist candidates", flush=True)
        shortlist = pd.concat(
            [
                family_stage1_df.head(config.shortlist_topk_cv),
                family_stage1_df.sort_values("holdout_r2", ascending=False).head(config.shortlist_topk_holdout),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["candidate_id"])
        shortlist_candidates = [next(candidate for candidate in candidates if candidate.candidate_id == row["candidate_id"]) for _, row in shortlist.iterrows()]
        family_panel_rows: list[dict[str, Any]] = []
        for candidate in shortlist_candidates:
            seed_scores: list[float] = []
            seed_maes: list[float] = []
            for seed in config.shortlist_seeds:
                result = _evaluate_candidate(candidate, seed, config)
                seed_scores.append(float(result["holdout_r2"]))
                seed_maes.append(float(result["holdout_mae"]))
            panel_row = {
                "family_id": family_id,
                "candidate_id": candidate.candidate_id,
                "backend": _backend_label(candidate.backend),
                "feature_mode": candidate.feature_mode,
                "signal_mode": candidate.signal_mode,
                "panel_mean_r2": float(np.mean(seed_scores)),
                "panel_std_r2": float(np.std(seed_scores)),
                "panel_worst_r2": float(np.min(seed_scores)),
                "panel_mean_mae": float(np.mean(seed_maes)),
            }
            family_panel_rows.append(panel_row)
            shortlist_rows.append(panel_row)
        family_panel_df = pd.DataFrame(family_panel_rows).sort_values(["panel_mean_r2", "panel_std_r2", "panel_worst_r2"], ascending=[False, True, False]).reset_index(drop=True)
        selected = next(candidate for candidate in candidates if candidate.candidate_id == str(family_panel_df.iloc[0]["candidate_id"]))
        print(f"[tune] {family_id}: selected {selected.candidate_id}", flush=True)
        selected_rows.append(
            {
                "family_id": family_id,
                "label": FAMILY_LABELS[family_id],
                "candidate_id": selected.candidate_id,
                "backend": _backend_label(selected.backend),
                "feature_mode": selected.feature_mode,
                "signal_mode": selected.signal_mode,
                "panel_mean_r2": float(family_panel_df.iloc[0]["panel_mean_r2"]),
                "panel_std_r2": float(family_panel_df.iloc[0]["panel_std_r2"]),
                "panel_worst_r2": float(family_panel_df.iloc[0]["panel_worst_r2"]),
            }
        )
        for seed in config.report_seeds:
            result = _evaluate_candidate(selected, seed, config)
            _write_selected_seed_outputs(config, family_id, seed, result)
            report_rows.append(
                {
                    "family_id": family_id,
                    "label": FAMILY_LABELS[family_id],
                    "candidate_id": selected.candidate_id,
                    "seed": seed,
                    "cv_r2": result["cv_r2"],
                    "cv_mae": result["cv_mae"],
                    "holdout_r2": result["holdout_r2"],
                    "holdout_mae": result["holdout_mae"],
                    "holdout_rmse": result["holdout_rmse"],
                }
            )

    stage1_df = pd.DataFrame(stage1_rows).sort_values(["family_id", "cv_r2", "holdout_r2"], ascending=[True, False, False]).reset_index(drop=True)
    shortlist_df = pd.DataFrame(shortlist_rows).sort_values(["family_id", "panel_mean_r2"], ascending=[True, False]).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_rows).sort_values("panel_mean_r2", ascending=False).reset_index(drop=True)
    report_seed_df = pd.DataFrame(report_rows).sort_values(["family_id", "seed"]).reset_index(drop=True)
    baseline_df = _baseline_report_summary(config.report_seeds)
    tuned_summary = (
        report_seed_df.groupby(["family_id", "label"], dropna=False)
        .agg(
            tuned_mean_r2=("holdout_r2", "mean"),
            tuned_median_r2=("holdout_r2", "median"),
            tuned_std_r2=("holdout_r2", "std"),
            tuned_worst_r2=("holdout_r2", "min"),
            tuned_mean_mae=("holdout_mae", "mean"),
        )
        .reset_index()
    )
    report_summary = tuned_summary.merge(baseline_df, on=["family_id"], how="left")
    report_summary["delta_mean_r2"] = report_summary["tuned_mean_r2"] - report_summary["mean_r2"]
    report_summary.rename(
        columns={
            "comparison_label": "baseline_label",
            "mean_r2": "baseline_mean_r2",
            "median_r2": "baseline_median_r2",
            "std_r2": "baseline_std_r2",
            "worst_r2": "baseline_worst_r2",
            "mean_mae": "baseline_mean_mae",
        },
        inplace=True,
    )
    report_summary = report_summary.merge(selected_df[["family_id", "candidate_id", "backend", "feature_mode", "signal_mode"]], on="family_id", how="left")
    report_summary = report_summary.sort_values("tuned_mean_r2", ascending=False).reset_index(drop=True)

    stage1_df.to_csv(config.table_root / "top4_stage1_seed42.csv", index=False)
    shortlist_df.to_csv(config.table_root / "top4_shortlist_panel.csv", index=False)
    selected_df.to_csv(config.table_root / "top4_selected_candidates.csv", index=False)
    report_seed_df.to_csv(config.table_root / "top4_report_seed_metrics.csv", index=False)
    report_summary.to_csv(config.table_root / "top4_report_summary.csv", index=False)
    (config.table_root / "top4_selected_candidates.json").write_text(
        json.dumps(
            {
                "search_seed": config.search_seed,
                "shortlist_seeds": list(config.shortlist_seeds),
                "report_seeds": list(config.report_seeds),
                "selected": [row for row in selected_rows],
            },
            indent=2,
            sort_keys=True,
        )
    )

    _make_figures(config, report_summary, report_seed_df)
    return {
        "stage1_csv": config.table_root / "top4_stage1_seed42.csv",
        "shortlist_csv": config.table_root / "top4_shortlist_panel.csv",
        "selected_csv": config.table_root / "top4_selected_candidates.csv",
        "report_csv": config.table_root / "top4_report_summary.csv",
    }


__all__ = ["run_top4_tuning"]
