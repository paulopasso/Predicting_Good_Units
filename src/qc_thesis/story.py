from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_dataset_bundle
from .domain import get_domain_shift_bundle
from .features import get_acg_columns, get_feature_view, get_waveform_columns, infer_numeric_feature_columns
from .limitations import get_limitations_bundle
from .modeling import (
    build_benchmark_table,
    build_family_summary,
    build_recording_summary,
    get_core_benchmark_recipe_ids,
    get_main_text_recipe_ids,
    get_recipe,
    get_recipe_groups,
    load_experiment,
)
from .modeling.recipes import DEFAULT_SWEEP_SEEDS
from .modeling.neural import prepare_feature_table
from .modeling.shared import build_stress_splits
from .modeling.specs import TransferBenchmarkConfig
from .modeling.stacking.backends import score_predictions
from .modeling.stacking.base_stack_runner import build_base_stack_state_from_split, evaluate_stack_variant
from .modeling.stacking.recipes import MLStatisticsWorkbenchConfig, _variant_specs_for_target
from .modeling.transfer import QCTransferBenchmark
from .paths import PRE_WAVEFORM_LABEL, THESIS_ROOT, WAVEFORM_AUGMENTED_LABEL, feature_group_display, normalize_model_label
from .xai import load_fpos_shap_group_summary, load_fpos_shap_importance


SPIKEFOREST_PAIRED_STUDY_CONTEXT: dict[str, dict[str, object]] = {
    # Paraphrased from SpikeForest / eLife 2020 Table 2.
    "PAIRED_BOYDEN": {
        "spikeforest_recordings": 19,
        "channel_context": "32-channel cortical subset",
        "duration_context": "6-10 min",
        "source_context": "mouse cortex, subset of larger probe recordings",
    },
    "PAIRED_CRCNS_HC1": {
        "spikeforest_recordings": 93,
        "channel_context": "4-6 channel tetrode or single-shank probe",
        "duration_context": "6-12 min",
        "source_context": "rat hippocampus",
    },
    "PAIRED_ENGLISH": {
        "spikeforest_recordings": 29,
        "channel_context": "4-32 channel juxtacellular / Si probe mix",
        "duration_context": "1-36 min",
        "source_context": "behaving mouse, multiple regions",
    },
    "PAIRED_KAMPFF": {
        "spikeforest_recordings": 15,
        "channel_context": "32-channel cortical subset",
        "duration_context": "9-20 min",
        "source_context": "mouse cortex, subset of larger probe recordings",
    },
    "PAIRED_MEA64C_YGER": {
        "spikeforest_recordings": 18,
        "channel_context": "64-channel MEA subset",
        "duration_context": "5 min",
        "source_context": "mouse retina",
    },
}


LOFO_LABELS = {
    "reference_anchor_stack_xgboost": PRE_WAVEFORM_LABEL,
    "wf_embed_anchor_stack_xgboost": WAVEFORM_AUGMENTED_LABEL,
}

FPOS_LOFO_MAIN_RECIPE_IDS = [
    "fpos_paired_raw",
    "fpos_paired_context",
    "fpos_context_stack",
    "fpos_pre_waveform_unlabeled",
    "fpos_waveform_winner",
]

FPOS_LOFO_MAIN_LABEL_ORDER = [get_recipe(recipe_id).label for recipe_id in FPOS_LOFO_MAIN_RECIPE_IDS]

FMISS_LOFO_MAIN_RECIPE_IDS = [
    "fmiss_paired_raw",
    "fmiss_paired_context",
    "fmiss_reduced_latent",
]

FMISS_LOFO_MAIN_LABEL_ORDER = [get_recipe(recipe_id).label for recipe_id in FMISS_LOFO_MAIN_RECIPE_IDS]


def _aggregate_recipe_dir(recipe_id: str) -> "Path":
    recipe = get_recipe(recipe_id)
    return THESIS_ROOT / "artifacts" / "runs_aggregate" / recipe.protocol / recipe.target / recipe.recipe_id


def _iqr(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def _load_seed_metrics(recipe_id: str) -> pd.DataFrame:
    path = _aggregate_recipe_dir(recipe_id) / "seed_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    recipe = get_recipe(recipe_id)
    df["recipe_id"] = recipe.recipe_id
    df["label"] = recipe.label
    return df


def _fpos_lofo_main_cache_paths() -> tuple[Path, Path]:
    base = THESIS_ROOT / "artifacts" / "aggregates" / "family_held_out" / "fpos"
    return (
        base / "leave_one_family_main5_by_family.csv",
        base / "leave_one_family_main5_aggregate.csv",
    )


def _fmiss_lofo_main_cache_paths() -> tuple[Path, Path]:
    base = THESIS_ROOT / "artifacts" / "aggregates" / "family_held_out" / "fmiss"
    return (
        base / "leave_one_family_main3_by_family.csv",
        base / "leave_one_family_main3_aggregate.csv",
    )


def _build_fpos_lofo_main_family_cache(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_family_path, aggregate_path = _fpos_lofo_main_cache_paths()
    if not force and by_family_path.exists() and aggregate_path.exists():
        by_family = pd.read_csv(by_family_path)
        aggregate = pd.read_csv(aggregate_path)
        if "label" in by_family.columns:
            by_family["label"] = by_family["label"].map(normalize_model_label)
        if "label" in aggregate.columns:
            aggregate["label"] = aggregate["label"].map(normalize_model_label)
        return by_family, aggregate

    recipe_map = {recipe_id: get_recipe(recipe_id) for recipe_id in FPOS_LOFO_MAIN_RECIPE_IDS}
    rows: list[dict[str, object]] = []

    lofo_bundle = get_limitations_bundle()
    anchor_map = {
        "reference_anchor_stack_xgboost": "fpos_pre_waveform_unlabeled",
        "wf_embed_anchor_stack_xgboost": "fpos_waveform_winner",
    }
    for _, row in lofo_bundle.lofo_family.iterrows():
        variant_id = str(row.get("variant_id", ""))
        recipe_id = anchor_map.get(variant_id)
        if recipe_id is None:
            continue
        rows.append(
            {
                "held_out_family": str(row["held_out_family"]),
                "recipe_id": recipe_id,
                "label": recipe_map[recipe_id].label,
                "r2": float(row["r2"]),
                "mae": float(row["mae"]),
                "seed_count": int(row.get("seed_count", len(DEFAULT_SWEEP_SEEDS))),
                "r2_std": float(row.get("r2_std", 0.0)),
                "mae_std": float(row.get("mae_std", 0.0)),
            }
        )

    extra_seed_rows: list[dict[str, object]] = []
    parquet_path = THESIS_ROOT / "data/raw/33000_ROWS.parquet"
    paired_raw_recipe = recipe_map["fpos_paired_raw"]
    paired_raw_config = TransferBenchmarkConfig(
        parquet_path=parquet_path,
        primary_targets=("fpos",),
        secondary_targets=(),
        feature_view_sets=(
            ("shape_only", ("shape_only",)),
            ("unit_raw", ("unit_raw",)),
        ),
        model_families=("paired_only",),
        protocol_modes=("inductive",),
        boosting_backends=("lightgbm",),
        include_ceiling_diagnostics=False,
        deploy_best_per_target=False,
        verbose=False,
    )
    paired_raw_benchmark = QCTransferBenchmark(paired_raw_config)
    transfer_prepared = paired_raw_benchmark.prepare_data()
    transfer_splits = build_stress_splits(transfer_prepared.df)
    context_variant_to_recipe = {
        "contextual_paired_only_lightgbm": "fpos_paired_context",
        "embed_nommd_fullctx_lightgbm": "fpos_context_stack",
    }
    stack_prepared = prepare_feature_table(parquet_path)
    stack_splits = build_stress_splits(stack_prepared.df)

    for seed in DEFAULT_SWEEP_SEEDS:
        paired_raw_config.random_state = int(seed)
        paired_raw_benchmark.config.random_state = int(seed)
        for split in transfer_splits:
            (
                _source_df,
                _y_source,
                paired_train,
                y_train,
                paired_test,
                y_test,
                source_target,
            ) = paired_raw_benchmark._target_frames("fpos", transfer_prepared.hybrid, split.paired_finetune, split.paired_test)
            paired_rows = paired_raw_benchmark._run_paired_only_family(
                phase="final_holdout",
                split_id=split.held_out_study,
                protocol_mode="inductive",
                prepared=transfer_prepared,
                eval_target="fpos",
                source_target=source_target,
                paired_train=paired_train,
                y_train=y_train,
                paired_test=paired_test,
                y_test=y_test,
                target_unlabeled_rows=0,
                used_unlabeled_paired=False,
            )
            for result in paired_rows:
                if str(result.get("candidate_id", "")) != paired_raw_recipe.result_key:
                    continue
                extra_seed_rows.append(
                    {
                        "seed": int(seed),
                        "held_out_family": split.held_out_study,
                        "recipe_id": paired_raw_recipe.recipe_id,
                        "label": paired_raw_recipe.label,
                        "r2": float(result["r2"]),
                        "mae": float(result["mae"]),
                    }
                )

        stack_config = MLStatisticsWorkbenchConfig(
            parquet_path=parquet_path,
            output_root=THESIS_ROOT / "artifacts" / "_tmp_lofo_main5",
            run_name=f"seed_{int(seed)}",
            targets=("fpos",),
            verbose=False,
            random_state=int(seed),
        )
        for split in stack_splits:
            state = build_base_stack_state_from_split(
                target="fpos",
                prepared=stack_prepared,
                split=split,
                config=stack_config,
            )
            no_pseudo_state = replace(state, trust_selected=state.trust_selected.head(0).copy())
            for idx, (variant_id, stack, no_pseudo) in enumerate(_variant_specs_for_target("fpos", state.base_stack), start=1):
                recipe_id = context_variant_to_recipe.get(str(variant_id))
                if recipe_id is None:
                    continue
                variant_state = no_pseudo_state if no_pseudo else state
                _, holdout_pred = evaluate_stack_variant(
                    state=variant_state,
                    stack=stack,
                    backend_id="lightgbm_default",
                    family_alpha=1.0,
                    random_state_offset=1000 + (100 * idx),
                )
                holdout_metrics = score_predictions(variant_state.y_test, holdout_pred)
                extra_seed_rows.append(
                    {
                        "seed": int(seed),
                        "held_out_family": split.held_out_study,
                        "recipe_id": recipe_id,
                        "label": recipe_map[recipe_id].label,
                        "r2": float(holdout_metrics["r2"]),
                        "mae": float(holdout_metrics["mae"]),
                    }
                )

    if extra_seed_rows:
        extra_df = pd.DataFrame(extra_seed_rows)
        aggregated_extra = (
            extra_df.groupby(["held_out_family", "recipe_id", "label"], as_index=False)
            .agg(
                r2=("r2", "mean"),
                mae=("mae", "mean"),
                seed_count=("seed", "nunique"),
                r2_std=("r2", lambda s: float(pd.Series(s).std(ddof=0)) if len(s) > 1 else 0.0),
                mae_std=("mae", lambda s: float(pd.Series(s).std(ddof=0)) if len(s) > 1 else 0.0),
            )
        )
        rows.extend(aggregated_extra.to_dict(orient="records"))

    by_family = pd.DataFrame(rows)
    if by_family.empty:
        empty = pd.DataFrame(columns=["held_out_family", "recipe_id", "label", "r2", "mae", "seed_count", "r2_std", "mae_std"])
        return empty, empty

    label_rank = {label: idx for idx, label in enumerate(FPOS_LOFO_MAIN_LABEL_ORDER)}
    by_family["label_rank"] = by_family["label"].map(label_rank).fillna(999).astype(int)
    by_family = by_family.sort_values(["held_out_family", "label_rank", "label"]).drop(columns=["label_rank"]).reset_index(drop=True)

    aggregate = (
        by_family.groupby(["recipe_id", "label"], as_index=False)
        .agg(
            macro_r2=("r2", "mean"),
            macro_mae=("mae", "mean"),
            worst_family_r2=("r2", "min"),
            best_family_r2=("r2", "max"),
            family_count=("held_out_family", "nunique"),
            seed_count=("seed_count", "max"),
        )
        .sort_values(["macro_r2", "macro_mae"], ascending=[False, True])
        .reset_index(drop=True)
    )

    by_family_path.parent.mkdir(parents=True, exist_ok=True)
    by_family.to_csv(by_family_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    return by_family, aggregate


def _build_fmiss_lofo_main_family_cache(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_family_path, aggregate_path = _fmiss_lofo_main_cache_paths()
    if not force and by_family_path.exists() and aggregate_path.exists():
        return pd.read_csv(by_family_path), pd.read_csv(aggregate_path)

    recipe_map = {recipe_id: get_recipe(recipe_id) for recipe_id in FMISS_LOFO_MAIN_RECIPE_IDS}
    extra_seed_rows: list[dict[str, object]] = []
    parquet_path = THESIS_ROOT / "data/raw/33000_ROWS.parquet"

    paired_raw_recipe = recipe_map["fmiss_paired_raw"]
    paired_raw_config = TransferBenchmarkConfig(
        parquet_path=parquet_path,
        primary_targets=("fmiss",),
        secondary_targets=(),
        feature_view_sets=(("shape_only", ("shape_only",)),),
        model_families=("paired_only",),
        protocol_modes=("inductive",),
        boosting_backends=("xgboost",),
        include_ceiling_diagnostics=False,
        deploy_best_per_target=False,
        verbose=False,
    )
    paired_raw_benchmark = QCTransferBenchmark(paired_raw_config)
    transfer_prepared = paired_raw_benchmark.prepare_data()
    transfer_splits = build_stress_splits(transfer_prepared.df)

    context_variant_to_recipe = {
        "contextual_paired_only_lightgbm": "fmiss_paired_context",
        "source_plus_reduced_latent_fullctx_lightgbm": "fmiss_reduced_latent",
    }
    stack_prepared = prepare_feature_table(parquet_path)
    stack_splits = build_stress_splits(stack_prepared.df)

    for seed in DEFAULT_SWEEP_SEEDS:
        paired_raw_config.random_state = int(seed)
        paired_raw_benchmark.config.random_state = int(seed)
        for split in transfer_splits:
            (
                _source_df,
                _y_source,
                paired_train,
                y_train,
                paired_test,
                y_test,
                source_target,
            ) = paired_raw_benchmark._target_frames("fmiss", transfer_prepared.hybrid, split.paired_finetune, split.paired_test)
            paired_rows = paired_raw_benchmark._run_paired_only_family(
                phase="final_holdout",
                split_id=split.held_out_study,
                protocol_mode="inductive",
                prepared=transfer_prepared,
                eval_target="fmiss",
                source_target=source_target,
                paired_train=paired_train,
                y_train=y_train,
                paired_test=paired_test,
                y_test=y_test,
                target_unlabeled_rows=0,
                used_unlabeled_paired=False,
            )
            for result in paired_rows:
                if str(result.get("candidate_id", "")) != paired_raw_recipe.result_key:
                    continue
                extra_seed_rows.append(
                    {
                        "seed": int(seed),
                        "held_out_family": split.held_out_study,
                        "recipe_id": paired_raw_recipe.recipe_id,
                        "label": paired_raw_recipe.label,
                        "r2": float(result["r2"]),
                        "mae": float(result["mae"]),
                    }
                )

        stack_config = MLStatisticsWorkbenchConfig(
            parquet_path=parquet_path,
            output_root=THESIS_ROOT / "artifacts" / "_tmp_lofo_main3_fmiss",
            run_name=f"seed_{int(seed)}",
            targets=("fmiss",),
            verbose=False,
            random_state=int(seed),
        )
        for split in stack_splits:
            state = build_base_stack_state_from_split(
                target="fmiss",
                prepared=stack_prepared,
                split=split,
                config=stack_config,
            )
            no_pseudo_state = replace(state, trust_selected=state.trust_selected.head(0).copy())
            for idx, (variant_id, stack, no_pseudo) in enumerate(_variant_specs_for_target("fmiss", state.base_stack), start=1):
                recipe_id = context_variant_to_recipe.get(str(variant_id))
                if recipe_id is None:
                    continue
                variant_state = no_pseudo_state if no_pseudo else state
                _, holdout_pred = evaluate_stack_variant(
                    state=variant_state,
                    stack=stack,
                    backend_id="lightgbm_default",
                    family_alpha=1.0,
                    random_state_offset=2000 + (100 * idx),
                )
                holdout_metrics = score_predictions(variant_state.y_test, holdout_pred)
                extra_seed_rows.append(
                    {
                        "seed": int(seed),
                        "held_out_family": split.held_out_study,
                        "recipe_id": recipe_id,
                        "label": recipe_map[recipe_id].label,
                        "r2": float(holdout_metrics["r2"]),
                        "mae": float(holdout_metrics["mae"]),
                    }
                )

    extra_df = pd.DataFrame(extra_seed_rows)
    if extra_df.empty:
        empty = pd.DataFrame(columns=["held_out_family", "recipe_id", "label", "r2", "mae", "seed_count", "r2_std", "mae_std"])
        return empty, empty

    by_family = (
        extra_df.groupby(["held_out_family", "recipe_id", "label"], as_index=False)
        .agg(
            r2=("r2", "mean"),
            mae=("mae", "mean"),
            seed_count=("seed", "nunique"),
            r2_std=("r2", lambda s: float(pd.Series(s).std(ddof=0)) if len(s) > 1 else 0.0),
            mae_std=("mae", lambda s: float(pd.Series(s).std(ddof=0)) if len(s) > 1 else 0.0),
        )
    )

    label_rank = {label: idx for idx, label in enumerate(FMISS_LOFO_MAIN_LABEL_ORDER)}
    by_family["label_rank"] = by_family["label"].map(label_rank).fillna(999).astype(int)
    by_family = by_family.sort_values(["held_out_family", "label_rank", "label"]).drop(columns=["label_rank"]).reset_index(drop=True)

    aggregate = (
        by_family.groupby(["recipe_id", "label"], as_index=False)
        .agg(
            macro_r2=("r2", "mean"),
            macro_mae=("mae", "mean"),
            worst_family_r2=("r2", "min"),
            best_family_r2=("r2", "max"),
            family_count=("held_out_family", "nunique"),
            seed_count=("seed_count", "max"),
        )
        .sort_values(["macro_r2", "macro_mae"], ascending=[False, True])
        .reset_index(drop=True)
    )

    by_family_path.parent.mkdir(parents=True, exist_ok=True)
    by_family.to_csv(by_family_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    return by_family, aggregate


def _legacy_backend_from_result_key(result_key: str) -> str:
    if "xgboost" in result_key:
        return "xgboost"
    if "catboost" in result_key:
        return "catboost"
    return "lightgbm"


def _legacy_feature_view_sets(feature_view: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    mapping = {
        "none": (("none", tuple()),),
        "unit_raw": (("unit_raw", ("unit_raw",)),),
        "shape_only": (("shape_only", ("shape_only",)),),
        "full_context": (("full_context", ("norm_swap", "recording_relative", "recording_context", "study_identity")),),
    }
    return mapping.get(feature_view, (("unit_raw", ("unit_raw",)),))


def _summarize_prediction_groups(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for key, part in frame.groupby(group_col, dropna=False, sort=False):
        true = pd.to_numeric(part["true"], errors="coerce")
        pred = pd.to_numeric(part["pred"], errors="coerce")
        mask = true.notna() & pred.notna()
        if not mask.any():
            continue
        true_v = true.loc[mask].to_numpy(dtype=float)
        pred_v = pred.loc[mask].to_numpy(dtype=float)
        err = pred_v - true_v
        r2 = float("nan")
        if len(true_v) >= 2 and pd.Series(true_v).nunique(dropna=True) > 1:
            ss_res = float(np.square(err).sum())
            ss_tot = float(np.square(true_v - true_v.mean()).sum())
            r2 = float("nan") if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))
        rows.append(
            {
                group_col: str(key),
                "row_count": int(mask.sum()),
                "mae": float(np.abs(err).mean()),
                "rmse": float(np.sqrt(np.square(err).mean())),
                "r2": r2,
                "bias": float(err.mean()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_group_rows(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    numeric_cols = [col for col in frame.columns if col not in set(group_cols) | {"seed"} and pd.api.types.is_numeric_dtype(frame[col])]
    grouped = frame.groupby(group_cols, dropna=False, sort=False)
    out = grouped[numeric_cols].mean().reset_index()
    out["seed_count"] = grouped.size().to_numpy()
    for col in numeric_cols:
        out[f"{col}_std"] = grouped[col].std(ddof=0).fillna(0.0).to_numpy()
    return out


def _rebuild_missing_legacy_paired_only_family_summary(recipe_id: str) -> pd.DataFrame:
    recipe = get_recipe(recipe_id)
    if recipe.model_family != "paired_only":
        return pd.DataFrame()

    seed_metrics = _load_seed_metrics(recipe_id)
    if seed_metrics.empty or "seed" not in seed_metrics.columns:
        return pd.DataFrame()

    from .modeling.specs import TransferBenchmarkConfig
    from .modeling.transfer import QCTransferBenchmark, _ensure_numeric_frame

    seeds = sorted(int(seed) for seed in seed_metrics["seed"].dropna().astype(int).unique().tolist())
    family_frames: list[pd.DataFrame] = []
    for seed in seeds:
        config = TransferBenchmarkConfig(
            parquet_path=THESIS_ROOT / "data/raw/33000_ROWS.parquet",
            primary_targets=(recipe.target,),
            secondary_targets=(),
            feature_view_sets=_legacy_feature_view_sets(recipe.feature_view),
            model_families=(recipe.model_family,),
            protocol_modes=("inductive",),
            boosting_backends=(_legacy_backend_from_result_key(recipe.result_key),),
            paired_weight_grid=(25.0,) if "w25" in recipe.result_key else (10.0,),
            include_ceiling_diagnostics=False,
            deploy_best_per_target=False,
            verbose=False,
            neural_verbose=False,
            random_state=seed,
        )
        benchmark = QCTransferBenchmark(config)
        prepared = benchmark.prepare_data()
        paired_train_all, paired_holdout, _ = benchmark._make_final_holdout(prepared.paired_matched)
        (
            _source_df,
            _y_source,
            paired_train,
            y_train,
            paired_test,
            y_test,
            _source_target,
        ) = benchmark._target_frames(recipe.target, prepared.hybrid, paired_train_all, paired_holdout)
        if paired_train.empty or paired_test.empty:
            continue
        feature_view = next(iter(prepared.feature_columns))
        cols = prepared.feature_columns.get(feature_view, [])
        if not cols:
            continue
        X_train = _ensure_numeric_frame(paired_train, cols)
        X_test = _ensure_numeric_frame(paired_test, cols)
        model = benchmark._make_backend_model(_legacy_backend_from_result_key(recipe.result_key))
        benchmark._fit_backend_model(model, X_train, y_train.values)
        pred = benchmark._predict_backend_model(model, X_test)
        family_summary = _summarize_prediction_groups(
            pd.DataFrame(
                {
                    "study_set": paired_test["study_set"].astype(str).to_numpy(),
                    "true": y_test.to_numpy(dtype=float),
                    "pred": pred,
                }
            ),
            "study_set",
        )
        if family_summary.empty:
            continue
        family_summary.insert(0, "recipe_id", recipe.recipe_id)
        family_summary.insert(1, "label", recipe.label)
        family_summary["seed"] = int(seed)
        family_frames.append(family_summary)

    if not family_frames:
        return pd.DataFrame()

    aggregate = _aggregate_group_rows(pd.concat(family_frames, ignore_index=True), ["recipe_id", "label", "study_set"])
    output_path = _aggregate_recipe_dir(recipe_id) / "per_family_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output_path, index=False)
    return aggregate


def build_dataset_partition_table() -> pd.DataFrame:
    bundle = build_dataset_bundle(with_context=False)
    rows = [
        {"subset": "full", "rows": len(bundle.full_df), "recordings": bundle.full_df["recording_key"].nunique(), "families": bundle.full_df["study_set"].nunique()},
        {"subset": "hybrid_labeled", "rows": len(bundle.hybrid_df), "recordings": bundle.hybrid_df["recording_key"].nunique(), "families": bundle.hybrid_df["study_set"].nunique()},
        {"subset": "paired_total", "rows": len(bundle.paired_df), "recordings": bundle.paired_df["recording_key"].nunique(), "families": bundle.paired_df["study_set"].nunique()},
        {"subset": "paired_labeled", "rows": len(bundle.paired_labeled_df), "recordings": bundle.paired_labeled_df["recording_key"].nunique(), "families": bundle.paired_labeled_df["study_set"].nunique()},
        {"subset": "paired_unlabeled", "rows": len(bundle.paired_unlabeled_df), "recordings": bundle.paired_unlabeled_df["recording_key"].nunique(), "families": bundle.paired_unlabeled_df["study_set"].nunique()},
    ]
    return pd.DataFrame(rows)


def build_target_missingness_table() -> pd.DataFrame:
    df = build_dataset_bundle(with_context=False).full_df
    rows = []
    for target in ["fpos", "fmiss", "accuracy"]:
        rows.append({
            "target": target,
            "missing_rows": int(df[target].isna().sum()),
            "missing_fraction": float(df[target].isna().mean()),
            "non_missing_rows": int(df[target].notna().sum()),
        })
    return pd.DataFrame(rows)


def build_family_coverage_table() -> pd.DataFrame:
    df = build_dataset_bundle(with_context=False).paired_df
    return (
        df.groupby("study_set", as_index=False)
        .agg(
            rows=("row_uid", "size"),
            recordings=("recording_key", "nunique"),
            sorters=("sorter_name", "nunique"),
            labeled_rows=("is_paired_labeled", "sum"),
        )
        .sort_values(["rows", "labeled_rows"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_feature_inventory_table() -> pd.DataFrame:
    df = build_dataset_bundle(with_context=True).full_df
    numeric = infer_numeric_feature_columns(df)
    waveform = get_waveform_columns(df)
    acg = get_acg_columns(df)
    full_context = get_feature_view(df, "full_context")
    reduced_latent = get_feature_view(df, "reduced_latent")
    waveform_embedding = get_feature_view(df, "waveform_embedding")
    context_only = get_feature_view(df, "contextual")
    rows = [
        {"feature_block": "numeric_total", "feature_count": len(numeric)},
        {"feature_block": "waveform_bins", "feature_count": len(waveform)},
        {"feature_block": "acg_bins", "feature_count": len(acg)},
        {"feature_block": "context_only", "feature_count": len(context_only)},
        {"feature_block": "full_context", "feature_count": len(full_context)},
        {"feature_block": "reduced_latent", "feature_count": len(reduced_latent)},
        {"feature_block": "waveform_embedding_view", "feature_count": len(waveform_embedding)},
    ]
    return pd.DataFrame(rows)


def build_benchmark_progress_table(
    target: str,
    protocol: str = "recording_disjoint_main",
    *,
    aggregate: bool = True,
    scope: str = "main_text",
) -> pd.DataFrame:
    if scope == "main_text":
        recipe_ids = get_main_text_recipe_ids(target)
        simple_transfer_id = {
            "fpos": "fpos_hybrid_plus_paired",
            "fmiss": "fmiss_hybrid_residual",
        }[target]
    elif scope == "core":
        recipe_ids = get_core_benchmark_recipe_ids(target)
        simple_transfer_id = {
            "fpos": "fpos_hybrid_only",
            "fmiss": "fmiss_hybrid_only",
        }[target]
    else:
        raise ValueError(f"Unsupported benchmark scope: {scope}")

    table = build_benchmark_table(target, recipe_ids, protocol, aggregate=aggregate).copy()
    if "label" in table.columns:
        table["label"] = table["label"].map(normalize_model_label)
    table["mae"] = pd.to_numeric(table["mae"], errors="coerce")
    table["r2"] = pd.to_numeric(table["r2"], errors="coerce")
    baseline = table.loc[table["recipe_id"].astype(str) == f"{target}_dummy"].iloc[0]
    simple_transfer = table.loc[table["recipe_id"].astype(str) == simple_transfer_id].iloc[0]
    table["delta_r2_vs_floor"] = table["r2"] - float(baseline["r2"])
    table["delta_r2_vs_basic_transfer"] = table["r2"] - float(simple_transfer["r2"])
    table["delta_mae_vs_floor"] = table["mae"] - float(baseline["mae"])
    table["delta_mae_vs_basic_transfer"] = table["mae"] - float(simple_transfer["mae"])
    return table


def build_ablation_winner_table(
    target: str,
    protocol: str = "recording_disjoint_main",
    *,
    aggregate: bool = True,
) -> pd.DataFrame:
    rows = []
    for group_name, recipe_ids in get_recipe_groups(target).items():
        table = build_benchmark_table(target, recipe_ids, protocol, aggregate=aggregate).copy()
        if "label" in table.columns:
            table["label"] = table["label"].map(normalize_model_label)
        table["r2"] = pd.to_numeric(table["r2"], errors="coerce")
        table["mae"] = pd.to_numeric(table["mae"], errors="coerce")
        best = table.sort_values(["r2", "mae"], ascending=[False, True]).iloc[0]
        rows.append({
            "ablation_group": group_name,
            "best_label": best["label"],
            "best_recipe_id": best["recipe_id"],
            "best_r2": float(best["r2"]),
            "best_mae": float(best["mae"]),
            "notes": best.get("notes", ""),
        })
    return pd.DataFrame(rows)


def build_main_wave_summary(
    target: str,
    recipe_ids: list[str],
    protocol: str = "recording_disjoint_main",
    *,
    aggregate: bool = True,
) -> dict[str, pd.DataFrame]:
    families = []
    recordings = []
    for recipe_id in recipe_ids:
        fam = build_family_summary(recipe_id, target, protocol, aggregate=aggregate)
        rec = build_recording_summary(recipe_id, target, protocol, aggregate=aggregate)
        if fam is not None:
            if "label" in fam.columns:
                fam["label"] = fam["label"].map(normalize_model_label)
            families.append(fam)
        if rec is not None:
            if "label" in rec.columns:
                rec["label"] = rec["label"].map(normalize_model_label)
            recordings.append(rec)
    return {
        "family": pd.concat(families, ignore_index=True) if families else pd.DataFrame(),
        "recording": pd.concat(recordings, ignore_index=True) if recordings else pd.DataFrame(),
    }


def build_seed_rank_bundle(
    target: str,
    recipe_ids: list[str],
    protocol: str = "recording_disjoint_main",
) -> dict[str, pd.DataFrame]:
    frames = []
    for recipe_id in recipe_ids:
        recipe = get_recipe(recipe_id)
        if recipe.target != target or recipe.protocol != protocol:
            raise ValueError(f"Recipe mismatch for {recipe_id}: expected {(target, protocol)}, got {(recipe.target, recipe.protocol)}")
        df = _load_seed_metrics(recipe_id)
        if df.empty:
            continue
        cols = [col for col in ["seed", "r2", "mae"] if col in df.columns]
        frames.append(df.loc[:, [*cols, "recipe_id", "label"]].copy())
    if not frames:
        empty = pd.DataFrame()
        return {"seed_rows": empty, "summary": empty, "rank_pivot": empty, "r2_pivot": empty, "winner_rows": empty}

    seed_rows = pd.concat(frames, ignore_index=True)
    seed_rows["seed"] = pd.to_numeric(seed_rows["seed"], errors="coerce").astype("Int64")
    seed_rows["r2"] = pd.to_numeric(seed_rows["r2"], errors="coerce")
    seed_rows["mae"] = pd.to_numeric(seed_rows["mae"], errors="coerce")
    ranked = (
        seed_rows.sort_values(["seed", "r2", "mae"], ascending=[True, False, True])
        .reset_index(drop=True)
        .copy()
    )
    ranked["rank"] = ranked.groupby("seed").cumcount() + 1
    ranked["winner_flag"] = ranked["rank"] == 1

    summary = (
        ranked.groupby(["recipe_id", "label"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", lambda s: float(s.std(ddof=0)) if len(s) > 1 else 0.0),
            min_r2=("r2", "min"),
            max_r2=("r2", "max"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", lambda s: float(s.std(ddof=0)) if len(s) > 1 else 0.0),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            winner_count=("winner_flag", "sum"),
            top3_count=("rank", lambda s: int((s <= 3).sum())),
        )
        .sort_values(["mean_rank", "mean_r2", "mean_mae"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    label_order = summary["label"].tolist()
    rank_pivot = ranked.pivot(index="label", columns="seed", values="rank").reindex(label_order)
    r2_pivot = ranked.pivot(index="label", columns="seed", values="r2").reindex(label_order)
    winner_rows = (
        ranked.loc[ranked["winner_flag"], ["seed", "recipe_id", "label", "r2", "mae", "rank"]]
        .sort_values("seed")
        .reset_index(drop=True)
    )
    return {
        "seed_rows": ranked,
        "summary": summary,
        "rank_pivot": rank_pivot,
        "r2_pivot": r2_pivot,
        "winner_rows": winner_rows,
    }


def build_seed_split_profile(
    recipe_id: str,
    target: str,
    protocol: str = "recording_disjoint_main",
    *,
    aggregate: bool = True,
) -> pd.DataFrame:
    result = load_experiment(recipe_id, target, protocol, aggregate=aggregate)
    preds = result.predictions.copy() if result.predictions is not None else pd.DataFrame()
    if preds.empty or "seed" not in preds.columns:
        return pd.DataFrame()
    preds["seed"] = pd.to_numeric(preds["seed"], errors="coerce").astype("Int64")
    counts = (
        preds.groupby(["seed", "study_set"], as_index=False)
        .agg(test_rows_family=("row_uid", "size"), family_recordings=("recording_key", "nunique"))
    )
    totals = (
        preds.groupby("seed", as_index=False)
        .agg(
            test_rows=("row_uid", "size"),
            test_recordings=("recording_key", "nunique"),
            true_mean=("true", "mean"),
            true_std=("true", "std"),
            true_iqr=("true", _iqr),
        )
        .sort_values("seed")
        .reset_index(drop=True)
    )
    counts["family_share"] = counts["test_rows_family"] / counts.groupby("seed")["test_rows_family"].transform("sum")
    row_wide = counts.pivot(index="seed", columns="study_set", values="test_rows_family").fillna(0.0)
    row_wide.columns = [f"rows__{col}" for col in row_wide.columns]
    share_wide = counts.pivot(index="seed", columns="study_set", values="family_share").fillna(0.0)
    share_wide.columns = [f"share__{col}" for col in share_wide.columns]
    return (
        totals.merge(row_wide.reset_index(), on="seed", how="left")
        .merge(share_wide.reset_index(), on="seed", how="left")
        .sort_values("seed")
        .reset_index(drop=True)
    )


def build_split_regime_bundle(
    target: str,
    recipe_ids: list[str],
    *,
    split_profile_recipe_id: str,
    protocol: str = "recording_disjoint_main",
) -> dict[str, pd.DataFrame]:
    rank_bundle = build_seed_rank_bundle(target, recipe_ids, protocol=protocol)
    split_profile = build_seed_split_profile(split_profile_recipe_id, target, protocol=protocol, aggregate=True)
    if rank_bundle["winner_rows"].empty or split_profile.empty:
        empty = pd.DataFrame()
        return {**rank_bundle, "split_profile": split_profile, "winner_table": empty, "winner_profile": empty}

    winner_table = (
        rank_bundle["winner_rows"]
        .merge(split_profile, on="seed", how="left")
        .sort_values("seed")
        .reset_index(drop=True)
    )
    share_cols = sorted(col for col in winner_table.columns if col.startswith("share__"))
    winner_profile = (
        winner_table.groupby(["recipe_id", "label"], as_index=False)
        .agg(
            winner_count=("seed", "size"),
            mean_true_mean=("true_mean", "mean"),
            mean_true_std=("true_std", "mean"),
            mean_true_iqr=("true_iqr", "mean"),
            **{col: (col, "mean") for col in share_cols},
        )
        .sort_values(["winner_count", "mean_true_std"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return {
        **rank_bundle,
        "split_profile": split_profile,
        "winner_table": winner_table,
        "winner_profile": winner_profile,
    }


def build_family_preference_table(
    target: str,
    recipe_ids: list[str],
    protocol: str = "recording_disjoint_main",
    *,
    aggregate: bool = True,
) -> dict[str, pd.DataFrame]:
    frames = []
    for recipe_id in recipe_ids:
        recipe = get_recipe(recipe_id)
        if recipe.target != target or recipe.protocol != protocol:
            raise ValueError(f"Recipe mismatch for {recipe_id}: expected {(target, protocol)}, got {(recipe.target, recipe.protocol)}")
        fam = build_family_summary(recipe_id, target, protocol, aggregate=aggregate)
        if (fam is None or fam.empty) and aggregate and recipe.model_family == "paired_only":
            fam = _rebuild_missing_legacy_paired_only_family_summary(recipe_id)
        if fam is None or fam.empty:
            continue
        part = fam.copy()
        if "label" in part.columns:
            part["label"] = part["label"].map(normalize_model_label)
        for col in ["row_count", "mae", "rmse", "r2", "bias", "seed_count", "r2_std", "mae_std"]:
            if col in part.columns:
                part[col] = pd.to_numeric(part[col], errors="coerce")
        frames.append(part)
    if not frames:
        empty = pd.DataFrame()
        return {"long": empty, "best": empty, "r2_pivot": empty, "mae_pivot": empty}

    long_df = pd.concat(frames, ignore_index=True)
    ranked = (
        long_df.sort_values(["study_set", "r2", "mae"], ascending=[True, False, True])
        .reset_index(drop=True)
        .copy()
    )
    ranked["family_rank"] = ranked.groupby("study_set").cumcount() + 1
    best = (
        ranked.loc[ranked["family_rank"] == 1, ["study_set", "recipe_id", "label", "r2", "mae", "family_rank"]]
        .rename(columns={"label": "best_label", "recipe_id": "best_recipe_id", "r2": "best_r2", "mae": "best_mae"})
        .reset_index(drop=True)
    )
    label_order = (
        ranked.groupby("label", as_index=False)
        .agg(mean_r2=("r2", "mean"), mean_mae=("mae", "mean"))
        .sort_values(["mean_r2", "mean_mae"], ascending=[False, True])["label"]
        .tolist()
    )
    r2_pivot = ranked.pivot(index="study_set", columns="label", values="r2").loc[:, label_order]
    mae_pivot = ranked.pivot(index="study_set", columns="label", values="mae").loc[:, label_order]
    return {"long": ranked, "best": best, "r2_pivot": r2_pivot, "mae_pivot": mae_pivot}


def build_fpos_lofo_main_family_preference_table(*, force_recompute: bool = False) -> dict[str, pd.DataFrame]:
    by_family, aggregate = _build_fpos_lofo_main_family_cache(force=force_recompute)
    if by_family.empty:
        empty = pd.DataFrame()
        return {"long": empty, "best": empty, "r2_pivot": empty, "mae_pivot": empty, "aggregate": empty}

    ranked = (
        by_family.sort_values(["held_out_family", "r2", "mae"], ascending=[True, False, True])
        .reset_index(drop=True)
        .copy()
    )
    ranked["label"] = ranked["label"].map(normalize_model_label)
    if not aggregate.empty and "label" in aggregate.columns:
        aggregate["label"] = aggregate["label"].map(normalize_model_label)
    ranked["family_rank"] = ranked.groupby("held_out_family").cumcount() + 1
    best = (
        ranked.loc[
            ranked["family_rank"] == 1,
            ["held_out_family", "recipe_id", "label", "r2", "r2_std", "mae", "mae_std", "family_rank"],
        ]
        .rename(
            columns={
                "label": "best_label",
                "recipe_id": "best_recipe_id",
                "r2": "best_r2",
                "r2_std": "best_r2_std",
                "mae": "best_mae",
                "mae_std": "best_mae_std",
            }
        )
        .reset_index(drop=True)
    )
    r2_pivot = ranked.pivot(index="held_out_family", columns="label", values="r2")
    mae_pivot = ranked.pivot(index="held_out_family", columns="label", values="mae")
    ordered_cols = [label for label in FPOS_LOFO_MAIN_LABEL_ORDER if label in r2_pivot.columns]
    r2_pivot = r2_pivot.loc[:, ordered_cols]
    mae_pivot = mae_pivot.loc[:, ordered_cols]
    return {"long": ranked, "best": best, "r2_pivot": r2_pivot, "mae_pivot": mae_pivot, "aggregate": aggregate}


def build_fmiss_lofo_main_family_preference_table(*, force_recompute: bool = False) -> dict[str, pd.DataFrame]:
    by_family, aggregate = _build_fmiss_lofo_main_family_cache(force=force_recompute)
    if by_family.empty:
        empty = pd.DataFrame()
        return {"long": empty, "best": empty, "r2_pivot": empty, "mae_pivot": empty, "aggregate": empty}

    ranked = (
        by_family.sort_values(["held_out_family", "r2", "mae"], ascending=[True, False, True])
        .reset_index(drop=True)
        .copy()
    )
    ranked["family_rank"] = ranked.groupby("held_out_family").cumcount() + 1
    best = (
        ranked.loc[ranked["family_rank"] == 1, ["held_out_family", "recipe_id", "label", "r2", "mae", "family_rank"]]
        .rename(columns={"label": "best_label", "recipe_id": "best_recipe_id", "r2": "best_r2", "mae": "best_mae"})
        .reset_index(drop=True)
    )
    r2_pivot = ranked.pivot(index="held_out_family", columns="label", values="r2")
    mae_pivot = ranked.pivot(index="held_out_family", columns="label", values="mae")
    ordered_cols = [label for label in FMISS_LOFO_MAIN_LABEL_ORDER if label in r2_pivot.columns]
    r2_pivot = r2_pivot.loc[:, ordered_cols]
    mae_pivot = mae_pivot.loc[:, ordered_cols]
    return {"long": ranked, "best": best, "r2_pivot": r2_pivot, "mae_pivot": mae_pivot, "aggregate": aggregate}


def build_family_structure_context_table(
    target: str,
    recipe_ids: list[str],
    protocol: str = "recording_disjoint_main",
    *,
    aggregate: bool = True,
) -> pd.DataFrame:
    coverage = build_family_coverage_table().copy()
    coverage = coverage.rename(columns={"rows": "packaged_rows", "recordings": "packaged_recordings", "labeled_rows": "packaged_labeled_rows"})
    coverage["packaged_unlabeled_rows"] = coverage["packaged_rows"] - coverage["packaged_labeled_rows"]

    bundle = build_dataset_bundle(with_context=False)
    paired_labeled = bundle.paired_labeled_df.copy()
    paired_labeled[target] = pd.to_numeric(paired_labeled[target], errors="coerce")
    target_stats = (
        paired_labeled.groupby("study_set", as_index=False)
        .agg(
            target_mean=(target, "mean"),
            target_std=(target, "std"),
            target_iqr=(target, _iqr),
            packaged_labeled_recordings=("recording_key", "nunique"),
        )
    )

    pca = get_domain_shift_bundle().pca_2d.copy()
    hybrid_centroid = pca.loc[pca["dataset_type"] == "hybrid", ["pc1", "pc2"]].mean()
    paired_centroids = (
        pca.loc[pca["dataset_type"] == "paired"]
        .groupby("study_set", as_index=False)
        .agg(pc1_mean=("pc1", "mean"), pc2_mean=("pc2", "mean"))
    )
    paired_centroids["paired_pca_distance_from_hybrid"] = np.sqrt(
        (paired_centroids["pc1_mean"] - float(hybrid_centroid["pc1"])) ** 2
        + (paired_centroids["pc2_mean"] - float(hybrid_centroid["pc2"])) ** 2
    )

    family_pref = build_family_preference_table(target, recipe_ids, protocol=protocol, aggregate=aggregate)
    best_recording = family_pref["best"].rename(
        columns={
            "best_label": "best_recording_disjoint_label",
            "best_recipe_id": "best_recording_disjoint_recipe_id",
            "best_r2": "best_recording_disjoint_r2",
            "best_mae": "best_recording_disjoint_mae",
        }
    )

    lofo = get_limitations_bundle().lofo_family.copy()
    if not lofo.empty:
        lofo = lofo.loc[lofo["variant_id"].astype(str).isin(LOFO_LABELS.keys())].copy()
        lofo["r2"] = pd.to_numeric(lofo["r2"], errors="coerce")
        lofo["mae"] = pd.to_numeric(lofo["mae"], errors="coerce")
        lofo_ranked = (
            lofo.sort_values(["held_out_family", "r2", "mae"], ascending=[True, False, True])
            .groupby("held_out_family", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        lofo_best = lofo_ranked.loc[:, ["held_out_family", "variant_id", "r2", "mae"]].rename(
            columns={
                "held_out_family": "study_set",
                "variant_id": "best_lofo_variant_id",
                "r2": "best_lofo_r2",
                "mae": "best_lofo_mae",
            }
        )
        lofo_best["best_lofo_label"] = lofo_best["best_lofo_variant_id"].map(lambda x: LOFO_LABELS.get(str(x), str(x)))
    else:
        lofo_best = pd.DataFrame(columns=["study_set", "best_lofo_variant_id", "best_lofo_r2", "best_lofo_mae", "best_lofo_label"])

    external = pd.DataFrame(
        [{"study_set": study_set, **payload} for study_set, payload in SPIKEFOREST_PAIRED_STUDY_CONTEXT.items()]
    )

    merged = (
        external.merge(coverage, on="study_set", how="outer")
        .merge(target_stats, on="study_set", how="left")
        .merge(paired_centroids.loc[:, ["study_set", "paired_pca_distance_from_hybrid"]], on="study_set", how="left")
        .merge(best_recording, on="study_set", how="left")
        .merge(lofo_best, on="study_set", how="left")
    )
    merged["packaged_recording_fraction_vs_spikeforest"] = (
        merged["packaged_recordings"] / merged["spikeforest_recordings"]
    )
    return merged.sort_values("study_set").reset_index(drop=True)


def build_extreme_groups_table(summary_df: pd.DataFrame, *, group_col: str, metric: str = "r2", top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = summary_df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    top = df.sort_values(metric, ascending=False).head(top_n).reset_index(drop=True)
    bottom = df.sort_values(metric, ascending=True).head(top_n).reset_index(drop=True)
    return top, bottom


def build_prediction_diagnostics(
    recipe_id: str,
    target: str,
    protocol: str = "recording_disjoint_main",
    *,
    aggregate: bool = True,
) -> dict[str, float | pd.DataFrame]:
    result = load_experiment(recipe_id, target, protocol, aggregate=aggregate)
    preds = result.predictions.copy() if result.predictions is not None else pd.DataFrame()
    if preds.empty:
        return {"bias": float("nan"), "mae": float("nan"), "correlation": float("nan"), "predictions": preds}
    preds["residual"] = preds["pred"] - preds["true"]
    corr = preds[["true", "pred"]].corr().iloc[0, 1]
    return {
        "bias": float(preds["residual"].mean()),
        "mae": float(preds["residual"].abs().mean()),
        "correlation": float(corr),
        "predictions": preds,
    }


def build_shap_group_summary() -> pd.DataFrame:
    return load_fpos_shap_group_summary().copy()
