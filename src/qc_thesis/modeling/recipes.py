from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score

from ..artifacts import make_run_artifact_paths, write_manifest
from ..paths import REDUCED_LATENT_CONTEXT_LABEL, THESIS_ROOT, WAVEFORM_AUGMENTED_LABEL
from .neural import run_ssl_with_config, run_mmd_with_config
from .transfer import QCTransferBenchmark
from .stacking.recipes import STACKING_FAMILIES


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    label: str
    target: str
    protocol: str
    ablation_group: str
    feature_view: str
    model_family: str
    stage: str
    runner_key: str
    result_key: str
    implementation_module: str
    result_filters: tuple[tuple[str, str], ...] = ()
    notes: str = ""
    main_text: bool = True


@dataclass(frozen=True)
class RecipeResult:
    recipe: RecipeSpec
    metrics: pd.DataFrame
    predictions: pd.DataFrame | None
    per_family_summary: pd.DataFrame | None
    per_recording_summary: pd.DataFrame | None
    output_dir: Path


DEFAULT_SWEEP_SEEDS = tuple(range(42, 62))


FPOS_RECIPE_GROUPS = {
    "source_supervision": [
        "fpos_dummy",
        "fpos_paired_raw",
        "fpos_hybrid_only",
    ],
    "context": [
        "fpos_paired_raw",
        "fpos_paired_context",
        "fpos_context_stack",
    ],
    "unlabeled_paired_structure": [
        "fpos_context_stack",
        "fpos_pre_waveform_unlabeled",
    ],
    "waveform_representation": [
        "fpos_pre_waveform_unlabeled",
        "fpos_waveform_winner",
    ],
    "robustness": [
        "fpos_pre_waveform_unlabeled",
        "fpos_waveform_winner",
    ],
}


FMISS_RECIPE_GROUPS = {
    "source_target_choice": [
        "fmiss_dummy",
        "fmiss_paired_raw",
        "fmiss_hybrid_only",
    ],
    "context": [
        "fmiss_paired_raw",
        "fmiss_paired_context",
        "fmiss_reduced_latent",
    ],
    "transfer_family": [
        "fmiss_hybrid_only",
        "fmiss_paired_context",
        "fmiss_reduced_latent",
    ],
}


CORE_BENCHMARK_RECIPE_IDS = {
    "fpos": [
        "fpos_dummy",
        "fpos_paired_raw",
        "fpos_hybrid_only",
        "fpos_paired_context",
        "fpos_context_stack",
        "fpos_pre_waveform_unlabeled",
        "fpos_waveform_winner",
    ],
    "fmiss": [
        "fmiss_dummy",
        "fmiss_paired_raw",
        "fmiss_hybrid_only",
        "fmiss_paired_context",
        "fmiss_reduced_latent",
    ],
}


def get_recipe_specs() -> list[RecipeSpec]:
    return [
        RecipeSpec("fpos_dummy", "Dummy paired mean", "fpos", "recording_disjoint_main", "source_supervision", "none", "dummy", "baseline", "legacy_transfer", "inductive|dummy|paired_mean|none", "qc_thesis.modeling.recipes", (("phase", "final_holdout"),), notes="Sanity floor"),
        RecipeSpec("fpos_paired_raw", "Paired-only raw", "fpos", "recording_disjoint_main", "source_supervision", "unit_raw", "paired_only", "baseline", "legacy_transfer", "inductive|paired_only|lightgbm|unit_raw", "qc_thesis.modeling.recipes", (("phase", "final_holdout"),), notes="Small paired baseline"),
        RecipeSpec("fpos_hybrid_only", "Hybrid-only XGBoost", "fpos", "recording_disjoint_main", "source_supervision", "unit_raw", "hybrid_only", "baseline", "legacy_transfer", "inductive|hybrid_only|xgboost|unit_raw", "qc_thesis.modeling.recipes", (("phase", "final_holdout"),), notes="Pure source failure baseline"),
        RecipeSpec("fpos_hybrid_calibrated", "Calibrated hybrid XGBoost", "fpos", "recording_disjoint_main", "source_supervision", "unit_raw", "hybrid_calibrated", "baseline", "legacy_transfer", "inductive|hybrid_calibrated|xgboost+isotonic|unit_raw", "qc_thesis.modeling.recipes", (("phase", "final_holdout"),), notes="Simple calibrated transfer", main_text=False),
        RecipeSpec("fpos_hybrid_plus_paired", "Hybrid+paired basic XGBoost", "fpos", "recording_disjoint_main", "source_supervision", "unit_raw", "hybrid_plus_paired", "baseline", "legacy_transfer", "inductive|hybrid_plus_paired|xgboost_w25|unit_raw", "qc_thesis.modeling.recipes", (("phase", "final_holdout"),), notes="Basic transfer baseline", main_text=False),
        RecipeSpec("fpos_hybrid_residual", "Hybrid stack residual", "fpos", "recording_disjoint_main", "source_supervision", "unit_raw", "hybrid_stack", "baseline", "legacy_transfer", "inductive|hybrid_stack|lightgbm_residual|unit_raw", "qc_thesis.modeling.recipes", (("phase", "final_holdout"),), notes="First nonlinear stack", main_text=False),
        RecipeSpec("fpos_paired_context", "Paired-only context", "fpos", "recording_disjoint_main", "context", "full_context", "context_first", "main_text", "context_stack", "contextual_paired_only_lightgbm", "qc_thesis.modeling.stacking.recipes", notes="Context-only target baseline"),
        RecipeSpec("fpos_context_stack", "Context-first source stack", "fpos", "recording_disjoint_main", "context", "full_context", "context_first", "main_text", "context_stack", "embed_nommd_fullctx_lightgbm", "qc_thesis.modeling.stacking.recipes", notes="Best pre-unlabeled context-first stack"),
        RecipeSpec("fpos_ssl", "Best SSL neural", "fpos", "recording_disjoint_main", "context", "contextual", "ssl_neural", "main_text", "ssl_neural", "ssl_fpos_contextual", "qc_thesis.modeling.recipes", notes="Best SSL comparison", main_text=False),
        RecipeSpec("fpos_mmd", "Best MMD neural", "fpos", "recording_disjoint_main", "context", "contextual", "mmd_neural", "main_text", "mmd_neural", "neural_mmd_finetuned", "qc_thesis.modeling.recipes", notes="Best MMD comparison", main_text=False),
        RecipeSpec("fpos_pre_waveform_unlabeled", "Pre-waveform unlabeled structure", "fpos", "recording_disjoint_main", "unlabeled_paired_structure", "full_context", "pseudo_anchor", "main_text", "anchor_stack", "source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default", "qc_thesis.modeling.stacking.recipes", notes="Strongest positive result before waveform embedding"),
        RecipeSpec("fpos_family_balanced", "Family-balanced robustness", "fpos", "recording_disjoint_main", "robustness", "full_context", "robustness_weighted", "archive_only", "robustness_stack", "family_balanced_strong_xgboost", "qc_thesis.modeling.stacking.recipes", notes="Archived robustness tradeoff variant", main_text=False),
        RecipeSpec("fpos_waveform_winner", WAVEFORM_AUGMENTED_LABEL, "fpos", "recording_disjoint_main", "waveform_representation", "waveform_embedding", "signal_embedding", "main_text", "waveform_stack", "wf_embed_anchor_stack_xgboost", "qc_thesis.modeling.stacking.recipes", notes="Final fpos waveform-augmented stack"),
        RecipeSpec("fmiss_dummy", "Dummy paired mean", "fmiss", "recording_disjoint_main", "source_target_choice", "none", "dummy", "baseline", "legacy_transfer", "inductive|dummy|paired_mean|none", "qc_thesis.modeling.recipes", (("phase", "final_holdout"),), notes="Sanity floor"),
        RecipeSpec("fmiss_paired_raw", "Paired-only raw", "fmiss", "recording_disjoint_main", "source_target_choice", "shape_only", "paired_only", "baseline", "legacy_transfer", "inductive|paired_only|xgboost|shape_only", "qc_thesis.modeling.recipes", (("phase", "final_holdout"),), notes="Strict paired-only baseline"),
        RecipeSpec("fmiss_hybrid_only", "Hybrid-only XGBoost", "fmiss", "recording_disjoint_main", "source_target_choice", "unit_raw", "hybrid_only", "baseline", "legacy_transfer", "inductive|hybrid_only|xgboost|unit_raw", "qc_thesis.modeling.recipes", (("phase", "final_holdout"),), notes="Pure source failure baseline"),
        RecipeSpec("fmiss_hybrid_calibrated", "Calibrated hybrid", "fmiss", "recording_disjoint_main", "source_target_choice", "unit_raw", "hybrid_calibrated", "baseline", "legacy_transfer", "inductive|hybrid_calibrated|lightgbm+linear|unit_raw", "qc_thesis.modeling.recipes", (("phase", "final_holdout"),), notes="Simple calibrated transfer", main_text=False),
        RecipeSpec("fmiss_hybrid_residual", "Hybrid stack residual", "fmiss", "recording_disjoint_main", "transfer_family", "unit_raw", "hybrid_stack", "baseline", "legacy_transfer", "inductive|hybrid_stack|lightgbm_residual|unit_raw", "qc_thesis.modeling.recipes", (("phase", "final_holdout"),), notes="First positive transfer baseline", main_text=False),
        RecipeSpec("fmiss_ssl", "Best SSL neural", "fmiss", "recording_disjoint_main", "transfer_family", "contextual", "ssl_neural", "main_text", "ssl_neural", "ssl_fmiss_contextual", "qc_thesis.modeling.recipes", notes="Best SSL comparison", main_text=False),
        RecipeSpec("fmiss_mmd", "Best MMD neural", "fmiss", "recording_disjoint_main", "transfer_family", "contextual", "mmd_neural", "main_text", "mmd_neural", "neural_mmd_finetuned", "qc_thesis.modeling.recipes", notes="Best MMD comparison", main_text=False),
        RecipeSpec("fmiss_paired_context", "Paired-only context", "fmiss", "recording_disjoint_main", "context", "full_context", "context_first", "main_text", "context_stack", "contextual_paired_only_lightgbm", "qc_thesis.modeling.stacking.recipes", notes="Context-only target baseline"),
        RecipeSpec("fmiss_source_reweighted", "Source-reweighted full context", "fmiss", "recording_disjoint_main", "context", "full_context", "context_first", "main_text", "context_stack", "source_reweighted_fullctx_lightgbm", "qc_thesis.modeling.stacking.recipes", notes="Source-weighted context-first stack", main_text=False),
        RecipeSpec("fmiss_embed_nommd", "Embed-nommd full context", "fmiss", "recording_disjoint_main", "representation_filtering", "full_context", "context_first", "main_text", "context_stack", "embed_nommd_fullctx_lightgbm", "qc_thesis.modeling.stacking.recipes", notes="Full latent context-first stack", main_text=False),
        RecipeSpec("fmiss_reduced_latent", REDUCED_LATENT_CONTEXT_LABEL, "fmiss", "recording_disjoint_main", "representation_filtering", "reduced_latent", "context_first", "main_text", "context_stack", "source_plus_reduced_latent_fullctx_lightgbm", "qc_thesis.modeling.stacking.recipes", notes="Final fmiss reduced-latent context stack"),
        RecipeSpec("fmiss_transplanted_fpos", "Transplanted fpos waveform recipe", "fmiss", "recording_disjoint_main", "unlabeled_paired_use", "waveform_embedding", "transplant", "main_text", "waveform_transfer", "fmiss_wf_embed_anchor_stack_xgboost", "qc_thesis.modeling.stacking.recipes", notes="Failed fpos transplant", main_text=False),
    ]


def get_recipe(recipe_id: str) -> RecipeSpec:
    for spec in get_recipe_specs():
        if spec.recipe_id == recipe_id:
            return spec
    raise KeyError(recipe_id)


def get_recipes(*, target: str | None = None, protocol: str | None = None, main_text: bool | None = None) -> list[RecipeSpec]:
    specs = get_recipe_specs()
    if target is not None:
        specs = [spec for spec in specs if spec.target == target]
    if protocol is not None:
        specs = [spec for spec in specs if spec.protocol == protocol]
    if main_text is not None:
        specs = [spec for spec in specs if spec.main_text == main_text]
    return specs


def get_recipe_groups(target: str) -> dict[str, list[str]]:
    if target == "fpos":
        return FPOS_RECIPE_GROUPS
    if target == "fmiss":
        return FMISS_RECIPE_GROUPS
    raise KeyError(target)


def get_main_text_recipe_ids(target: str) -> list[str]:
    return [spec.recipe_id for spec in get_recipes(target=target, protocol="recording_disjoint_main", main_text=True)]


def get_core_benchmark_recipe_ids(target: str) -> list[str]:
    try:
        return list(CORE_BENCHMARK_RECIPE_IDS[target])
    except KeyError as exc:
        raise KeyError(target) from exc


_STACKING_ONLY = STACKING_FAMILIES


def list_recipes(*, target: str | None = None, protocol: str | None = None, main_text: bool | None = None) -> list[RecipeSpec]:
    return get_recipes(target=target, protocol=protocol, main_text=main_text)


def build_recipe_inventory() -> pd.DataFrame:
    rows = [
        {
            "recipe_id": spec.recipe_id,
            "target": spec.target,
            "protocol": spec.protocol,
            "runner_key": spec.runner_key,
            "implementation_module": spec.implementation_module,
            "result_key": spec.result_key,
            "main_text": spec.main_text,
        }
        for spec in get_recipe_specs()
    ]
    return pd.DataFrame(rows).sort_values(["target", "recipe_id"]).reset_index(drop=True)


def validate_recipe_registry() -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for spec in get_recipe_specs():
        if spec.recipe_id in seen:
            errors.append(f"Duplicate recipe_id: {spec.recipe_id}")
        seen.add(spec.recipe_id)
        if spec.runner_key not in _RUNNERS:
            errors.append(f"Missing runner for recipe {spec.recipe_id}: {spec.runner_key}")
            continue
        try:
            importlib.import_module(spec.implementation_module)
        except Exception as exc:  # pragma: no cover
            errors.append(f"Failed to import {spec.implementation_module} for {spec.recipe_id}: {exc}")
    return errors


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _apply_recipe_filters(frame: pd.DataFrame, recipe: RecipeSpec, *, include_result_key: bool = True) -> pd.DataFrame:
    filtered = frame.copy()
    if "target" in filtered.columns:
        filtered = filtered[filtered["target"] == recipe.target].copy()
    for column, value in recipe.result_filters:
        if column in filtered.columns:
            filtered = filtered[filtered[column].astype(str) == str(value)].copy()
    if include_result_key:
        for key_column in ("variant_id", "candidate_id", "model_id"):
            if key_column in filtered.columns:
                values = filtered[key_column].astype(str)
                mask = (
                    (values == recipe.result_key)
                    | (values == "main_" + recipe.result_key)
                    | values.str.endswith("|" + recipe.result_key)
                )
                filtered = filtered[mask].copy()
                break
    return filtered.reset_index(drop=True)


def _can_accept_relaxed_rows(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    for key_column in ("variant_id", "candidate_id", "model_id"):
        if key_column in frame.columns:
            return frame[key_column].astype(str).nunique(dropna=False) == 1
    return True


def _select_recipe_rows(frame: pd.DataFrame, recipe: RecipeSpec, source_path: Path) -> pd.DataFrame:
    filtered = _apply_recipe_filters(frame, recipe)
    if not filtered.empty:
        return filtered

    relaxed = _apply_recipe_filters(frame, recipe, include_result_key=False)
    if _can_accept_relaxed_rows(relaxed):
        return relaxed.reset_index(drop=True)

    raise KeyError(f"No rows matched recipe {recipe.recipe_id} in {source_path}")


def _maybe_add_recording_key(frame: pd.DataFrame) -> pd.DataFrame:
    if "recording_key" in frame.columns:
        return frame
    if {"study_set", "recording_name"}.issubset(frame.columns):
        enriched = frame.copy()
        enriched["recording_key"] = enriched["study_set"].astype(str) + "::" + enriched["recording_name"].astype(str)
        return enriched
    return frame


def _summarize_predictions(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(group_col, dropna=False):
        true = part["true"].astype(float)
        pred = part["pred"].astype(float)
        err = pred - true
        r2 = float("nan")
        if len(part) >= 2 and true.nunique(dropna=True) > 1:
            r2 = float(r2_score(true, pred))
        rows.append(
            {
                group_col: key,
                "row_count": int(len(part)),
                "mae": float(err.abs().mean()),
                "rmse": float((err.pow(2).mean()) ** 0.5),
                "r2": r2,
                "bias": float(err.mean()),
            }
        )
    return pd.DataFrame(rows)


def _load_recipe_output(recipe: RecipeSpec, output_dir: Path) -> RecipeResult:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = _select_recipe_rows(pd.read_csv(metrics_path), recipe, metrics_path)
    metrics["recipe_id"] = recipe.recipe_id
    metrics["label"] = recipe.label
    metrics["protocol"] = recipe.protocol
    metrics["notes"] = recipe.notes

    predictions = _read_optional_csv(output_dir / "predictions.csv")
    family_summary = _read_optional_csv(output_dir / "per_family_summary.csv")
    recording_summary = _read_optional_csv(output_dir / "per_recording_summary.csv")
    if predictions is not None and not predictions.empty:
        predictions = _select_recipe_rows(predictions, recipe, output_dir / "predictions.csv")
        predictions = _maybe_add_recording_key(predictions)
        predictions["recipe_id"] = recipe.recipe_id
        predictions["label"] = recipe.label
    if family_summary is not None and not family_summary.empty:
        family_summary = _select_recipe_rows(family_summary, recipe, output_dir / "per_family_summary.csv")
        family_summary = family_summary.reset_index(drop=True)
        family_summary["recipe_id"] = recipe.recipe_id
        family_summary["label"] = recipe.label
    elif predictions is not None and not predictions.empty and "study_set" in predictions.columns and {"true", "pred"}.issubset(predictions.columns):
        family_summary = _summarize_predictions(predictions, "study_set")
        family_summary.insert(0, "recipe_id", recipe.recipe_id)
        family_summary.insert(1, "label", recipe.label)
    else:
        family_summary = None
    if recording_summary is not None and not recording_summary.empty:
        recording_summary = _select_recipe_rows(recording_summary, recipe, output_dir / "per_recording_summary.csv")
        recording_summary = recording_summary.reset_index(drop=True)
        recording_summary["recipe_id"] = recipe.recipe_id
        recording_summary["label"] = recipe.label
    elif predictions is not None and not predictions.empty and "recording_key" in predictions.columns and {"true", "pred"}.issubset(predictions.columns):
        recording_summary = _summarize_predictions(predictions, "recording_key")
        recording_summary.insert(0, "recipe_id", recipe.recipe_id)
        recording_summary.insert(1, "label", recipe.label)
    else:
        recording_summary = None

    return RecipeResult(
        recipe=recipe,
        metrics=metrics.reset_index(drop=True),
        predictions=predictions.reset_index(drop=True) if predictions is not None else None,
        per_family_summary=family_summary.reset_index(drop=True) if family_summary is not None else None,
        per_recording_summary=recording_summary.reset_index(drop=True) if recording_summary is not None else None,
        output_dir=output_dir,
    )


def _write_recipe_artifacts(result: RecipeResult) -> None:
    paths = make_run_artifact_paths(
        result.recipe.recipe_id,
        result.recipe.target,
        result.recipe.protocol,
        output_dir=result.output_dir,
    )
    result.metrics.to_csv(paths.metrics_path, index=False)
    if result.predictions is not None:
        result.predictions.to_csv(paths.predictions_path, index=False)
    if result.per_family_summary is not None:
        result.per_family_summary.to_csv(paths.family_summary_path, index=False)
    if result.per_recording_summary is not None:
        result.per_recording_summary.to_csv(paths.recording_summary_path, index=False)
    write_manifest(
        paths.manifest_path,
        {
            "recipe_id": result.recipe.recipe_id,
            "target": result.recipe.target,
            "protocol": result.recipe.protocol,
            "runner_key": result.recipe.runner_key,
            "implementation_module": result.recipe.implementation_module,
            "result_key": result.recipe.result_key,
            "output_dir": str(result.output_dir),
        },
    )


def run_recipe(
    recipe_id: str,
    output_dir: Path | None = None,
    force: bool = False,
    smoke: bool = False,
    parquet_path: Path | None = None,
    random_state: int | None = None,
) -> RecipeResult:
    recipe = get_recipe(recipe_id)
    paths = make_run_artifact_paths(recipe.recipe_id, recipe.target, recipe.protocol, output_dir=output_dir)
    if force or not paths.metrics_path.exists():
        runner = _RUNNERS[recipe.runner_key]
        runner.run(recipe, paths.root, smoke=smoke, parquet_path=parquet_path, random_state=random_state)
    result = _load_recipe_output(recipe, paths.root)
    _write_recipe_artifacts(result)
    return result


def load_recipe_result(recipe_id: str, output_dir: Path | None = None, *, run_if_missing: bool = True) -> RecipeResult:
    recipe = get_recipe(recipe_id)
    paths = make_run_artifact_paths(recipe.recipe_id, recipe.target, recipe.protocol, output_dir=output_dir)
    if not paths.metrics_path.exists():
        if run_if_missing:
            return run_recipe(recipe_id, output_dir=output_dir, force=False)
        raise FileNotFoundError(paths.metrics_path)
    return _load_recipe_output(recipe, paths.root)


def seed_output_dir(recipe_id: str, target: str, protocol: str, seed: int) -> Path:
    run_root = THESIS_ROOT / "artifacts" / ("runs" if int(seed) == 42 else f"runs_seed{int(seed)}")
    return run_root / protocol / target / recipe_id


def aggregate_output_dir(recipe_id: str, target: str, protocol: str) -> Path:
    return THESIS_ROOT / "artifacts" / "runs_aggregate" / protocol / target / recipe_id


def materialize_recipe_output(
    recipe_id: str,
    source_output_dir: Path,
    *,
    dest_output_dir: Path | None = None,
) -> RecipeResult:
    recipe = get_recipe(recipe_id)
    loaded = _load_recipe_output(recipe, source_output_dir)
    paths = make_run_artifact_paths(recipe.recipe_id, recipe.target, recipe.protocol, output_dir=dest_output_dir)
    result = RecipeResult(
        recipe=loaded.recipe,
        metrics=loaded.metrics.copy(),
        predictions=loaded.predictions.copy() if loaded.predictions is not None else None,
        per_family_summary=loaded.per_family_summary.copy() if loaded.per_family_summary is not None else None,
        per_recording_summary=loaded.per_recording_summary.copy() if loaded.per_recording_summary is not None else None,
        output_dir=paths.root,
    )
    _write_recipe_artifacts(result)
    return result


def load_aggregate_result(recipe_id: str) -> RecipeResult:
    recipe = get_recipe(recipe_id)
    output_dir = aggregate_output_dir(recipe.recipe_id, recipe.target, recipe.protocol)
    if not (output_dir / "metrics.csv").exists():
        return load_recipe_result(recipe_id)
    return load_recipe_result(recipe_id, output_dir=output_dir, run_if_missing=False)


def load_experiment(recipe_id: str, target: str, protocol: str, *, aggregate: bool = False) -> RecipeResult:
    recipe = get_recipe(recipe_id)
    if recipe.target != target or recipe.protocol != protocol:
        raise ValueError(f"Recipe mismatch for {recipe_id}: expected {(target, protocol)}, got {(recipe.target, recipe.protocol)}")
    if aggregate:
        return load_aggregate_result(recipe_id)
    return load_recipe_result(recipe_id)


def run_experiment(
    recipe_id: str,
    target: str,
    protocol: str,
    output_dir: Path | None = None,
    force: bool = False,
    *,
    random_state: int | None = None,
) -> RecipeResult:
    recipe = get_recipe(recipe_id)
    if recipe.target != target or recipe.protocol != protocol:
        raise ValueError(f"Recipe mismatch for {recipe_id}: expected {(target, protocol)}, got {(recipe.target, recipe.protocol)}")
    return run_recipe(recipe_id, output_dir=output_dir, force=force, random_state=random_state)


def load_or_run_experiment(
    recipe_id: str,
    target: str,
    protocol: str,
    output_dir: Path | None = None,
    force: bool = False,
    *,
    random_state: int | None = None,
) -> RecipeResult:
    return run_experiment(recipe_id, target, protocol, output_dir=output_dir, force=force, random_state=random_state)


def build_benchmark_table(target: str, recipe_ids: list[str], protocol: str, *, aggregate: bool = False) -> pd.DataFrame:
    rows = [load_experiment(recipe_id, target, protocol, aggregate=aggregate).metrics.iloc[0] for recipe_id in recipe_ids]
    df = pd.DataFrame(rows)
    preferred = ["label", "recipe_id", "mae", "rmse", "r2", "bias", "calibration_slope", "notes", "protocol"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df.loc[:, cols]


def build_ablation_bundle(target: str, ablation_group: str, protocol: str, *, aggregate: bool = False) -> dict[str, pd.DataFrame]:
    recipe_ids = get_recipe_groups(target)[ablation_group]
    table = build_benchmark_table(target, recipe_ids, protocol, aggregate=aggregate)
    families = [build_family_summary(recipe_id, target, protocol, aggregate=aggregate) for recipe_id in recipe_ids]
    recordings = [build_recording_summary(recipe_id, target, protocol, aggregate=aggregate) for recipe_id in recipe_ids]
    return {
        "benchmark": table,
        "per_family": pd.concat([df for df in families if df is not None], ignore_index=True) if any(df is not None for df in families) else pd.DataFrame(),
        "per_recording": pd.concat([df for df in recordings if df is not None], ignore_index=True) if any(df is not None for df in recordings) else pd.DataFrame(),
    }


def build_recording_summary(recipe_id: str, target: str, protocol: str, *, aggregate: bool = False) -> pd.DataFrame | None:
    return load_experiment(recipe_id, target, protocol, aggregate=aggregate).per_recording_summary


def build_family_summary(recipe_id: str, target: str, protocol: str, *, aggregate: bool = False) -> pd.DataFrame | None:
    return load_experiment(recipe_id, target, protocol, aggregate=aggregate).per_family_summary


def get_main_benchmark_table(target: str, *, aggregate: bool = False) -> pd.DataFrame:
    return build_benchmark_table(target, get_main_text_recipe_ids(target), "recording_disjoint_main", aggregate=aggregate)


__all__ = [
    "CORE_BENCHMARK_RECIPE_IDS",
    "FMISS_RECIPE_GROUPS",
    "FPOS_RECIPE_GROUPS",
    "RecipeResult",
    "RecipeSpec",
    "build_ablation_bundle",
    "build_benchmark_table",
    "build_family_summary",
    "build_recipe_inventory",
    "build_recording_summary",
    "get_core_benchmark_recipe_ids",
    "get_main_benchmark_table",
    "get_main_text_recipe_ids",
    "get_recipe",
    "get_recipe_groups",
    "get_recipe_specs",
    "get_recipes",
    "list_recipes",
    "load_experiment",
    "load_or_run_experiment",
    "load_recipe_result",
    "run_experiment",
    "run_recipe",
    "validate_recipe_registry",
    "ExperimentResult",
    "LegacyTransferFamily",
    "SSLFamily",
    "MMDFamily",
    "RUNNER_FAMILIES",
]


# ── absorbed from runners.py ──────────────────────────────────────────────────

def _legacy_feature_view_sets(feature_view: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    mapping = {
        "none": (("none", tuple()),),
        "unit_raw": (("unit_raw", ("unit_raw",)),),
        "shape_only": (("shape_only", ("shape_only",)),),
        "full_context": (
            (
                "full_context",
                ("norm_swap", "recording_relative", "recording_context", "study_identity"),
            ),
        ),
    }
    return mapping.get(feature_view, (("unit_raw", ("unit_raw",)),))


def _legacy_backend_from_result_key(result_key: str) -> tuple[str, ...]:
    if "xgboost" in result_key:
        return ("xgboost",)
    if "lightgbm" in result_key:
        return ("lightgbm",)
    if "catboost" in result_key:
        return ("catboost",)
    return ("lightgbm",)


class LegacyTransferFamily:
    implementation_module = "qc_thesis.modeling.recipes"

    def _build_config(
        self,
        recipe: Any,
        output_dir: Path,
        *,
        smoke: bool,
        parquet_path: Path | None,
        random_state: int | None,
    ) -> Any:
        from .specs import TransferBenchmarkConfig
        from qc_thesis.paths import THESIS_ROOT
        config = TransferBenchmarkConfig(
            parquet_path=parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet"),
            primary_targets=(recipe.target,),
            secondary_targets=(),
            feature_view_sets=_legacy_feature_view_sets(recipe.feature_view),
            model_families=(recipe.model_family,),
            protocol_modes=("inductive",),
            boosting_backends=_legacy_backend_from_result_key(recipe.result_key),
            paired_weight_grid=(25.0,) if "w25" in recipe.result_key else (10.0,),
            include_ceiling_diagnostics=False,
            deploy_best_per_target=False,
            verbose=False,
            neural_verbose=False,
        )
        if random_state is not None:
            config.random_state = int(random_state)
        if smoke:
            config.paired_final_holdout_rows = 24
            config.model_selection_splits = 2
            config.lightgbm_estimators = 24
            config.xgboost_estimators = 24
            config.neural_pretrain_epochs = 2
            config.neural_finetune_epochs = 2
        return config

    def run(
        self,
        recipe: Any,
        output_dir: Path,
        *,
        smoke: bool = False,
        parquet_path: Path | None = None,
        random_state: int | None = None,
    ) -> Path:
        import json
        output_dir.mkdir(parents=True, exist_ok=True)
        benchmark = QCTransferBenchmark(
            self._build_config(
                recipe,
                output_dir,
                smoke=smoke,
                parquet_path=parquet_path,
                random_state=random_state,
            )
        )
        artifacts = benchmark.run()
        metrics_df = artifacts.get("final_holdout_results", pd.DataFrame())
        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        for name, frame in artifacts.items():
            if name == "final_holdout_results":
                continue
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frame.to_csv(output_dir / f"{name}.csv", index=False)
        config_payload = benchmark._config_dict()
        (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
        return output_dir


class SSLFamily:
    implementation_module = "qc_thesis.modeling.recipes"

    def run(
        self,
        recipe: Any,
        output_dir: Path,
        *,
        smoke: bool = False,
        parquet_path: Path | None = None,
        random_state: int | None = None,
    ) -> Path:
        from .specs import SSLDomainAdaptationConfig
        from qc_thesis.paths import THESIS_ROOT
        config = SSLDomainAdaptationConfig(
            parquet_path=parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet"),
            output_root=output_dir.parent,
            run_name=output_dir.name,
            targets=(recipe.target,),
            save_checkpoints=False,
            run_stress_test=False,
            verbose=False,
        )
        if random_state is not None:
            config.random_state = int(random_state)
        if smoke:
            config.paired_final_holdout_rows = 24
            config.ssl_pretrain_epochs = 2
            config.source_warmup_epochs = 1
            config.source_train_epochs = 2
            config.finetune_epochs = 2
            config.batch_size = 64
            config.latent_dim = 16
            config.hidden_dim = 32
            config.context_hidden_dim = 16
        return run_ssl_with_config(config)


class MMDFamily:
    implementation_module = "qc_thesis.modeling.recipes"

    def run(
        self,
        recipe: Any,
        output_dir: Path,
        *,
        smoke: bool = False,
        parquet_path: Path | None = None,
        random_state: int | None = None,
    ) -> Path:
        from .specs import MMDDomainAdaptationConfig
        from qc_thesis.paths import THESIS_ROOT
        config = MMDDomainAdaptationConfig(
            parquet_path=parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet"),
            output_root=output_dir.parent,
            run_name=output_dir.name,
            targets=(recipe.target,),
            save_checkpoints=False,
            verbose=False,
        )
        if random_state is not None:
            config.random_state = int(random_state)
        if smoke:
            config.paired_final_holdout_rows = 24
            config.lambda_grid = (0.0, 0.05)
            config.source_train_epochs = 2
            config.finetune_epochs = 2
            config.batch_size = 64
            config.latent_dim = 16
            config.hidden_dim = 32
        return run_mmd_with_config(config)


RUNNER_FAMILIES = {
    "legacy_transfer": LegacyTransferFamily(),
    "ssl_neural": SSLFamily(),
    "mmd_neural": MMDFamily(),
}

_RUNNERS = {**RUNNER_FAMILIES, **_STACKING_ONLY}

# ── absorbed from experiments.py ─────────────────────────────────────────────
ExperimentResult = RecipeResult
