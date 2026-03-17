from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score

from ...artifacts import make_run_artifact_paths, write_manifest
from ..neural.families import NEURAL_FAMILIES
from ..stacking.recipes import STACKING_FAMILIES
from ..transfer.families import LegacyTransferFamily
from .catalog import RecipeSpec, get_main_text_recipe_ids, get_recipe, get_recipe_groups, get_recipe_specs, get_recipes


@dataclass(frozen=True)
class RecipeResult:
    recipe: RecipeSpec
    metrics: pd.DataFrame
    predictions: pd.DataFrame | None
    per_family_summary: pd.DataFrame | None
    per_recording_summary: pd.DataFrame | None
    output_dir: Path


_RUNNERS = {
    "legacy_transfer": LegacyTransferFamily(),
    **STACKING_FAMILIES,
    **NEURAL_FAMILIES,
}


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
        except Exception as exc:  # pragma: no cover - surfaced via validation script/tests
            errors.append(f"Failed to import {spec.implementation_module} for {spec.recipe_id}: {exc}")
    return errors


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _apply_recipe_filters(frame: pd.DataFrame, recipe: RecipeSpec) -> pd.DataFrame:
    filtered = frame.copy()
    if "target" in filtered.columns:
        filtered = filtered[filtered["target"] == recipe.target].copy()
    for column, value in recipe.result_filters:
        if column in filtered.columns:
            filtered = filtered[filtered[column].astype(str) == str(value)].copy()
    for key_column in ("variant_id", "candidate_id", "model_id"):
        if key_column in filtered.columns:
            filtered = filtered[filtered[key_column].astype(str) == recipe.result_key].copy()
            break
    return filtered.reset_index(drop=True)


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
    metrics = _apply_recipe_filters(pd.read_csv(metrics_path), recipe)
    if metrics.empty:
        raise KeyError(f"No metrics row matched recipe {recipe.recipe_id} in {metrics_path}")
    metrics["recipe_id"] = recipe.recipe_id
    metrics["label"] = recipe.label
    metrics["protocol"] = recipe.protocol
    metrics["notes"] = recipe.notes

    predictions = _read_optional_csv(output_dir / "predictions.csv")
    family_summary = None
    recording_summary = None
    if predictions is not None and not predictions.empty:
        predictions = _apply_recipe_filters(predictions, recipe)
        predictions = _maybe_add_recording_key(predictions)
        predictions["recipe_id"] = recipe.recipe_id
        predictions["label"] = recipe.label
        if "study_set" in predictions.columns and {"true", "pred"}.issubset(predictions.columns):
            family_summary = _summarize_predictions(predictions, "study_set")
            family_summary.insert(0, "recipe_id", recipe.recipe_id)
            family_summary.insert(1, "label", recipe.label)
        if "recording_key" in predictions.columns and {"true", "pred"}.issubset(predictions.columns):
            recording_summary = _summarize_predictions(predictions, "recording_key")
            recording_summary.insert(0, "recipe_id", recipe.recipe_id)
            recording_summary.insert(1, "label", recipe.label)

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
) -> RecipeResult:
    recipe = get_recipe(recipe_id)
    paths = make_run_artifact_paths(recipe.recipe_id, recipe.target, recipe.protocol, output_dir=output_dir)
    if force or not paths.metrics_path.exists():
        runner = _RUNNERS[recipe.runner_key]
        runner.run(recipe, paths.root, smoke=smoke, parquet_path=parquet_path)
    result = _load_recipe_output(recipe, paths.root)
    _write_recipe_artifacts(result)
    return result


def load_recipe_result(recipe_id: str, output_dir: Path | None = None) -> RecipeResult:
    recipe = get_recipe(recipe_id)
    paths = make_run_artifact_paths(recipe.recipe_id, recipe.target, recipe.protocol, output_dir=output_dir)
    if not paths.metrics_path.exists():
        return run_recipe(recipe_id, output_dir=output_dir, force=False)
    return _load_recipe_output(recipe, paths.root)


def load_experiment(recipe_id: str, target: str, protocol: str) -> RecipeResult:
    recipe = get_recipe(recipe_id)
    if recipe.target != target or recipe.protocol != protocol:
        raise ValueError(f"Recipe mismatch for {recipe_id}: expected {(target, protocol)}, got {(recipe.target, recipe.protocol)}")
    return load_recipe_result(recipe_id)


def run_experiment(recipe_id: str, target: str, protocol: str, output_dir: Path | None = None, force: bool = False) -> RecipeResult:
    recipe = get_recipe(recipe_id)
    if recipe.target != target or recipe.protocol != protocol:
        raise ValueError(f"Recipe mismatch for {recipe_id}: expected {(target, protocol)}, got {(recipe.target, recipe.protocol)}")
    return run_recipe(recipe_id, output_dir=output_dir, force=force)


def load_or_run_experiment(recipe_id: str, target: str, protocol: str, output_dir: Path | None = None, force: bool = False) -> RecipeResult:
    return run_experiment(recipe_id, target, protocol, output_dir=output_dir, force=force)


def build_benchmark_table(target: str, recipe_ids: list[str], protocol: str) -> pd.DataFrame:
    rows = [load_experiment(recipe_id, target, protocol).metrics.iloc[0] for recipe_id in recipe_ids]
    df = pd.DataFrame(rows)
    preferred = ["label", "recipe_id", "mae", "rmse", "r2", "bias", "calibration_slope", "notes", "protocol"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df.loc[:, cols]


def build_ablation_bundle(target: str, ablation_group: str, protocol: str) -> dict[str, pd.DataFrame]:
    recipe_ids = get_recipe_groups(target)[ablation_group]
    table = build_benchmark_table(target, recipe_ids, protocol)
    families = [build_family_summary(recipe_id, target, protocol) for recipe_id in recipe_ids]
    recordings = [build_recording_summary(recipe_id, target, protocol) for recipe_id in recipe_ids]
    return {
        "benchmark": table,
        "per_family": pd.concat([df for df in families if df is not None], ignore_index=True) if any(df is not None for df in families) else pd.DataFrame(),
        "per_recording": pd.concat([df for df in recordings if df is not None], ignore_index=True) if any(df is not None for df in recordings) else pd.DataFrame(),
    }


def build_recording_summary(recipe_id: str, target: str, protocol: str) -> pd.DataFrame | None:
    return load_experiment(recipe_id, target, protocol).per_recording_summary


def build_family_summary(recipe_id: str, target: str, protocol: str) -> pd.DataFrame | None:
    return load_experiment(recipe_id, target, protocol).per_family_summary


def get_main_benchmark_table(target: str) -> pd.DataFrame:
    return build_benchmark_table(target, get_main_text_recipe_ids(target), "recording_disjoint_main")
