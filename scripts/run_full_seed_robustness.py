from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from qc_thesis.paths import THESIS_ROOT
from qc_thesis.artifacts import write_manifest
from qc_thesis.modeling.neural import prepare_feature_table, run_mmd_with_config, run_ssl_with_config
from qc_thesis.modeling.recipes import (
    DEFAULT_SWEEP_SEEDS,
    aggregate_output_dir,
    get_recipe,
    get_recipe_specs,
    load_recipe_result,
    materialize_recipe_output,
    run_recipe,
    seed_output_dir,
)
from qc_thesis.modeling.shared import build_stress_splits
from qc_thesis.modeling.specs import MMDDomainAdaptationConfig, SSLDomainAdaptationConfig, TransferBenchmarkConfig
from qc_thesis.modeling.stacking.augmenters import fit_signal_embeddings
from qc_thesis.modeling.stacking.base_stack_runner import build_base_stack_state_from_split, build_embedding_universe
from qc_thesis.modeling.stacking.recipes import (
    FPosSignalEmbeddingStackConfig,
    FPosFamilyRobustnessConfig,
    MLStatisticsWorkbenchConfig,
    run_context_stack,
    run_robustness_stack,
)
from qc_thesis.modeling.stacking.stack_recipe_runner import StackVariantSpec, run_stack_recipe
from qc_thesis.modeling.transfer import QCTransferBenchmark


def log(message: str) -> None:
    print(message, flush=True)


def parse_seed_spec(raw: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            seeds.extend(list(range(start, end + step, step)))
        else:
            seeds.append(int(token))
    ordered = sorted(dict.fromkeys(seeds))
    if not ordered:
        raise ValueError("No seeds parsed from seed specification")
    return tuple(ordered)


def run_root_for_seed(seed: int) -> Path:
    return THESIS_ROOT / "artifacts" / ("runs" if int(seed) == 42 else f"runs_seed{int(seed)}")


def shared_output_dir(seed: int, name: str) -> Path:
    return run_root_for_seed(seed) / "_shared" / name


def recipe_exists(recipe_id: str, seed: int) -> bool:
    recipe = get_recipe(recipe_id)
    return (seed_output_dir(recipe.recipe_id, recipe.target, recipe.protocol, seed) / "metrics.csv").exists()


def all_recipes_exist(recipe_ids: list[str], seed: int) -> bool:
    return all(recipe_exists(recipe_id, seed) for recipe_id in recipe_ids)


def save_transfer_outputs(output_dir: Path, artifacts: dict[str, Any], config_payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = artifacts.get("final_holdout_results", pd.DataFrame())
    if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    for name, frame in artifacts.items():
        if name == "final_holdout_results":
            continue
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")


def _legacy_feature_view_set(feature_view: str) -> tuple[str, tuple[str, ...]]:
    mapping = {
        "none": ("none", tuple()),
        "unit_raw": ("unit_raw", ("unit_raw",)),
        "shape_only": ("shape_only", ("shape_only",)),
        "full_context": (
            "full_context",
            ("norm_swap", "recording_relative", "recording_context", "study_identity"),
        ),
    }
    if feature_view not in mapping:
        raise KeyError(f"Unsupported grouped legacy feature view: {feature_view}")
    return mapping[feature_view]


def _legacy_group_config(recipe_ids: list[str], seed: int) -> TransferBenchmarkConfig:
    specs = [get_recipe(recipe_id) for recipe_id in recipe_ids]
    targets = sorted({spec.target for spec in specs})
    if len(targets) != 1:
        raise ValueError(f"Legacy group must share exactly one target: {recipe_ids}")
    feature_view_sets = tuple(
        dict.fromkeys(_legacy_feature_view_set(spec.feature_view) for spec in specs)
    )
    model_families = tuple(dict.fromkeys(spec.model_family for spec in specs))
    backends = []
    for spec in specs:
        if "xgboost" in spec.result_key:
            backends.append("xgboost")
        if "lightgbm" in spec.result_key:
            backends.append("lightgbm")
        if "catboost" in spec.result_key:
            backends.append("catboost")
    boosting_backends = tuple(dict.fromkeys(backends)) or ("lightgbm",)
    paired_weights = [10.0]
    if any("w25" in spec.result_key for spec in specs):
        paired_weights.append(25.0)
    return TransferBenchmarkConfig(
        parquet_path=THESIS_ROOT / "data/raw/33000_ROWS.parquet",
        primary_targets=(targets[0],),
        secondary_targets=(),
        feature_view_sets=feature_view_sets,
        model_families=model_families,
        protocol_modes=("inductive",),
        boosting_backends=boosting_backends,
        paired_weight_grid=tuple(dict.fromkeys(paired_weights)),
        include_ceiling_diagnostics=False,
        deploy_best_per_target=False,
        verbose=False,
        neural_verbose=False,
        random_state=int(seed),
    )


def _write_group_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(path, payload)


def run_individual_recipe(seed: int, recipe_id: str, *, force: bool) -> dict[str, Any]:
    recipe = get_recipe(recipe_id)
    start = time.perf_counter()
    dest_dir = seed_output_dir(recipe.recipe_id, recipe.target, recipe.protocol, seed)
    if not force and (dest_dir / "metrics.csv").exists():
        log(f"[seed {seed}] {recipe.recipe_id}: skip existing")
        return {"seed": seed, "group": recipe.recipe_id, "status": "skipped_existing", "elapsed_seconds": 0.0}
    log(f"[seed {seed}] {recipe.recipe_id}: run")
    run_recipe(
        recipe.recipe_id,
        output_dir=dest_dir,
        force=force,
        random_state=int(seed),
    )
    elapsed = time.perf_counter() - start
    return {"seed": seed, "group": recipe.recipe_id, "status": "ran", "elapsed_seconds": round(elapsed, 2), "output_dir": str(dest_dir)}


def run_grouped_context(seed: int, recipe_ids: list[str], *, force: bool) -> dict[str, Any]:
    start = time.perf_counter()
    if not force and all_recipes_exist(recipe_ids, seed):
        log(f"[seed {seed}] context_stack: skip existing")
        return {"seed": seed, "group": "context_stack", "status": "skipped_existing", "elapsed_seconds": 0.0}
    output_dir = shared_output_dir(seed, "context_stack")
    log(f"[seed {seed}] context_stack: run shared family")
    config = MLStatisticsWorkbenchConfig(
        parquet_path=THESIS_ROOT / "data/raw/33000_ROWS.parquet",
        output_root=output_dir.parent,
        run_name=output_dir.name,
        targets=tuple(sorted({get_recipe(recipe_id).target for recipe_id in recipe_ids})),
        save_checkpoints=False,
        verbose=False,
        random_state=int(seed),
    )
    run_context_stack(config)
    for recipe_id in recipe_ids:
        recipe = get_recipe(recipe_id)
        materialize_recipe_output(
            recipe_id,
            output_dir,
            dest_output_dir=seed_output_dir(recipe.recipe_id, recipe.target, recipe.protocol, seed),
        )
    elapsed = time.perf_counter() - start
    return {"seed": seed, "group": "context_stack", "status": "ran", "elapsed_seconds": round(elapsed, 2), "output_dir": str(output_dir)}


def run_grouped_ssl(seed: int, recipe_ids: list[str], *, force: bool) -> dict[str, Any]:
    start = time.perf_counter()
    if not force and all_recipes_exist(recipe_ids, seed):
        log(f"[seed {seed}] ssl_neural: skip existing")
        return {"seed": seed, "group": "ssl_neural", "status": "skipped_existing", "elapsed_seconds": 0.0}
    output_dir = shared_output_dir(seed, "ssl_neural")
    log(f"[seed {seed}] ssl_neural: run shared family")
    config = SSLDomainAdaptationConfig(
        parquet_path=THESIS_ROOT / "data/raw/33000_ROWS.parquet",
        output_root=output_dir.parent,
        run_name=output_dir.name,
        targets=tuple(sorted({get_recipe(recipe_id).target for recipe_id in recipe_ids})),
        save_checkpoints=False,
        run_stress_test=False,
        verbose=False,
        random_state=int(seed),
    )
    run_ssl_with_config(config)
    for recipe_id in recipe_ids:
        recipe = get_recipe(recipe_id)
        materialize_recipe_output(
            recipe_id,
            output_dir,
            dest_output_dir=seed_output_dir(recipe.recipe_id, recipe.target, recipe.protocol, seed),
        )
    elapsed = time.perf_counter() - start
    return {"seed": seed, "group": "ssl_neural", "status": "ran", "elapsed_seconds": round(elapsed, 2), "output_dir": str(output_dir)}


def run_grouped_mmd(seed: int, recipe_ids: list[str], *, force: bool) -> dict[str, Any]:
    start = time.perf_counter()
    if not force and all_recipes_exist(recipe_ids, seed):
        log(f"[seed {seed}] mmd_neural: skip existing")
        return {"seed": seed, "group": "mmd_neural", "status": "skipped_existing", "elapsed_seconds": 0.0}
    output_dir = shared_output_dir(seed, "mmd_neural")
    log(f"[seed {seed}] mmd_neural: run shared family")
    config = MMDDomainAdaptationConfig(
        parquet_path=THESIS_ROOT / "data/raw/33000_ROWS.parquet",
        output_root=output_dir.parent,
        run_name=output_dir.name,
        targets=tuple(sorted({get_recipe(recipe_id).target for recipe_id in recipe_ids})),
        save_checkpoints=False,
        verbose=False,
        random_state=int(seed),
    )
    run_mmd_with_config(config)
    for recipe_id in recipe_ids:
        recipe = get_recipe(recipe_id)
        materialize_recipe_output(
            recipe_id,
            output_dir,
            dest_output_dir=seed_output_dir(recipe.recipe_id, recipe.target, recipe.protocol, seed),
        )
    elapsed = time.perf_counter() - start
    return {"seed": seed, "group": "mmd_neural", "status": "ran", "elapsed_seconds": round(elapsed, 2), "output_dir": str(output_dir)}


def run_grouped_robustness(seed: int, recipe_ids: list[str], *, force: bool) -> dict[str, Any]:
    start = time.perf_counter()
    if not force and all_recipes_exist(recipe_ids, seed):
        log(f"[seed {seed}] robustness_stack: skip existing")
        return {"seed": seed, "group": "robustness_stack", "status": "skipped_existing", "elapsed_seconds": 0.0}
    output_dir = shared_output_dir(seed, "robustness_stack")
    log(f"[seed {seed}] robustness_stack: run shared family")
    config = FPosFamilyRobustnessConfig(
        parquet_path=THESIS_ROOT / "data/raw/33000_ROWS.parquet",
        output_root=output_dir.parent,
        run_name=output_dir.name,
        verbose=False,
        random_state=int(seed),
    )
    run_robustness_stack(config)
    for recipe_id in recipe_ids:
        recipe = get_recipe(recipe_id)
        materialize_recipe_output(
            recipe_id,
            output_dir,
            dest_output_dir=seed_output_dir(recipe.recipe_id, recipe.target, recipe.protocol, seed),
        )
    elapsed = time.perf_counter() - start
    return {"seed": seed, "group": "robustness_stack", "status": "ran", "elapsed_seconds": round(elapsed, 2), "output_dir": str(output_dir)}


def run_main_seed_sweep(seeds: tuple[int, ...], *, force: bool) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    individual: list[str] = []
    for spec in get_recipe_specs():
        if spec.runner_key == "legacy_transfer":
            individual.append(spec.recipe_id)
        elif spec.runner_key == "context_stack":
            grouped.setdefault("context_stack", []).append(spec.recipe_id)
        elif spec.runner_key == "ssl_neural":
            grouped.setdefault("ssl_neural", []).append(spec.recipe_id)
        elif spec.runner_key == "mmd_neural":
            grouped.setdefault("mmd_neural", []).append(spec.recipe_id)
        elif spec.runner_key == "robustness_stack":
            grouped.setdefault("robustness_stack", []).append(spec.recipe_id)
    status_rows: list[dict[str, Any]] = []
    for seed in seeds:
        log(f"[seed {seed}] main sweep start")
        for recipe_id in individual:
            status_rows.append(run_individual_recipe(seed, recipe_id, force=force))
        for group_name, recipe_ids in grouped.items():
            if group_name == "context_stack":
                status_rows.append(run_grouped_context(seed, recipe_ids, force=force))
            elif group_name == "ssl_neural":
                status_rows.append(run_grouped_ssl(seed, recipe_ids, force=force))
            elif group_name == "mmd_neural":
                status_rows.append(run_grouped_mmd(seed, recipe_ids, force=force))
            elif group_name == "robustness_stack":
                status_rows.append(run_grouped_robustness(seed, recipe_ids, force=force))
    manifest_path = THESIS_ROOT / "artifacts" / "seed_sweep_full.json"
    _write_group_manifest(
        manifest_path,
        {
            "script": str(Path(__file__).resolve()),
            "seeds": list(seeds),
            "individual_recipes": individual,
            "groups": sorted(grouped),
            "results": status_rows,
        },
    )
    return status_rows


def run_family_holdout_seed(seed: int, *, force: bool) -> list[dict[str, Any]]:
    log(f"[seed {seed}] family holdout start")
    prepared = prepare_feature_table(THESIS_ROOT / "data/raw/33000_ROWS.parquet")
    config = FPosSignalEmbeddingStackConfig(
        parquet_path=THESIS_ROOT / "data/raw/33000_ROWS.parquet",
        output_root=run_root_for_seed(seed) / "family_held_out" / "fpos",
        run_name="family_holdout_stress",
        verbose=False,
        random_state=int(seed),
    )
    rows: list[dict[str, Any]] = []
    stress_root = config.output_root / config.run_name
    stress_root.mkdir(parents=True, exist_ok=True)
    for split in build_stress_splits(prepared.df):
        output_dir = stress_root / split.held_out_study.lower()
        if not force and (output_dir / "metrics.csv").exists():
            log(f"[seed {seed}] family holdout {split.held_out_study}: skip existing")
            rows.append({"seed": seed, "held_out_family": split.held_out_study, "status": "skipped_existing", "output_dir": str(output_dir)})
            continue
        log(f"[seed {seed}] family holdout {split.held_out_study}: run")
        output_dir.mkdir(parents=True, exist_ok=True)
        state = build_base_stack_state_from_split(
            target="fpos",
            prepared=prepared,
            split=split,
            config=config,
        )
        universe_df = build_embedding_universe(
            split=state.split,
            max_rows=config.universe_max_rows,
            random_state=int(seed),
        )
        wf_bundle = fit_signal_embeddings(
            train_signal=state.target_train_df.loc[:, state.prepared.wf_feature_cols],
            train_universe_signal=universe_df.loc[:, state.prepared.wf_feature_cols],
            unlabeled_signal=state.unlabeled_df.loc[:, state.prepared.wf_feature_cols],
            test_signal=state.target_test_df.loc[:, state.prepared.wf_feature_cols],
            prefix="wf",
            config=config,
        )
        variants = [
            StackVariantSpec(
                variant_id="reference_anchor_stack_xgboost",
                stack=state.base_stack,
                notes="family-held-out anchor reference",
                backend_id="xgboost_default",
                family_alpha=0.0,
            ),
            StackVariantSpec(
                variant_id="family_balanced_strong_xgboost",
                stack=state.base_stack,
                notes="family-held-out family-balanced robustness variant",
                backend_id="xgboost_default",
                family_alpha=1.0,
            ),
            StackVariantSpec(
                variant_id="wf_embed_anchor_stack_xgboost",
                stack=state.base_stack.apply(wf_bundle.as_augmenter()),
                notes="family-held-out waveform-augmented structure variant",
                backend_id="xgboost_default",
                family_alpha=1.0,
            ),
        ]
        run_stack_recipe(
            config=config,
            output_dir=output_dir,
            state=state,
            target="fpos",
            family="family_holdout_stress",
            source_target="fpos",
            variants=variants,
            extra_outputs={
                "embedding_history.csv": pd.DataFrame({"wf": wf_bundle.history}),
            },
        )
        metrics_df = pd.read_csv(output_dir / "metrics.csv")
        cv_df = pd.read_csv(output_dir / "variant_cv_summary.csv")
        split_manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
        metrics_df["held_out_family"] = split.held_out_study
        metrics_df["n_test_rows"] = int(split_manifest.get("paired_test_rows", len(state.target_test_df)))
        metrics_df["n_test_recordings"] = int(split_manifest.get("paired_test_recordings", state.target_test_df["recording_key"].nunique()))
        metrics_df = metrics_df.merge(
            cv_df.loc[:, ["variant_id", "selected_rows"]].rename(columns={"selected_rows": "pseudo_rows"}),
            on="variant_id",
            how="left",
        )
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        preds_df = pd.read_csv(output_dir / "predictions.csv")
        preds_df["held_out_family"] = split.held_out_study
        preds_df.to_csv(output_dir / "predictions.csv", index=False)
        rows.append({"seed": seed, "held_out_family": split.held_out_study, "status": "ran", "output_dir": str(output_dir)})
    return rows


def run_family_holdout_sweep(seeds: tuple[int, ...], *, force: bool) -> list[dict[str, Any]]:
    status_rows: list[dict[str, Any]] = []
    for seed in seeds:
        status_rows.extend(run_family_holdout_seed(seed, force=force))
    manifest_path = THESIS_ROOT / "artifacts" / "family_holdout_seed_sweep.json"
    _write_group_manifest(
        manifest_path,
        {
            "script": str(Path(__file__).resolve()),
            "seeds": list(seeds),
            "results": status_rows,
        },
    )
    return status_rows


def aggregate_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    base = df.iloc[0].copy()
    if "seed" in base.index:
        base = base.drop(labels=["seed"])
    numeric_cols = [col for col in df.columns if col != "seed" and pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_cols:
        base[col] = float(df[col].mean())
        base[f"{col}_std"] = float(df[col].std(ddof=0)) if len(df) > 1 else 0.0
        base[f"{col}_min"] = float(df[col].min())
        base[f"{col}_max"] = float(df[col].max())
    base["seed_count"] = int(len(df))
    base["seed_values"] = ",".join(str(int(seed)) for seed in sorted(df["seed"].astype(int).unique().tolist()))
    return pd.DataFrame([base])


def aggregate_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [col for col in df.columns if col != "seed" and pd.api.types.is_numeric_dtype(df[col])]
    group_cols = [col for col in df.columns if col not in set(numeric_cols) | {"seed"}]
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    mean_df = grouped[numeric_cols].mean().reset_index()
    mean_df["seed_count"] = grouped.size().to_numpy()
    for col in numeric_cols:
        mean_df[f"{col}_std"] = grouped[col].std(ddof=0).fillna(0.0).to_numpy()
    return mean_df


def aggregate_main_seed_results(seeds: tuple[int, ...]) -> None:
    log("[aggregate] recording_disjoint aggregates start")
    for spec in get_recipe_specs():
        metric_rows: list[pd.Series] = []
        prediction_frames: list[pd.DataFrame] = []
        family_frames: list[pd.DataFrame] = []
        recording_frames: list[pd.DataFrame] = []
        for seed in seeds:
            output_dir = seed_output_dir(spec.recipe_id, spec.target, spec.protocol, seed)
            metrics_path = output_dir / "metrics.csv"
            if not metrics_path.exists():
                continue
            result = load_recipe_result(spec.recipe_id, output_dir=output_dir, run_if_missing=False)
            row = result.metrics.iloc[0].copy()
            row["seed"] = int(seed)
            metric_rows.append(row)
            if result.predictions is not None and not result.predictions.empty:
                frame = result.predictions.copy()
                frame["seed"] = int(seed)
                prediction_frames.append(frame)
            if result.per_family_summary is not None and not result.per_family_summary.empty:
                frame = result.per_family_summary.copy()
                frame["seed"] = int(seed)
                family_frames.append(frame)
            if result.per_recording_summary is not None and not result.per_recording_summary.empty:
                frame = result.per_recording_summary.copy()
                frame["seed"] = int(seed)
                recording_frames.append(frame)
        if not metric_rows:
            continue
        output_dir = aggregate_output_dir(spec.recipe_id, spec.target, spec.protocol)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_df = pd.DataFrame(metric_rows)
        aggregate_numeric_frame(metrics_df).to_csv(output_dir / "metrics.csv", index=False)
        metrics_df.sort_values("seed").to_csv(output_dir / "seed_metrics.csv", index=False)
        if prediction_frames:
            pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
        if family_frames:
            aggregate_group_summary(pd.concat(family_frames, ignore_index=True)).to_csv(output_dir / "per_family_summary.csv", index=False)
        if recording_frames:
            aggregate_group_summary(pd.concat(recording_frames, ignore_index=True)).to_csv(output_dir / "per_recording_summary.csv", index=False)
        _write_group_manifest(
            output_dir / "aggregate_manifest.json",
            {
                "recipe_id": spec.recipe_id,
                "target": spec.target,
                "protocol": spec.protocol,
                "seeds": list(seeds),
            },
        )


def aggregate_family_holdout_results(seeds: tuple[int, ...]) -> None:
    log("[aggregate] family_holdout aggregates start")
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in seeds:
        stress_root = run_root_for_seed(seed) / "family_held_out" / "fpos" / "family_holdout_stress"
        if not stress_root.exists():
            continue
        for metrics_path in sorted(stress_root.glob("*/metrics.csv")):
            metrics_df = pd.read_csv(metrics_path)
            metrics_df["seed"] = int(seed)
            metric_frames.append(metrics_df)
            predictions_path = metrics_path.with_name("predictions.csv")
            if predictions_path.exists():
                preds_df = pd.read_csv(predictions_path)
                preds_df["seed"] = int(seed)
                prediction_frames.append(preds_df)
    if not metric_frames:
        return
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    by_family = aggregate_group_summary(all_metrics)
    aggregate_rows = []
    for variant_id, part in by_family.groupby("variant_id", dropna=False):
        raw_weight = part["n_test_rows"].astype(float).to_numpy(dtype=float)
        if raw_weight.sum() > 0:
            weight = raw_weight / raw_weight.sum()
        else:
            weight = raw_weight
        aggregate_rows.append(
            {
                "variant_id": str(variant_id),
                "macro_mae": float(part["mae"].mean()),
                "macro_r2": float(part["r2"].mean()),
                "weighted_mae": float((part["mae"].to_numpy(dtype=float) * weight).sum()),
                "weighted_r2": float((part["r2"].to_numpy(dtype=float) * weight).sum()),
                "worst_family_r2": float(part["r2"].min()),
                "best_family_r2": float(part["r2"].max()),
                "family_count": int(part["held_out_family"].nunique()),
                "seed_count": int(all_metrics.loc[all_metrics["variant_id"].astype(str) == str(variant_id), "seed"].nunique()),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(["macro_r2", "macro_mae"], ascending=[False, True]).reset_index(drop=True)
    output_dir = THESIS_ROOT / "artifacts" / "aggregates" / "family_held_out" / "fpos"
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_df.to_csv(output_dir / "leave_one_family_aggregate.csv", index=False)
    by_family.sort_values(["held_out_family", "variant_id"]).to_csv(output_dir / "leave_one_family_by_family.csv", index=False)
    all_metrics.sort_values(["seed", "held_out_family", "variant_id"]).to_csv(output_dir / "leave_one_family_seed_rows.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "leave_one_family_predictions.csv", index=False)
    _write_group_manifest(
        output_dir / "aggregate_manifest.json",
        {
            "seeds": list(seeds),
            "variants": sorted(aggregate_df["variant_id"].astype(str).unique().tolist()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and aggregate the full 20-seed robustness benchmark.")
    parser.add_argument("--seeds", default=f"{DEFAULT_SWEEP_SEEDS[0]}-{DEFAULT_SWEEP_SEEDS[-1]}")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-family", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--skip-notebooks", action="store_true")
    args = parser.parse_args()

    seeds = parse_seed_spec(args.seeds)
    log(f"[run] seeds={list(seeds)} force={args.force}")

    if not args.skip_main:
        run_main_seed_sweep(seeds, force=args.force)
    if not args.skip_family:
        run_family_holdout_sweep(seeds, force=args.force)
    if not args.skip_aggregate:
        aggregate_main_seed_results(seeds)
        aggregate_family_holdout_results(seeds)
    if not args.skip_notebooks:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(THESIS_ROOT / "src")
        subprocess.run(
            ["python", str(THESIS_ROOT / "scripts" / "rebuild_story_notebooks.py")],
            check=True,
            cwd=str(THESIS_ROOT),
            env=env,
        )
        subprocess.run(
            ["python", str(THESIS_ROOT / "scripts" / "execute_notebooks.py"), "--timeout", "1200"],
            check=True,
            cwd=str(THESIS_ROOT),
            env=env,
        )


if __name__ == "__main__":
    main()
