from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qc_thesis.paths import THESIS_ROOT
from qc_thesis.modeling.neural import prepare_feature_table
from qc_thesis.modeling.shared import build_stress_splits
from qc_thesis.modeling.stacking.augmenters import fit_signal_embeddings
from qc_thesis.modeling.stacking.backends import fit_tree_regressor, predict_tree_regressor, score_predictions
from qc_thesis.modeling.stacking.base_stack_runner import (
    assemble_training_data,
    build_base_stack_state_from_split,
    build_embedding_universe,
)
from qc_thesis.modeling.stacking.recipes import FPosSignalEmbeddingStackConfig


DEFAULT_SEEDS = tuple(range(42, 62))
TARGET = "fpos"
VARIANT_ID = "wf_embed_anchor_stack_xgboost"
PROTOCOL = "leave_one_family_out"
STACK_RANDOM_STATE_SEED = 2200
WAVEFORM_VARIANT_POSITION = 3
FINAL_FIT_RANDOM_STATE_OFFSET = STACK_RANDOM_STATE_SEED + (100 * WAVEFORM_VARIANT_POSITION) + 100


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
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(token))
    ordered = tuple(sorted(dict.fromkeys(seeds)))
    if not ordered:
        raise ValueError("No seeds parsed from seed specification")
    return ordered


def _feature_group(feature: str) -> str:
    if feature.startswith("wf_"):
        return "waveform"
    if feature.startswith("acg_"):
        return "acg"
    if "_recording_" in feature:
        return "recording_context"
    if feature.startswith("amp_"):
        return "amplitude"
    if feature.startswith("si_") or feature.startswith("isi_"):
        return "spikeinterface_qc"
    if feature in {"source_pred", "anchor_fp_score", "matched_support_conf", "anchor_support_margin"}:
        return "source_transfer"
    return "other"


def _safe_std(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=0))


def _compute_mean_abs_shap(
    *,
    fitted_model,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    import shap

    X_imp = fitted_model.imputer.transform(feature_frame)
    explainer = shap.TreeExplainer(fitted_model.model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_imp)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 1:
        return np.abs(arr)
    return np.abs(arr).mean(axis=0)


def _detail_output_dir() -> Path:
    return THESIS_ROOT / "artifacts" / "aggregates" / "family_held_out" / "fpos"


def _detail_path() -> Path:
    return _detail_output_dir() / "fpos_waveform_lofo_shap_by_seed_family.csv"


def _run_summary_path() -> Path:
    return _detail_output_dir() / "fpos_waveform_lofo_shap_run_summary.csv"


def _aggregate_feature_path() -> Path:
    return _detail_output_dir() / "fpos_waveform_lofo_shap_aggregate.csv"


def _aggregate_group_path() -> Path:
    return _detail_output_dir() / "fpos_waveform_lofo_shap_group_aggregate.csv"


def _aggregate_feature_by_family_path() -> Path:
    return _detail_output_dir() / "fpos_waveform_lofo_shap_by_family_aggregate.csv"


def _aggregate_group_by_family_path() -> Path:
    return _detail_output_dir() / "fpos_waveform_lofo_shap_group_by_family_aggregate.csv"


def _manifest_path() -> Path:
    return _detail_output_dir() / "fpos_waveform_lofo_shap_manifest.json"


def _completed_pairs(detail_df: pd.DataFrame) -> set[tuple[int, str]]:
    if detail_df.empty:
        return set()
    pairs = detail_df.loc[:, ["seed", "held_out_family"]].drop_duplicates()
    return {(int(row.seed), str(row.held_out_family)) for row in pairs.itertuples(index=False)}


def _aggregate_outputs(detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if detail_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    feature_agg = (
        detail_df.groupby(["feature", "feature_group"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", _safe_std),
            run_count=("mean_abs_shap", "size"),
            mean_rank=("rank", "mean"),
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    feature_agg["rank"] = feature_agg["mean_abs_shap"].rank(ascending=False, method="min").astype(int)

    family_feature_agg = (
        detail_df.groupby(["held_out_family", "feature", "feature_group"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", _safe_std),
            seed_count=("mean_abs_shap", "size"),
            mean_rank=("rank", "mean"),
        )
        .sort_values(["held_out_family", "mean_abs_shap"], ascending=[True, False])
        .reset_index(drop=True)
    )

    group_runs = (
        detail_df.groupby(["seed", "held_out_family", "feature_group"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "sum"),
            feature_count=("feature", "size"),
        )
    )
    group_agg = (
        group_runs.groupby("feature_group", as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", _safe_std),
            run_count=("mean_abs_shap", "size"),
            mean_feature_count=("feature_count", "mean"),
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    group_agg["rank"] = group_agg["mean_abs_shap"].rank(ascending=False, method="min").astype(int)

    family_group_agg = (
        group_runs.groupby(["held_out_family", "feature_group"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", _safe_std),
            seed_count=("mean_abs_shap", "size"),
            mean_feature_count=("feature_count", "mean"),
        )
        .sort_values(["held_out_family", "mean_abs_shap"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return feature_agg, group_agg, family_feature_agg, family_group_agg


def _append_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    header = not path.exists()
    frame.to_csv(path, mode="a", header=header, index=False)


def run_export(*, seeds: tuple[int, ...], resume: bool) -> dict[str, object]:
    output_dir = _detail_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = _detail_path()
    run_summary_path = _run_summary_path()
    if not resume:
        for path in (
            detail_path,
            run_summary_path,
            _aggregate_feature_path(),
            _aggregate_group_path(),
            _aggregate_feature_by_family_path(),
            _aggregate_group_by_family_path(),
            _manifest_path(),
        ):
            if path.exists():
                path.unlink()

    existing_detail = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    completed_pairs = _completed_pairs(existing_detail)

    prepared = prepare_feature_table(THESIS_ROOT / "data/raw/33000_ROWS.parquet")
    splits = build_stress_splits(prepared.df)
    held_out_families = [split.held_out_study for split in splits]

    detail_rows_written = 0
    run_rows_written = 0
    for seed in seeds:
        config = FPosSignalEmbeddingStackConfig(
            parquet_path=THESIS_ROOT / "data/raw/33000_ROWS.parquet",
            output_root=output_dir,
            run_name="fpos_waveform_lofo_shap",
            verbose=False,
            random_state=int(seed),
        )
        for split in splits:
            pair = (int(seed), str(split.held_out_study))
            if pair in completed_pairs:
                print(f"[skip] seed {seed} | held_out_family {split.held_out_study}", flush=True)
                continue

            print(f"[run] seed {seed} | held_out_family {split.held_out_study}", flush=True)
            state = build_base_stack_state_from_split(
                target=TARGET,
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
            waveform_stack = state.base_stack.apply(wf_bundle.as_augmenter())
            full_train, full_y, full_weight = assemble_training_data(
                labeled_frame=waveform_stack.train,
                labeled_y=state.y_train,
                labeled_meta_df=state.target_train_df,
                pseudo_selected=state.trust_selected,
                unlabeled_frame=waveform_stack.unlabeled,
                family_alpha=1.0,
            )
            fitted = fit_tree_regressor(
                full_train,
                full_y,
                backend="xgboost",
                random_state=int(seed) + FINAL_FIT_RANDOM_STATE_OFFSET,
                target_transform_mode=state.base.resolved_target_transform(),
                lightgbm_estimators=state.base.lightgbm_estimators,
                lightgbm_learning_rate=state.base.lightgbm_learning_rate,
                lightgbm_num_leaves=state.base.lightgbm_num_leaves,
                xgboost_estimators=state.base.xgboost_estimators,
                xgboost_learning_rate=state.base.xgboost_learning_rate,
                xgboost_max_depth=state.base.xgboost_max_depth,
                xgboost_min_child_weight=state.base.xgboost_min_child_weight,
                xgboost_subsample=state.base.xgboost_subsample,
                xgboost_colsample_bytree=state.base.xgboost_colsample_bytree,
                sample_weight=full_weight,
            )

            test_frame = waveform_stack.test.reset_index(drop=True)
            mean_abs_shap = _compute_mean_abs_shap(
                fitted_model=fitted,
                feature_frame=test_frame,
            )
            detail_frame = pd.DataFrame(
                {
                    "seed": int(seed),
                    "held_out_family": str(split.held_out_study),
                    "target": TARGET,
                    "protocol": PROTOCOL,
                    "variant_id": VARIANT_ID,
                    "feature": test_frame.columns.tolist(),
                    "feature_group": [_feature_group(col) for col in test_frame.columns.tolist()],
                    "mean_abs_shap": mean_abs_shap.astype(float),
                    "n_test_rows": int(len(test_frame)),
                    "n_train_rows": int(len(full_train)),
                    "n_pseudo_rows": int(len(state.trust_selected)),
                }
            ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            detail_frame["rank"] = detail_frame["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
            _append_csv(detail_frame, detail_path)
            detail_rows_written += int(len(detail_frame))

            holdout_pred = predict_tree_regressor(fitted, test_frame)
            holdout_metrics = score_predictions(state.y_test, holdout_pred)
            run_summary = pd.DataFrame(
                [
                    {
                        "seed": int(seed),
                        "held_out_family": str(split.held_out_study),
                        "target": TARGET,
                        "protocol": PROTOCOL,
                        "variant_id": VARIANT_ID,
                        "r2": float(holdout_metrics["r2"]),
                        "mae": float(holdout_metrics["mae"]),
                        "n_test_rows": int(len(test_frame)),
                        "n_train_rows": int(len(full_train)),
                        "n_pseudo_rows": int(len(state.trust_selected)),
                    }
                ]
            )
            _append_csv(run_summary, run_summary_path)
            run_rows_written += 1

    detail_df = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    feature_agg, group_agg, family_feature_agg, family_group_agg = _aggregate_outputs(detail_df)
    feature_agg.to_csv(_aggregate_feature_path(), index=False)
    group_agg.to_csv(_aggregate_group_path(), index=False)
    family_feature_agg.to_csv(_aggregate_feature_by_family_path(), index=False)
    family_group_agg.to_csv(_aggregate_group_by_family_path(), index=False)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "target": TARGET,
        "protocol": PROTOCOL,
        "variant_id": VARIANT_ID,
        "seeds": list(seeds),
        "held_out_families": held_out_families,
        "aggregation": "mean and std over seed x held-out-family runs",
        "rows_explained": "held-out test rows only",
        "resume": bool(resume),
        "detail_rows_written_this_run": int(detail_rows_written),
        "run_rows_written_this_run": int(run_rows_written),
        "total_run_count": int(detail_df.loc[:, ["seed", "held_out_family"]].drop_duplicates().shape[0]) if not detail_df.empty else 0,
    }
    _manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LOFO SHAP summaries for the fpos waveform-augmented stack.")
    parser.add_argument(
        "--seeds",
        default="42-61",
        help="Seed specification, for example '42-61' or '42,43,44'.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing seed/family outputs and recompute from scratch.",
    )
    args = parser.parse_args()

    seeds = parse_seed_spec(args.seeds)
    manifest = run_export(seeds=seeds, resume=not args.no_resume)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
