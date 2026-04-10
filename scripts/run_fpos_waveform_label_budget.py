"""Run a fixed-split label-budget sweep for the fpos waveform stack.

This script keeps the recording-disjoint holdout fixed at seed 42, varies only
the labeled paired fine-tuning subset, and evaluates the canonical waveform
variant (`wf_embed_anchor_stack_xgboost`) on grouped recording-level subsets.

Outputs:
  - tables/07_limitations_and_data_budget/fpos_waveform_label_budget_runs.csv
  - tables/07_limitations_and_data_budget/fpos_waveform_label_budget_summary.csv
  - figures/07_limitations_and_data_budget/label_budget_combined.png
  - artifacts/label_budget_waveform/fpos_waveform_seed42/...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qc_thesis.modeling.neural import prepare_feature_table
from qc_thesis.modeling.shared import MainExperimentSplit, make_main_split
from qc_thesis.modeling.stacking.backends import score_predictions
from qc_thesis.modeling.stacking.base_stack_runner import (
    build_base_stack_state_from_split,
    build_embedding_universe,
    evaluate_stack_variant,
)
from qc_thesis.modeling.stacking.recipes import FPosSignalEmbeddingStackConfig
from qc_thesis.modeling.stacking.augmenters import fit_signal_embeddings


TARGET = "fpos"
VARIANT_ID = "wf_embed_anchor_stack_xgboost"
RECIPE_ID = "fpos_waveform_winner"
FIXED_SPLIT_SEED = 42
FIXED_HOLDOUT_ROWS = 100
WAVEFORM_RANDOM_STATE_OFFSET = 2400

TABLE_DIR = REPO / "tables" / "07_limitations_and_data_budget"
FIG_DIR = REPO / "figures" / "07_limitations_and_data_budget"
ARTIFACT_ROOT = REPO / "artifacts" / "label_budget_waveform" / "fpos_waveform_seed42"

RUNS_CSV = TABLE_DIR / "fpos_waveform_label_budget_runs.csv"
SUMMARY_CSV = TABLE_DIR / "fpos_waveform_label_budget_summary.csv"
FIG_PATH = FIG_DIR / "label_budget_combined.png"

CANONICAL_METRICS = REPO / "artifacts" / "runs" / "recording_disjoint_main" / "fpos" / RECIPE_ID / "metrics.csv"
CANONICAL_SPLIT = REPO / "artifacts" / "runs" / "recording_disjoint_main" / "fpos" / RECIPE_ID / "split_manifest.json"


BUDGET_REPEATS: dict[int, tuple[int, ...]] = {
    20: (42, 43),
    60: (42, 43),
    107: (42,),
}


def _grouped_budget_subset(df: pd.DataFrame, *, target_rows: int, random_state: int, n_candidates: int = 256) -> pd.DataFrame:
    if target_rows >= len(df):
        return df.copy().reset_index(drop=True)
    test_size = min(0.95, max(0.05, target_rows / max(len(df), 1)))
    splitter = GroupShuffleSplit(n_splits=n_candidates, test_size=test_size, random_state=random_state)
    best_subset: pd.DataFrame | None = None
    best_gap: int | None = None
    for _, subset_idx in splitter.split(df, groups=df["recording_key"].astype(str)):
        subset = df.iloc[subset_idx].copy()
        gap = abs(len(subset) - target_rows)
        if best_gap is None or gap < best_gap:
            best_subset = subset
            best_gap = gap
            if gap == 0:
                break
    assert best_subset is not None
    return best_subset.reset_index(drop=True)


def _budget_split(base_split: MainExperimentSplit, *, budget_rows: int, subset_seed: int) -> MainExperimentSplit:
    subset = _grouped_budget_subset(
        base_split.paired_finetune,
        target_rows=budget_rows,
        random_state=subset_seed,
    )
    manifest = dict(base_split.manifest)
    manifest.update(
        {
            "protocol": "recording_disjoint_budget_waveform",
            "base_random_state": FIXED_SPLIT_SEED,
            "label_budget_rows_target": int(budget_rows),
            "label_budget_subset_seed": int(subset_seed),
            "paired_finetune_rows": int(len(subset)),
            "paired_finetune_recordings": int(subset["recording_key"].nunique()),
        }
    )
    return MainExperimentSplit(
        hybrid_source=base_split.hybrid_source,
        paired_finetune=subset,
        paired_test=base_split.paired_test,
        paired_non_test_all=base_split.paired_non_test_all,
        ssl_pool=base_split.ssl_pool,
        holdout_recordings=base_split.holdout_recordings,
        manifest=manifest,
    )


def _save_run_outputs(output_dir: Path, *, state, holdout_pred, metrics: dict[str, float], cv_metrics: dict[str, float], config_dict: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = state.target_test_df.loc[:, ["row_uid", "recording_key", "study_set"]].copy().reset_index(drop=True)
    pred_df["true"] = state.y_test
    pred_df["pred"] = holdout_pred
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame([{**metrics, "cv_r2": float(cv_metrics["r2"]), "cv_mae": float(cv_metrics["mae"])}]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )
    with open(output_dir / "split_manifest.json", "w") as handle:
        json.dump(state.split.manifest, handle, indent=2)
    with open(output_dir / "config.json", "w") as handle:
        json.dump(config_dict, handle, indent=2)


def _run_budget_point(prepared, base_split: MainExperimentSplit, *, budget_rows: int, subset_seed: int) -> dict[str, object]:
    budget_split = _budget_split(base_split, budget_rows=budget_rows, subset_seed=subset_seed)
    run_name = f"budget_{budget_rows}_subsetseed_{subset_seed}"
    output_dir = ARTIFACT_ROOT / run_name

    config = FPosSignalEmbeddingStackConfig(
        parquet_path=REPO / "data/raw/33000_ROWS.parquet",
        output_root=ARTIFACT_ROOT,
        run_name=run_name,
        random_state=FIXED_SPLIT_SEED,
        paired_final_holdout_rows=FIXED_HOLDOUT_ROWS,
        verbose=False,
    )
    state = build_base_stack_state_from_split(
        target=TARGET,
        prepared=prepared,
        split=budget_split,
        config=config,
    )
    universe_df = build_embedding_universe(
        split=state.split,
        max_rows=config.universe_max_rows,
        random_state=config.random_state,
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
    cv_metrics, holdout_pred = evaluate_stack_variant(
        state=state,
        stack=waveform_stack,
        backend_id="xgboost_default",
        family_alpha=1.0,
        random_state_offset=WAVEFORM_RANDOM_STATE_OFFSET,
    )
    metrics = score_predictions(state.y_test, holdout_pred)
    _save_run_outputs(
        output_dir,
        state=state,
        holdout_pred=holdout_pred,
        metrics=metrics,
        cv_metrics=cv_metrics,
        config_dict=config.to_dict(),
    )
    return {
        "target": TARGET,
        "recipe_id": RECIPE_ID,
        "variant_id": VARIANT_ID,
        "budget_rows": int(budget_rows),
        "subset_seed": int(subset_seed),
        "actual_budget_rows": int(len(state.target_train_df)),
        "subset_recordings": int(state.target_train_df["recording_key"].nunique()),
        "holdout_seed": FIXED_SPLIT_SEED,
        "r2": float(metrics["r2"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "bias": float(metrics["bias"]),
        "cv_r2": float(cv_metrics["r2"]),
        "cv_mae": float(cv_metrics["mae"]),
        "output_dir": str(output_dir),
    }


def _load_canonical_full_budget() -> dict[str, object]:
    metrics = pd.read_csv(CANONICAL_METRICS)
    row = metrics.loc[metrics["variant_id"] == VARIANT_ID].iloc[0]
    split_manifest = json.loads(CANONICAL_SPLIT.read_text())
    output_dir = CANONICAL_METRICS.parent
    return {
        "target": TARGET,
        "recipe_id": RECIPE_ID,
        "variant_id": VARIANT_ID,
        "budget_rows": 107,
        "subset_seed": FIXED_SPLIT_SEED,
        "actual_budget_rows": int(split_manifest["paired_finetune_rows"]),
        "subset_recordings": int(split_manifest["paired_finetune_recordings"]),
        "holdout_seed": FIXED_SPLIT_SEED,
        "r2": float(row["r2"]),
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
        "bias": float(row["bias"]),
        "cv_r2": float("nan"),
        "cv_mae": float("nan"),
        "output_dir": str(output_dir),
        "source": "canonical_saved_run",
    }


def _summarize_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        run_df.groupby(["target", "recipe_id", "variant_id", "budget_rows"], as_index=False)
        .agg(
            repeats=("subset_seed", "size"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_bias=("bias", "mean"),
            mean_subset_recordings=("subset_recordings", "mean"),
            mean_actual_budget_rows=("actual_budget_rows", "mean"),
        )
        .sort_values("budget_rows")
        .reset_index(drop=True)
    )
    for col in ["std_mae", "std_r2"]:
        grouped[col] = grouped[col].fillna(0.0)
    return grouped


def _plot_summary(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df.sort_values("budget_rows").reset_index(drop=True)
    xs = plot_df["budget_rows"].to_numpy(dtype=float)
    ys_r2 = plot_df["mean_r2"].to_numpy(dtype=float)
    sds_r2 = plot_df["std_r2"].to_numpy(dtype=float)
    ys_mae = plot_df["mean_mae"].to_numpy(dtype=float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax_mae = ax.twinx()
    color_r2 = "#1B7F3B"
    color_mae = "#D4691E"

    r2_line = ax.plot(xs, ys_r2, marker="o", linewidth=2.4, markersize=7, color=color_r2, label="Mean R²")[0]
    mae_line = ax_mae.plot(
        xs,
        ys_mae,
        marker="s",
        linewidth=2.1,
        markersize=6,
        color=color_mae,
        linestyle="--",
        label="Mean MAE",
    )[0]
    mask = sds_r2 > 0
    if mask.any():
        ax.fill_between(xs[mask], ys_r2[mask] - sds_r2[mask], ys_r2[mask] + sds_r2[mask], color=color_r2, alpha=0.15)

    point_offsets_r2 = {20: 0.05, 60: 0.04, 107: 0.0}
    point_offsets_mae = {20: -0.010, 60: -0.008, 107: 0.008}
    for x, y in zip(xs, ys_r2):
        if int(x) == 107:
            continue
        ax.text(
            x,
            y + point_offsets_r2.get(int(x), 0.04),
            f"{y:.2f}",
            color=color_r2,
            fontsize=8,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
        )
    for x, y in zip(xs, ys_mae):
        if int(x) == 107:
            continue
        ax_mae.text(
            x,
            y + point_offsets_mae.get(int(x), -0.008),
            f"{y:.3f}",
            color=color_mae,
            fontsize=8,
            ha="center",
            va="top" if point_offsets_mae.get(int(x), -0.008) < 0 else "bottom",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
        )

    x_min = float(xs.min()) - 3
    x_max = float(xs.max()) + 7
    ax.axhline(0, color="#888888", linewidth=1.1, linestyle="--", zorder=1)
    ax.fill_between([x_min, x_max], [0, 0], [-1.2, -1.2], color="#cccccc", alpha=0.25, zorder=0)
    final_r2 = float(plot_df.loc[plot_df["budget_rows"] == 107, "mean_r2"].iloc[0])
    final_mae = float(plot_df.loc[plot_df["budget_rows"] == 107, "mean_mae"].iloc[0])
    ax.axvline(107, color="black", linewidth=1.1, linestyle=":", zorder=3)
    ax.text(
        107 - 0.8,
        max(0.565, final_r2 + 0.08),
        "fixed split\nseed 42\nn=107",
        fontsize=8.2,
        va="top",
        ha="right",
        color="black",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#d0d0d0", alpha=0.9),
    )
    ax.annotate(
        f"Waveform-augmented stack\nR²={final_r2:.2f}",
        xy=(107, final_r2),
        xytext=(108.8, final_r2 + 0.04),
        textcoords="data",
        fontsize=8.5,
        fontweight="bold",
        color=color_r2,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color_r2, alpha=0.92),
    )
    ax_mae.annotate(
        f"MAE={final_mae:.3f}",
        xy=(107, final_mae),
        xytext=(108.8, final_mae - 0.004),
        textcoords="data",
        fontsize=8.3,
        fontweight="bold",
        color=color_mae,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color_mae, alpha=0.92),
    )

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(sorted(set(int(x) for x in xs.tolist())))
    ax.set_ylim(-1.22, max(0.58, final_r2 + 0.12))
    ax.set_xlabel("Number of labeled paired training rows", fontsize=11)
    ax.set_ylabel("Mean R²", fontsize=11)
    ax_mae.set_ylabel("Mean MAE", fontsize=11, color=color_mae)
    ax_mae.tick_params(axis="y", colors=color_mae)
    ax.set_title("Label efficiency of the fpos waveform-augmented stack", fontsize=13, fontweight="bold")
    ax.legend(
        handles=[r2_line, mae_line],
        loc="lower left",
        bbox_to_anchor=(0.015, 0.02),
        ncol=2,
        frameon=True,
        framealpha=0.9,
        borderaxespad=0.2,
    )

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    prepared = prepare_feature_table(REPO / "data/raw/33000_ROWS.parquet")
    base_split = make_main_split(
        prepared.df,
        paired_final_holdout_rows=FIXED_HOLDOUT_ROWS,
        random_state=FIXED_SPLIT_SEED,
    )

    run_rows: list[dict[str, object]] = []
    for budget_rows, subset_seeds in BUDGET_REPEATS.items():
        for subset_seed in subset_seeds:
            if budget_rows == 107 and CANONICAL_METRICS.exists() and CANONICAL_SPLIT.exists():
                print(f"[budget {budget_rows}] reusing canonical waveform run from seed {FIXED_SPLIT_SEED}", flush=True)
                run_rows.append(_load_canonical_full_budget())
                continue
            print(f"[budget {budget_rows}] running grouped subset seed {subset_seed}", flush=True)
            run_rows.append(
                _run_budget_point(
                    prepared,
                    base_split,
                    budget_rows=budget_rows,
                    subset_seed=subset_seed,
                )
            )

    run_df = pd.DataFrame(run_rows).sort_values(["budget_rows", "subset_seed"]).reset_index(drop=True)
    summary_df = _summarize_runs(run_df)
    run_df.to_csv(RUNS_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    _plot_summary(summary_df)

    print(f"Saved runs: {RUNS_CSV}")
    print(f"Saved summary: {SUMMARY_CSV}")
    print(f"Saved figure: {FIG_PATH}")


if __name__ == "__main__":
    main()
