from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot family robustness comparisons for fpos")
    parser.add_argument("run_dir", help="Path to a fpos_family_robustness output directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    metrics_path = run_dir / "metrics.csv"
    family_path = run_dir / "family_performance.csv"

    metrics_df = pd.read_csv(metrics_path)
    family_df = pd.read_csv(family_path)

    focus_variants = [
        "reference_xgboost",
        "family_balanced_strong_xgboost",
        "reference_plus_strong_blend_050",
    ]
    label_map = {
        "reference_xgboost": "Reference",
        "family_balanced_strong_xgboost": "Family-balanced",
        "reference_plus_strong_blend_050": "50/50 blend",
    }
    palette = {
        "Reference": "#1f77b4",
        "Family-balanced": "#d62728",
        "50/50 blend": "#2ca02c",
    }

    metrics_focus = metrics_df[metrics_df["variant_id"].isin(focus_variants)].copy()
    metrics_focus["label"] = metrics_focus["variant_id"].map(label_map)

    family_focus = family_df[
        (family_df["split"].astype(str) == "holdout")
        & (family_df["variant_id"].isin(focus_variants))
    ].copy()
    family_focus["label"] = family_focus["variant_id"].map(label_map)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    r2_ax = axes[0]
    order = ["Reference", "50/50 blend", "Family-balanced"]
    sns.barplot(
        data=metrics_focus,
        x="label",
        y="r2",
        order=order,
        palette=palette,
        ax=r2_ax,
    )
    r2_ax.set_title("Overall Holdout R²")
    r2_ax.set_xlabel("")
    r2_ax.set_ylabel("R²")
    for container in r2_ax.containers:
        r2_ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10)

    family_r2_ax = axes[1]
    sns.barplot(
        data=family_focus,
        x="study_set",
        y="r2",
        hue="label",
        hue_order=order,
        palette=palette,
        ax=family_r2_ax,
    )
    family_r2_ax.set_title("Per-Family Holdout R²")
    family_r2_ax.set_xlabel("")
    family_r2_ax.set_ylabel("R²")
    family_r2_ax.tick_params(axis="x", rotation=25)
    family_r2_ax.legend(title="", loc="lower right", fontsize=10)

    family_mae_ax = axes[2]
    sns.barplot(
        data=family_focus,
        x="study_set",
        y="mae",
        hue="label",
        hue_order=order,
        palette=palette,
        ax=family_mae_ax,
    )
    family_mae_ax.set_title("Per-Family Holdout MAE")
    family_mae_ax.set_xlabel("")
    family_mae_ax.set_ylabel("MAE")
    family_mae_ax.tick_params(axis="x", rotation=25)
    family_mae_ax.legend_.remove()

    fig.suptitle("fpos Family Robustness Tradeoff", fontsize=18, y=1.02)
    fig.tight_layout()
    out_path = run_dir / "family_robustness_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
