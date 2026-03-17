from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot diagnostics for a saved winner run")
    parser.add_argument("run_dir", help="Path to a run output directory containing metrics.csv and predictions.csv")
    parser.add_argument("--variant-id", default=None, help="Specific variant_id to plot; defaults to best holdout R2")
    return parser.parse_args()


def _select_variant(run_dir: Path, variant_id: str | None) -> str:
    if variant_id:
        return variant_id
    metrics = pd.read_csv(run_dir / "metrics.csv")
    best = metrics.sort_values(["r2", "mae"], ascending=[False, True]).iloc[0]
    return str(best["variant_id"])


def _study_table(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("study_set", dropna=False)
        .agg(
            n=("true", "size"),
            true_mean=("true", "mean"),
            pred_mean=("pred", "mean"),
            mae=("abs_error", "mean"),
            bias=("pred", lambda s: float(np.mean(s - df.loc[s.index, "true"]))),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )
    return out


def make_plots(run_dir: Path, variant_id: str | None = None) -> tuple[Path, Path]:
    run_dir = run_dir.resolve()
    predictions = pd.read_csv(run_dir / "predictions.csv")
    variant = _select_variant(run_dir, variant_id)
    df = predictions[predictions["variant_id"].astype(str) == variant].copy()
    if df.empty:
        raise ValueError(f"Variant {variant!r} not found in {run_dir / 'predictions.csv'}")

    df["residual"] = df["pred"] - df["true"]
    study_df = _study_table(df)

    plt.style.use("seaborn-v0_8-whitegrid")

    panel_path = run_dir / f"{variant}_diagnostics.png"
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.scatter(df["true"], df["pred"], alpha=0.75, s=36, edgecolor="none", color="#1f77b4")
    lims = [0.0, max(1.0, float(max(df["true"].max(), df["pred"].max())))]
    ax.plot(lims, lims, "--", color="black", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True fpos")
    ax.set_ylabel("Predicted fpos")
    ax.set_title("Predicted vs True")

    ax = axes[0, 1]
    ax.scatter(df["true"], df["residual"], alpha=0.75, s=36, edgecolor="none", color="#d62728")
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("True fpos")
    ax.set_ylabel("Residual (pred - true)")
    ax.set_title("Residuals vs True")

    ax = axes[1, 0]
    bins = np.linspace(0.0, 1.0, 16)
    ax.hist(df["true"], bins=bins, alpha=0.55, label="True", color="#2ca02c")
    ax.hist(df["pred"], bins=bins, alpha=0.55, label="Pred", color="#ff7f0e")
    ax.set_xlabel("fpos")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of True and Predicted Values")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ordered = study_df.sort_values("true_mean")
    x = np.arange(len(ordered))
    ax.scatter(x, ordered["true_mean"], label="True mean", color="#2ca02c", s=60)
    ax.scatter(x, ordered["pred_mean"], label="Pred mean", color="#ff7f0e", s=60)
    for i, row in ordered.iterrows():
        idx = ordered.index.get_loc(i)
        ax.plot([idx, idx], [row["true_mean"], row["pred_mean"]], color="#999999", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["study_set"], rotation=25, ha="right")
    ax.set_ylabel("Mean fpos")
    ax.set_title("Per-Study Mean True vs Predicted")
    ax.legend(frameon=False)

    fig.suptitle(f"Winner diagnostics: {variant}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    study_path = run_dir / f"{variant}_study_breakdown.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ordered = study_df.sort_values("mae", ascending=True)
    ax.barh(ordered["study_set"], ordered["mae"], color="#4c78a8")
    ax.set_xlabel("MAE")
    ax.set_title("Per-Study MAE")

    ax = axes[1]
    ordered = study_df.sort_values("bias")
    colors = ["#d62728" if b < 0 else "#1f77b4" for b in ordered["bias"]]
    ax.barh(ordered["study_set"], ordered["bias"], color=colors)
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Bias (pred - true)")
    ax.set_title("Per-Study Bias")

    fig.suptitle(f"Study breakdown: {variant}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(study_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    study_df.to_csv(run_dir / f"{variant}_study_summary.csv", index=False)
    return panel_path, study_path


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    panel_path, study_path = make_plots(run_dir=run_dir, variant_id=args.variant_id)
    print(panel_path)
    print(study_path)


if __name__ == "__main__":
    main()
