"""
Regenerate the main-text benchmark ladder figures using only the retained
methodology models.

This script is intentionally standalone so it can run even when the local PATH
does not expose the Python version expected by the package itself.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_MODEL_COLORS = {
    "Dummy paired mean": "#7A7A7A",
    "Paired-only raw": "#2F2F2F",
    "Hybrid-only XGBoost": "#E07A1F",
    "Paired-only context": "#2F6690",
    "Context-first source stack": "#4EA1D3",
    "Pre-waveform unlabeled structure": "#2A9D8F",
    "Waveform-augmented structure": "#1B7F3B",
    "Reduced-latent context": "#7B61FF",
}

TARGET_SPECS = {
    "fpos": {
        "table": ROOT / "tables" / "04_fpos_model_progression" / "fpos_benchmark_ladder.csv",
        "figure_dir": ROOT / "figures" / "04_fpos_model_progression",
        "order": [
            "Waveform-augmented structure",
            "Pre-waveform unlabeled structure",
            "Context-first source stack",
            "Paired-only context",
            "Paired-only raw",
            "Hybrid-only XGBoost",
            "Dummy paired mean",
        ],
        "r2_xlim": (-1.05, 0.62),
        "mae_xlim": (0.0, 0.36),
    },
    "fmiss": {
        "table": ROOT / "tables" / "05_fmiss_model_progression" / "fmiss_benchmark_ladder.csv",
        "figure_dir": ROOT / "figures" / "05_fmiss_model_progression",
        "order": [
            "Reduced-latent context",
            "Paired-only context",
            "Paired-only raw",
            "Hybrid-only XGBoost",
            "Dummy paired mean",
        ],
        "r2_xlim": (-1.05, 0.08),
        "mae_xlim": (0.0, 0.46),
    },
}


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 220,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _prepare_table(table_path: Path, order: list[str]) -> pd.DataFrame:
    df = pd.read_csv(table_path)
    df = df[df["label"].isin(order)].copy()
    missing = [label for label in order if label not in set(df["label"])]
    if missing:
        raise ValueError(f"Missing expected labels in {table_path}: {missing}")
    return df.set_index("label").loc[order].reset_index()


def _annotate_bar_values(ax: plt.Axes, bars, raw_values: list[float], shown_values: list[float]) -> None:
    for bar, raw_value, shown_value in zip(bars, raw_values, shown_values):
        y = bar.get_y() + bar.get_height() / 2
        if raw_value < shown_value:
            ax.annotate(
                f"{raw_value:.3f}",
                xy=(shown_value, y),
                xytext=(shown_value + 0.06, y),
                va="center",
                ha="left",
                fontsize=8,
                color="#334155",
                arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=0.8),
            )
        elif raw_value >= 0:
            ax.text(raw_value + 0.01, y, f"{raw_value:.3f}", va="center", ha="left", fontsize=8, color="#334155")
        else:
            ax.text(raw_value - 0.01, y, f"{raw_value:.3f}", va="center", ha="right", fontsize=8, color="#334155")


def _plot_metric(target: str, df: pd.DataFrame, metric: str, *, xlim: tuple[float, float]) -> plt.Figure:
    is_r2 = metric == "r2"
    raw_values = pd.to_numeric(df[metric], errors="coerce").tolist()
    shown_values = raw_values
    if is_r2:
        clip_floor = xlim[0]
        shown_values = [max(value, clip_floor) for value in raw_values]

    colors = [BENCHMARK_MODEL_COLORS[label] for label in df["label"].tolist()]

    fig, ax = plt.subplots(figsize=(9.5, max(4.6, 0.58 * len(df))))
    y = np.arange(len(df))
    bars = ax.barh(y, shown_values, color=colors, edgecolor="white", linewidth=0.7, height=0.72)

    if is_r2:
        clip_floor = xlim[0]
        for bar, raw_value in zip(bars, raw_values):
            if raw_value < clip_floor:
                bar.set_hatch("///")
                bar.set_alpha(0.9)
        ax.axvline(0, color="#475569", linestyle="--", linewidth=1.0, zorder=2)
        ax.text(
            0.02,
            0.03,
            "Hatched bars are clipped on the left for readability.",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#64748B",
            ha="left",
            va="bottom",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist())
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color="#DCE3EB", linewidth=0.8, alpha=0.9)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel("R²" if is_r2 else "MAE")
    ax.set_title(f"{target}: retained benchmark ladder by {'R²' if is_r2 else 'MAE'}")

    _annotate_bar_values(ax, bars, raw_values, shown_values)
    fig.tight_layout()
    return fig


def main() -> None:
    for target, spec in TARGET_SPECS.items():
        df = _prepare_table(spec["table"], spec["order"])
        spec["figure_dir"].mkdir(parents=True, exist_ok=True)

        fig = _plot_metric(target, df, "r2", xlim=spec["r2_xlim"])
        fig.savefig(spec["figure_dir"] / f"{target}_benchmark_r2.png", bbox_inches="tight", facecolor="white", pad_inches=0.08)
        plt.close(fig)

        fig = _plot_metric(target, df, "mae", xlim=spec["mae_xlim"])
        fig.savefig(spec["figure_dir"] / f"{target}_benchmark_mae.png", bbox_inches="tight", facecolor="white", pad_inches=0.08)
        plt.close(fig)

        print(f"Updated {target} benchmark figures in {spec['figure_dir']}")


if __name__ == "__main__":
    main()
