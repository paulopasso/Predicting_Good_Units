"""
Generate missing thesis figures from existing CSV tables and seed artifacts.

Figures produced:
  figures/04_fpos_model_progression/fpos_seed_stability.png
  figures/04_fpos_model_progression/fpos_fmiss_milestone_trajectory.png
  figures/03_paired_family_structure/recording_vs_family_disjoint_r2_gap.png
  figures/06_fpos_vs_fmiss_xai/fpos_fmiss_per_family_r2.png
  figures/07_limitations_and_data_budget/headroom_summary.png
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
ARTIFACTS = ROOT / "artifacts"
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qc_thesis.paths import PRE_WAVEFORM_LABEL, WAVEFORM_AUGMENTED_LABEL
from qc_thesis.plots import BENCHMARK_MODEL_COLORS

# ── colour palette ────────────────────────────────────────────────────────────
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
GREY   = "#8C8C8C"

FPOS_COLOR  = BLUE
FMISS_COLOR = ORANGE


# ─────────────────────────────────────────────────────────────────────────────
# 1. fpos seed stability  (box plot across aggregate seeds 42-61)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fpos_seed_stability():
    """Violin plots of R² and MAE across 20 seeds for all retained fpos recipes."""
    recipes = {
        "Paired-only raw": "fpos_paired_raw",
        "Paired-only context": "fpos_paired_context",
        "Context-first source stack": "fpos_context_stack",
        PRE_WAVEFORM_LABEL: "fpos_pre_waveform_unlabeled",
        WAVEFORM_AUGMENTED_LABEL: "fpos_waveform_winner",
    }

    rows = []
    for label, recipe in recipes.items():
        mfile = ARTIFACTS / "runs_aggregate" / "recording_disjoint_main" / "fpos" / recipe / "seed_metrics.csv"
        if not mfile.exists():
            continue
        df = pd.read_csv(mfile)
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
        df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
        df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
        df = df.dropna(subset=["seed", "r2", "mae"])
        df = df[df["seed"].between(42, 61)].copy()
        for _, row in df.iterrows():
            rows.append({"seed": int(row["seed"]), "model": label,
                          "r2": float(row["r2"]),
                          "mae": float(row["mae"])})

    if not rows:
        print("WARNING: no seed data found for fpos stability plot, skipping.")
        return

    df = pd.DataFrame(rows)
    model_order = [
        "Paired-only raw",
        "Paired-only context",
        "Context-first source stack",
        PRE_WAVEFORM_LABEL,
        WAVEFORM_AUGMENTED_LABEL,
    ]
    display_labels = model_order
    colors = [BENCHMARK_MODEL_COLORS.get(label, GREY) for label in model_order]

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 8.1), sharey=True)

    for ax, metric, ylabel, title in zip(
        axes,
        ["r2", "mae"],
        ["R²", "MAE"],
        [
            "R² across the five main compared `fpos` models\n(20 recording-disjoint seeds, 42-61)",
            "MAE across the five main compared `fpos` models\n(20 recording-disjoint seeds, 42-61)",
        ],
    ):
        data = [df.loc[df["model"] == m, metric].values for m in model_order]
        positions = np.arange(1, len(model_order) + 1)
        vp = ax.violinplot(
            data,
            positions=positions,
            vert=False,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.82,
        )
        for body, color in zip(vp["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.30)

        for pos, values, color in zip(positions, data, colors):
            if len(values) == 0:
                continue
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            low = np.min(values)
            high = np.max(values)
            ax.hlines(pos, low, high, color=color, linewidth=1.0, alpha=0.9, zorder=3)
            ax.hlines(pos, q1, q3, color=color, linewidth=4.4, alpha=0.95, zorder=4)
            ax.scatter([med], [pos], color=color, edgecolors="white", linewidths=0.7, s=34, zorder=5)

        ax.set_yticks(positions)
        ax.set_yticklabels(display_labels, fontsize=8.5)
        ax.set_xlabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        if metric == "r2":
            ax.axvline(0, color="#444444", linewidth=0.9, linestyle="--", zorder=1)
        else:
            dummy_mae = df.loc[df["model"] == "Dummy paired mean", "mae"].mean()
            ax.axvline(dummy_mae, color="#666666", linewidth=0.9, linestyle="--", zorder=1)

    fig.suptitle("fpos: seed stability across the retained compared models\n(recording-disjoint aggregate, seeds 42-61)", fontsize=12, y=0.995)
    fig.tight_layout()
    out = FIGURES / "04_fpos_model_progression" / "fpos_seed_stability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. fpos + fmiss milestone trajectory side-by-side
# ─────────────────────────────────────────────────────────────────────────────
def plot_milestone_trajectory():
    fpos_df = pd.read_csv(TABLES / "04_fpos_model_progression" / "fpos_milestone_trajectory.csv")
    fmiss_df = pd.read_csv(TABLES / "05_fmiss_model_progression" / "fmiss_milestone_trajectory.csv")

    def _clean_stage(s):
        # strip leading "N  " prefix if present
        return str(s).split("  ", 1)[-1].strip()

    fpos_df["stage_label"] = fpos_df["stage"].apply(_clean_stage)
    fmiss_df["stage_label"] = fmiss_df["stage"].apply(_clean_stage)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, df, color, title in zip(
        axes,
        [fpos_df, fmiss_df],
        [FPOS_COLOR, FMISS_COLOR],
        ["fpos — model progression (20 recording-disjoint seeds; final holdout R²)",
         "fmiss — model progression (20 recording-disjoint seeds; final holdout R²)"],
    ):
        n = len(df)
        xs = np.arange(n)
        bars = ax.bar(xs, df["r2"], color=color, alpha=0.75, width=0.6, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        # annotate bars
        for bar, val in zip(bars, df["r2"]):
            vert = bar.get_height()
            ypos = vert + 0.02 if vert >= 0 else vert - 0.06
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(xs)
        ax.set_xticklabels(df["stage_label"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("R²")
        ax.set_title(title, fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle("Model progression: fpos vs fmiss\n(20 recording-disjoint seeds; held-out recordings)", fontsize=13, y=1.03)
    fig.tight_layout()
    out = FIGURES / "04_fpos_model_progression" / "fpos_fmiss_milestone_trajectory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Recording-disjoint vs family-disjoint R² gap
# ─────────────────────────────────────────────────────────────────────────────
def plot_disjoint_gap():
    protocol = pd.read_csv(TABLES / "04_fpos_model_progression" / "fpos_protocol_regime_preference.csv")
    wave_row = protocol.loc[protocol["label"] == WAVEFORM_AUGMENTED_LABEL].iloc[0]
    winner_rd = float(wave_row["recording_disjoint_r2"])
    winner_fd = float(wave_row["macro_r2"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["Recording-disjoint\n(14 held-out recordings)", "Family-disjoint\n(LOFO, 5 families)"]
    values = [winner_rd, winner_fd]
    bar_colors = [BLUE, ORANGE]

    bars = ax.bar(labels, values, color=bar_colors, alpha=0.8, width=0.45, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"R² = {val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # gap annotation
    ax.annotate(
        "",
        xy=(1, winner_fd), xytext=(1, winner_rd),
        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5),
    )
    ax.text(1.27, (winner_rd + winner_fd) / 2,
            f"Δ = {winner_rd - winner_fd:.3f}", color=RED, fontsize=9, va="center")

    ax.set_ylabel(f"R²  ({WAVEFORM_AUGMENTED_LABEL.lower()})")
    ax.set_title("Generalization gap: recording-disjoint vs family-disjoint\n(20 recording-disjoint seeds vs LOFO family-held-out)", fontsize=10)
    ax.set_ylim(0, winner_rd + 0.12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES / "03_paired_family_structure" / "recording_vs_family_disjoint_r2_gap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. fpos vs fmiss per-family R² side-by-side grouped bar
# ─────────────────────────────────────────────────────────────────────────────
def plot_family_r2_comparison():
    df = pd.read_csv(TABLES / "06_fpos_vs_fmiss_xai" / "fpos_vs_fmiss_family_comparison.csv")

    # short family names
    df["family_short"] = df["study_set"].str.replace("PAIRED_", "", regex=False)

    families = df["family_short"].tolist()
    n = len(families)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_fpos  = ax.bar(x - width / 2, df["fpos_r2"],  width, label="fpos",
                        color=FPOS_COLOR, alpha=0.82, zorder=3)
    bars_fmiss = ax.bar(x + width / 2, df["fmiss_r2"], width, label="fmiss",
                        color=FMISS_COLOR, alpha=0.82, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for bar in list(bars_fpos) + list(bars_fmiss):
        h = bar.get_height()
        ypos = h + 0.02 if h >= 0 else h - 0.07
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=10)
    ax.set_ylabel("R²")
    ax.set_title("Per-family R²: lead `fpos` vs lead `fmiss` models\n(paired-family summaries from 20 recording-disjoint seeds)", fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES / "06_fpos_vs_fmiss_xai" / "fpos_fmiss_per_family_r2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Headroom summary
# ─────────────────────────────────────────────────────────────────────────────
def plot_headroom_summary():
    df = pd.read_csv(TABLES / "07_limitations_and_data_budget" / "headroom_summary.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(df))
    width = 0.28

    bars_basic   = ax.bar(x - width,     df["basic_transfer_r2"], width, label="Dummy floor",
                          color=GREY, alpha=0.75, zorder=3)
    bars_current = ax.bar(x,             df["current_best_r2"],   width, label="Current best",
                          color=BLUE, alpha=0.85, zorder=3)
    # optional: a "ceiling" bar if we had one — skipped for now

    for bar in list(bars_basic) + list(bars_current):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    # gain annotation
    for i, row in df.iterrows():
        mid_x = x[i] + width / 2
        ax.annotate(
            "",
            xy=(mid_x, row["current_best_r2"]),
            xytext=(mid_x, row["basic_transfer_r2"]),
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5),
        )
        ax.text(mid_x + 0.15, (row["basic_transfer_r2"] + row["current_best_r2"]) / 2,
                f"+{row['gain_r2']:.3f}", color=GREEN, fontsize=9, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(df["target"].str.upper(), fontsize=12)
    ax.set_ylabel("R²")
    ax.set_title("Headroom: dummy floor vs current best\n(20 recording-disjoint seeds; held-out recordings)", fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, df["current_best_r2"].max() + 0.12)

    fig.tight_layout()
    out = FIGURES / "07_limitations_and_data_budget" / "headroom_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating missing thesis figures...")
    plot_fpos_seed_stability()
    plot_milestone_trajectory()
    plot_disjoint_gap()
    plot_family_r2_comparison()
    plot_headroom_summary()
    print("Done.")
