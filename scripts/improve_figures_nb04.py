"""
Improved figures for notebook 04: fpos model progression.

Run from repo root:
    python scripts/improve_figures_nb04.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
TABLE_DIR = REPO / "tables" / "04_fpos_model_progression"
FIG_DIR = REPO / "figures" / "04_fpos_model_progression"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from qc_thesis.plots import BENCHMARK_MODEL_COLORS

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

C_NEURAL = "#CC6677"
C_WINNER = BENCHMARK_MODEL_COLORS["Waveform-augmented structure"]

# Family colors (5 families)
FAMILY_COLORS = {
    "BOYDEN":    "#4477AA",
    "CRCNS_HC1": "#EE6677",
    "ENGLISH":   "#228833",
    "KAMPFF":    "#CCBB44",
    "MEA64C_YGER": "#AA3377",
}

DISPLAY_SEP = " · "

# ---------------------------------------------------------------------------
# Helper: label → short name
# ---------------------------------------------------------------------------
SHORT = {
    "Dummy paired mean":               "Dummy baseline",
    "Paired-only raw":                 "Paired raw",
    "Hybrid-only XGBoost":             "Hybrid only",
    "Calibrated hybrid XGBoost":       "Hybrid calibrated",
    "Hybrid+paired basic XGBoost":     "Hybrid + paired",
    "Hybrid stack residual":           "Hybrid residual",
    "Paired-only context":             "Paired context",
    "Context-first source stack":      "Context stack",
    "Best SSL neural":                 "SSL (neural)",
    "Best MMD neural":                 "MMD (neural)",
    "Pre-waveform unlabeled structure":"Pseudo + anchor",
    "Waveform-augmented structure":    "Waveform-augmented",
}

def _model_color(label: str) -> str:
    return BENCHMARK_MODEL_COLORS.get(str(label), "#7A7A7A")


def _first_present_column(df: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"Expected one of {names}, found {list(df.columns)}")


def _extract_family(label: object) -> str:
    text = str(label)
    if "::" in text:
        return text.split("::")[0].replace("PAIRED_", "")
    if DISPLAY_SEP in text:
        return text.split(DISPLAY_SEP, 1)[0].strip()
    return text.replace("PAIRED_", "")


def _extract_recording_name(label: object) -> str:
    text = str(label)
    if "::" in text:
        return text.split("::")[-1]
    if DISPLAY_SEP in text:
        return text.split(DISPLAY_SEP, 1)[1].strip()
    return text

# ---------------------------------------------------------------------------
# Load benchmark ladder
# ---------------------------------------------------------------------------
ladder = pd.read_csv(TABLE_DIR / "fpos_benchmark_ladder.csv")
ladder["short"] = ladder["label"].map(lambda x: SHORT.get(x, x))
ladder["color"] = ladder["label"].map(_model_color)


# ===========================================================================
# 1. fpos_benchmark_r2_annotated.png — main figure
# ===========================================================================
def plot_r2_benchmark(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of R² for the core main-text recipes."""
    d = df.sort_values("r2", ascending=True).copy()  # ascending so top is highest on chart

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        d["short"], d["r2"],
        color=d["color"],
        height=0.65,
        edgecolor="white",
        linewidth=0.5,
    )

    # Vertical dashed line at R²=0
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.0, zorder=3)

    ax.set_xlabel("R²  (20 recording-disjoint seeds; held-out recordings)", fontsize=10)
    ax.set_title("fpos core benchmark ladder — R² by recipe\n(20 recording-disjoint seeds, 42-61)", fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(-1.35, 0.72)

    fig.tight_layout()
    return fig


fig = plot_r2_benchmark(ladder)
out = FIG_DIR / "fpos_benchmark_r2_annotated.png"
fig.savefig(out, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved: {out}")


# ===========================================================================
# 2. fpos_benchmark_mae.png — same layout for MAE
# ===========================================================================
def plot_mae_benchmark(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of MAE for the core main-text recipes."""
    d = df.sort_values("mae", ascending=False).copy()  # ascending=False so best (lowest) is at top

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        d["short"], d["mae"],
        color=d["color"],
        height=0.65,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("MAE  (20 recording-disjoint seeds; held-out recordings)", fontsize=10)
    ax.set_title("fpos core benchmark ladder — MAE by recipe\n(20 recording-disjoint seeds, 42-61)", fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(0, 0.34)

    fig.tight_layout()
    return fig


fig = plot_mae_benchmark(ladder)
out = FIG_DIR / "fpos_benchmark_mae.png"
fig.savefig(out, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved: {out}")


# ===========================================================================
# 3. fpos_winner_prediction_scatter.png
#    No unit-level predictions stored in tables; simulate from per-recording R²
#    using per-family row counts and aggregate metrics from benchmark_ladder.
#    We use the family_wave_summary per-family mae and family r2 to make a
#    representative scatter using the recording-level r2 pivot as proxy data.
# ===========================================================================
def plot_winner_scatter_from_recording_r2(df_rec_r2: pd.DataFrame) -> plt.Figure:
    """
    Scatter of per-recording R² values colored by family.
    X-axis: R² for Context stack (best pre-waveform baseline)
    Y-axis: R² for the waveform-augmented structure stack
    This shows per-recording improvement, which is informative for the thesis.
    """
    d = df_rec_r2.copy()
    key_col = _first_present_column(d, "recording_key", "index")
    d["family"] = d[key_col].map(_extract_family)

    fig, ax = plt.subplots(figsize=(6, 5))

    for fam, grp in d.groupby("family"):
        color = FAMILY_COLORS.get(fam, "#999999")
        short_fam = fam.replace("_", " ").title()
        ax.scatter(
            grp["Context stack"],
            grp["Waveform+"],
            color=color,
            s=50,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.4,
            label=short_fam,
            zorder=3,
        )

    # Diagonal y=x line
    lim_min = min(d["Context stack"].min(), d["Waveform+"].min()) - 0.1
    lim_max = max(d["Context stack"].max(), d["Waveform+"].max()) + 0.1
    lim = (lim_min, lim_max)
    ax.plot(lim, lim, color="#555555", linestyle="--", linewidth=1.0, zorder=2, label="y = x")

    ax.set_xlabel("R²  —  Context stack", fontsize=10)
    ax.set_ylabel("R²  —  Waveform-augmented structure", fontsize=10)
    ax.set_title("Per-recording mean R²: Context stack vs waveform-augmented structure\n(recording summaries pooled across 20 recording-disjoint seeds)", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")

    # Text annotation with overall metrics
    ax.text(
        0.98, 0.04,
        "Waveform-augmented structure\nR²=0.533  MAE=0.116",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
    )

    fig.tight_layout()
    return fig


rec_r2 = pd.read_csv(TABLE_DIR / "fpos_recording_r2_pivot.csv").rename(columns={
    "Waveform-augmented structure": "Waveform+",
    "Waveform-augmented structure": "Waveform+",
})
fig = plot_winner_scatter_from_recording_r2(rec_r2)
out = FIG_DIR / "fpos_winner_prediction_scatter.png"
fig.savefig(out, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved: {out}")


# ===========================================================================
# 4. fpos_winner_protocol_comparison.png
# ===========================================================================
def plot_protocol_comparison(df: pd.DataFrame) -> plt.Figure:
    protocols = df["protocol"].tolist()
    r2_vals   = df["r2"].tolist()
    mae_vals  = df["mae"].tolist()

    # Friendly labels
    label_map = {
        "recording_disjoint_main": "Recording-disjoint\n(20-seed aggregate)",
        "family_disjoint_lofo":    "Family-held-out\n(LOFO aggregate)",
    }
    labels = [label_map.get(p, p) for p in protocols]
    colors = [C_WINNER, C_NEURAL]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # R²
    bars0 = axes[0].bar(labels, r2_vals, color=colors, width=0.45, edgecolor="white")
    axes[0].set_ylabel("R²", fontsize=10)
    axes[0].set_title("R² by evaluation protocol", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 0.65)
    for bar, val in zip(bars0, r2_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, val + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # MAE
    bars1 = axes[1].bar(labels, mae_vals, color=colors, width=0.45, edgecolor="white")
    axes[1].set_ylabel("MAE", fontsize=10)
    axes[1].set_title("MAE by evaluation protocol", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 0.165)
    for bar, val in zip(bars1, mae_vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, val + 0.001,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Delta annotations
    dr2  = df["delta_r2_vs_recording"].iloc[1]
    dmae = df["delta_mae_vs_recording"].iloc[1]
    axes[0].annotate(
        f"Δ={dr2:+.3f}",
        xy=(1, r2_vals[1]), xytext=(1, r2_vals[1] + 0.07),
        ha="center", fontsize=8.5, color="#CC3333",
        arrowprops=dict(arrowstyle="-|>", color="#CC3333", lw=1.0),
    )

    fig.suptitle("Waveform-augmented structure — protocol gap\n(recording-disjoint 20-seed aggregate vs family-held-out LOFO)", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


proto = pd.read_csv(TABLE_DIR / "fpos_winner_protocol_comparison.csv")
fig = plot_protocol_comparison(proto)
out = FIG_DIR / "fpos_winner_protocol_comparison.png"
fig.savefig(out, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved: {out}")


# ===========================================================================
# 5. fpos_family_r2_by_wave.png — per-family R² heatmap
# ===========================================================================
def plot_family_r2_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of per-family R² across key model waves."""
    d = df.copy()
    group_col = _first_present_column(d, "study_set", "index")
    d["family"] = d[group_col].map(_extract_family)
    d = d.set_index("family").drop(columns=[group_col])

    # Shorten column names
    col_map = {
        "Context stack":   "Context\nstack",
        "Paired context":  "Paired\ncontext",
        "Pseudo+anchor":   "Pseudo\n+anchor",
        "Waveform-augmented structure": "Waveform\naugmented",
    }
    d = d.rename(columns=col_map)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    vals = d.values.astype(float)
    # Clip negative values to -0.5 for display purposes
    vals_clipped = np.clip(vals, -0.5, 1.0)

    im = ax.imshow(vals_clipped, cmap="RdYlGn", vmin=-0.5, vmax=0.75, aspect="auto")
    ax.grid(False)

    ax.set_xticks(range(len(d.columns)))
    ax.set_xticklabels(d.columns, fontsize=9)
    ax.set_yticks(range(len(d.index)))
    ax.set_yticklabels(d.index, fontsize=9)
    ax.tick_params(bottom=False, left=False)

    # Annotate cells
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt_color = "white" if abs(vals_clipped[i, j]) > 0.4 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color=txt_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("R²", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Per-family mean R² across model waves\n(paired-family summaries from 20 recording-disjoint seeds)", fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    return fig


fam_pivot = pd.read_csv(TABLE_DIR / "fpos_family_r2_pivot.csv")
fig = plot_family_r2_heatmap(fam_pivot)
out = FIG_DIR / "fpos_family_r2_by_wave.png"
fig.savefig(out, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved: {out}")


# ===========================================================================
# 6. fpos_winner_residuals.png — NEW: per-recording residuals
#    Since we have per-recording R² (not unit-level predictions), we use the
#    recording-level R² deviation from mean as a proxy for where the model
#    struggles. We also color by family.
# ===========================================================================
def plot_winner_residuals(df_rec_r2: pd.DataFrame, df_top: pd.DataFrame, df_worst: pd.DataFrame) -> plt.Figure:
    """
    Show the extreme best and worst held-out recordings without letting a
    single outlier flatten the rest of the panel.
    """
    key_col = _first_present_column(df_top, "recording_key", "index")

    def _prep(df: pd.DataFrame, ascending: bool, n: int) -> pd.DataFrame:
        d = df.copy()
        d["family"] = d[key_col].map(_extract_family)
        d["short_key"] = d[key_col].map(_extract_recording_name)
        d = d.sort_values("r2", ascending=ascending).drop_duplicates(key_col).head(n)
        return d

    worst = _prep(df_worst, ascending=True, n=6).iloc[::-1].reset_index(drop=True)
    best = _prep(df_top, ascending=False, n=6).iloc[::-1].reset_index(drop=True)
    overall_r2 = 0.5332

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    for ax, data, title, xlim in [
        (axes[0], worst, "Worst held-out recordings", (min(-12.5, worst["r2"].min() * 1.05), 0.15)),
        (axes[1], best, "Best held-out recordings", (-0.1, max(1.0, best["r2"].max() * 1.08))),
    ]:
        colors = [FAMILY_COLORS.get(f, "#999999") for f in data["family"]]
        bars = ax.barh(
            data["short_key"],
            data["r2"],
            color=colors,
            height=0.72,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axvline(0, color="#333333", linestyle="-", linewidth=0.9, zorder=3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(xlim)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=9)

        for bar, val in zip(bars, data["r2"]):
            if val >= 0:
                xpos = val + 0.03
                ha = "left"
            else:
                xpos = val - 0.18
                ha = "right"
            ax.text(
                xpos,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                ha=ha,
                fontsize=8.5,
                color="#222222",
            )

    axes[1].axvline(
        overall_r2,
        color=C_WINNER,
        linestyle="--",
        linewidth=1.2,
        zorder=3,
    )
    axes[1].text(
        overall_r2 + 0.02,
        0.04,
        f"Overall R²={overall_r2:.3f}",
        transform=axes[1].get_xaxis_transform(),
        color=C_WINNER,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    axes[0].text(
        0.03,
        0.03,
        "Showing only the six strongest failures avoids compressing the entire panel.",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#666666",
    )

    fig.suptitle("Waveform-augmented structure — best and worst recording-level mean R²\n(recording summaries pooled across 20 recording-disjoint seeds)", fontsize=12, fontweight="bold", y=1.04)
    fig.supxlabel("Per-recording mean R²  (Waveform-augmented structure; 20 recording-disjoint seeds)")

    family_patches = [
        mpatches.Patch(color=v, label=k.replace("_", " ").title())
        for k, v in FAMILY_COLORS.items()
    ]
    family_patches.append(Line2D([0], [0], color=C_WINNER, linestyle="--", lw=1.2, label=f"Overall R²={overall_r2:.3f}"))
    fig.legend(handles=family_patches, fontsize=8, framealpha=0.9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    return fig


top_rec   = pd.read_csv(TABLE_DIR / "fpos_top_recordings.csv")
worst_rec = pd.read_csv(TABLE_DIR / "fpos_worst_recordings.csv")
fig = plot_winner_residuals(rec_r2, top_rec, worst_rec)
out = FIG_DIR / "fpos_winner_residuals.png"
fig.savefig(out, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved: {out}")


print("\nAll figures saved to:", FIG_DIR)
