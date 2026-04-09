"""
Improved figures for notebook 05: fmiss model progression.

Figures produced:
  figures/05_fmiss_model_progression/fmiss_benchmark_r2.png
  figures/05_fmiss_model_progression/fmiss_benchmark_mae.png
  figures/05_fmiss_model_progression/fmiss_winner_prediction_scatter.png
  figures/05_fmiss_model_progression/fmiss_family_r2_by_wave.png
  figures/05_fmiss_model_progression/fmiss_seed_stability.png
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
TABLES = ROOT / "tables" / "05_fmiss_model_progression"
FIGURES = ROOT / "figures" / "05_fmiss_model_progression"
ARTIFACTS = ROOT / "artifacts"
FIGURES.mkdir(parents=True, exist_ok=True)

from qc_thesis.plots import BENCHMARK_MODEL_COLORS

# ── colour palette ─────────────────────────────────────────────────────────
FMISS_ORANGE   = "#D4691E"
BASELINE_GREY  = BENCHMARK_MODEL_COLORS["Dummy paired mean"]
NEURAL_RED     = BENCHMARK_MODEL_COLORS["Best SSL neural"]
TRANSFER_ORANGE = BENCHMARK_MODEL_COLORS["Hybrid-only XGBoost"]
CONTEXT_BLUE  = BENCHMARK_MODEL_COLORS["Paired-only context"]
WINNER_GREEN   = BENCHMARK_MODEL_COLORS["Reduced-latent context"]
ZERO_LINE      = "#333333"
SLATE          = "#5b6675"
R2_CLIP_MIN    = -1.05

# ── style ──────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ── shared display labels ──────────────────────────────────────────────────
# Keep retained fmiss figure labels aligned with the manuscript naming.
DISPLAY_LABEL = {
    "Dummy paired mean":                   "Dummy paired mean",
    "Paired-only raw":                     "Paired-only raw",
    "Hybrid-only XGBoost":                 "Hybrid-only XGBoost",
    "Calibrated hybrid":                   "Calibrated hybrid",
    "Hybrid stack residual":               "Hybrid stack residual",
    "Best SSL neural":                     "Best SSL neural",
    "Best MMD neural":                     "Best MMD neural",
    "Paired-only context":                 "Paired-only context",
    "Paired context":                      "Paired-only context",
    "Reduced-latent context":              "Reduced-latent context",
    "Reduced latent":                      "Reduced-latent context",
}

def _short(label: str) -> str:
    return DISPLAY_LABEL.get(label, label)


def _approach_color(recipe_id: str, label: str) -> str:
    if str(label) in BENCHMARK_MODEL_COLORS:
        return BENCHMARK_MODEL_COLORS[str(label)]
    rid = str(recipe_id).lower()
    if rid == "fmiss_ssl" or rid.endswith("_ssl"):
        return NEURAL_RED
    if rid == "fmiss_mmd" or rid.endswith("_mmd") or "neural_mmd" in rid:
        return NEURAL_RED
    if "reduced_latent" in rid:
        return WINNER_GREEN
    if "paired_context" in rid:
        return CONTEXT_BLUE
    return TRANSFER_ORANGE


# ═══════════════════════════════════════════════════════════════════════════
# 1. fmiss_benchmark_r2.png
# ═══════════════════════════════════════════════════════════════════════════
def plot_benchmark_r2():
    df = pd.read_csv(TABLES / "fmiss_benchmark_ladder.csv")
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    # Keep original research-stage order
    labels   = [_short(l) for l in df["label"]]
    values   = df["r2"].tolist()
    colors   = [_approach_color(r, l) for r, l in zip(df["recipe_id"], df["label"])]
    display_values = np.clip(values, R2_CLIP_MIN, None)

    fig, ax = plt.subplots(figsize=(10.5, max(4.7, 0.58 * len(df))))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, display_values, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    # R²=0 reference
    ax.axvline(0, color=ZERO_LINE, linewidth=1.0, linestyle="--", zorder=3)
    ax.text(0.01, 0.03, "R² = 0", transform=ax.get_xaxis_transform(),
            color=ZERO_LINE, fontsize=8, ha="left", va="bottom", style="italic")

    for i, (bar, val, shown) in enumerate(zip(bars, values, display_values)):
        if val < R2_CLIP_MIN:
            bar.set_hatch("///")
            bar.set_alpha(0.9)
            ax.annotate(
                f"{val:.3f}",
                xy=(R2_CLIP_MIN, i),
                xytext=(R2_CLIP_MIN + 0.10, i),
                va="center",
                ha="left",
                fontsize=8,
                color="#333333",
                arrowprops=dict(arrowstyle="-|>", color="#666666", lw=0.8),
            )
        else:
            x_offset = 0.010 if val >= 0 else -0.012
            ha = "left" if val >= 0 else "right"
            ax.text(val + x_offset, i, f"{val:.3f}", va="center", ha=ha, fontsize=8)

    ax.text(
        0.02,
        0.96,
        "Hatched bars are clipped on the left for display.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#666666",
        ha="left",
        va="top",
    )

    ax.set_xlim(R2_CLIP_MIN - 0.08, 0.40)
    ax.set_xlabel("R²  (20 recording-disjoint seeds; held-out recordings)", fontsize=10)
    ax.set_title("fmiss core progression on the held-out recordings\n(20 recording-disjoint seeds, 42-61)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out = FIGURES / "fmiss_benchmark_r2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. fmiss_benchmark_mae.png
# ═══════════════════════════════════════════════════════════════════════════
def plot_benchmark_mae():
    df = pd.read_csv(TABLES / "fmiss_benchmark_ladder.csv")
    df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
    # Sort ascending for MAE (lower is better at bottom)
    df = df.sort_values("mae", ascending=False).reset_index(drop=True)
    labels  = [_short(l) for l in df["label"]]
    values  = df["mae"].tolist()
    colors  = [_approach_color(r, l) for r, l in zip(df["recipe_id"], df["label"])]

    fig, ax = plt.subplots(figsize=(10.5, max(4.7, 0.58 * len(df))))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    # Dummy baseline reference
    dummy_val = float(df.loc[df["recipe_id"] == "fmiss_dummy", "mae"].iloc[0])
    ax.axvline(dummy_val, color=SLATE, linewidth=1.2, linestyle="--",
               label=f"Dummy baseline ({dummy_val:.3f})")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)

    ax.text(dummy_val + 0.004, 0.02, f"Dummy {dummy_val:.3f}",
            transform=ax.get_xaxis_transform(), fontsize=7.5, color=SLATE, ha="left", va="bottom")

    ax.set_xlim(0, 0.46)
    ax.set_xlabel("MAE  (20 recording-disjoint seeds; held-out recordings)", fontsize=10)
    ax.set_title("fmiss core holdout MAE: reduced-latent versus the retained comparisons\n(20 recording-disjoint seeds, 42-61)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "fmiss_benchmark_mae.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. fmiss_winner_prediction_scatter.png
# ═══════════════════════════════════════════════════════════════════════════
def plot_winner_scatter():
    pred_path = (ARTIFACTS / "runs" / "recording_disjoint_main" / "fmiss"
                 / "fmiss_reduced_latent" / "predictions.csv")
    if not pred_path.exists():
        print(f"  WARNING: {pred_path} not found, skipping scatter plot")
        return

    df = pd.read_csv(pred_path)
    df["true"] = pd.to_numeric(df["true"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df = df.dropna(subset=["true", "pred"])

    from sklearn.metrics import r2_score, mean_absolute_error
    r2  = r2_score(df["true"], df["pred"])
    mae = mean_absolute_error(df["true"], df["pred"])

    # Family color map
    families = sorted(df["study_set"].unique()) if "study_set" in df.columns else []
    fam_palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9467BD"]
    fam_color = {f: fam_palette[i % len(fam_palette)] for i, f in enumerate(families)}

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    if families:
        for fam in families:
            sub = df[df["study_set"] == fam]
            short_fam = fam.replace("PAIRED_", "")
            ax.scatter(sub["true"], sub["pred"], s=22, alpha=0.65,
                       color=fam_color[fam], label=short_fam, zorder=3)
        ax.legend(fontsize=8, framealpha=0.85, title="Family", title_fontsize=8,
                  loc="upper left")
    else:
        ax.scatter(df["true"], df["pred"], s=22, alpha=0.65, color=FMISS_ORANGE, zorder=3)

    lo = min(df["true"].min(), df["pred"].min()) - 0.02
    hi = max(df["true"].max(), df["pred"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=SLATE, linewidth=1.2, zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True fmiss", fontsize=10)
    ax.set_ylabel("Predicted fmiss", fontsize=10)
    ax.set_title("Reduced-latent context: observed vs predicted\n(pooled holdout predictions from 20 recording-disjoint seeds)", fontsize=12)

    ax.text(0.97, 0.05, f"R² = {r2:.3f}\nMAE = {mae:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"))

    fig.tight_layout()
    out = FIGURES / "fmiss_winner_prediction_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. fmiss_family_r2_by_wave.png
# ═══════════════════════════════════════════════════════════════════════════
def plot_family_r2_heatmap():
    pivot_path = TABLES / "fmiss_family_r2_pivot.csv"
    if not pivot_path.exists():
        print(f"  WARNING: {pivot_path} not found, skipping heatmap")
        return

    df = pd.read_csv(pivot_path)
    # First column is study_set
    group_col = df.columns[0]
    df = df.set_index(group_col)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Shorten column names, then keep only retained core contextual models.
    df.columns = [DISPLAY_LABEL.get(c, c) for c in df.columns]
    keep_cols = [col for col in ["Paired-only context", "Reduced-latent context"] if col in df.columns]
    df = df[keep_cols]

    # Shorten row index
    df.index = [str(i).replace("PAIRED_", "") for i in df.index]

    # Diverging colormap: red=negative, white=0, green=positive
    # Clamp vmin/vmax to avoid PAIRED_KAMPFF outlier dominating
    vmin, vmax = -1.5, 0.9
    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(df.columns)), max(4, 0.65 * len(df))))
    im = ax.imshow(df.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.8, label="R²  (per-family holdout)", pad=0.02)

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns.tolist(), rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index.tolist(), fontsize=9)

    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            val = df.values[r, c]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.8 * max(abs(vmin), abs(vmax)) else "black"
                # Clamp display for extreme negative KAMPFF values
                display_val = max(val, vmin)
                ax.text(c, r, f"{display_val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=text_color,
                        fontweight="bold" if val < -0.5 or val > 0.5 else "normal")

    ax.set_title("fmiss mean R² by paired family across retained core contextual models\n"
                 "(paired-family summaries from 20 recording-disjoint seeds)", fontsize=12)

    fig.tight_layout()
    out = FIGURES / "fmiss_family_r2_by_wave.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# 4b. fmiss_lofo_family_heatmap.png
# ═══════════════════════════════════════════════════════════════════════════
def plot_lofo_family_heatmap():
    pivot_path = TABLES / "fmiss_lofo_family_preference_r2.csv"
    if not pivot_path.exists():
        print(f"  WARNING: {pivot_path} not found, skipping LOFO heatmap")
        return

    df = pd.read_csv(pivot_path)
    group_col = df.columns[0]
    df = df.set_index(group_col)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = [str(i).replace("PAIRED_", "") for i in df.index]

    # Match the fpos LOFO heatmap logic: diverging red-yellow-green scale centered at R² = 0.
    vmin, vmax = -1.10, 0.85
    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(df.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.grid(False)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns.tolist(), rotation=24, ha="right")
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index.tolist())
    ax.set_title("Mean held-out-family `R²` across the main compared `fmiss` models\n(20-seed leave-one-family-out aggregate)", pad=10)

    for row_idx, family_name in enumerate(df.index):
        for col_idx, label in enumerate(df.columns):
            value = df.loc[family_name, label]
            if pd.notna(value):
                text_color = "white" if abs(value) >= 0.72 else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean R²")
    fig.tight_layout()
    out = FIGURES / "fmiss_lofo_family_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# 5. fmiss_seed_stability.png
# ═══════════════════════════════════════════════════════════════════════════
def plot_seed_stability():
    """Violin plots of R² and MAE across seeds for the retained compared fmiss recipes."""
    recipes = {
        "Paired-only raw":        "fmiss_paired_raw",
        "Hybrid-only XGBoost":    "fmiss_hybrid_only",
        "Paired-only context":    "fmiss_paired_context",
        "Reduced-latent context": "fmiss_reduced_latent",
    }

    rows = []
    for label, recipe in recipes.items():
        mfile = ARTIFACTS / "runs_aggregate" / "recording_disjoint_main" / "fmiss" / recipe / "seed_metrics.csv"
        if not mfile.exists():
            continue
        try:
            m = pd.read_csv(mfile)
        except Exception:
            continue
        m["seed"] = pd.to_numeric(m["seed"], errors="coerce")
        m["r2"] = pd.to_numeric(m["r2"], errors="coerce")
        m["mae"] = pd.to_numeric(m["mae"], errors="coerce")
        m = m.dropna(subset=["seed", "r2", "mae"])
        m = m[m["seed"].between(42, 61)].copy()
        for _, row in m.iterrows():
            rows.append({
                "seed": int(row["seed"]),
                "model": label,
                "r2": float(row["r2"]),
                "mae": float(row["mae"]),
            })

    if not rows:
        print("  WARNING: no seed data found for fmiss stability plot, skipping")
        return

    df = pd.DataFrame(rows)
    model_order = list(recipes.keys())
    colors = [BENCHMARK_MODEL_COLORS.get(label, WINNER_GREEN) for label in model_order]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.6), sharey=True)

    for ax, metric, ylabel, title in zip(
        axes,
        ["r2", "mae"],
        ["R²", "MAE"],
        [
            "R² across the retained compared `fmiss` models\n(20 recording-disjoint seeds, 42-61)",
            "MAE across the retained compared `fmiss` models\n(20 recording-disjoint seeds, 42-61)",
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

        if metric == "r2":
            ax.axvline(0, color="#444444", linewidth=0.9, linestyle="--", zorder=1)
        ax.set_yticks(positions)
        ax.set_yticklabels(model_order, fontsize=8.5)
        ax.set_xlabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)

    fig.suptitle(
        "fmiss: seed stability across the retained compared models\n(recording-disjoint aggregate, seeds 42-61)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    out = FIGURES / "fmiss_seed_stability.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating improved figures for notebook 05 (fmiss)...")
    plot_benchmark_r2()
    plot_benchmark_mae()
    plot_winner_scatter()
    plot_family_r2_heatmap()
    plot_lofo_family_heatmap()
    plot_seed_stability()
    print("Done.")
