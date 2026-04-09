"""
improve_figures_nb02.py
-----------------------
Produces improved figures for notebook 02 (domain shift analysis).
All figures saved to figures/02_domain_shift_hybrid_vs_paired/.

Run with:
    /opt/homebrew/Caskroom/miniconda/base/envs/spike-qc-local/bin/python \
        scripts/improve_figures_nb02.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures" / "02_domain_shift_hybrid_vs_paired"
TABLE_DIR = ROOT / "tables" / "02_domain_shift_hybrid_vs_paired"
FROZEN_DIR = ROOT / "data" / "frozen_inputs" / "domain_shift"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "font.family": "sans-serif",
})

# ---------------------------------------------------------------------------
# Domain-level colours (specified by task)
# ---------------------------------------------------------------------------
C_HYBRID = "#888888"
C_PAIRED = "#2E6DA4"

# 5-color qualitative palette for families (tab10 first 5)
import matplotlib.cm as cm
TAB10 = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
FAMILY_COLORS = {
    "BOYDEN":       TAB10[0],   # blue
    "CRCNS_HC1":    TAB10[1],   # orange
    "ENGLISH":      TAB10[2],   # green
    "KAMPFF":       TAB10[3],   # red
    "MEA64C_YGER":  TAB10[4],   # purple
}

# Feature-group color palette
GROUP_COLORS = {
    "waveform_bins": "#4C72B0",
    "amplitude":     "#DD8452",
    "confusability": "#55A868",
    "spikeinterface": "#C44E52",
    "acg_bins":      "#8172B2",
    "other":         "#937860",
    "recording_context": "#DA8BC3",
    "recording_relative": "#8C8C8C",
}


def _savefig(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
domain_summary = pd.read_csv(TABLE_DIR / "domain_classifier_summary.csv")
feature_group = pd.read_csv(TABLE_DIR / "feature_group_shift_summary.csv")
feature_shift = pd.read_csv(TABLE_DIR / "feature_shift_summary_top50.csv")
pca_paired = pd.read_csv(TABLE_DIR / "pca_2d_paired_points.csv")
centroids = pd.read_csv(TABLE_DIR / "paired_family_pca_centroids.csv")

# Frozen PCA data (contains both hybrid and paired, 1500 each)
pca_frozen = pd.read_csv(FROZEN_DIR / "pca_projection.csv")
pca_frozen_3d = pd.read_csv(FROZEN_DIR / "pca_projection_3d.csv")

print("Data loaded.")
print(f"  domain_summary: {domain_summary.shape}")
print(f"  feature_group: {feature_group.shape}")
print(f"  feature_shift: {feature_shift.shape}")
print(f"  pca_paired (paired-only, full dataset): {pca_paired.shape}")
print(f"  pca_frozen (balanced 1500+1500): {pca_frozen.shape}")
print(f"  pca_frozen_3d: {pca_frozen_3d.shape}")

# ---------------------------------------------------------------------------
# FIGURE 1: pca_2d_by_domain.png
# Improved PCA scatter: hybrid cloud (transparent) + paired points (solid)
# ---------------------------------------------------------------------------
print("\n[1/6] pca_2d_by_domain.png ...")

pca_df = pca_frozen.copy()
hybrid_pts = pca_df[pca_df["dataset_type"] == "hybrid"]
paired_pts = pca_df[pca_df["dataset_type"] == "paired"]

n_hybrid = len(hybrid_pts)
n_paired = len(paired_pts)

fig, ax = plt.subplots(figsize=(7, 5.5))

# Hybrid: dense cloud, very transparent
ax.scatter(
    hybrid_pts["pc1"], hybrid_pts["pc2"],
    s=8, alpha=0.15, color=C_HYBRID, linewidths=0,
    rasterized=True, zorder=1,
)
# Paired: solid, visible on top
ax.scatter(
    paired_pts["pc1"], paired_pts["pc2"],
    s=14, alpha=0.65, color=C_PAIRED, linewidths=0,
    rasterized=True, zorder=2,
)

# Legend with n= counts
legend_handles = [
    mpatches.Patch(color=C_HYBRID, alpha=0.55, label=f"Hybrid (n={n_hybrid:,})"),
    mpatches.Patch(color=C_PAIRED, label=f"Paired (n={n_paired:,})"),
]
ax.legend(handles=legend_handles, frameon=False, fontsize=9, loc="upper right")

ax.set_xlabel("PC1", fontsize=11)
ax.set_ylabel("PC2", fontsize=11)
ax.set_title("PCA projection: hybrid vs paired (unit features)", fontsize=13)
fig.tight_layout()
_savefig(fig, "pca_2d_by_domain")


# ---------------------------------------------------------------------------
# FIGURE 2: pca_2d_paired_by_family.png
# PCA of paired-only points colored by family, larger markers
# ---------------------------------------------------------------------------
print("[2/6] pca_2d_paired_by_family.png ...")


def _strip_prefix(name: str) -> str:
    return name.replace("PAIRED_", "")


pca_paired_df = pca_paired.copy()

fig, ax = plt.subplots(figsize=(7, 5.5))

families_sorted = sorted(pca_paired_df["study_set"].unique())
for study_set in families_sorted:
    short = _strip_prefix(study_set)
    color = FAMILY_COLORS.get(short, "#333333")
    pts = pca_paired_df[pca_paired_df["study_set"] == study_set]
    n = len(pts)
    ax.scatter(
        pts["pc1"], pts["pc2"],
        s=20, alpha=0.55, color=color, linewidths=0,
        rasterized=True, zorder=2,
        label=f"{short} (n={n:,})",
    )

# Add centroid markers
for _, row in centroids.iterrows():
    short = _strip_prefix(row["study_set"])
    color = FAMILY_COLORS.get(short, "#333333")
    ax.scatter(
        row["pc1_mean"], row["pc2_mean"],
        s=120, marker="*", color=color, edgecolors="white",
        linewidths=0.8, zorder=5,
    )

ax.set_xlabel("PC1", fontsize=11)
ax.set_ylabel("PC2", fontsize=11)
ax.set_title("PCA projection: paired units by family", fontsize=13)
ax.legend(frameon=False, fontsize=9, loc="upper right",
          title="Family", title_fontsize=9)
fig.tight_layout()
_savefig(fig, "pca_2d_paired_by_family")


# ---------------------------------------------------------------------------
# FIGURE 3: pca_3d_by_domain.png
# Regenerated 3D PCA with proper axis labels and alpha
# ---------------------------------------------------------------------------
print("[3/6] pca_3d_by_domain.png ...")

pca3d = pca_frozen_3d.copy()
hybrid_3d = pca3d[pca3d["dataset_type"] == "hybrid"]
paired_3d = pca3d[pca3d["dataset_type"] == "paired"]

fig = plt.figure(figsize=(8, 6), facecolor="white")
ax3d = fig.add_subplot(111, projection="3d")
ax3d.set_facecolor("white")

ax3d.scatter(
    hybrid_3d["pc1"], hybrid_3d["pc2"], hybrid_3d["pc3"],
    s=6, alpha=0.12, color=C_HYBRID, linewidths=0, rasterized=True,
    label=f"Hybrid (n={len(hybrid_3d):,})",
)
ax3d.scatter(
    paired_3d["pc1"], paired_3d["pc2"], paired_3d["pc3"],
    s=12, alpha=0.55, color=C_PAIRED, linewidths=0, rasterized=True,
    label=f"Paired (n={len(paired_3d):,})",
)

ax3d.set_xlabel("PC1", fontsize=10, labelpad=6)
ax3d.set_ylabel("PC2", fontsize=10, labelpad=6)
ax3d.set_zlabel("PC3", fontsize=10, labelpad=6)  # type: ignore[attr-defined]
ax3d.set_title("3D PCA: hybrid vs paired (unit features)", fontsize=12)
ax3d.tick_params(axis="both", labelsize=8)
ax3d.legend(frameon=False, fontsize=9, loc="upper left")

fig.tight_layout()
_savefig(fig, "pca_3d_by_domain")


# ---------------------------------------------------------------------------
# FIGURE 4: domain_separability.png
# Clean bar chart with AUC annotations and danger zone
# ---------------------------------------------------------------------------
print("[4/6] domain_separability.png ...")

# Use only the hybrid_vs_paired_all comparison (the main finding)
dc_main = domain_summary[domain_summary["comparison"] == "hybrid_vs_paired_all"].copy()
dc_main = dc_main.sort_values("cv_auc_mean")

# Human-readable labels
label_map = {
    "unit":         "Unit features only\n(164 features)",
    "full_context": "Full context\n(685 features)",
}
dc_main["label"] = dc_main["feature_set"].map(label_map).fillna(dc_main["feature_set"])

SLATE = "#5b6675"
RED = "#b33a3a"

fig, ax = plt.subplots(figsize=(7, 3.5))

bar_colors = [C_PAIRED if v < 0.99 else RED for v in dc_main["cv_auc_mean"]]
bars = ax.bar(
    dc_main["label"], dc_main["cv_auc_mean"],
    color=bar_colors, edgecolor="white", width=0.45,
)

# Reference lines
ax.axhline(0.5,  linestyle=":",  linewidth=1.2, color=SLATE,    alpha=0.7, label="Random (0.50)")
ax.axhline(0.75, linestyle="--", linewidth=1.2, color="#d98e04", alpha=0.8, label="Strong (0.75)")
ax.axhline(0.90, linestyle="-",  linewidth=1.4, color=RED,       alpha=0.9, label="Very strong / transfer-unsafe (0.90)")

# Danger zone
ax.axhspan(0.90, 1.025, alpha=0.07, color=RED)
ax.text(1.38, 0.96, "Transfer\nunsafe zone", color=RED, ha="right",
        fontsize=8, style="italic", va="center")

# Value annotations on bars
for bar, (_, row) in zip(bars, dc_main.iterrows()):
    v = row["cv_auc_mean"]
    std = row.get("cv_auc_std", 0.0)
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v + 0.005,
        f"AUC = {v:.4f}",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold",
    )
    if std > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v - 0.025,
            f"±{std:.4f}",
            ha="center", va="top", fontsize=8, color="white", alpha=0.9,
        )

ax.set_ylim(0.0, 1.06)
ax.set_ylabel("5-fold CV AUC", fontsize=11)
ax.set_title("Domain separability: hybrid vs paired", fontsize=13)
ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

# Remove gridlines on x-axis for cleaner look
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
ax.tick_params(axis="x", which="both", bottom=False)

fig.tight_layout()
_savefig(fig, "domain_separability")


# ---------------------------------------------------------------------------
# FIGURE 5: top_feature_shift.png
# Horizontal bar chart, top 20 features, colored by feature group
# Exclude meta-features (domain_label, is_hybrid_labeled, is_paired_unlabeled)
# ---------------------------------------------------------------------------
print("[5/6] top_feature_shift.png ...")

SKIP_FEATURES = {"domain_label", "is_hybrid_labeled", "is_paired_unlabeled"}
plot_shift = (
    feature_shift[~feature_shift["feature"].isin(SKIP_FEATURES)]
    .sort_values("abs_smd", ascending=False)
    .head(20)
    .iloc[::-1]   # reverse for horizontal bar (best at top)
    .reset_index(drop=True)
)

fig, ax = plt.subplots(figsize=(9, max(5, 0.38 * len(plot_shift))))

bar_colors_shift = [
    GROUP_COLORS.get(g, "#888888") for g in plot_shift["feature_group"]
]
bars = ax.barh(
    plot_shift["feature"], plot_shift["abs_smd"],
    color=bar_colors_shift, edgecolor="white", linewidth=0.4,
)

# Value labels at end of bars
for bar, val in zip(bars, plot_shift["abs_smd"]):
    ax.text(
        val + 0.01, bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}", va="center", ha="left", fontsize=7.5, color="#333333",
    )

ax.set_xlabel("Absolute standardized mean difference (|SMD|)", fontsize=11)
ax.set_title("Top 20 shifted features: hybrid vs paired", fontsize=13)
ax.set_xlim(0, plot_shift["abs_smd"].max() * 1.18)

# Group legend (unique groups in this top-20)
seen_groups = plot_shift["feature_group"].unique()
legend_patches = [
    mpatches.Patch(color=GROUP_COLORS.get(g, "#888888"), label=g.replace("_", " "))
    for g in sorted(seen_groups)
]
ax.legend(handles=legend_patches, frameon=False, fontsize=8,
          loc="lower right", title="Feature group", title_fontsize=8)

fig.tight_layout()
_savefig(fig, "top_feature_shift")


# ---------------------------------------------------------------------------
# FIGURE 6 (NEW): feature_group_shift_summary.png
# Grouped bar chart: mean |SMD| by feature group for the "unit" feature set
# comparing hybrid_vs_paired_all (main finding)
# ---------------------------------------------------------------------------
print("[6/6] feature_group_shift_summary.png (NEW) ...")

# Filter to the main comparison, unit feature set only
# (this is the cleanest view for the thesis narrative)
unit_groups = (
    feature_group[
        (feature_group["comparison"] == "hybrid_vs_paired_all") &
        (feature_group["feature_set"] == "unit")
    ]
    .copy()
)

# Sort descending by mean_abs_smd
unit_groups = unit_groups.sort_values("mean_abs_smd", ascending=False).reset_index(drop=True)

# For each group, we show mean_abs_smd and max_abs_smd side by side
fig, ax = plt.subplots(figsize=(9, max(4, 0.55 * len(unit_groups))))

y_pos = np.arange(len(unit_groups))
bar_h = 0.35

bar_colors_groups = [
    GROUP_COLORS.get(g, "#888888") for g in unit_groups["feature_group"]
]

# Mean |SMD|
bars_mean = ax.barh(
    y_pos + bar_h / 2,
    unit_groups["mean_abs_smd"],
    height=bar_h,
    color=bar_colors_groups, edgecolor="white", linewidth=0.4,
    label="Mean |SMD|",
)
# Max |SMD| (lighter, outlines only)
bars_max = ax.barh(
    y_pos - bar_h / 2,
    unit_groups["max_abs_smd"],
    height=bar_h,
    color=bar_colors_groups, edgecolor="white", linewidth=0.4,
    alpha=0.35,
    label="Max |SMD|",
)

# Value labels on mean bars
for bar, val in zip(bars_mean, unit_groups["mean_abs_smd"]):
    ax.text(
        val + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}",
        va="center", ha="left", fontsize=7.5,
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(
    [g.replace("_", " ") for g in unit_groups["feature_group"]],
    fontsize=9,
)
ax.set_xlabel("Standardized Mean Difference (|SMD|)", fontsize=11)
ax.set_title(
    "Feature-group domain shift: hybrid vs paired (unit features)",
    fontsize=13,
)
ax.set_xlim(0, unit_groups["max_abs_smd"].max() * 1.2)

# Inset: n_features annotation
for i, (_, row) in enumerate(unit_groups.iterrows()):
    ax.text(
        unit_groups["max_abs_smd"].max() * 1.18,
        i,
        f"n={int(row['n_features'])}",
        va="center", ha="right", fontsize=7, color="#666666",
    )

# Legend
legend_handles_grp = [
    mpatches.Patch(facecolor="#555555", alpha=1.0, label="Mean |SMD| (solid)"),
    mpatches.Patch(facecolor="#555555", alpha=0.35, label="Max |SMD| (faded)"),
]
ax.legend(handles=legend_handles_grp, frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()
_savefig(fig, "feature_group_shift_summary")


print("\nAll 6 figures saved to:")
print(f"  {FIG_DIR}")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"    {f.name}")
