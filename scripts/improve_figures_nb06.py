"""Improved figures for notebook 06: fpos vs fmiss XAI and target asymmetry.

Generates / replaces:
  - fpos_shap_top.png          : top-20 SHAP features, colored by feature group, with legend
  - fpos_shap_group_summary.png: feature-group aggregate importance bar chart (NEW)
  - fpos_family_behavior.png   : per-family fpos R², family colors + value labels
  - fmiss_family_behavior.png  : per-family fmiss R², family colors + value labels
  - target_asymmetry_combined.png: improved side-by-side barh with family colors
  - fpos_fmiss_family_r2_scatter.png: scatter x=fpos R², y=fmiss R², per family (NEW)
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
REPO = Path(__file__).resolve().parent.parent
TABLE_DIR = REPO / "tables" / "06_fpos_vs_fmiss_xai"
FIG_DIR   = REPO / "figures" / "06_fpos_vs_fmiss_xai"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.7,
})

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
FPOS_COLOR  = "#2E6DA4"   # blue
FMISS_COLOR = "#D4691E"   # orange

FAMILY_COLORS = {
    "BOYDEN":      "#E64B35",
    "CRCNS_HC1":   "#4DBBD5",
    "ENGLISH":     "#00A087",
    "KAMPFF":      "#3C5488",
    "MEA64C_YGER": "#F39B7F",
}

# Feature group colors (consistent across all plots)
GROUP_COLORS = {
    "recording_context": "#1B3A6B",   # navy
    "spikeinterface_qc": "#2D8B57",   # green
    "waveform":          "#1F7A8C",   # teal
    "source_transfer":   "#7B3F9E",   # purple-ish (source_pred is its own thing)
    "acg":               "#6A3D9A",   # purple
    "amplitude":         "#E05C5C",   # coral
    "other":             "#888888",   # grey
}

GROUP_LABELS = {
    "recording_context": "Recording context",
    "spikeinterface_qc": "SpikeInterface QC (quality metrics)",
    "waveform":          "Waveform",
    "source_transfer":   "Source transfer",
    "acg":               "ACG",
    "amplitude":         "Amplitude",
    "other":             "Other",
}

FMISS_DISPLAY_MIN = -1.2
FMISS_DISPLAY_MAX = 0.95

# ---------------------------------------------------------------------------
# Helper: assign feature group from feature name
# ---------------------------------------------------------------------------
def _assign_group(feat: str) -> str:
    """Heuristic group assignment matching the group_summary CSV categories."""
    f = feat.lower()
    # source_pred is its own source_transfer group
    if f == "source_pred":
        return "source_transfer"
    # Recording-context features have _recording_ in the name
    if "_recording_" in f:
        return "recording_context"
    # SpikeInterface QC features (quality metrics) start with si_
    if f.startswith("si_"):
        return "spikeinterface_qc"
    # Waveform features
    if f.startswith("wf_") or f in (
        "peak_to_trough_uv", "post_trough_peak_uv", "pre_trough_peak_uv",
        "trough_time_ms", "half_width_ms", "repolarization_slope",
        "wf_pre_trough_slope", "wf_post_peak_slope", "wf_prepeak_ratio",
        "wf_prepeak_to_trough_ms", "wf_rebound_ratio", "wf_asymmetry",
        "wf_trough_width_75_ms", "wf_trough_width_50_ms", "wf_trough_width_25_ms",
        "wf_zero_cross_pre_ms", "wf_pos_area_uv_ms", "wf_neg_area_uv_ms",
        "wf_pos_neg_area_ratio", "wf_energy",
        "spread_um", "spread_weighted_um", "center_of_mass_um",
        "n_active_channels", "n_channels", "peak_channel_depth_um",
    ):
        return "waveform"
    # ACG features
    if f.startswith("acg_") or f in ("burst_index",):
        return "acg"
    # ISI / firing-rate features → bucket into ISI / other
    if f.startswith("isi_") or f.startswith("firing_rate") or f in (
        "presence_ratio", "isi_cv", "isi_skew", "isi_std_ms",
        "isi_mean_ms", "isi_violation_rate",
    ):
        return "other"
    # Amplitude features
    if f.startswith("amp_") or f in ("snr_recording_pct", "snr_recording_delta", "snr_recording_std",):
        return "amplitude"
    # Confusability / cosine / neighbour → recording context (unit-level context)
    if any(k in f for k in ("cosine", "confusable", "neighbor", "template_norm",
                             "anchor", "matched_support", "isolation", "max_cosine",
                             "mean_cosine", "median_cosine", "std_cosine", "mean_waveform",
                             "median_waveform", "std_waveform", "max_waveform")):
        return "other"
    return "other"


def _strip(label: str) -> str:
    return str(label).replace("PAIRED_", "")


def _clip_fmiss(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), FMISS_DISPLAY_MIN, FMISS_DISPLAY_MAX)


def _annotate_clipped_barh(ax: plt.Axes, bar, actual: float, display: float, *, pad: float) -> None:
    y = bar.get_y() + bar.get_height() / 2
    if actual < FMISS_DISPLAY_MIN:
        ax.annotate(
            f"{actual:.2f}",
            xy=(FMISS_DISPLAY_MIN, y),
            xytext=(FMISS_DISPLAY_MIN + 0.14, y),
            va="center",
            ha="left",
            fontsize=8.5,
            color="#333333",
            arrowprops=dict(arrowstyle="-|>", color="#666666", lw=0.8),
        )
    elif actual > FMISS_DISPLAY_MAX:
        ax.annotate(
            f"{actual:.2f}",
            xy=(FMISS_DISPLAY_MAX, y),
            xytext=(FMISS_DISPLAY_MAX - 0.14, y),
            va="center",
            ha="right",
            fontsize=8.5,
            color="#333333",
            arrowprops=dict(arrowstyle="-|>", color="#666666", lw=0.8),
        )
    else:
        xpos = display + pad if actual >= 0 else display - pad
        ha = "left" if actual >= 0 else "right"
        ax.text(xpos, y, f"{actual:.2f}", va="center", ha=ha, fontsize=8.5, color="#222222")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
shap_df    = pd.read_csv(TABLE_DIR / "fpos_shap_importance.csv")
group_df   = pd.read_csv(TABLE_DIR / "fpos_shap_group_summary.csv")
family_cmp = pd.read_csv(TABLE_DIR / "fpos_vs_fmiss_family_comparison.csv")
fpos_fam   = pd.read_csv(TABLE_DIR / "fpos_family_summary.csv")
fmiss_fam  = pd.read_csv(TABLE_DIR / "fmiss_family_summary.csv")

# ---------------------------------------------------------------------------
# Figure 1: fpos_shap_top.png — top-20 features colored by group
# ---------------------------------------------------------------------------
TOP_N = 14
plot_shap = shap_df.head(TOP_N).copy()
plot_shap["group"] = plot_shap["feature"].apply(_assign_group)
# Reverse so best feature is at top
plot_shap = plot_shap.iloc[::-1].reset_index(drop=True)

bar_colors = [GROUP_COLORS[g] for g in plot_shap["group"]]
xerr_top = plot_shap["std_abs_shap"] if "std_abs_shap" in plot_shap.columns else None
max_top_x = float((plot_shap["mean_abs_shap"] + (plot_shap["std_abs_shap"] if "std_abs_shap" in plot_shap.columns else 0.0)).max())
label_pad_top = max(max_top_x * 0.015, 0.004)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(
    plot_shap["feature"],
    plot_shap["mean_abs_shap"],
    xerr=xerr_top,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.4,
    error_kw={"ecolor": "#444444", "elinewidth": 0.8, "capsize": 2.0},
)

# Value labels on bars
for idx, (bar, val) in enumerate(zip(bars, plot_shap["mean_abs_shap"])):
    err = float(xerr_top.iloc[idx]) if xerr_top is not None else 0.0
    ax.text(
        val + err + label_pad_top,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center", ha="left", fontsize=7.5, color="#444444",
    )

ax.set_xlabel("Mean |SHAP| across LOFO runs", fontsize=11)
ax.set_title(
    "Top SHAP features across LOFO runs\nWaveform-augmented stack",
    fontsize=13,
    fontweight="bold",
    pad=10,
)
ax.set_xlim(right=max_top_x + (4.5 * label_pad_top))

# Legend for feature groups — only show groups actually present
present_groups = plot_shap["group"].unique()
legend_patches = [
    mpatches.Patch(facecolor=GROUP_COLORS[g], label=GROUP_LABELS[g])
    for g in ["recording_context", "spikeinterface_qc", "amplitude",
              "waveform", "acg", "source_transfer", "other"]
    if g in present_groups
]
ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9,
          title="Feature group", title_fontsize=8.5)

fig.tight_layout()
out = FIG_DIR / "fpos_shap_top.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Figure 2: fpos_shap_group_summary.png — aggregate group importance
# ---------------------------------------------------------------------------
# Sort by mean_abs_shap descending, then reverse for horizontal bar (best at top)
group_plot = group_df.sort_values("mean_abs_shap", ascending=True).copy()
group_plot["label"] = group_plot["feature_group"].map(GROUP_LABELS).fillna(group_plot["feature_group"])
bar_colors_g = [GROUP_COLORS.get(g, "#888888") for g in group_plot["feature_group"]]
xerr_group = group_plot["std_abs_shap"] if "std_abs_shap" in group_plot.columns else None
max_group_x = float((group_plot["mean_abs_shap"] + (group_plot["std_abs_shap"] if "std_abs_shap" in group_plot.columns else 0.0)).max())
label_pad_group = max(max_group_x * 0.015, 0.01)

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(
    group_plot["label"],
    group_plot["mean_abs_shap"],
    xerr=xerr_group,
    color=bar_colors_g,
    edgecolor="white",
    linewidth=0.4,
    error_kw={"ecolor": "#444444", "elinewidth": 0.8, "capsize": 2.0},
)

# Value labels
for idx, (bar, val) in enumerate(zip(bars, group_plot["mean_abs_shap"])):
    err = float(xerr_group.iloc[idx]) if xerr_group is not None else 0.0
    ax.text(
        val + err + label_pad_group, bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center", ha="left", fontsize=9, color="#333333",
    )

# Feature count annotations inside bars
for bar, (_, row) in zip(bars, group_plot.iterrows()):
    if "feature_count" in row.index:
        cnt_label = f"n={int(row['feature_count'])}"
    elif "mean_feature_count" in row.index:
        cnt_label = f"n≈{int(round(float(row['mean_feature_count'])))}"
    else:
        cnt_label = None
    if cnt_label is None:
        continue
    ax.text(
        0.001, bar.get_y() + bar.get_height() / 2,
        cnt_label,
        va="center", ha="left", fontsize=7.5, color="white", fontweight="bold",
    )

ax.set_xlabel("Aggregate mean |SHAP| across LOFO runs", fontsize=11)
ax.set_title(
    "Feature group importance across LOFO runs\nWaveform-augmented stack",
    fontsize=13,
    fontweight="bold",
    pad=10,
)
ax.set_xlim(right=max_group_x + (4.5 * label_pad_group))

fig.tight_layout()
out = FIG_DIR / "fpos_shap_group_summary.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Figure 3: fpos_family_behavior.png — per-family fpos R² with family colors
# ---------------------------------------------------------------------------
fpos_fam_plot = fpos_fam.copy()
fpos_fam_plot["family"] = fpos_fam_plot["study_set"].apply(_strip)
# Sort by R² ascending so best is at top in barh
fpos_fam_plot = fpos_fam_plot.sort_values("r2", ascending=True).reset_index(drop=True)
colors_fp = [FAMILY_COLORS.get(f, "#888888") for f in fpos_fam_plot["family"]]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(
    fpos_fam_plot["family"],
    fpos_fam_plot["r2"],
    color=colors_fp,
    edgecolor="white",
    linewidth=0.4,
)

# Value labels
for bar, val in zip(bars, fpos_fam_plot["r2"]):
    xpos = val + 0.005 if val >= 0 else val - 0.005
    ha = "left" if val >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha=ha, fontsize=9, color="#222222")

ax.axvline(0, color="#aaaaaa", linewidth=0.8)
ax.set_xlabel("R²  (recording-disjoint holdout)", fontsize=11)
ax.set_title("fpos prediction: per-family R²\n(Waveform-augmented stack model)", fontsize=12, fontweight="bold", pad=8)
ax.set_xlim(left=-0.1, right=fpos_fam_plot["r2"].max() * 1.22)

# Family color legend
legend_patches = [
    mpatches.Patch(facecolor=FAMILY_COLORS[f], label=f)
    for f in FAMILY_COLORS
]
ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9,
          title="Family", title_fontsize=8.5, fontsize=8)

fig.tight_layout()
out = FIG_DIR / "fpos_family_behavior.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Figure 4: fmiss_family_behavior.png — per-family fmiss R² with family colors
# ---------------------------------------------------------------------------
fmiss_fam_plot = fmiss_fam.copy()
fmiss_fam_plot["family"] = fmiss_fam_plot["study_set"].apply(_strip)
fmiss_fam_plot = fmiss_fam_plot.sort_values("r2", ascending=True).reset_index(drop=True)
colors_fm = [FAMILY_COLORS.get(f, "#888888") for f in fmiss_fam_plot["family"]]
fmiss_fam_plot["display_r2"] = _clip_fmiss(fmiss_fam_plot["r2"])

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(
    fmiss_fam_plot["family"],
    fmiss_fam_plot["display_r2"],
    color=colors_fm,
    edgecolor="white",
    linewidth=0.4,
)

for bar, actual, display in zip(bars, fmiss_fam_plot["r2"], fmiss_fam_plot["display_r2"]):
    if actual != display:
        bar.set_hatch("///")
        bar.set_alpha(0.9)
    _annotate_clipped_barh(ax, bar, float(actual), float(display), pad=0.03)

ax.axvline(0, color="#888888", linewidth=1.0, linestyle="--")
ax.set_xlabel("R²  (recording-disjoint holdout)", fontsize=11)
ax.set_title("fmiss prediction: per-family R²\n(Reduced-latent context model)", fontsize=12, fontweight="bold", pad=8)
ax.set_xlim(FMISS_DISPLAY_MIN - 0.1, FMISS_DISPLAY_MAX + 0.08)
ax.text(
    0.02,
    0.03,
    "Left tail clipped at -1.2 for display; hatch marks indicate off-scale values.",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=7.5,
    color="#666666",
)

# Add note about KAMPFF being problematic
kampff_r2 = fmiss_fam_plot.loc[fmiss_fam_plot["family"] == "KAMPFF", "r2"]
if len(kampff_r2):
    ax.annotate(
        "KAMPFF: near-perfect fpos\nbut fmiss collapses",
        xy=(float(kampff_r2.values[0]), fmiss_fam_plot[fmiss_fam_plot["family"] == "KAMPFF"].index[0]),
        xytext=(0.1, fmiss_fam_plot[fmiss_fam_plot["family"] == "KAMPFF"].index[0] - 0.7),
        fontsize=7.5, color="#3C5488",
        arrowprops=dict(arrowstyle="->", color="#3C5488", lw=1.0),
    )

legend_patches = [
    mpatches.Patch(facecolor=FAMILY_COLORS[f], label=f)
    for f in FAMILY_COLORS
]
ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9,
          title="Family", title_fontsize=8.5, fontsize=8)

fig.tight_layout()
out = FIG_DIR / "fmiss_family_behavior.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Figure 5: target_asymmetry_combined.png — improved side-by-side per family
# ---------------------------------------------------------------------------
cmp = family_cmp.copy()
cmp["family"] = cmp["study_set"].apply(_strip)
cmp = cmp.sort_values("family").reset_index(drop=True)
cmp["fmiss_display"] = _clip_fmiss(cmp["fmiss_r2"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

fam_names  = cmp["family"].tolist()
fam_colors = [FAMILY_COLORS.get(f, "#888888") for f in fam_names]

# Left: fpos
bars_fp = axes[0].barh(fam_names, cmp["fpos_r2"], color=fam_colors, edgecolor="white", linewidth=0.4)
axes[0].axvline(0, color="#999999", linewidth=0.8)
axes[0].set_xlabel("R²  (recording-disjoint holdout)", fontsize=10)
axes[0].set_title("fpos (Waveform-augmented stack)", fontsize=11, fontweight="bold", color=FPOS_COLOR)
axes[0].set_xlim(-0.1, 0.85)
for bar, val in zip(bars_fp, cmp["fpos_r2"]):
    axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", ha="left", fontsize=8.5)

# Right: fmiss — use a lighter set of the same family colors with hatching for negative bars
bars_fm = axes[1].barh(fam_names, cmp["fmiss_display"], color=fam_colors, edgecolor="white", linewidth=0.4)
axes[1].axvline(0, color="#999999", linewidth=1.0, linestyle="--")
axes[1].set_xlabel("R²  (recording-disjoint holdout)", fontsize=10)
axes[1].set_title("fmiss (Reduced-latent context)", fontsize=11, fontweight="bold", color=FMISS_COLOR)
axes[1].set_xlim(FMISS_DISPLAY_MIN - 0.1, FMISS_DISPLAY_MAX + 0.08)

for bar, actual, display in zip(bars_fm, cmp["fmiss_r2"], cmp["fmiss_display"]):
    if actual != display:
        bar.set_hatch("///")
        bar.set_alpha(0.9)
    _annotate_clipped_barh(axes[1], bar, float(actual), float(display), pad=0.03)

axes[1].text(
    0.02,
    0.03,
    "fmiss axis clipped at -1.2 so family-level contrast stays visible.",
    transform=axes[1].transAxes,
    ha="left",
    va="bottom",
    fontsize=7.5,
    color="#666666",
)

fig.suptitle(
    "Target asymmetry: fpos vs fmiss per recording family",
    fontsize=13, fontweight="bold", y=1.01,
)

# Shared family color legend in figure
legend_patches = [
    mpatches.Patch(facecolor=FAMILY_COLORS[f], label=f)
    for f in fam_names  # same order as y-axis
]
fig.legend(handles=legend_patches, loc="lower center", ncol=5,
           framealpha=0.9, fontsize=8, title="Recording family",
           title_fontsize=8.5, bbox_to_anchor=(0.5, -0.08))

fig.tight_layout()
out = FIG_DIR / "target_asymmetry_combined.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Figure 6: fpos_fmiss_family_r2_scatter.png — novel scatter (NEW)
# ---------------------------------------------------------------------------
cmp2 = family_cmp.copy()
cmp2["family"] = cmp2["study_set"].apply(_strip)
cmp2["fmiss_display"] = _clip_fmiss(cmp2["fmiss_r2"])

# Annotation offsets (dx, dy) per family for readable labels
OFFSETS = {
    "BOYDEN":      ( 0.02, -0.09),
    "CRCNS_HC1":   ( 0.02, -0.09),
    "ENGLISH":     (-0.05,  0.06),
    "KAMPFF":      ( 0.02,  0.04),
    "MEA64C_YGER": ( 0.02,  0.04),
}

fig, ax = plt.subplots(figsize=(7, 6))

for _, row in cmp2.iterrows():
    fam = row["family"]
    c = FAMILY_COLORS.get(fam, "#888888")
    y = row["fmiss_display"]
    marker = "v" if row["fmiss_r2"] != row["fmiss_display"] else "o"
    ax.scatter(row["fpos_r2"], y,
               s=190 if marker == "v" else 160,
               marker=marker,
               color=c,
               edgecolors="white",
               linewidths=1.2,
               zorder=5)
    dx, dy = OFFSETS.get(fam, (0.02, 0.04))
    label = fam if row["fmiss_r2"] == row["fmiss_display"] else f"{fam} (actual {row['fmiss_r2']:.2f})"
    ax.annotate(
        label,
        xy=(row["fpos_r2"], y),
        xytext=(row["fpos_r2"] + dx, y + dy),
        fontsize=9, fontweight="bold", color=c,
        arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.7),
    )

# Reference lines
ax.axhline(0, color="#cccccc", linewidth=0.9, linestyle="--")
ax.axvline(0, color="#cccccc", linewidth=0.9, linestyle="--")

# Diagonal reference (perfect symmetry)
lim_lo = min(cmp2["fpos_r2"].min(), cmp2["fmiss_display"].min()) - 0.1
lim_hi = max(cmp2["fpos_r2"].max(), cmp2["fmiss_display"].max()) + 0.1
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
        linestyle=":", linewidth=1.2, color="#aaaaaa", label="Symmetry line (fpos=fmiss)")

# Quadrant shading: top-left = good fpos, bad fmiss (asymmetric)
ax.axhspan(lim_lo, 0, alpha=0.05, color="#D4691E")
ax.text(0.68, -0.2, "Good fpos\nPoor fmiss", fontsize=7.5,
        color=FMISS_COLOR, alpha=0.75, style="italic", ha="center")
ax.text(0.37, 0.7, "Good fmiss\nPoor fpos", fontsize=7.5,
        color=FPOS_COLOR, alpha=0.75, style="italic", ha="center")

ax.set_xlabel("fpos R²  (Waveform-augmented stack)", fontsize=11)
ax.set_ylabel("fmiss R²  (Reduced-latent context)", fontsize=11)
ax.set_title(
    "Per-family predictability: fpos vs fmiss\nKAMPFF shows strong target asymmetry",
    fontsize=12, fontweight="bold", pad=10,
)
ax.set_xlim(0.25, 0.82)
ax.set_ylim(FMISS_DISPLAY_MIN - 0.1, 1.05)
ax.text(
    0.02,
    0.03,
    "Downward triangles mark off-scale fmiss values clipped at -1.2.",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=7.5,
    color="#666666",
)

ax.legend(fontsize=8, framealpha=0.85, loc="upper left")

fig.tight_layout()
out = FIG_DIR / "fpos_fmiss_family_r2_scatter.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nAll figures saved to:", FIG_DIR)
saved = sorted(FIG_DIR.glob("*.png"))
for p in saved:
    print(f"  {p.name}")
