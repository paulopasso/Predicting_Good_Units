"""Improved figures for notebook 07: limitations and data budget.

Saves all outputs to figures/07_limitations_and_data_budget/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FIG_DIR = REPO / "figures" / "07_limitations_and_data_budget"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = REPO / "tables" / "07_limitations_and_data_budget"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
budget = pd.read_csv(TABLE_DIR / "label_budget_summary.csv")
headroom = pd.read_csv(TABLE_DIR / "headroom_summary.csv")
lofo_family = pd.read_csv(TABLE_DIR / "leave_one_family_by_family.csv")
lofo_agg = pd.read_csv(TABLE_DIR / "leave_one_family_aggregate.csv")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Brand colours
C_FPOS  = "#2E6DA4"   # blue
C_FMISS = "#D4691E"   # orange
C_ZERO  = "#888888"
CURRENT_N = 107       # operating point

# ---------------------------------------------------------------------------
# Helper: variant display names
# ---------------------------------------------------------------------------
VARIANT_LABELS = {
    "contextual_paired_only_lightgbm":    "Paired-only context",
    "embed_nommd_fullctx_lightgbm_filtered": "Embed full context",
    "source_reweighted_fullctx_lightgbm": "Source-reweighted full context",
}

# ---------------------------------------------------------------------------
# 1. label_budget_fpos.png — dual subplot: R² and MAE
# ---------------------------------------------------------------------------
def plot_label_budget_single(target: str, color: str, out_stem: str) -> None:
    df = budget[budget["target"] == target].copy()
    df["budget_rows"] = pd.to_numeric(df["budget_rows"], errors="coerce")
    df["mean_r2"]     = pd.to_numeric(df["mean_r2"],     errors="coerce")
    df["std_r2"]      = pd.to_numeric(df["std_r2"],      errors="coerce")
    df["mean_mae"]    = pd.to_numeric(df["mean_mae"],    errors="coerce")
    df["std_mae"]     = pd.to_numeric(df["std_mae"],     errors="coerce")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    fig.subplots_adjust(wspace=0.32)

    linestyles = ["-", "--", ":"]
    markers    = ["o", "s", "^"]

    variants = sorted(df["variant_id"].unique())

    for idx, vid in enumerate(variants):
        part = df[df["variant_id"] == vid].sort_values("budget_rows")
        xs   = part["budget_rows"].values
        ys_r2  = part["mean_r2"].values
        ys_mae = part["mean_mae"].values
        sd_r2  = part["std_r2"].values
        sd_mae = part["std_mae"].values
        label  = VARIANT_LABELS.get(vid, vid)
        ls = linestyles[idx % len(linestyles)]
        mk = markers[idx % len(markers)]
        alpha_shade = 0.18

        # R² panel
        ax1.plot(xs, ys_r2, marker=mk, linestyle=ls, color=color,
                 linewidth=2, markersize=6, label=label, alpha=0.9)
        # Shade ± 1 SD where repeats > 1
        mask = sd_r2 > 0
        if mask.any():
            ax1.fill_between(xs[mask],
                             ys_r2[mask] - sd_r2[mask],
                             ys_r2[mask] + sd_r2[mask],
                             color=color, alpha=alpha_shade)

        # MAE panel
        ax2.plot(xs, ys_mae, marker=mk, linestyle=ls, color=color,
                 linewidth=2, markersize=6, label=label, alpha=0.9)
        if mask.any():
            ax2.fill_between(xs[mask],
                             ys_mae[mask] - sd_mae[mask],
                             ys_mae[mask] + sd_mae[mask],
                             color=color, alpha=alpha_shade)

    # ---- R² panel decorations ----
    ax1.axhline(0, color=C_ZERO, linewidth=1.1, linestyle="--", zorder=1)
    ax1.text(22, 0.03, "R² = 0", fontsize=8, color=C_ZERO, va="bottom")

    ax1.axvline(CURRENT_N, color="black", linewidth=1.1, linestyle=":", zorder=2)
    y_annot_r2 = df.loc[df["budget_rows"] == CURRENT_N, "mean_r2"].mean()
    ax1.annotate(f"n={CURRENT_N}\n(current)",
                 xy=(CURRENT_N, y_annot_r2),
                 xytext=(CURRENT_N - 30, y_annot_r2 - 0.22),
                 fontsize=8, ha="center",
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.9))

    ax1.set_xlabel("Number of labeled paired training rows")
    ax1.set_ylabel("Mean R²")
    ax1.set_title(f"{target.upper()} — R² vs label budget")
    ax1.legend(frameon=True, framealpha=0.85, loc="upper left")

    # ---- MAE panel decorations ----
    ax2.axvline(CURRENT_N, color="black", linewidth=1.1, linestyle=":", zorder=2)
    y_annot_mae = df.loc[df["budget_rows"] == CURRENT_N, "mean_mae"].mean()
    ax2.annotate(f"n={CURRENT_N}\n(current)",
                 xy=(CURRENT_N, y_annot_mae),
                 xytext=(CURRENT_N - 30, y_annot_mae + 0.04),
                 fontsize=8, ha="center",
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.9))

    ax2.set_xlabel("Number of labeled paired training rows")
    ax2.set_ylabel("Mean MAE")
    ax2.set_title(f"{target.upper()} — MAE vs label budget")

    fig.suptitle(
        f"Data efficiency: how performance scales with labeled paired data ({target.upper()})",
        fontsize=12, y=1.01,
    )
    out = FIG_DIR / f"{out_stem}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


plot_label_budget_single("fpos",  C_FPOS,  "label_budget_fpos")
plot_label_budget_single("fmiss", C_FMISS, "label_budget_fmiss")


# ---------------------------------------------------------------------------
# 2. label_budget_combined.png — fpos and fmiss R² on the same axes
# ---------------------------------------------------------------------------
def plot_label_budget_combined() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    cfg = [
        ("fpos",  C_FPOS,  "contextual_paired_only_lightgbm",     "fpos (paired-only context)"),
        ("fmiss", C_FMISS, "contextual_paired_only_lightgbm",     "fmiss (paired-only context)"),
    ]

    all_xs = []
    endpoint_meta = []
    point_label_offsets = {
        "fpos": {20: 0.05, 60: 0.04, 107: 0.0},
        "fmiss": {20: 0.05, 60: 0.04, 107: 0.0},
    }

    for target, color, vid, label in cfg:
        df = budget[(budget["target"] == target) & (budget["variant_id"] == vid)].copy()
        df = df.sort_values("budget_rows")
        xs  = df["budget_rows"].values.astype(float)
        ys  = df["mean_r2"].values
        sds = df["std_r2"].values
        all_xs.extend(xs.tolist())

        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=7, color=color, label=label)
        mask = sds > 0
        if mask.any():
            ax.fill_between(xs[mask], ys[mask] - sds[mask], ys[mask] + sds[mask],
                            color=color, alpha=0.15)
        for x, y in zip(xs, ys):
            if int(x) == CURRENT_N:
                continue
            dy = point_label_offsets.get(target, {}).get(int(x), 0.04)
            ax.text(
                x,
                y + dy,
                f"{y:.2f}",
                color=color,
                fontsize=8,
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
            )
        endpoint_meta.append({"target": target, "color": color, "label": label, "x": xs[-1], "y": ys[-1]})

    x_min = min(all_xs) - 3
    x_max = max(all_xs) + 7

    # Below-zero shading
    ax.axhline(0, color=C_ZERO, linewidth=1.1, linestyle="--", zorder=1)
    ax.fill_between([x_min, x_max], [0, 0], [-1.2, -1.2],
                    color="#cccccc", alpha=0.25, zorder=0, label="Below chance (R² < 0)")
    ax.text(
        x_min + 2,
        -0.07,
        "below chance",
        fontsize=8,
        color=C_ZERO,
        va="top",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.7),
    )

    # Current operating point
    ax.axvline(CURRENT_N, color="black", linewidth=1.1, linestyle=":", zorder=3)
    ax.text(
        CURRENT_N - 0.8,
        0.565,
        f"canonical split\nn={CURRENT_N}",
        fontsize=8.2,
        va="top",
        ha="right",
        color="black",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#d0d0d0", alpha=0.9),
    )

    endpoint_offsets = {
        "fpos": (1.8, 0.04),
        "fmiss": (1.8, -0.02),
    }
    endpoint_align = {
        "fpos": ("left", "bottom"),
        "fmiss": ("left", "top"),
    }
    for meta in endpoint_meta:
        dx, dy = endpoint_offsets.get(meta["target"], (1.8, 0.0))
        ha, va = endpoint_align.get(meta["target"], ("left", "center"))
        ax.annotate(
            f"{meta['label']}\nR²={meta['y']:.2f}",
            xy=(meta["x"], meta["y"]),
            xytext=(meta["x"] + dx, meta["y"] + dy),
            textcoords="data",
            fontsize=8.5,
            fontweight="bold",
            color=meta["color"],
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=meta["color"], alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=meta["color"], lw=1.0, shrinkA=0, shrinkB=4),
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(sorted(set(all_xs)))
    ax.set_ylim(-1.22, 0.58)
    ax.set_xlabel("Number of labeled paired training rows", fontsize=11)
    ax.set_ylabel("Mean R²", fontsize=11)
    ax.set_title("Label efficiency of paired-only context models", fontsize=13, fontweight="bold")
    ax.text(
        0.02,
        0.03,
        "Low-budget points are 2-repeat means; n=107 is the canonical full-budget split.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#666666",
        ha="left",
        va="bottom",
    )
    ax.legend(frameon=True, framealpha=0.88, loc="upper left")

    fig.tight_layout()

    out = FIG_DIR / "label_budget_combined.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


plot_label_budget_combined()


# ---------------------------------------------------------------------------
# 3. leave_one_family_r2.png — LOFO bar chart with family colours
# ---------------------------------------------------------------------------
FAMILY_COLORS = {
    "PAIRED_BOYDEN":      "#E64B35",
    "PAIRED_CRCNS_HC1":   "#4DBBD5",
    "PAIRED_ENGLISH":     "#00A087",
    "PAIRED_KAMPFF":      "#3C5488",
    "PAIRED_MEA64C_YGER": "#F39B7F",
}
# Clean short names for display
FAMILY_SHORT = {
    "PAIRED_BOYDEN":      "BOYDEN",
    "PAIRED_CRCNS_HC1":   "CRCNS_HC1",
    "PAIRED_ENGLISH":     "ENGLISH",
    "PAIRED_KAMPFF":      "KAMPFF",
    "PAIRED_MEA64C_YGER": "MEA64C_YGER",
}

def plot_lofo_r2() -> None:
    df = lofo_family.copy()
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")

    variants = df["variant_id"].unique().tolist()
    families = [
        "PAIRED_BOYDEN", "PAIRED_CRCNS_HC1", "PAIRED_ENGLISH",
        "PAIRED_KAMPFF", "PAIRED_MEA64C_YGER",
    ]

    # Overall reference R² from aggregate (macro mean across best variant)
    # Use the first variant's macro_r2 as the reference line
    ref_r2 = lofo_agg["macro_r2"].iloc[0]  # reference_anchor_stack_xgboost

    n_families = len(families)
    n_variants = len(variants)
    bar_h = 0.32
    spacing = 0.12
    group_h = n_variants * bar_h + spacing

    fig, ax = plt.subplots(figsize=(10, 5.5))

    variant_labels = {
        "reference_anchor_stack_xgboost": "Reference anchor stack",
        "wf_embed_anchor_stack_xgboost":  "Waveform embed anchor stack",
    }

    for vi, vid in enumerate(variants):
        part = df[df["variant_id"] == vid]
        label = variant_labels.get(vid, vid)
        # Use hatching to distinguish variants; colour comes from family
        hatch = "" if vi == 0 else "///"
        edge_alpha = 0.85

        for fi, fam in enumerate(families):
            row = part[part["held_out_family"] == fam]
            if row.empty:
                continue
            r2_val = float(row["r2"].iloc[0])
            y_pos  = fi * group_h + vi * bar_h
            fc = FAMILY_COLORS.get(fam, "#888888")
            # Lighten second variant slightly
            face = fc if vi == 0 else fc + "bb"
            ax.barh(y_pos, r2_val, height=bar_h - 0.03,
                    color=face, edgecolor="white", linewidth=0.5,
                    hatch=hatch, alpha=0.92)
            # Value label
            ax.text(r2_val + 0.008, y_pos, f"{r2_val:.3f}",
                    va="center", ha="left", fontsize=7.5, color="#333333")

    # Reference line: all-families macro R²
    ax.axvline(ref_r2, color="#444444", linewidth=1.4, linestyle="--",
               label=f"All-families macro R² = {ref_r2:.3f}")
    ax.text(ref_r2 + 0.005, -0.3, f"all-fam\n{ref_r2:.3f}",
            fontsize=7.5, color="#444444", va="top", ha="left")

    # Y-tick positions: centre of each group
    ytick_pos = [fi * group_h + (n_variants - 1) * bar_h / 2 for fi in range(n_families)]
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([FAMILY_SHORT.get(f, f) for f in families], fontsize=10)
    ax.set_ylabel("Held-out Family", fontsize=11)
    ax.set_xlabel("R²  (held-out family test set)", fontsize=11)
    ax.set_title("Leave-one-family-out R²  (fpos)", fontsize=13)
    ax.set_xlim(0, 0.82)

    # Family colour legend patches
    family_patches = [
        mpatches.Patch(facecolor=FAMILY_COLORS[f], label=FAMILY_SHORT[f])
        for f in families
    ]
    # Variant hatch legend
    hatch_patches = [
        mpatches.Patch(facecolor="#aaaaaa", hatch="",    label=variant_labels.get(variants[0], variants[0])),
        mpatches.Patch(facecolor="#aaaaaa", hatch="///", label=variant_labels.get(variants[1], variants[1]) if len(variants) > 1 else ""),
    ]
    leg1 = ax.legend(handles=family_patches, title="Family", loc="lower right",
                     fontsize=8, title_fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    if len(variants) > 1:
        ax.legend(handles=hatch_patches + [
            mpatches.Patch(facecolor="none", edgecolor="#444444",
                           linestyle="--", linewidth=1.2, label=f"All-families ref ({ref_r2:.3f})")
        ], loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out = FIG_DIR / "leave_one_family_r2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# The older LOFO-by-family bar chart relies on a non-retained anchor variant and is
# intentionally not regenerated for the main thesis figure set.


# ---------------------------------------------------------------------------
# 4. headroom_summary.png — improved styling with clear gain arrows
# ---------------------------------------------------------------------------
def plot_headroom() -> None:
    df = headroom.copy()
    df["basic_transfer_r2"] = pd.to_numeric(df["basic_transfer_r2"], errors="coerce")
    df["current_best_r2"]   = pd.to_numeric(df["current_best_r2"],   errors="coerce")
    df["gain_r2"]           = pd.to_numeric(df["gain_r2"],           errors="coerce")

    df["target_label"] = df["target"].str.upper()
    df["color"] = df["target"].map({"fpos": C_FPOS, "fmiss": C_FMISS})
    df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(df))
    gain_color = "#228B22"

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.axvline(0, color=C_ZERO, linewidth=1.1, linestyle="--", zorder=1)

    for yi, row in zip(y, df.itertuples()):
        ax.hlines(yi, row.basic_transfer_r2, row.current_best_r2, color="#c7c7c7", linewidth=3.0, zorder=2)
        ax.scatter(row.basic_transfer_r2, yi, s=90, color="#8f8f8f", edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(row.current_best_r2, yi, s=120, color=row.color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(
            "",
            xy=(row.current_best_r2, yi),
            xytext=(row.basic_transfer_r2, yi),
            arrowprops=dict(arrowstyle="->", color=gain_color, lw=1.6),
        )
        ax.text(row.basic_transfer_r2 - 0.03, yi - 0.12, f"{row.basic_transfer_r2:.3f}",
                ha="right", va="center", fontsize=8.5, color="#555555")
        ax.text(row.current_best_r2 + 0.03, yi, f"{row.current_best_r2:.3f}",
                ha="left", va="center", fontsize=9.5, color=row.color, fontweight="bold")
        ax.text((row.basic_transfer_r2 + row.current_best_r2) / 2, yi + 0.16, f"ΔR² = +{row.gain_r2:.3f}",
                ha="center", va="bottom", fontsize=8.5, color=gain_color, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(df["target_label"], fontsize=12)
    ax.set_xlabel("R²  (recording-disjoint holdout)", fontsize=11)
    ax.set_title("Improvement beyond the baseline floor", fontsize=13, fontweight="bold")
    ax.set_xlim(min(df["basic_transfer_r2"]) - 0.10, max(df["current_best_r2"]) + 0.16)
    ax.margins(y=0.22)

    legend_handles = [
        Line2D([0], [0], marker="o", color="#c7c7c7", markerfacecolor="#8f8f8f", markersize=8, linewidth=3, label="Dummy baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_FPOS, markeredgecolor="white", markersize=9, label="fpos current best"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_FMISS, markeredgecolor="white", markersize=9, label="fmiss current best"),
        Line2D([0], [0], color=gain_color, linewidth=1.6, label="Gain"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.88, loc="lower right")

    fig.tight_layout()
    out = FIG_DIR / "headroom_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


plot_headroom()

print("\nAll figures saved successfully.")
