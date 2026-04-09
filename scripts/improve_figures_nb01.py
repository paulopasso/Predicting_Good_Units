"""
improve_figures_nb01.py
=======================
Regenerates publication-quality figures for notebook 01
(data_extraction_and_audit) and saves them to
figures/01_data_extraction_and_audit/.

Run with the spike-qc-local conda env:
    /opt/homebrew/Caskroom/miniconda/base/envs/spike-qc-local/bin/python \
        scripts/improve_figures_nb01.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "01_data_extraction_and_audit"
TABLE_DIR = ROOT / "tables" / "01_data_extraction_and_audit"
DATA_DIR = ROOT / "data" / "raw"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Thesis style
# ---------------------------------------------------------------------------
HYBRID_COLOR  = "#888888"   # gray
PAIRED_COLOR  = "#2E6DA4"   # blue
FPOS_COLOR    = "#2E6DA4"   # blue
FMISS_COLOR   = "#D4691E"   # orange

FAMILY_COLORS = {
    "ENGLISH":     "#4C72B0",
    "MEA64C_YGER": "#DD8452",
    "KAMPFF":      "#55A868",
    "BOYDEN":      "#C44E52",
    "CRCNS_HC1":   "#8172B3",
}

BLOCK_COLORS = {
    "acg_bins":           "#4C72B0",
    "waveform_bins":      "#55A868",
    "amplitude":          "#DD8452",
    "waveform_shape":     "#C44E52",
    "spikeinterface":     "#8172B3",
    "recording_context":  "#937860",
    "isi":                "#DA8BC3",
}

CATEGORY_COLORS = {
    "acg_bins":                 "#4C72B0",
    "waveform_bins":            "#55A868",
    "waveform_summary":         "#C44E52",
    "amplitude":                "#DD8452",
    "spikeinterface_qc":        "#8172B3",
    "confusability_isolation":  "#937860",
    "spike_train_summary":      "#DA8BC3",
    "spatial_template":         "#64B5CD",
    "recording_metadata":       "#8C8C8C",
    "other":                    "#B0B0B0",
}

CATEGORY_LABELS = {
    "acg_bins":                "Autocorrelogram bins",
    "waveform_bins":           "Waveform bins",
    "waveform_summary":        "Waveform summaries",
    "amplitude":               "Amplitude statistics",
    "spikeinterface_qc":       "SpikeInterface QC",
    "confusability_isolation": "Confusability / isolation",
    "spike_train_summary":     "Spike-train summaries",
    "spatial_template":        "Spatial / template",
    "recording_metadata":      "Recording metadata",
    "other":                   "Other",
}

UNIT_FEATURE_EXCLUDE = {
    "row_uid",
    "study_set",
    "study_name",
    "recording_name",
    "sorter_name",
    "unit_id",
    "group_key",
    "matched_gt_unit_id",
    "hungarian_gt_unit_id",
    "match_status",
    "fpos",
    "fmiss",
    "accuracy",
    "agreement_score",
    "tp",
    "fp",
    "fn",
    "num_gt",
    "num_tested",
    "recall",
    "precision",
    "__fragment_index",
    "__batch_index",
    "__last_in_fragment",
}

WAVEFORM_SUMMARY_COLS = {
    "snr",
    "peak_to_trough_uv",
    "half_width_ms",
    "repolarization_slope",
    "pre_trough_peak_uv",
    "post_trough_peak_uv",
    "trough_time_ms",
    "wf_energy",
    "wf_ptp_ratio",
    "wf_asymmetry",
    "wf_prepeak_ratio",
    "wf_rebound_ratio",
    "wf_peak_trough_ms",
    "wf_prepeak_to_trough_ms",
    "wf_trough_to_peak_ms",
    "wf_zero_cross_pre_ms",
    "wf_zero_cross_post_ms",
    "wf_trough_width_25_ms",
    "wf_trough_width_50_ms",
    "wf_trough_width_75_ms",
    "wf_neg_area_uv_ms",
    "wf_pos_area_uv_ms",
    "wf_pos_neg_area_ratio",
    "wf_line_length_norm",
    "wf_curvature_norm",
    "wf_trough_sharpness",
    "wf_pre_trough_slope",
    "wf_post_peak_slope",
}

SPIKE_TRAIN_SUMMARY_COLS = {
    "firing_rate_hz",
    "presence_ratio",
    "burst_index",
    "spike_count",
}

SPATIAL_TEMPLATE_COLS = {
    "template_norm",
    "n_active_channels",
    "spread_um",
    "center_of_mass_um",
    "spread_weighted_um",
    "peak_channel_depth_um",
    "n_channels",
}

CONFUSABILITY_EXTRA_COLS = {
    "n_confusable",
    "isolation_score",
    "max_waveform_cosine",
    "mean_waveform_cosine_top3",
    "median_waveform_cosine",
    "std_waveform_cosine",
    "n_waveform_confusable",
    "min_neighbor_dist_um",
    "mean_neighbor_dist_um",
}

SORTER_ORDER = [
    "HerdingSpikes2", "IronClust", "JRClust", "KiloSort",
    "KiloSort2", "Klusta", "MountainSort4", "SpykingCircus", "Tridesclous",
]
FAMILY_ORDER = ["ENGLISH", "MEA64C_YGER", "KAMPFF", "BOYDEN", "CRCNS_HC1"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.color":       "#eeeeee",
    "grid.linewidth":   0.6,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "legend.frameon":   False,
    "figure.dpi":       100,
    "savefig.dpi":      200,
})


def _save(fig: plt.Figure, stem: str) -> None:
    path = FIG_DIR / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved -> {path.name}")


def _load_raw_df() -> pd.DataFrame:
    parquet = DATA_DIR / "33000_ROWS.parquet"
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    df = pd.read_parquet(parquet)
    study = df["study_set"].astype(str)
    is_paired = study.str.startswith("PAIRED_")
    df["dataset_kind"] = np.where(is_paired, "paired", "hybrid")
    df["paired_family"] = np.where(is_paired, study.str.replace("PAIRED_", "", regex=False), pd.NA)
    return df


def _unit_feature_category(column: str) -> str:
    name = str(column)
    if name.startswith("wf_bin_"):
        return "waveform_bins"
    if name.startswith("acg_"):
        return "acg_bins"
    if name.startswith("amp_"):
        return "amplitude"
    if name.startswith("si_"):
        return "spikeinterface_qc"
    if name.startswith("isi_") or name in SPIKE_TRAIN_SUMMARY_COLS:
        return "spike_train_summary"
    if name in WAVEFORM_SUMMARY_COLS:
        return "waveform_summary"
    if name in SPATIAL_TEMPLATE_COLS:
        return "spatial_template"
    if name.startswith("rec_"):
        return "recording_metadata"
    if "cosine" in name or name in CONFUSABILITY_EXTRA_COLS:
        return "confusability_isolation"
    return "other"


def _build_unit_feature_category_counts(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in UNIT_FEATURE_EXCLUDE
    ]
    rows = [{"category": _unit_feature_category(col), "feature": col} for col in numeric_cols]
    out = (
        pd.DataFrame(rows)
        .groupby("category", as_index=False)
        .agg(feature_count=("feature", "size"))
        .sort_values("feature_count", ascending=False)
        .reset_index(drop=True)
    )
    out["label"] = out["category"].map(CATEGORY_LABELS).fillna(out["category"])
    return out


# ---------------------------------------------------------------------------
# Load tables
# ---------------------------------------------------------------------------
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


dataset_summary      = load("dataset_summary.csv")
partition            = load("dataset_partition_table.csv")
feature_inv          = load("feature_inventory.csv")
top_corr             = load("top_feature_correlations.csv")
family_sorter_counts = load("paired_labeled_support_by_family_sorter.csv")
split_main           = load("recording_disjoint_protocol_summary.csv")
family_heldout       = load("family_held_out_protocol_summary.csv")
family_profile       = load("family_profile_summary.csv")
target_miss          = load("target_missingness.csv")


# ---------------------------------------------------------------------------
# Figure 1 – Dataset composition audit  (IMPROVED)
# ---------------------------------------------------------------------------
def fig_dataset_composition() -> None:
    print("Figure 1: dataset_composition_audit")

    ds = dict(zip(dataset_summary["summary_item"], dataset_summary["value"].astype(int)))
    part_sub = (
        partition[partition["subset"].isin(["hybrid_labeled", "paired_labeled", "paired_unlabeled"])]
        .set_index("subset")["rows"]
    )

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38, left=0.06, right=0.97, top=0.88, bottom=0.18)

    # --- Panel A: Operational subset sizes ---
    ax0 = fig.add_subplot(gs[0])
    labels_a  = ["Hybrid\n(labeled)", "Paired\n(labeled)", "Paired\n(unlabeled)"]
    counts_a  = [
        part_sub.get("hybrid_labeled", 0),
        part_sub.get("paired_labeled", 0),
        part_sub.get("paired_unlabeled", 0),
    ]
    colors_a = [HYBRID_COLOR, PAIRED_COLOR, "#88BBDD"]
    bars = ax0.bar(labels_a, counts_a, color=colors_a, width=0.55, zorder=2)
    ax0.set_ylabel("Units (rows)")
    ax0.set_title("A  –  Operational subset sizes", loc="left", fontsize=11, fontweight="bold")
    ax0.yaxis.grid(True, zorder=0)
    ax0.set_axisbelow(True)
    for bar, n in zip(bars, counts_a):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f"{n:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax0.set_ylim(0, max(counts_a) * 1.15)
    ax0.tick_params(axis="x", length=0)

    # --- Panel B: Target label availability ---
    ax1 = fig.add_subplot(gs[1])
    tm = target_miss.set_index("target")
    targets = ["fpos", "fmiss"]
    total   = ds["total_rows"]
    avail   = [int(tm.loc[t, "non_missing_rows"]) for t in targets]
    pct     = [v / total * 100 for v in avail]
    bar_colors = [FPOS_COLOR, FMISS_COLOR]
    bars2 = ax1.bar(targets, pct, color=bar_colors, width=0.4, zorder=2)
    ax1.set_ylabel("Available rows (%)")
    ax1.set_ylim(0, 115)
    ax1.set_title("B  –  Target label availability", loc="left", fontsize=11, fontweight="bold")
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    ax1.axhline(100, color="#aaaaaa", lw=0.8, ls="--")
    for bar, p, n in zip(bars2, pct, avail):
        ax1.text(bar.get_x() + bar.get_width() / 2, p + 1.5,
                 f"{p:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(np.arange(len(targets)))
    ax1.set_xticklabels(["fpos", "fmiss"], fontsize=10)
    ax1.tick_params(axis="x", length=0)

    # --- Panel C: Paired family coverage ---
    ax2 = fig.add_subplot(gs[2])
    fp = family_profile.set_index("family").loc[
        [f for f in FAMILY_ORDER if f in family_profile["family"].values]
    ]
    labeled_counts = family_sorter_counts.set_index("paired_family").sum(axis=1)
    fp["labeled"] = labeled_counts.reindex(fp.index).fillna(0).astype(int)
    fp["label_rate"] = fp["labeled"] / fp["rows"] * 100

    x = np.arange(len(fp))
    w = 0.45
    fcolors = [FAMILY_COLORS.get(f, "#999999") for f in fp.index]
    bars3 = ax2.bar(x, fp["rows"], color=fcolors, width=w, zorder=2, label="Total units")
    ax2.set_ylabel("Total paired units")
    ax2.set_title("C  –  Paired family coverage", loc="left", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f.replace("MEA64C_YGER", "MEA64C\nYGER") for f in fp.index],
        fontsize=8.5,
    )
    ax2.tick_params(axis="x", length=0)
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)

    ax2r = ax2.twinx()
    ax2r.plot(x, fp["label_rate"], color="#333333", marker="o", ms=6,
              lw=1.5, zorder=3, label="Labeled rate (%)")
    ax2r.set_ylabel("Labeled rate (%)", fontsize=9)
    ax2r.set_ylim(0, fp["label_rate"].max() * 2.2)
    ax2r.spines["right"].set_visible(True)
    ax2r.spines["right"].set_color("#cccccc")
    for xi, (rate, labeled) in enumerate(zip(fp["label_rate"], fp["labeled"])):
        ax2r.text(xi, rate + 0.3, f"n={int(labeled)}", ha="center", va="bottom",
                  fontsize=7.5, color="#333333")

    h1 = mpatches.Patch(color=HYBRID_COLOR, label="Hybrid domain")
    h2 = mpatches.Patch(color=PAIRED_COLOR, label="Paired domain")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.01), fontsize=9, frameon=False)

    fig.suptitle("Dataset composition — 33,206 units across 61 recordings, 6 families",
                 fontsize=12, y=0.97)
    _save(fig, "dataset_composition_audit")


# ---------------------------------------------------------------------------
# Figure 2 – Feature distributions hybrid vs paired  (IMPROVED)
# ---------------------------------------------------------------------------
def fig_feature_distributions() -> None:
    print("Figure 2: feature_distributions_hybrid_vs_paired")

    # Try to load raw data
    parquet = DATA_DIR / "33000_ROWS.parquet"
    if not parquet.exists():
        print("  [SKIP] parquet not found locally:", parquet)
        return

    df = _load_raw_df()
    df["domain"] = np.where(df["dataset_kind"].eq("hybrid"), "Hybrid", "Paired")

    features = ["snr", "amp_cv", "wf_asymmetry", "acg_00", "isi_cv", "amp_bimodality"]
    feature_labels = {
        "snr":            "SNR",
        "amp_cv":         "Amplitude CV",
        "wf_asymmetry":   "Waveform asymmetry",
        "acg_00":         "ACG bin 0 (refractory)",
        "isi_cv":         "ISI CV",
        "amp_bimodality": "Amplitude bimodality",
    }
    features = [f for f in features if f in df.columns]

    sample = df.sample(n=min(10000, len(df)), random_state=42)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for ax, feat in zip(axes, features):
        h_vals = sample.loc[sample["domain"] == "Hybrid", feat].dropna()
        p_vals = sample.loc[sample["domain"] == "Paired", feat].dropna()

        # Clip to 1–99th percentile for readability
        lo = min(h_vals.quantile(0.01), p_vals.quantile(0.01))
        hi = max(h_vals.quantile(0.99), p_vals.quantile(0.99))
        bins = np.linspace(lo, hi, 45)

        ax.hist(h_vals.clip(lo, hi), bins=bins, density=True, alpha=0.55,
                color=HYBRID_COLOR, label=f"Hybrid (n={len(h_vals):,})")
        ax.hist(p_vals.clip(lo, hi), bins=bins, density=True, alpha=0.60,
                color=PAIRED_COLOR, label=f"Paired (n={len(p_vals):,})")

        ax.set_xlabel(feature_labels.get(feat, feat), fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(feature_labels.get(feat, feat), fontsize=11)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    # Hide any unused axes
    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle(
        "Feature distributions: Hybrid vs Paired domain\n"
        "(histograms clipped to 1st–99th percentile, kernel density normalized)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "feature_distributions_hybrid_vs_paired")


# ---------------------------------------------------------------------------
# Figure 3 – Split protocol diagnostics  (IMPROVED)
# ---------------------------------------------------------------------------
def fig_split_protocol() -> None:
    print("Figure 3: split_protocol_diagnostics")

    row = split_main.iloc[0]
    tr, te = int(row["train_rows"]), int(row["test_rows"])
    total = tr + te

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40, left=0.07, right=0.97,
                             top=0.87, bottom=0.15)

    # --- Panel A: recording-disjoint main split ---
    ax0 = fig.add_subplot(gs[0])
    split_labels = ["Train\n(recording-disjoint)", "Test\n(recording-disjoint)"]
    split_vals   = [tr, te]
    split_colors = [FPOS_COLOR, FMISS_COLOR]
    bars = ax0.bar(split_labels, split_vals, color=split_colors, width=0.45, zorder=2)
    ax0.set_ylabel("Labeled paired units")
    ax0.set_title("A  –  Recording-disjoint main split", loc="left",
                  fontsize=11, fontweight="bold")
    ax0.yaxis.grid(True, zorder=0)
    ax0.set_axisbelow(True)
    ax0.tick_params(axis="x", length=0)
    for bar, n in zip(bars, split_vals):
        pct = n / total * 100
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{n} ({pct:.0f}%)", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")
    ax0.set_ylim(0, max(split_vals) * 1.2)

    # Annotations explaining the guarantee
    ax0.text(0.5, 0.07,
             "No recording appears in both train and test.\n"
             f"Leakage blocked: {int(row.get('recording_leakage', 0))==0}   "
             f"({int(row['train_recordings'])} train / {int(row['test_recordings'])} test recordings)",
             transform=ax0.transAxes, ha="center", va="bottom", fontsize=8.5,
             color="#444444", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#cccccc"))

    # --- Panel B: leave-one-family-out holdout sizes ---
    ax1 = fig.add_subplot(gs[1])
    fho = family_heldout.set_index("held_out_family").loc[
        [f for f in FAMILY_ORDER if f in family_heldout["held_out_family"].values]
    ]
    x      = np.arange(len(fho))
    w      = 0.35
    fcolors = [FAMILY_COLORS.get(f, "#999999") for f in fho.index]
    b1 = ax1.bar(x - w/2, fho["train_rows"], width=w, color="#aaaaaa",
                 zorder=2, label="Train rows")
    b2 = ax1.bar(x + w/2, fho["test_rows"],  width=w, color=fcolors,
                 zorder=2, label="Test rows (held-out family)")
    ax1.set_ylabel("Labeled paired units")
    ax1.set_title("B  –  Leave-one-family-out fold sizes", loc="left",
                  fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f.replace("MEA64C_YGER", "MEA64C\nYGER") for f in fho.index], fontsize=8.5
    )
    ax1.tick_params(axis="x", length=0)
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=8)

    for bar, n in zip(b2, fho["test_rows"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 str(int(n)), ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Split protocol diagnostics — recording-disjoint leakage prevention",
        fontsize=12, y=0.97,
    )
    _save(fig, "split_protocol_diagnostics")


# ---------------------------------------------------------------------------
# Figure 4 – Top target correlations  (IMPROVED)
# ---------------------------------------------------------------------------
def fig_top_correlations() -> None:
    print("Figure 4: top_target_correlations")

    fpos_df = top_corr[top_corr["target"] == "fpos"].nlargest(12, "abs_corr")
    fmiss_df = top_corr[top_corr["target"] == "fmiss"].nlargest(12, "abs_corr")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

    def _clean_label(s: str) -> str:
        return s.replace("si_", "SI: ").replace("amp_", "amp_").replace("_", " ")

    for ax, df_t, color, target in [
        (ax0, fpos_df, FPOS_COLOR, "fpos"),
        (ax1, fmiss_df, FMISS_COLOR, "fmiss"),
    ]:
        df_t = df_t.sort_values("abs_corr", ascending=True)
        labels = [_clean_label(f) for f in df_t["feature"]]
        values = df_t["abs_corr"].tolist()

        bars = ax.barh(labels, values, color=color, alpha=0.82, height=0.65, zorder=2)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("Absolute Pearson correlation  |r|", fontsize=10)
        ax.set_title(f"Top features correlated with  {target}",
                     fontsize=12, pad=8)
        ax.set_xlim(0, max(values) * 1.25)
        for bar, val in zip(bars, values):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=8)
        ax.tick_params(axis="y", length=0)

    fig.suptitle(
        "Feature–target correlations  (paired-labeled units, n=207)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "top_target_correlations")


# ---------------------------------------------------------------------------
# Figure 5 – Family × sorter target heatmaps  (IMPROVED)
# ---------------------------------------------------------------------------
def fig_family_sorter_heatmaps() -> None:
    print("Figure 5: family_sorter_target_heatmaps")

    # Re-derive from the support counts + family order
    support = family_sorter_counts.set_index("paired_family")
    sorters  = [s for s in SORTER_ORDER if s in support.columns]
    families = [f for f in FAMILY_ORDER if f in support.index]
    support  = support.loc[families, sorters].astype(float)

    # Load parquet for actual mean fpos/fmiss per cell if available
    parquet = DATA_DIR / "33000_ROWS.parquet"
    if parquet.exists():
        df = _load_raw_df()
        pl = df[df["dataset_kind"].eq("paired") & df["fpos"].notna()].copy()
        fpos_mat = (
            pl.pivot_table(index="paired_family", columns="sorter_name",
                           values="fpos", aggfunc="mean")
            .reindex(index=families, columns=sorters)
        )
        pl_fm = df[df["dataset_kind"].eq("paired") & df["fmiss"].notna()].copy()
        fmiss_mat = (
            pl_fm.pivot_table(index="paired_family", columns="sorter_name",
                              values="fmiss", aggfunc="mean")
            .reindex(index=families, columns=sorters)
        )
    else:
        # Fallback: random noise placeholder (should not happen in normal run)
        fpos_mat  = pd.DataFrame(np.nan, index=families, columns=sorters)
        fmiss_mat = pd.DataFrame(np.nan, index=families, columns=sorters)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    import matplotlib.colors as mcolors

    for ax, mat, title, cmap_name in [
        (axes[0], fpos_mat,  "Mean fpos  (false-positive rate)\nby paired family × sorter", "Blues"),
        (axes[1], fmiss_mat, "Mean fmiss (false-negative rate)\nby paired family × sorter", "Oranges"),
    ]:
        arr = mat.values.astype(float)
        masked = np.ma.masked_invalid(arr)

        im = ax.imshow(masked, aspect="auto", cmap=cmap_name,
                       vmin=0.0, vmax=0.75, interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.85, label="Mean rate")

        ax.set_xticks(range(len(sorters)))
        ax.set_xticklabels(sorters, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels([f.replace("MEA64C_YGER", "MEA64C_YGER") for f in families],
                           fontsize=9)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("Sorter", fontsize=10)
        ax.set_ylabel("Paired family", fontsize=10)

        # Annotate each cell
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                val = arr[r, c]
                n   = int(support.iloc[r, c]) if not np.isnan(support.iloc[r, c]) else 0
                if not np.isnan(val):
                    text_color = "white" if val > 0.45 else "black"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=7.5, color=text_color, fontweight="bold")
                    ax.text(c, r + 0.35, f"n={n}", ha="center", va="center",
                            fontsize=6, color=text_color, alpha=0.75)
                else:
                    ax.text(c, r, "–", ha="center", va="center",
                            fontsize=9, color="#999999")

    fig.suptitle(
        "Quality targets by paired family and sorter  "
        "(paired-labeled units only, showing mean ± cell support)",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "family_sorter_target_heatmaps")


# ---------------------------------------------------------------------------
# Figure 6 – Feature block inventory  (IMPROVED)
# ---------------------------------------------------------------------------
def fig_feature_block_inventory() -> None:
    print("Figure 6: feature_block_inventory")

    fi = feature_inv.copy()
    fi = fi.sort_values("feature_count", ascending=True)
    blocks   = fi["feature_block"].tolist()
    counts   = fi["feature_count"].tolist()
    has_missingness = "mean_missing_pct" in fi.columns
    miss_pct = fi["mean_missing_pct"].fillna(0.0).tolist() if has_missingness else [0.0] * len(fi)
    colors   = [BLOCK_COLORS.get(b, "#aaaaaa") for b in blocks]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: feature counts
    bars0 = ax0.barh(blocks, counts, color=colors, height=0.6, zorder=2)
    ax0.xaxis.grid(True, zorder=0)
    ax0.set_axisbelow(True)
    ax0.set_xlabel("Number of features", fontsize=10)
    ax0.set_title("A  –  Features per block", loc="left", fontsize=11, fontweight="bold")
    ax0.tick_params(axis="y", length=0)
    total = sum(counts)
    for bar, n in zip(bars0, counts):
        ax0.text(n + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{n}  ({n/total*100:.0f}%)", va="center", ha="left", fontsize=9)
    ax0.set_xlim(0, max(counts) * 1.35)
    ax0.text(0.97, 0.02, f"Total: {total} features", transform=ax0.transAxes,
             ha="right", va="bottom", fontsize=9, color="#555555")

    # Panel B: missingness
    miss_colors = ["#D94F4F" if m > 5 else "#F4A443" if m > 0.5 else "#5BAD72"
                   for m in miss_pct]
    bars1 = ax1.barh(blocks, miss_pct, color=miss_colors, height=0.6, zorder=2)
    ax1.xaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_xlabel("Mean missingness (%)", fontsize=10)
    ax1.set_title("B  –  Missingness by block", loc="left", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", length=0)
    if has_missingness:
        for bar, m in zip(bars1, miss_pct):
            label = f"{m:.1f}%" if m > 0.05 else "0%"
            ax1.text(m + 0.1, bar.get_y() + bar.get_height() / 2,
                     label, va="center", ha="left", fontsize=9)
        ax1.set_xlim(0, max(miss_pct) * 1.35 if max(miss_pct) > 0 else 1)

        leg_patches = [
            mpatches.Patch(color="#5BAD72", label="< 0.5% — complete"),
            mpatches.Patch(color="#F4A443", label="0.5–5% — minor gaps"),
            mpatches.Patch(color="#D94F4F", label="> 5% — notable missing"),
        ]
        ax1.legend(handles=leg_patches, fontsize=8, loc="lower right")
    else:
        ax1.set_xlim(0, 1)
        ax1.text(
            0.5,
            0.56,
            "Block-level missingness\nis not available in the\ncurrent compact inventory table.",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
        )
        ax1.text(
            0.5,
            0.12,
            "Counts remain accurate; this panel is shown as unavailable.",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#888888",
        )

    fig.suptitle(
        f"Feature block inventory  —  {total} features across 7 blocks",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "feature_block_inventory")


# ---------------------------------------------------------------------------
# NEW Figure – Unit feature category counts
# ---------------------------------------------------------------------------
def fig_feature_category_counts() -> None:
    print("Figure NEW: feature_category_counts")
    strict_counts = pd.read_csv(
        ROOT / "data" / "frozen_inputs" / "domain_shift" / "feature_group_summary.csv"
    )
    strict_counts = strict_counts[
        (strict_counts["comparison"] == "hybrid_vs_paired_all")
        & (strict_counts["feature_set"].isin(["unit", "full_context"]))
    ].copy()

    label_map = {
        "recording_context": "Recording context",
        "recording_relative": "Recording-relative",
        "other": "Other unit descriptors",
        "acg_bins": "Autocorrelogram bins",
        "waveform_bins": "Waveform bins",
        "amplitude": "Amplitude statistics",
        "confusability": "Confusability / isolation",
        "spikeinterface": "SpikeInterface QC",
    }
    color_map = {
        "recording_context": "#937860",
        "recording_relative": "#DA8BC3",
        "other": "#9AA5B1",
        "acg_bins": "#4C72B0",
        "waveform_bins": "#55A868",
        "amplitude": "#DD8452",
        "confusability": "#C44E52",
        "spikeinterface": "#8172B3",
    }
    order = [
        "recording_context",
        "recording_relative",
        "other",
        "acg_bins",
        "waveform_bins",
        "amplitude",
        "confusability",
        "spikeinterface",
    ]
    strict_counts["feature_group"] = pd.Categorical(
        strict_counts["feature_group"], categories=order, ordered=True
    )
    strict_counts = strict_counts.sort_values(["feature_set", "feature_group"]).reset_index(drop=True)
    strict_counts["label"] = strict_counts["feature_group"].map(label_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    panel_meta = [
        ("unit", "A  –  Strict unit-only view", 164),
        ("full_context", "B  –  Strict full-context view", 685),
    ]

    for ax, (feature_set, title, total) in zip(axes, panel_meta):
        panel = strict_counts[strict_counts["feature_set"] == feature_set].copy()
        bars = ax.barh(
            panel["label"],
            panel["n_features"],
            color=[color_map.get(group, "#aaaaaa") for group in panel["feature_group"]],
            height=0.65,
            zorder=2,
        )
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of features", fontsize=10)
        ax.set_xlim(0, max(panel["n_features"]) * 1.22)
        for bar, count in zip(bars, panel["n_features"]):
            ax.text(
                count + max(panel["n_features"]) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{int(count)}",
                va="center",
                ha="left",
                fontsize=9,
            )
        ax.text(
            0.98,
            0.02,
            f"Total: {total}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#555555",
        )

    axes[1].tick_params(labelleft=True)
    fig.suptitle(
        "Strict feature-group counts from the canonical modelling pipeline\n"
        "Different model families had access to different feature views, so counts depend on the view.",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    svg_path = FIG_DIR / "feature_category_counts.svg"
    png_path = FIG_DIR / "feature_category_counts.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  saved -> {svg_path.name}")
    print(f"  saved -> {png_path.name}")


# ---------------------------------------------------------------------------
# Figure 7 – Domain projection PCA  (IMPROVED)
# ---------------------------------------------------------------------------
def fig_domain_pca() -> None:
    print("Figure 7: domain_projection_pca")

    parquet = DATA_DIR / "33000_ROWS.parquet"
    if not parquet.exists():
        print("  [SKIP] parquet not found:", parquet)
        return

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = _load_raw_df()

    wf_bins = [c for c in df.columns if c.startswith("wf_bin_")][:20]
    acg     = [c for c in df.columns if c.startswith("acg_")][:20]
    amp     = [c for c in df.columns if c.startswith("amp_")][:8]
    si      = [c for c in df.columns if c.startswith("si_")][:8]
    rec_ctx = ["rec_sampling_rate", "n_active_channels",
               "rec_noise_std_uv", "rec_n_units"]
    rec_ctx = [c for c in rec_ctx if c in df.columns]

    feat_cols = list(dict.fromkeys(wf_bins + acg + amp + si + rec_ctx))
    feat_cols = [c for c in feat_cols if c in df.columns]

    sample_n = min(8000, len(df))
    sample   = df.sample(n=sample_n, random_state=42).copy()
    X = sample[feat_cols].fillna(sample[feat_cols].median()).values
    X = StandardScaler().fit_transform(X)

    pca  = PCA(n_components=3, random_state=42)
    Xpc  = pca.fit_transform(X)
    sample["pc1"] = Xpc[:, 0]
    sample["pc2"] = Xpc[:, 1]

    var = pca.explained_variance_ratio_
    hybrid_mask = sample["dataset_kind"].eq("hybrid")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: scatter coloured by domain
    ax0.scatter(sample.loc[hybrid_mask, "pc1"], sample.loc[hybrid_mask, "pc2"],
                s=7, alpha=0.35, color=HYBRID_COLOR, label=f"Hybrid (n={hybrid_mask.sum():,})", rasterized=True)
    ax0.scatter(sample.loc[~hybrid_mask, "pc1"], sample.loc[~hybrid_mask, "pc2"],
                s=12, alpha=0.60, color=PAIRED_COLOR, label=f"Paired (n={(~hybrid_mask).sum():,})", rasterized=True)
    ax0.set_xlabel(f"PC1  ({var[0]*100:.1f}% var)", fontsize=10)
    ax0.set_ylabel(f"PC2  ({var[1]*100:.1f}% var)", fontsize=10)
    ax0.set_title("A  –  Domain overlap in PC space", loc="left", fontsize=11, fontweight="bold")
    ax0.legend(markerscale=2.5, fontsize=9)

    # Panel B: coloured by family
    sample["group"] = np.where(hybrid_mask, "HYBRID", sample["paired_family"].fillna("?"))
    group_colors = {"HYBRID": HYBRID_COLOR}
    group_colors.update(FAMILY_COLORS)

    for grp in ["HYBRID"] + FAMILY_ORDER:
        mask = sample["group"] == grp
        if mask.sum() == 0:
            continue
        color = group_colors.get(grp, "#aaaaaa")
        alpha = 0.25 if grp == "HYBRID" else 0.75
        size  = 6 if grp == "HYBRID" else 16
        ax1.scatter(sample.loc[mask, "pc1"], sample.loc[mask, "pc2"],
                    s=size, alpha=alpha, color=color, label=grp, rasterized=True)

    ax1.set_xlabel(f"PC1  ({var[0]*100:.1f}% var)", fontsize=10)
    ax1.set_ylabel(f"PC2  ({var[1]*100:.1f}% var)", fontsize=10)
    ax1.set_title("B  –  Paired families in PC space", loc="left", fontsize=11, fontweight="bold")
    ax1.legend(markerscale=2.0, fontsize=8.5, ncol=2)

    fig.suptitle(
        "PCA projection of waveform, ACG, amplitude & QC features\n"
        f"(n={sample_n:,} sampled units; first two PCs shown)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "domain_projection_pca")


# ---------------------------------------------------------------------------
# NEW Figure – Target distribution overview (fpos & fmiss histograms)
# ---------------------------------------------------------------------------
def fig_target_distributions() -> None:
    print("Figure NEW: target_distributions")

    parquet = DATA_DIR / "33000_ROWS.parquet"
    if not parquet.exists():
        print("  [SKIP] parquet not found:", parquet)
        return

    df = _load_raw_df()
    paired_labeled = df[df["dataset_kind"].eq("paired") & df["fpos"].notna()].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    bins_01 = np.linspace(0, 1, 40)

    # --- Panel A: fpos by family ---
    ax0 = axes[0]
    for fam in FAMILY_ORDER:
        vals = paired_labeled.loc[paired_labeled["paired_family"].eq(fam), "fpos"].dropna()
        if len(vals) == 0:
            continue
        color = FAMILY_COLORS.get(fam, "#999999")
        ax0.hist(vals, bins=bins_01, density=True, alpha=0.55, color=color,
                 label=f"{fam} (n={len(vals)})", histtype="stepfilled")
    ax0.set_xlabel("fpos  (false-positive rate)", fontsize=10)
    ax0.set_ylabel("Density", fontsize=10)
    ax0.set_title("A  –  fpos by paired family", loc="left", fontsize=11, fontweight="bold")
    ax0.legend(fontsize=7.5)

    # --- Panel B: fmiss by family ---
    ax1 = axes[1]
    paired_labeled_fm = df[df["dataset_kind"].eq("paired") & df["fmiss"].notna()].copy()
    for fam in FAMILY_ORDER:
        vals = paired_labeled_fm.loc[paired_labeled_fm["paired_family"].eq(fam), "fmiss"].dropna()
        if len(vals) == 0:
            continue
        color = FAMILY_COLORS.get(fam, "#999999")
        ax1.hist(vals, bins=bins_01, density=True, alpha=0.55, color=color,
                 label=f"{fam} (n={len(vals)})", histtype="stepfilled")
    ax1.set_xlabel("fmiss  (false-negative rate)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("B  –  fmiss by paired family", loc="left", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=7.5)

    # --- Panel C: fpos vs fmiss scatter (paired labeled) ---
    ax2 = axes[2]
    both = df[df["dataset_kind"].eq("paired") & df["fpos"].notna() & df["fmiss"].notna()].copy()
    for fam in FAMILY_ORDER:
        fam_df = both[both["paired_family"].eq(fam)]
        if len(fam_df) == 0:
            continue
        color = FAMILY_COLORS.get(fam, "#999999")
        ax2.scatter(fam_df["fpos"], fam_df["fmiss"], s=14, alpha=0.6, color=color,
                    label=f"{fam} (n={len(fam_df)})")

    # Diagonal: fpos = fmiss reference
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="fpos = fmiss")
    ax2.set_xlabel("fpos  (false-positive rate)", fontsize=10)
    ax2.set_ylabel("fmiss  (false-negative rate)", fontsize=10)
    ax2.set_title("C  –  fpos vs fmiss  (all paired labeled)", loc="left",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7.5, ncol=1)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    fig.suptitle(
        "Target variable distributions across paired families\n"
        "(fpos always available; fmiss missing for unmatched / false-positive units)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "target_distributions_by_family")


# ---------------------------------------------------------------------------
# NEW Figure – Sorter support heatmap (count only)
# ---------------------------------------------------------------------------
def fig_sorter_support_heatmap() -> None:
    print("Figure NEW: sorter_support_heatmap")

    support = family_sorter_counts.set_index("paired_family")
    sorters  = [s for s in SORTER_ORDER if s in support.columns]
    families = [f for f in FAMILY_ORDER if f in support.index]
    mat      = support.loc[families, sorters].astype(float)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(mat.values, aspect="auto", cmap="YlOrBr",
                   vmin=0, vmax=mat.values.max(), interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Labeled units (n)", fontsize=9)

    ax.set_xticks(range(len(sorters)))
    ax.set_xticklabels(sorters, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=10)
    ax.set_xlabel("Sorter", fontsize=10)
    ax.set_ylabel("Paired family", fontsize=10)
    ax.set_title(
        "Labeled unit support by paired family × sorter\n"
        "(cells with 0 = sorter not run on this family)",
        fontsize=11,
    )

    for r, fam in enumerate(families):
        for c, sor in enumerate(sorters):
            n = int(mat.iloc[r, c])
            text_color = "white" if n > mat.values.max() * 0.65 else "black"
            ax.text(c, r, str(n), ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    fig.tight_layout()
    _save(fig, "sorter_support_heatmap")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Output directory: {FIG_DIR}\n")

    fig_dataset_composition()
    fig_feature_block_inventory()
    fig_feature_category_counts()
    fig_top_correlations()
    fig_split_protocol()
    fig_feature_distributions()
    fig_domain_pca()
    fig_family_sorter_heatmaps()
    fig_target_distributions()
    fig_sorter_support_heatmap()

    print("\nDone — all figures saved.")
