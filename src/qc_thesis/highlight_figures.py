from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from .paths import THESIS_ROOT, WAVEFORM_AUGMENTED_LABEL, feature_group_display, normalize_model_label
from .plots import (
    THESIS_HIGHLIGHT_FAMILY_COLORS,
    THESIS_HIGHLIGHT_MODEL_COLORS,
    THESIS_HIGHLIGHT_MODEL_SHORT,
    apply_thesis_style,
)


FAMILY_SHORT = {
    "PAIRED_BOYDEN": "BOYDEN",
    "PAIRED_CRCNS_HC1": "CRCNS_HC1",
    "PAIRED_ENGLISH": "ENGLISH",
    "PAIRED_KAMPFF": "KAMPFF",
    "PAIRED_MEA64C_YGER": "MEA64C_YGER",
}

FAMILY_SHARE_COLS = [
    "share__PAIRED_BOYDEN",
    "share__PAIRED_CRCNS_HC1",
    "share__PAIRED_ENGLISH",
    "share__PAIRED_KAMPFF",
    "share__PAIRED_MEA64C_YGER",
]

FAMILY_ROW_COLS = [
    "rows__PAIRED_BOYDEN",
    "rows__PAIRED_CRCNS_HC1",
    "rows__PAIRED_ENGLISH",
    "rows__PAIRED_KAMPFF",
    "rows__PAIRED_MEA64C_YGER",
]


@dataclass(frozen=True)
class HighlightFigureData:
    root: Path
    out_fig_dir: Path
    out_tab_dir: Path
    domain_clf: pd.DataFrame
    feat_shift: pd.DataFrame
    pca_centroids: pd.DataFrame
    seed_winner: pd.DataFrame
    fam_pref_r2: pd.DataFrame
    fam_ctx: pd.DataFrame
    rec_pivot: pd.DataFrame
    protocol_table: pd.DataFrame
    sm_raw: pd.DataFrame
    sm_wave: pd.DataFrame
    sm_pre: pd.DataFrame
    sm_ctx: pd.DataFrame


def highlight_output_dirs(root: Path | str = THESIS_ROOT) -> tuple[Path, Path]:
    base = Path(root)
    out_fig_dir = base / "figures" / "09_thesis_highlight_figures"
    out_tab_dir = base / "tables" / "09_thesis_highlight_figures"
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_tab_dir.mkdir(parents=True, exist_ok=True)
    return out_fig_dir, out_tab_dir


def _load_seed_r2(path: Path, column: str = "r2") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["seed", column]].copy()


def _normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ("label", "winner_label", "best_recording_disjoint_label", "best_lofo_label"):
        if column in out.columns:
            out[column] = out[column].map(normalize_model_label)
    return out


def load_highlight_figure_data(root: Path | str = THESIS_ROOT) -> HighlightFigureData:
    base = Path(root)
    tables = base / "tables"
    artifacts = base / "artifacts"
    out_fig_dir, out_tab_dir = highlight_output_dirs(base)
    return HighlightFigureData(
        root=base,
        out_fig_dir=out_fig_dir,
        out_tab_dir=out_tab_dir,
        domain_clf=pd.read_csv(tables / "02_domain_shift_hybrid_vs_paired" / "domain_classifier_summary.csv"),
        feat_shift=pd.read_csv(tables / "02_domain_shift_hybrid_vs_paired" / "feature_group_shift_summary.csv"),
        pca_centroids=pd.read_csv(tables / "02_domain_shift_hybrid_vs_paired" / "paired_family_pca_centroids.csv"),
        seed_winner=_normalize_label_columns(pd.read_csv(tables / "04_fpos_model_progression" / "fpos_seed_winner_by_split.csv")),
        fam_pref_r2=_normalize_label_columns(pd.read_csv(tables / "04_fpos_model_progression" / "fpos_family_model_preference_r2.csv")),
        fam_ctx=_normalize_label_columns(pd.read_csv(tables / "04_fpos_model_progression" / "fpos_family_structure_context.csv")),
        rec_pivot=pd.read_csv(tables / "04_fpos_model_progression" / "fpos_recording_r2_pivot.csv", index_col=0).rename(columns=lambda label: normalize_model_label(label)),
        protocol_table=_normalize_label_columns(pd.read_csv(tables / "04_fpos_model_progression" / "fpos_protocol_regime_preference.csv")),
        sm_raw=_load_seed_r2(artifacts / "runs_aggregate/recording_disjoint_main/fpos/fpos_paired_raw/seed_metrics.csv").rename(columns={"r2": "r2_raw"}),
        sm_wave=_load_seed_r2(artifacts / "runs_aggregate/recording_disjoint_main/fpos/fpos_waveform_winner/seed_metrics.csv").rename(columns={"r2": "r2_waveform"}),
        sm_pre=_load_seed_r2(artifacts / "runs_aggregate/recording_disjoint_main/fpos/fpos_pre_waveform_unlabeled/seed_metrics.csv").rename(columns={"r2": "r2_pre_waveform"}),
        sm_ctx=_load_seed_r2(artifacts / "runs_aggregate/recording_disjoint_main/fpos/fpos_paired_context/seed_metrics.csv").rename(columns={"r2": "r2_context"}),
    )


def _configure_style() -> None:
    apply_thesis_style()
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.7,
    })


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)


def _family_from_key(key: object) -> str:
    text = str(key).upper()
    for family in THESIS_HIGHLIGHT_FAMILY_COLORS:
        if family in text:
            return family
    return "OTHER"


def _cluster_seed_order(seed_winner: pd.DataFrame) -> pd.DataFrame:
    matrix = seed_winner[FAMILY_SHARE_COLS].to_numpy(dtype=float)
    if len(seed_winner) <= 2:
        return seed_winner.reset_index(drop=True)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage

        order = leaves_list(linkage(matrix, method="average", metric="euclidean"))
        ordered = seed_winner.iloc[order]
    except Exception:
        dominant = seed_winner[FAMILY_SHARE_COLS].idxmax(axis=1).str.replace("share__PAIRED_", "", regex=False)
        ordered = (
            seed_winner.assign(_dominant=dominant)
            .sort_values(
                by=["_dominant", *FAMILY_SHARE_COLS, "true_iqr", "seed"],
                ascending=[True, False, False, False, False, False, False, True],
            )
            .drop(columns="_dominant")
        )
    return ordered.reset_index(drop=True)


def _build_domain_shift_story(data: HighlightFigureData) -> dict[str, str]:
    auc_data = data.domain_clf[
        (data.domain_clf["comparison"] == "hybrid_vs_paired_all")
        & (data.domain_clf["feature_set"].isin(["unit", "full_context"]))
    ].copy()
    auc_data["feat_label"] = auc_data["feature_set"].map({
        "unit": "Unit features\n(164)",
        "full_context": "Full context\n(685)",
    })
    auc_data = auc_data.sort_values("cv_auc_mean")

    group_shift = data.feat_shift[
        (data.feat_shift["comparison"] == "hybrid_vs_paired_all")
        & (data.feat_shift["feature_set"] == "full_context")
        & (data.feat_shift["feature_group"] != "recording_relative")
    ].copy()
    group_shift = group_shift.sort_values("mean_abs_smd", ascending=True)
    group_shift["display_group"] = group_shift["feature_group"].map(feature_group_display)

    centroids = data.pca_centroids.copy()
    centroids["family"] = centroids["study_set"].map(FAMILY_SHORT)
    paired_center = np.average(
        centroids[["pc1_mean", "pc2_mean"]],
        axis=0,
        weights=centroids["n_rows"].to_numpy(dtype=float),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.8, 4.2),
        gridspec_kw={"width_ratios": [1.0, 1.1, 1.0]},
    )
    ax1, ax2, ax3 = axes
    fig.suptitle("Domain shift story: severe source-target gap and structured target domain", fontsize=14, fontweight="bold", y=1.03)

    colors_a = ["#2E6DA4" if feature == "unit" else "#117733" for feature in auc_data["feature_set"]]
    bars = ax1.barh(
        auc_data["feat_label"],
        auc_data["cv_auc_mean"],
        xerr=auc_data["cv_auc_std"],
        color=colors_a,
        height=0.52,
        capsize=4,
        error_kw={"linewidth": 1.1},
    )
    ax1.axvline(0.5, color="#A3A3A3", lw=1.1, ls="--", zorder=0)
    ax1.axvline(1.0, color="#D4D4D4", lw=0.9, ls=":", zorder=0)
    ax1.set_xlim(0.45, 1.03)
    ax1.set_xlabel("Cross-validated AUC")
    ax1.set_title("A  Hybrid vs paired classifier AUC", loc="left", fontsize=11, fontweight="bold")
    for bar, value in zip(bars, auc_data["cv_auc_mean"].tolist()):
        ax1.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=8.5)

    group_colors = {
        "other": "#888888",
        "waveform_bins": THESIS_HIGHLIGHT_FAMILY_COLORS["BOYDEN"],
        "amplitude": THESIS_HIGHLIGHT_FAMILY_COLORS["CRCNS_HC1"],
        "confusability": THESIS_HIGHLIGHT_FAMILY_COLORS["ENGLISH"],
        "recording_context": "#CC6677",
        "spikeinterface": THESIS_HIGHLIGHT_FAMILY_COLORS["KAMPFF"],
        "acg_bins": THESIS_HIGHLIGHT_FAMILY_COLORS["MEA64C_YGER"],
    }
    ax2.barh(
        group_shift["display_group"],
        group_shift["mean_abs_smd"],
        color=[group_colors.get(group, "#888888") for group in group_shift["feature_group"]],
        height=0.64,
    )
    ax2.set_xlabel("Mean |standardized mean difference|")
    ax2.set_title("B  Full-context feature-group shift", loc="left", fontsize=11, fontweight="bold")
    ax2.axvline(0.5, color="#A3A3A3", lw=1.1, ls="--", zorder=0)

    ax3.scatter(
        paired_center[0],
        paired_center[1],
        marker="X",
        s=130,
        color="#1F2937",
        edgecolors="white",
        linewidths=0.7,
        label="Paired centroid",
        zorder=3,
    )
    for _, row in centroids.iterrows():
        family = row["family"]
        size = 55 + 1.8 * np.sqrt(float(row["n_rows"]))
        ax3.scatter(
            row["pc1_mean"],
            row["pc2_mean"],
            s=size,
            color=THESIS_HIGHLIGHT_FAMILY_COLORS[family],
            edgecolors="white",
            linewidths=0.6,
            alpha=0.95,
            zorder=4,
        )
        ax3.annotate(
            family,
            xy=(row["pc1_mean"], row["pc2_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=THESIS_HIGHLIGHT_FAMILY_COLORS[family],
            fontweight="bold",
        )
    ax3.set_xlabel("PC1 centroid")
    ax3.set_ylabel("PC2 centroid")
    ax3.set_title("C  Paired-family PCA centroids", loc="left", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        0.5,
        -0.04,
        "Claim: The hybrid→paired gap is severe even on unit features, becomes near-perfect with full context, and the paired target domain is internally structured rather than one compact regime.",
        ha="center",
        fontsize=8,
        color="#555555",
        style="italic",
    )
    fig.tight_layout()
    _savefig(fig, data.out_fig_dir / "domain_shift_story.png")
    return {
        "figure_filename": "domain_shift_story.png",
        "scientific_claim": "The hybrid→paired gap is severe even on unit features, becomes near-perfect with full context, and the paired target domain is internally structured rather than one compact regime.",
        "primary_data_sources": "domain_classifier_summary.csv, feature_group_shift_summary.csv, paired_family_pca_centroids.csv",
        "recommended_usage": "main_text",
        "notes": "Use early in the thesis to motivate transfer difficulty without over-assigning causality to one feature block.",
    }


def _build_split_regime_map(data: HighlightFigureData) -> dict[str, str]:
    ordered = _cluster_seed_order(data.seed_winner.copy())
    x = np.arange(len(ordered))

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.9, 7.1),
        gridspec_kw={"height_ratios": [2.9, 1.8]},
        sharex=True,
    )
    fig.suptitle("Split-regime map: clustered test-set composition changes the leading recipe", fontsize=14, fontweight="bold")

    bottom = np.zeros(len(ordered))
    families = ["BOYDEN", "CRCNS_HC1", "ENGLISH", "KAMPFF", "MEA64C_YGER"]
    for family, col in zip(families, FAMILY_SHARE_COLS):
        values = ordered[col].to_numpy(dtype=float)
        ax1.bar(
            x,
            values,
            bottom=bottom,
            color=THESIS_HIGHLIGHT_FAMILY_COLORS[family],
            label=family,
            width=0.84,
            edgecolor="none",
        )
        bottom += values

    seen_labels: set[str] = set()
    for idx, row in ordered.iterrows():
        label = row["label"]
        ax1.scatter(
            idx,
            1.06,
            color=THESIS_HIGHLIGHT_MODEL_COLORS.get(label, "#888888"),
            marker="D",
            s=60,
            zorder=5,
            label=THESIS_HIGHLIGHT_MODEL_SHORT.get(label, label) if label not in seen_labels else "",
        )
        seen_labels.add(label)

    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("Family share of test set")
    ax1.set_title("A  Family composition per seed (ordered by clustered share profile)", loc="left", fontsize=10.5, fontweight="bold")
    family_handles = [mpatches.Patch(color=THESIS_HIGHLIGHT_FAMILY_COLORS[family], label=family) for family in families]
    winner_handles = [
        Line2D([0], [0], marker="D", color=THESIS_HIGHLIGHT_MODEL_COLORS[label], ls="", ms=6, label=THESIS_HIGHLIGHT_MODEL_SHORT.get(label, label))
        for label in ordered["label"].drop_duplicates().tolist()
    ]
    family_legend = ax1.legend(handles=family_handles, title="Family", fontsize=8, ncol=5, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax1.add_artist(family_legend)
    ax1.legend(handles=winner_handles, title="Top recipe", fontsize=8, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.02))

    ax2.plot(x, ordered["r2"], color="#1F2937", linewidth=1.25, zorder=2)
    for label, part in ordered.groupby("label"):
        ax2.scatter(
            part.index.to_numpy(),
            part["r2"],
            color=THESIS_HIGHLIGHT_MODEL_COLORS.get(label, "#888888"),
            edgecolors="white",
            linewidths=0.4,
            s=62,
            zorder=4,
            label=THESIS_HIGHLIGHT_MODEL_SHORT.get(label, label),
        )
    ax2.set_ylabel("Leading-recipe R²")
    ax2.set_title("B  Outcome and difficulty across the same clustered seed order", loc="left", fontsize=10.5, fontweight="bold")

    ax2b = ax2.twinx()
    ax2b.plot(
        x,
        ordered["true_iqr"],
        color="#D97706",
        linewidth=1.2,
        linestyle="--",
        marker="o",
        markersize=3.5,
        alpha=0.9,
        label="Target IQR",
    )
    ax2b.set_ylabel("Target IQR", color="#B45309")
    ax2b.tick_params(axis="y", colors="#B45309")

    ax2.set_xticks(x)
    ax2.set_xticklabels(ordered["seed"].astype(int).astype(str).tolist(), rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("Seed")
    line_handles = [
        Line2D([0], [0], color="#1F2937", lw=1.2, label="Lead R²"),
        Line2D([0], [0], color="#D97706", lw=1.2, ls="--", marker="o", ms=4, label="Target IQR"),
    ]
    ax2.legend(handles=line_handles, fontsize=8, loc="upper left")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax2b.spines["top"].set_visible(False)

    fig.text(
        0.5,
        -0.01,
        "Claim: No single model leads consistently; after ordering seeds by clustered family composition, the top recipe still changes with test-set regime and target difficulty.",
        ha="center",
        fontsize=8,
        color="#555555",
        style="italic",
    )
    fig.tight_layout()
    _savefig(fig, data.out_fig_dir / "split_regime_map.png")
    return {
        "figure_filename": "split_regime_map.png",
        "scientific_claim": "No single model leads consistently; after ordering seeds by clustered family composition, the top recipe still changes with test-set regime and target difficulty.",
        "primary_data_sources": "fpos_seed_winner_by_split.csv",
        "recommended_usage": "supporting_notebook",
        "notes": "Useful in notebook 04 to show regime dependence, but not required in the main thesis if space is tight.",
    }


def _build_family_structure_preference_map(data: HighlightFigureData) -> dict[str, str]:
    pref = data.fam_pref_r2.set_index("study_set").copy()
    pref.index = [FAMILY_SHORT.get(name, name) for name in pref.index]
    pref["_best"] = pref.max(axis=1)
    pref = pref.sort_values("_best", ascending=False).drop(columns="_best")
    pref = pref[pref.mean(axis=0).sort_values(ascending=False).index]
    pref_display = pref.rename(columns=lambda label: THESIS_HIGHLIGHT_MODEL_SHORT.get(label, label))

    fc = data.fam_ctx.copy()
    fc["family"] = fc["study_set"].map(FAMILY_SHORT)
    fc["best_short"] = fc["best_recording_disjoint_label"].map(lambda label: THESIS_HIGHLIGHT_MODEL_SHORT.get(label, label))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.6),
        gridspec_kw={"width_ratios": [1.7, 1.15]},
    )
    fig.suptitle("Family structure and preference: the target domain is heterogeneous", fontsize=14, fontweight="bold")

    sns.heatmap(
        pref_display,
        ax=ax1,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        vmin=-0.5,
        vmax=0.85,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        linewidths=0.5,
        linecolor="#E5E7EB",
        cbar_kws={"shrink": 0.75, "label": "Mean R²"},
    )
    ax1.set_title("A  Family × model preference heatmap", loc="left", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Model variant")
    ax1.set_ylabel("Family")
    ax1.tick_params(axis="x", rotation=20)
    ax1.tick_params(axis="y", rotation=0)
    for tick in ax1.get_yticklabels():
        family = tick.get_text()
        tick.set_color(THESIS_HIGHLIGHT_FAMILY_COLORS.get(family, "#111827"))
        tick.set_fontweight("bold")

    for _, row in fc.iterrows():
        family = row["family"]
        label = row["best_recording_disjoint_label"]
        ax2.scatter(
            row["paired_pca_distance_from_hybrid"],
            row["target_iqr"],
            s=40 + 4.0 * float(row["packaged_labeled_rows"]),
            color=THESIS_HIGHLIGHT_MODEL_COLORS.get(label, "#888888"),
            edgecolors=THESIS_HIGHLIGHT_FAMILY_COLORS[family],
            linewidths=1.3,
            alpha=0.95,
            zorder=4,
        )
        ax2.annotate(
            family,
            xy=(row["paired_pca_distance_from_hybrid"], row["target_iqr"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color=THESIS_HIGHLIGHT_FAMILY_COLORS[family],
            fontweight="bold",
        )
    ax2.set_xlabel("PCA distance from hybrid centroid")
    ax2.set_ylabel("Target IQR")
    ax2.set_title("B  Structure map (color = best recording-disjoint model)", loc="left", fontsize=10.5, fontweight="bold")

    legend_labels = fc["best_recording_disjoint_label"].drop_duplicates().tolist()
    legend_handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor=THESIS_HIGHLIGHT_MODEL_COLORS.get(label, "#888888"), markeredgecolor="#374151", markersize=7, lw=0, label=THESIS_HIGHLIGHT_MODEL_SHORT.get(label, label))
        for label in legend_labels
    ]
    ax2.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax2.text(
        0.02,
        0.98,
        "Point size = packaged labeled rows",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#6B7280",
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.text(
        0.5,
        -0.03,
        "Claim: Model preference aligns with family regime differences: KAMPFF is farthest from hybrid and easiest, CRCNS_HC1 is high-variance and hardest, and no single recipe dominates all families.",
        ha="center",
        fontsize=8,
        color="#555555",
        style="italic",
    )
    fig.tight_layout()
    _savefig(fig, data.out_fig_dir / "family_structure_preference_map.png")
    return {
        "figure_filename": "family_structure_preference_map.png",
        "scientific_claim": "Model preference aligns with family regime differences: KAMPFF is farthest from hybrid and easiest, CRCNS_HC1 is high-variance and hardest, and no single recipe dominates all families.",
        "primary_data_sources": "fpos_family_model_preference_r2.csv, fpos_family_structure_context.csv",
        "recommended_usage": "main_text",
        "notes": "This is the clearest family-heterogeneity figure because it combines outcome preference with structural context.",
    }


def _build_raw_vs_waveform_complementarity(data: HighlightFigureData) -> dict[str, str]:
    join = (
        data.sm_raw
        .merge(data.sm_wave, on="seed")
        .merge(data.sm_pre, on="seed")
        .merge(data.sm_ctx, on="seed")
    )
    join["delta_r2_waveform_minus_raw"] = join["r2_waveform"] - join["r2_raw"]

    split_profile = data.seed_winner[
        ["seed", "label", "r2", "true_mean", "true_std", "true_iqr", *FAMILY_SHARE_COLS, *FAMILY_ROW_COLS]
    ].rename(columns={"label": "winner_label", "r2": "winner_r2"})
    join = join.merge(split_profile, on="seed")
    join["dominant_family"] = (
        join[FAMILY_SHARE_COLS]
        .rename(columns=lambda col: col.replace("share__PAIRED_", ""))
        .idxmax(axis=1)
    )
    join["winner_group"] = np.where(
        join["winner_label"] == WAVEFORM_AUGMENTED_LABEL,
        WAVEFORM_AUGMENTED_LABEL,
        np.where(join["winner_label"] == "Paired-only raw", "Paired-only raw", "Other winner"),
    )
    join.to_csv(data.out_tab_dir / "raw_vs_waveform_seed_join.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4))
    fig.suptitle(f"{WAVEFORM_AUGMENTED_LABEL} vs paired-only raw: regime-dependent complementarity", fontsize=14, fontweight="bold")
    ax1, ax2, ax3 = axes

    lower = min(join["r2_raw"].min(), join["r2_waveform"].min()) - 0.03
    upper = max(join["r2_raw"].max(), join["r2_waveform"].max()) + 0.03
    ax1.plot([lower, upper], [lower, upper], color="#A3A3A3", lw=1.0, ls="--", zorder=0)
    marker_map = {WAVEFORM_AUGMENTED_LABEL: "D", "Paired-only raw": "o", "Other winner": "s"}
    for winner_group, marker in marker_map.items():
        subset = join[join["winner_group"] == winner_group]
        for family, part in subset.groupby("dominant_family"):
            ax1.scatter(
                part["r2_raw"],
                part["r2_waveform"],
                color=THESIS_HIGHLIGHT_FAMILY_COLORS.get(family, "#888888"),
                marker=marker,
                s=64 if winner_group != "Other winner" else 56,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
    ax1.set_xlabel("R² (paired-only raw)")
    ax1.set_ylabel(f"R² ({WAVEFORM_AUGMENTED_LABEL.lower()})")
    ax1.set_title("A  Seed-level comparison", loc="left", fontsize=10.5, fontweight="bold")
    for _, row in join.nlargest(2, "delta_r2_waveform_minus_raw").iterrows():
        ax1.annotate(f"s{int(row['seed'])}", xy=(row["r2_raw"], row["r2_waveform"]), xytext=(5, -10), textcoords="offset points", fontsize=7, color="#117733")
    for _, row in join.nsmallest(2, "delta_r2_waveform_minus_raw").iterrows():
        ax1.annotate(f"s{int(row['seed'])}", xy=(row["r2_raw"], row["r2_waveform"]), xytext=(5, 5), textcoords="offset points", fontsize=7, color="#444444")
    family_handles = [
        mpatches.Patch(color=THESIS_HIGHLIGHT_FAMILY_COLORS[family], label=family)
        for family in join["dominant_family"].drop_duplicates().tolist()
    ]
    legend_labels = {
        WAVEFORM_AUGMENTED_LABEL: "Waveform-augmented leads",
        "Paired-only raw": "Raw leads",
        "Other winner": "Other recipe leads",
    }
    winner_handles = [
        Line2D([0], [0], marker=marker_map[group], color="#666666", ls="", ms=7, label=legend_labels[group])
        for group in [WAVEFORM_AUGMENTED_LABEL, "Paired-only raw", "Other winner"]
    ]
    ax1.legend(handles=family_handles + winner_handles, fontsize=7.5, ncol=2, loc="upper left")

    sorted_join = join.sort_values("delta_r2_waveform_minus_raw").reset_index(drop=True)
    ax2.bar(
        range(len(sorted_join)),
        sorted_join["delta_r2_waveform_minus_raw"],
        color=np.where(sorted_join["delta_r2_waveform_minus_raw"] >= 0, "#117733", "#444444"),
        width=0.82,
    )
    ax2.axhline(0, color="#9CA3AF", lw=0.9)
    ax2.set_xticks(range(len(sorted_join)))
    ax2.set_xticklabels(sorted_join["seed"].astype(int).astype(str).tolist(), rotation=45, ha="right", fontsize=7.5)
    for tick, family in zip(ax2.get_xticklabels(), sorted_join["dominant_family"]):
        tick.set_color(THESIS_HIGHLIGHT_FAMILY_COLORS.get(family, "#888888"))
    ax2.set_ylabel("ΔR² (waveform - raw)")
    ax2.set_title("B  Per-seed waveform margin", loc="left", fontsize=10.5, fontweight="bold")
    ax2.legend(
        handles=[
            mpatches.Patch(color="#117733", label="Waveform leads"),
            mpatches.Patch(color="#444444", label="Raw leads"),
        ],
        fontsize=8,
        loc="upper left",
    )

    group_specs = [
        (WAVEFORM_AUGMENTED_LABEL, "Waveform-augmented leads", "#117733"),
        ("Paired-only raw", "Raw wins", "#444444"),
        ("Other winner", "Other recipe leads", "#9CA3AF"),
    ]
    families = ["BOYDEN", "CRCNS_HC1", "ENGLISH", "KAMPFF", "MEA64C_YGER"]
    x_positions = np.arange(len(families))
    width = 0.24
    offsets = [-width, 0, width]
    summary_lines: list[str] = []
    for offset, (group_key, group_label, edgecolor) in zip(offsets, group_specs):
        subset = join[join["winner_group"] == group_key]
        if subset.empty:
            continue
        means = [subset[f"share__PAIRED_{family}"].mean() for family in families]
        bars = ax3.bar(
            x_positions + offset,
            means,
            width=width * 0.92,
            color=[THESIS_HIGHLIGHT_FAMILY_COLORS[family] for family in families],
            alpha=0.86 if group_key != "Other winner" else 0.55,
            edgecolor=edgecolor,
            linewidth=1.1,
            label=f"{group_label} (n={len(subset)})",
        )
        for bar, value in zip(bars, means):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
                color="#374151",
            )
        summary_lines.append(
            f"{group_label}: KAMPFF {subset['share__PAIRED_KAMPFF'].mean():.2f}, ENGLISH {subset['share__PAIRED_ENGLISH'].mean():.2f}, IQR {subset['true_iqr'].mean():.3f}"
        )
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels([family.replace("_", "\n") for family in families], fontsize=8)
    ax3.set_ylabel("Mean family share")
    ax3.set_title("C  Regime summary by leading recipe", loc="left", fontsize=10.5, fontweight="bold")
    ax3.legend(fontsize=7.5, loc="upper right")
    ax3.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#4B5563",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F9FAFB", "edgecolor": "#E5E7EB"},
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    mean_raw = join["r2_raw"].mean()
    mean_wave = join["r2_waveform"].mean()
    fig.text(
        0.5,
        -0.03,
        f"Claim: Raw has the stronger mean R² ({mean_raw:.3f} vs {mean_wave:.3f}); waveform gains are regime-dependent and coincide with slightly higher KAMPFF share, slightly lower ENGLISH share, and lower target IQR.",
        ha="center",
        fontsize=8,
        color="#555555",
        style="italic",
    )
    fig.tight_layout()
    _savefig(fig, data.out_fig_dir / "raw_vs_waveform_complementarity.png")
    return {
        "figure_filename": "raw_vs_waveform_complementarity.png",
        "scientific_claim": f"Raw has the stronger mean R² ({mean_raw:.3f} vs {mean_wave:.3f}); waveform gains are regime-dependent and coincide with slightly higher KAMPFF share, slightly lower ENGLISH share, and lower target IQR.",
        "primary_data_sources": "fpos_paired_raw/seed_metrics.csv, fpos_waveform_winner/seed_metrics.csv, fpos_seed_winner_by_split.csv",
        "recommended_usage": "main_text",
        "notes": "This is the cleanest explanation of why waveform is complementary rather than universally best.",
    }


def _build_recording_performance_landscape(data: HighlightFigureData) -> dict[str, str]:
    preferred_cols = [
        ("Paired-only raw", "Paired raw"),
        ("Pseudo+anchor", "Pre-waveform"),
        ("Waveform+", "Waveform+"),
    ]
    fallback_cols = [
        ("Pseudo+anchor", "Pre-waveform"),
        ("Waveform+", "Waveform+"),
        ("Paired context", "Paired context"),
    ]
    selected = preferred_cols if all(col in data.rec_pivot.columns for col, _ in preferred_cols) else fallback_cols
    df = data.rec_pivot[[col for col, _ in selected]].rename(columns=dict(selected)).copy()
    clip_low = -2.0
    clipped = df.clip(lower=clip_low)
    clipped["family"] = [_family_from_key(index) for index in clipped.index]
    clipped["_median"] = clipped[list(clipped.columns[:-1])].median(axis=1)
    clipped = clipped.sort_values("_median", ascending=True)

    model_color_lookup = {
        "Paired raw": THESIS_HIGHLIGHT_MODEL_COLORS["Paired-only raw"],
        "Pre-waveform": THESIS_HIGHLIGHT_MODEL_COLORS["Pre-waveform unlabeled structure"],
        "Waveform+": THESIS_HIGHLIGHT_MODEL_COLORS[WAVEFORM_AUGMENTED_LABEL],
        "Paired context": THESIS_HIGHLIGHT_MODEL_COLORS["Paired-only context"],
    }

    fig, ax = plt.subplots(figsize=(8.7, 6.5))
    y_positions = np.arange(len(clipped))
    offsets = np.linspace(-0.22, 0.22, len(selected))
    for offset, label in zip(offsets, [short for _, short in selected]):
        ax.scatter(
            clipped[label],
            y_positions + offset,
            color=model_color_lookup[label],
            s=44,
            edgecolors="white",
            linewidths=0.3,
            zorder=4,
            label=label,
        )

    strip_x = clip_low - 0.15
    for y_index, (_, row) in enumerate(clipped.iterrows()):
        family = row["family"]
        ax.plot([strip_x - 0.04, strip_x + 0.02], [y_index, y_index], color=THESIS_HIGHLIGHT_FAMILY_COLORS.get(family, "#888888"), lw=5, solid_capstyle="butt")

    short_labels = []
    for record in clipped.index:
        parts = str(record).split("·")
        short_labels.append(parts[-1].strip()[:30] if len(parts) > 1 else str(record)[:30])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(short_labels, fontsize=7.5)
    for tick, (_, row) in zip(ax.get_yticklabels(), clipped.iterrows()):
        tick.set_color(THESIS_HIGHLIGHT_FAMILY_COLORS.get(row["family"], "#374151"))

    clipped_any = df.lt(clip_low).any(axis=1)
    if clipped_any.any():
        clipped_records = clipped.index[clipped_any.reindex(clipped.index, fill_value=False)]
        clipped_y = [y_positions[clipped.index.get_loc(record)] for record in clipped_records]
        ax.scatter([clip_low] * len(clipped_y), clipped_y, marker="<", color="#CC4444", s=28, zorder=5, label=f"Clipped (< {clip_low})")

    ax.axvline(0, color="#9CA3AF", lw=0.9, ls="--", zorder=0)
    ax.set_xlabel("R² (clipped at -2)")
    ax.set_title("Supporting figure: recording heterogeneity among competitive `fpos` models", fontsize=12.5, fontweight="bold")
    family_handles = [mpatches.Patch(color=THESIS_HIGHLIGHT_FAMILY_COLORS[family], label=family) for family in THESIS_HIGHLIGHT_FAMILY_COLORS]
    family_legend = ax.legend(handles=family_handles, fontsize=8, loc="upper left", title="Family", title_fontsize=8)
    ax.add_artist(family_legend)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.5,
        -0.01,
        "Claim: Within the competitive models that have recording-level summaries, recording difficulty dominates model-to-model differences; KAMPFF recordings remain consistently strong while many ENGLISH recordings stay unstable.",
        ha="center",
        fontsize=8,
        color="#555555",
        style="italic",
    )
    fig.tight_layout()
    _savefig(fig, data.out_fig_dir / "recording_performance_landscape.png")
    return {
        "figure_filename": "recording_performance_landscape.png",
        "scientific_claim": "Within the competitive models that have recording-level summaries, recording difficulty dominates model-to-model differences; KAMPFF recordings remain consistently strong while many ENGLISH recordings stay unstable.",
        "primary_data_sources": "fpos_recording_r2_pivot.csv",
        "recommended_usage": "appendix_or_supporting",
        "notes": "Keep as a supporting or appendix figure unless per-recording raw summaries become available.",
    }


def _build_protocol_tradeoff(data: HighlightFigureData) -> dict[str, str]:
    protocol = data.protocol_table.copy().sort_values("recording_disjoint_r2", ascending=False).reset_index(drop=True)
    protocol["short_label"] = protocol["label"].map(lambda label: THESIS_HIGHLIGHT_MODEL_SHORT.get(label, label))

    fig, ax = plt.subplots(figsize=(7.7, 4.3))
    fig.suptitle("Protocol tradeoff: recording-disjoint vs LOFO family-held-out", fontsize=14, fontweight="bold")

    x = np.arange(len(protocol))
    width = 0.22
    metric_specs = [
        ("recording_disjoint_r2", "Recording-disjoint R²", "#444444", -width),
        ("macro_r2", "LOFO macro R²", "#2E6DA4", 0.0),
        ("weighted_r2", "LOFO weighted R²", "#44AA99", width),
    ]
    for column, label, color, offset in metric_specs:
        ax.bar(x + offset, protocol[column], width=width * 0.9, color=color, label=label, zorder=3)
    ax.scatter(
        x,
        protocol["worst_family_r2"],
        color="#CC4444",
        marker="_",
        s=180,
        linewidths=2.5,
        zorder=5,
        label="Worst-family R² (LOFO)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(protocol["short_label"], fontsize=10)
    ax.set_ylabel("R²")
    ax.set_ylim(0, 0.55)

    rec_winner_idx = int(protocol["recording_disjoint_r2"].idxmax())
    lofo_winner_idx = int(protocol["macro_r2"].idxmax())
    if rec_winner_idx != lofo_winner_idx:
        ax.annotate(
            "Rec-disjoint leader",
            xy=(rec_winner_idx - width, protocol.loc[rec_winner_idx, "recording_disjoint_r2"]),
            xytext=(rec_winner_idx - 0.55, protocol.loc[rec_winner_idx, "recording_disjoint_r2"] + 0.04),
            fontsize=8,
            color="#444444",
            arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.8},
        )
        ax.annotate(
            "LOFO macro leader",
            xy=(lofo_winner_idx, protocol.loc[lofo_winner_idx, "macro_r2"]),
            xytext=(lofo_winner_idx + 0.05, protocol.loc[lofo_winner_idx, "macro_r2"] + 0.04),
            fontsize=8,
            color="#2E6DA4",
            arrowprops={"arrowstyle": "->", "color": "#2E6DA4", "lw": 0.8},
        )
    for idx, value in enumerate(protocol["worst_family_r2"].tolist()):
        ax.text(idx, value - 0.025, f"{value:.2f}", ha="center", fontsize=7.5, color="#CC4444")

    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5,
        -0.03,
        "Claim: Recording-disjoint and LOFO protocols do not reward the same variant; the protocol choice materially changes which model appears safest.",
        ha="center",
        fontsize=8,
        color="#555555",
        style="italic",
    )
    fig.tight_layout()
    _savefig(fig, data.out_fig_dir / "protocol_tradeoff.png")
    return {
        "figure_filename": "protocol_tradeoff.png",
        "scientific_claim": "Recording-disjoint and LOFO protocols do not reward the same variant; the protocol choice materially changes which model appears safest.",
        "primary_data_sources": "fpos_protocol_regime_preference.csv",
        "recommended_usage": "main_text",
        "notes": "Use with the family-heterogeneity chapter to justify reporting both protocols.",
    }


def _write_manifest(data: HighlightFigureData, rows: list[dict[str, str]]) -> pd.DataFrame:
    manifest = pd.DataFrame(rows)
    manifest.to_csv(data.out_tab_dir / "figure_manifest.csv", index=False)
    return manifest


def build_highlight_figures(root: Path | str = THESIS_ROOT) -> pd.DataFrame:
    _configure_style()
    data = load_highlight_figure_data(root)
    manifest_rows = [
        _build_domain_shift_story(data),
        _build_split_regime_map(data),
        _build_family_structure_preference_map(data),
        _build_raw_vs_waveform_complementarity(data),
        _build_recording_performance_landscape(data),
        _build_protocol_tradeoff(data),
    ]
    manifest = _write_manifest(data, manifest_rows)
    print(f"Saved highlight figures to {data.out_fig_dir}")
    print(f"Saved highlight tables to {data.out_tab_dir}")
    return manifest
