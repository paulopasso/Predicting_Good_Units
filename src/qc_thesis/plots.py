from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import (
    PRE_WAVEFORM_LABEL,
    THESIS_ROOT,
    WAVEFORM_AUGMENTED_LABEL,
    REDUCED_LATENT_CONTEXT_LABEL,
    feature_group_display,
)


PALETTE = {
    "blue": "#2957a4",
    "teal": "#1f8a70",
    "orange": "#d98e04",
    "red": "#b33a3a",
    "slate": "#5b6675",
    "navy": "#18314f",
    "purple": "#6c4a9a",
    "grey": "#9aa5b1",
    "light_blue": "#9ec5fe",
    "light_orange": "#f6c177",
}

FAMILY_COLORS = {
    "BOYDEN": "#e64b35",
    "CRCNS_HC1": "#4dbbd5",
    "ENGLISH": "#00a087",
    "KAMPFF": "#3c5488",
    "MEA64C_YGER": "#f39b7f",
}

THESIS_HIGHLIGHT_FAMILY_COLORS = {
    "BOYDEN": "#4477AA",
    "CRCNS_HC1": "#EE6677",
    "ENGLISH": "#228833",
    "KAMPFF": "#CCBB44",
    "MEA64C_YGER": "#AA3377",
}

THESIS_HIGHLIGHT_MODEL_COLORS = {
    "Paired-only raw": "#444444",
    "Paired-only context": "#2E6DA4",
    PRE_WAVEFORM_LABEL: "#44AA99",
    WAVEFORM_AUGMENTED_LABEL: "#117733",
    "Waveform winner": "#117733",
    "Context-first source stack": "#888888",
}

THESIS_HIGHLIGHT_MODEL_SHORT = {
    "Paired-only raw": "Raw",
    "Paired-only context": "Context",
    PRE_WAVEFORM_LABEL: "Trust-filtered",
    WAVEFORM_AUGMENTED_LABEL: "Waveform+",
    "Waveform winner": "Waveform+",
    "Context-first source stack": "Ctx-stack",
}

GROUP_COLORS = {
    "recording_context": PALETTE["navy"],
    "spikeinterface_qc": "#2d8b57",
    "waveform": "#1f7a8c",
    "source_transfer": PALETTE["purple"],
    "acg": "#6a3d9a",
    "amplitude": "#e05c5c",
    "other": PALETTE["grey"],
}


LABEL_SHORT = {
    "Dummy paired mean": "Dummy",
    "Paired-only raw": "Paired-only raw",
    "Hybrid-only XGBoost": "Hybrid-only XGBoost",
    "Calibrated hybrid XGBoost": "Hybrid calibrated",
    "Hybrid+paired basic XGBoost": "Hybrid+paired",
    "Hybrid stack residual": "Hybrid residual",
    "Paired-only context": "Paired-only context",
    "Context-first source stack": "Context stack",
    "Best SSL neural": "SSL",
    "Best MMD neural": "MMD",
    PRE_WAVEFORM_LABEL: "Trust-filtered",
    WAVEFORM_AUGMENTED_LABEL: "Waveform+",
    "Waveform winner": "Waveform+",
    "Calibrated hybrid": "Hybrid calibrated",
    "Source-reweighted full context": "Source reweighted",
    "Embed-nommd full context": "Embed full ctx",
    REDUCED_LATENT_CONTEXT_LABEL: "Reduced-latent context",
    "Reduced-latent winner": "Reduced-latent context",
    "Transplanted `fpos` waveform recipe": "Transplanted fpos",
}


BENCHMARK_MODEL_COLORS = {
    "Dummy paired mean": "#7A7A7A",
    "Paired-only raw": "#2F2F2F",
    "Hybrid-only XGBoost": "#E07A1F",
    "Calibrated hybrid XGBoost": "#F2A65A",
    "Calibrated hybrid": "#F2A65A",
    "Hybrid+paired basic XGBoost": "#C06C84",
    "Hybrid stack residual": "#C65D0E",
    "Paired-only context": "#2F6690",
    "Context-first source stack": "#4EA1D3",
    "Source-reweighted full context": "#6BAED6",
    "Embed-nommd full context": "#1B9E77",
    PRE_WAVEFORM_LABEL: "#2A9D8F",
    WAVEFORM_AUGMENTED_LABEL: "#1B7F3B",
    "Waveform winner": "#1B7F3B",
    REDUCED_LATENT_CONTEXT_LABEL: "#7B61FF",
    "Reduced-latent winner": "#7B61FF",
    "Transplanted fpos waveform recipe": "#C9A227",
    "Transplanted `fpos` waveform recipe": "#C9A227",
    "Best SSL neural": "#C44E52",
    "Best MMD neural": "#DD5A8C",
    "Hybrid residual": "#C65D0E",
    "Reduced-latent context": "#7B61FF",
    "Transplanted fpos": "#C9A227",
}


def apply_thesis_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#d6dde6",
        "axes.linewidth": 0.8,
        "grid.color": "#dce3eb",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "axes.grid.axis": "y",
        "axes.grid.which": "major",
        "legend.frameon": False,
        "figure.autolayout": False,
    })
    # Prevent double figure display in Jupyter notebooks.
    # With %matplotlib inline, IPython registers a flush_figures() post-execute hook
    # that auto-shows all open figures at cell end. Notebook cells already return `fig`
    # as the last expression (shown once via IPython rich output), so unregistering the
    # hook avoids every plot appearing twice.
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            try:
                from matplotlib_inline.backend_inline import set_matplotlib_close
                set_matplotlib_close(True)
            except Exception:
                pass
            # Remove every inline flush callback by module/name, not just one object
            # identity, because some notebook environments can register wrappers
            # and trigger duplicate figure renders.
            callbacks = list(ip.events.callbacks.get("post_execute", []))
            for callback in callbacks:
                module = getattr(callback, "__module__", "")
                name = getattr(callback, "__name__", "")
                callback_repr = repr(callback)
                if (
                    (module.startswith("matplotlib_inline") and name == "flush_figures")
                    or "flush_figures" in callback_repr
                ):
                    try:
                        ip.events.unregister("post_execute", callback)
                    except ValueError:
                        pass
    except ImportError:
        pass


def notebook_output_dirs(notebook_stem: str) -> tuple[Path, Path]:
    fig_dir = THESIS_ROOT / "figures" / notebook_stem
    table_dir = THESIS_ROOT / "tables" / notebook_stem
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def save_figure(fig, output_dir: Path, stem: str) -> Path:
    path = output_dir / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    return path


def save_table(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    path = output_dir / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def _shorten(labels: list[str]) -> list[str]:
    return [LABEL_SHORT.get(x, x) for x in labels]


def _display_group_label(raw: object) -> str:
    text = str(raw)
    if "::" in text:
        family, *_middle, tail = text.split("::")
        return f"{family.replace('PAIRED_', '')} · {tail}"
    return text.replace("PAIRED_", "")


def _feature_group(feature: str) -> str:
    feat = str(feature).lower()
    if feat == "source_pred":
        return "source_transfer"
    if "_recording_" in feat:
        return "recording_context"
    if feat.startswith("si_"):
        return "spikeinterface_qc"
    if feat.startswith("wf_") or feat in {
        "peak_to_trough_uv",
        "post_trough_peak_uv",
        "pre_trough_peak_uv",
        "trough_time_ms",
        "half_width_ms",
        "repolarization_slope",
        "wf_pre_trough_slope",
        "wf_post_peak_slope",
        "wf_prepeak_ratio",
        "wf_prepeak_to_trough_ms",
        "wf_rebound_ratio",
        "wf_asymmetry",
        "wf_trough_width_75_ms",
        "wf_trough_width_50_ms",
        "wf_trough_width_25_ms",
        "wf_zero_cross_pre_ms",
        "wf_pos_area_uv_ms",
        "wf_neg_area_uv_ms",
        "wf_pos_neg_area_ratio",
        "wf_energy",
        "spread_um",
        "spread_weighted_um",
        "center_of_mass_um",
        "n_active_channels",
        "n_channels",
        "peak_channel_depth_um",
    }:
        return "waveform"
    if feat.startswith("acg_") or feat == "burst_index":
        return "acg"
    if feat.startswith("amp_"):
        return "amplitude"
    return "other"


def _benchmark_color(label: str, *, is_best: bool = False) -> str:
    if label in BENCHMARK_MODEL_COLORS:
        return BENCHMARK_MODEL_COLORS[label]
    lowered = label.lower()
    if "dummy" in lowered:
        return PALETTE["grey"]
    if "ssl" in lowered or "mmd" in lowered:
        return "#cc6677"
    if "waveform" in lowered or "transplant" in lowered:
        return PALETTE["purple"]
    if "family-balanced" in lowered:
        return PALETTE["orange"]
    return PALETTE["blue"]


def _series_colors(n: int) -> list[str]:
    base = [
        PALETTE["blue"],
        PALETTE["teal"],
        PALETTE["orange"],
        PALETTE["red"],
        PALETTE["purple"],
        PALETTE["slate"],
    ]
    if n <= len(base):
        return base[:n]
    repeats = (n // len(base)) + 1
    return (base * repeats)[:n]


def _clean_axes(ax, *, grid_axis: str = "x", zero_line: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d6dde6")
    ax.spines["bottom"].set_color("#d6dde6")
    ax.grid(axis=grid_axis, color="#dce3eb", linewidth=0.8, alpha=0.9)
    if grid_axis == "x":
        ax.grid(axis="y", visible=False)
    elif grid_axis == "y":
        ax.grid(axis="x", visible=False)
    if zero_line:
        ax.axvline(0, color=PALETTE["slate"], linestyle="--", linewidth=1.0, zorder=1)


def _annotate_barh(ax, bars, values: list[float]) -> None:
    finite = [float(v) for v in values if pd.notna(v)]
    span = (max(finite) - min(finite)) if finite else 1.0
    pad = max(span * 0.02, 0.004)
    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue
        x = float(value)
        y = bar.get_y() + bar.get_height() / 2
        if x >= 0:
            ax.text(x + pad, y, f"{x:.3f}", va="center", ha="left", fontsize=8, color="#334155")
        else:
            ax.text(x - pad, y, f"{x:.3f}", va="center", ha="right", fontsize=8, color="#334155")


def _category_color(key: object, idx: int) -> str:
    label = _display_group_label(key)
    if label in FAMILY_COLORS:
        return FAMILY_COLORS[label]
    lowered = str(key).lower()
    if lowered == "hybrid":
        return PALETTE["blue"]
    if lowered == "paired":
        return PALETTE["orange"]
    return _series_colors(idx + 1)[idx]


def plot_benchmark_metric(table: pd.DataFrame, metric: str = "r2", title: str | None = None):
    df = table.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    ascending = metric.lower() == "mae"
    df = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.52 * len(df))))
    best_idx = 0 if not df.empty else -1
    colors = [
        _benchmark_color(str(label), is_best=(idx == best_idx))
        for idx, label in enumerate(df["label"].tolist())
    ]
    bars = ax.barh(_shorten(df["label"].tolist()), df[metric].tolist(), color=colors, edgecolor="white", linewidth=0.7)
    ax.invert_yaxis()
    _clean_axes(ax, grid_axis="x", zero_line=(metric.lower() != "mae"))
    _annotate_barh(ax, bars, df[metric].tolist())
    if metric.lower() == "r2":
        ax.set_xlabel("R²")
    else:
        ax.set_xlabel("MAE")
    ax.set_title(title or f"Benchmark comparison by {metric.upper()}")
    fig.tight_layout()
    return fig, ax


def plot_group_metric(summary_df: pd.DataFrame, group_col: str, metric: str = "r2", title: str | None = None):
    column_col = "label" if "label" in summary_df.columns else "variant_id"
    pivot = summary_df.pivot_table(index=group_col, columns=column_col, values=metric, aggfunc="mean").sort_index()
    pivot = pivot.rename(columns=LABEL_SHORT)
    pivot.index = [_display_group_label(x) for x in pivot.index]
    metric_lower = metric.lower()

    if metric_lower == "mae":
        ordering = pivot.min(axis=1, skipna=True).sort_values(ascending=True).index
    else:
        ordering = pivot.max(axis=1, skipna=True).sort_values(ascending=False).index
    pivot = pivot.loc[ordering]

    if pivot.shape[1] == 1:
        series = pd.to_numeric(pivot.iloc[:, 0], errors="coerce")
        fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(series))))
        labels = list(series.index)
        colors = [_category_color(label, idx) for idx, label in enumerate(labels)]
        bars = ax.barh(labels, series.tolist(), color=colors, edgecolor="white", linewidth=0.7)
        ax.invert_yaxis()
        _clean_axes(ax, grid_axis="x", zero_line=(metric_lower != "mae"))
        _annotate_barh(ax, bars, series.tolist())
    elif len(pivot) > 10:
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(pivot.columns) + 1.5), max(5, 0.34 * len(pivot) + 1.8)))
        values = pivot.to_numpy(dtype=float)
        if metric_lower == "mae":
            cmap = plt.get_cmap("YlOrBr_r")
            norm = mcolors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
        else:
            cmap = plt.get_cmap("RdYlGn")
            limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
            norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
        ax.grid(False)
        cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label("MAE" if metric_lower == "mae" else "R²")
        if pivot.shape[0] * pivot.shape[1] <= 80:
            for row_idx in range(pivot.shape[0]):
                for col_idx in range(pivot.shape[1]):
                    value = values[row_idx, col_idx]
                    if np.isnan(value):
                        continue
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7, color="#111827")
    else:
        fig, ax = plt.subplots(figsize=(11, max(4.5, 0.48 * len(pivot))))
        y = np.arange(len(pivot))
        offsets = np.linspace(-0.28, 0.28, len(pivot.columns))
        bar_height = min(0.65 / max(len(pivot.columns), 1), 0.22)
        colors = _series_colors(len(pivot.columns))
        for idx, column in enumerate(pivot.columns):
            values = pd.to_numeric(pivot[column], errors="coerce").tolist()
            bars = ax.barh(y + offsets[idx], values, height=bar_height, color=colors[idx], edgecolor="white", linewidth=0.6, label=column)
            if len(pivot) <= 7:
                _annotate_barh(ax, bars, values)
        ax.set_yticks(y)
        ax.set_yticklabels(pivot.index.tolist())
        ax.invert_yaxis()
        _clean_axes(ax, grid_axis="x", zero_line=(metric_lower != "mae"))
        ax.legend(loc="lower right", ncol=min(2, len(pivot.columns)))

    ax.set_xlabel("MAE" if metric_lower == "mae" else "R²")
    ax.set_ylabel(group_col.replace("_", " ").title())
    ax.set_title(title or f"{metric.upper()} by {group_col}")
    fig.tight_layout()
    return fig, ax, pivot.reset_index()


def plot_domain_shift_separability(df: pd.DataFrame):
    return plot_domain_auc_annotated(df)


def plot_top_feature_shift(df: pd.DataFrame, top_n: int = 15):
    plot_df = df.sort_values("abs_smd", ascending=False).head(top_n).iloc[::-1].copy()
    plot_df["feature_group"] = plot_df["feature"].map(_feature_group)
    colors = [GROUP_COLORS.get(group, PALETTE["grey"]) for group in plot_df["feature_group"]]
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.4 * len(plot_df))))
    bars = ax.barh(plot_df["feature"], plot_df["abs_smd"], color=colors, edgecolor="white", linewidth=0.7)
    _clean_axes(ax, grid_axis="x")
    _annotate_barh(ax, bars, plot_df["abs_smd"].tolist())
    present_groups = list(dict.fromkeys(plot_df["feature_group"].tolist()))
    if present_groups:
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor=GROUP_COLORS[group], label=feature_group_display(group))
            for group in present_groups
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8, ncol=1)
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_title("Top shifted features")
    fig.tight_layout()
    return fig, ax


def plot_pca_projection(df: pd.DataFrame, color_col: str = "dataset_type"):
    fig, ax = plt.subplots(figsize=(8.5, 6.3))
    grouped = list(df.groupby(color_col, dropna=False))
    for idx, (key, part) in enumerate(grouped):
        color = _category_color(key, idx)
        label = _display_group_label(key)
        ax.scatter(
            part["pc1"],
            part["pc2"],
            s=18,
            alpha=0.68,
            label=f"{label} (n={len(part)})",
            color=color,
            edgecolors="white",
            linewidths=0.25,
            rasterized=len(df) > 1500,
        )
        ax.scatter(
            float(part["pc1"].mean()),
            float(part["pc2"].mean()),
            marker="X",
            s=120,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection")
    ax.grid(True, color="#e8edf3", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig, ax


def plot_pca_projection_3d(df: pd.DataFrame, color_col: str = "dataset_type"):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    grouped = list(df.groupby(color_col, dropna=False))
    for idx, (key, part) in enumerate(grouped):
        color = _category_color(key, idx)
        ax.scatter(part["pc1"], part["pc2"], part["pc3"], s=12, alpha=0.62, label=_display_group_label(key), color=color)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA projection")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_label_budget(df: pd.DataFrame, target: str = "fpos"):
    plot_df = df[df["target"] == target].copy()
    plot_df["budget_rows"] = pd.to_numeric(plot_df["budget_rows"], errors="coerce")
    plot_df["mean_r2"] = pd.to_numeric(plot_df["mean_r2"], errors="coerce")
    if "std_r2" in plot_df.columns:
        plot_df["std_r2"] = pd.to_numeric(plot_df["std_r2"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = _series_colors(plot_df["variant_id"].nunique())
    current_n = 107 if 107 in plot_df["budget_rows"].dropna().tolist() else None
    for idx, ((label, part), color) in enumerate(zip(plot_df.groupby("variant_id"), colors * 10)):
        part = part.sort_values("budget_rows")
        ax.plot(part["budget_rows"], part["mean_r2"], marker="o", linewidth=2.1, markersize=6, label=LABEL_SHORT.get(label, label), color=color)
        if "std_r2" in part.columns:
            std = pd.to_numeric(part["std_r2"], errors="coerce")
            if std.notna().any() and std.fillna(0).gt(0).any():
                ax.fill_between(part["budget_rows"], part["mean_r2"] - std.fillna(0), part["mean_r2"] + std.fillna(0), color=color, alpha=0.12)
        end_x = float(part["budget_rows"].iloc[-1])
        end_y = float(part["mean_r2"].iloc[-1])
        ax.text(end_x + 1.5, end_y, LABEL_SHORT.get(label, label), color=color, fontsize=8, va="center")
    _clean_axes(ax, grid_axis="y", zero_line=True)
    if current_n is not None:
        ax.axvline(current_n, color=PALETTE["slate"], linestyle=":", linewidth=1.1)
        ax.text(current_n + 2, ax.get_ylim()[0] + 0.03, "current n", fontsize=8, color=PALETTE["slate"])
    ax.set_xlabel("Paired adaptation rows")
    ax.set_ylabel("Mean R²")
    ax.set_title(f"Label-budget curve: {target}")
    fig.tight_layout()
    return fig, ax


def plot_fpos_shap_top(df: pd.DataFrame, top_n: int = 15):
    plot_df = df.head(top_n).iloc[::-1].copy()
    plot_df["feature_group"] = plot_df["feature"].map(_feature_group)
    colors = [GROUP_COLORS.get(group, PALETTE["grey"]) for group in plot_df["feature_group"]]
    fig, ax = plt.subplots(figsize=(8.8, max(4.5, 0.38 * len(plot_df))))
    xerr = plot_df["std_abs_shap"].to_numpy(dtype=float) if "std_abs_shap" in plot_df.columns else None
    max_x = float(np.max(plot_df["mean_abs_shap"] + (plot_df["std_abs_shap"] if "std_abs_shap" in plot_df.columns else 0.0)))
    label_pad = max(max_x * 0.015, 0.004)
    bars = ax.barh(
        plot_df["feature"],
        plot_df["mean_abs_shap"],
        xerr=xerr,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        error_kw={"ecolor": "#444444", "elinewidth": 0.8, "capsize": 2.0},
    )
    _clean_axes(ax, grid_axis="x")
    label_positions = plot_df["mean_abs_shap"].to_numpy(dtype=float)
    if xerr is not None:
        label_positions = label_positions + xerr
    for bar, xpos, val in zip(bars, label_positions, plot_df["mean_abs_shap"].tolist()):
        ax.text(
            float(xpos) + label_pad,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#333333",
        )
    present_groups = list(dict.fromkeys(plot_df["feature_group"].tolist()))
    if present_groups:
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(facecolor=GROUP_COLORS[group], label=feature_group_display(group))
            for group in present_groups
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax.set_xlabel("Mean |SHAP| across LOFO runs")
    ax.set_title(
        f"Top SHAP features across LOFO runs\n{WAVEFORM_AUGMENTED_LABEL}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(right=max_x + (4.5 * label_pad))
    fig.tight_layout()
    return fig, ax


def plot_target_asymmetry(joined: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    plot_df = joined.copy()
    plot_df["family"] = plot_df["study_set"].map(_display_group_label)
    plot_df["r2_gap"] = pd.to_numeric(plot_df["fpos_r2"], errors="coerce") - pd.to_numeric(plot_df["fmiss_r2"], errors="coerce")
    plot_df = plot_df.sort_values("r2_gap", ascending=False)
    family_colors = [FAMILY_COLORS.get(fam, PALETTE["blue"]) for fam in plot_df["family"]]
    axes[0].barh(plot_df["family"], plot_df["fpos_r2"], color=family_colors, edgecolor="white", linewidth=0.7)
    axes[0].set_title("`fpos` R² by held-out family")
    axes[0].set_xlabel("R²")
    axes[1].barh(plot_df["family"], plot_df["fmiss_r2"], color=family_colors, edgecolor="white", linewidth=0.7)
    axes[1].set_title("`fmiss` R² by held-out family")
    axes[1].set_xlabel("R²")
    limits = np.array([
        pd.to_numeric(plot_df["fpos_r2"], errors="coerce").min(),
        pd.to_numeric(plot_df["fmiss_r2"], errors="coerce").min(),
        pd.to_numeric(plot_df["fpos_r2"], errors="coerce").max(),
        pd.to_numeric(plot_df["fmiss_r2"], errors="coerce").max(),
    ], dtype=float)
    limit = max(abs(np.nanmin(limits)), abs(np.nanmax(limits))) if np.isfinite(limits).any() else 1.0
    for ax in axes:
        ax.set_xlim(-limit * 1.08, limit * 1.08)
        _clean_axes(ax, grid_axis="x", zero_line=True)
    for ax, column in zip(axes, ["fpos_r2", "fmiss_r2"]):
        for patch, value in zip(ax.patches, pd.to_numeric(plot_df[column], errors="coerce").tolist()):
            if pd.isna(value):
                continue
            y = patch.get_y() + patch.get_height() / 2
            x = float(value)
            align = "left" if x >= 0 else "right"
            offset = 0.015 * limit if limit else 0.01
            ax.text(x + (offset if x >= 0 else -offset), y, f"{x:.2f}", va="center", ha=align, fontsize=8, color="#334155")
    fig.suptitle("Target asymmetry under unseen-family extrapolation")
    fig.tight_layout()
    return fig, axes


def plot_prediction_scatter(predictions: pd.DataFrame, *, title: str):
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.scatter(predictions["true"], predictions["pred"], s=28, alpha=0.72, color=PALETTE["teal"], edgecolors="white", linewidths=0.35)
    lo = min(predictions["true"].min(), predictions["pred"].min())
    hi = max(predictions["true"].max(), predictions["pred"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=PALETTE["slate"])
    try:
        coeffs = np.polyfit(pd.to_numeric(predictions["true"], errors="coerce"), pd.to_numeric(predictions["pred"], errors="coerce"), 1)
        x_line = np.linspace(lo, hi, 100)
        ax.plot(x_line, coeffs[0] * x_line + coeffs[1], color=PALETTE["orange"], linewidth=1.5, alpha=0.9)
    except Exception:
        pass
    corr = predictions[["true", "pred"]].corr().iloc[0, 1]
    mae = (predictions["pred"] - predictions["true"]).abs().mean()
    ax.text(
        0.03,
        0.97,
        f"r = {corr:.3f}\nMAE = {mae:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d6dde6", "boxstyle": "round,pad=0.35"},
    )
    _clean_axes(ax, grid_axis="both")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# New storytelling plots (added for notebooks 02, 04, 05 richer visuals)
# ---------------------------------------------------------------------------

_APPROACH_COLOR = {
    "dummy":                 "#aaaaaa",   # grey
    "neural pipeline":       "#c0504d",   # salmon/red
    "tabular":               "#2957a4",   # blue
    "tabular + SSL features": "#7030a0",  # purple
    "best":                  "#1f8a70",   # teal
}

# Maps recipe_id substrings / model_family values → approach bucket
def _infer_approach(row) -> str:
    rid = str(row.get("recipe_id", "")).lower()
    mf  = str(row.get("model_family", "")).lower()
    label = str(row.get("label", "")).lower()
    if "dummy" in rid or "dummy" in label:
        return "dummy"
    if "ssl" in mf or "mmd" in mf or ("ssl" in rid and "waveform" not in rid):
        return "neural pipeline"
    if "waveform" in rid or "waveform" in label or "transplant" in rid or "transplant" in label:
        return "tabular + SSL features"
    return "tabular"


def bootstrap_r2_significance(
    true: "pd.Series | list",
    pred_model: "pd.Series | list",
    pred_baseline: "pd.Series | list",
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap test for ΔR² = R²(model) − R²(baseline) vs. H₀: ΔR² ≤ 0.

    Returns a dict with keys: delta_r2, ci_low, ci_high (95%), p_value, stars.
    p_value is one-sided: P(boot ΔR² ≤ 0) under the observed data distribution.
    """
    import numpy as np
    from sklearn.metrics import r2_score as _r2

    t = pd.to_numeric(pd.Series(true), errors="coerce").to_numpy()
    pm = pd.to_numeric(pd.Series(pred_model), errors="coerce").to_numpy()
    pb = pd.to_numeric(pd.Series(pred_baseline), errors="coerce").to_numpy()

    # Drop rows where any column is NaN
    mask = ~(pd.isna(t) | pd.isna(pm) | pd.isna(pb))
    t, pm, pb = t[mask], pm[mask], pb[mask]

    if len(t) < 4:
        return {"delta_r2": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_value": float("nan"), "stars": "—"}

    delta_obs = float(_r2(t, pm) - _r2(t, pb))
    rng = np.random.default_rng(seed)
    n = len(t)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        d = float(_r2(t[idx], pm[idx]) - _r2(t[idx], pb[idx]))
        deltas.append(d)
    deltas = np.array(deltas)

    ci_low, ci_high = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    # One-sided p-value: fraction of bootstrap samples where ΔR² ≤ 0
    p_value = float((deltas <= 0).mean())

    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "n.s."

    return {
        "delta_r2": delta_obs,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "stars": stars,
    }


def plot_waterfall_progression(
    table: pd.DataFrame,
    *,
    metric: str = "r2",
    title: str | None = None,
    sig_stars: "dict[str, str] | None" = None,
):
    """Horizontal bars in research-stage order, coloured by approach type.

    Expects columns: label (str), metric column (float), and optionally
    recipe_id / model_family for approach colouring.
    A dashed vertical reference line is drawn at the second row value
    (basic transfer baseline). The best row gets an approach colour of
    'best' (teal) and a gain annotation.
    """
    import numpy as np

    df = table.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    # Keep presentation order (do NOT sort by value)
    labels   = _shorten(df["label"].tolist())
    values   = df[metric].tolist()
    approaches = [_infer_approach(r) for _, r in df.iterrows()]

    best_idx = int(df[metric].idxmax())
    # Identify basic-transfer row (second row by convention, index 1)
    basic_val = values[1] if len(values) > 1 else 0.0

    colors = []
    for i, ap in enumerate(approaches):
        if i == best_idx:
            colors.append(_APPROACH_COLOR["best"])
        else:
            colors.append(_APPROACH_COLOR.get(ap, _APPROACH_COLOR["tabular"]))

    fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(df))))
    y_pos = range(len(labels))
    ax.barh(list(y_pos), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    xlabel = metric.upper()
    if metric.lower() == "r2":
        xlabel = "R²  (recording-disjoint final holdout)"
    elif metric.lower() == "mae":
        xlabel = "MAE  (recording-disjoint final holdout)"
    ax.set_xlabel(xlabel)
    ax.set_title(title or f"Model progression by {metric.upper()}")

    # Reference line at basic transfer
    ax.axvline(basic_val, linestyle="--", linewidth=1.2, color=PALETTE["slate"],
               label=f"Basic transfer ({basic_val:.3f})")

    # Gain annotation on best bar
    best_val = values[best_idx]
    gain = best_val - basic_val
    if abs(gain) > 0.001:
        sign = "+" if gain > 0 else ""
        ax.text(best_val + 0.005, best_idx, f"{sign}{gain:.3f}",
                va="center", ha="left", fontsize=8, color=_APPROACH_COLOR["best"])

    # Significance star annotations (if provided)
    if sig_stars:
        x_max = max(v for v in values if v == v)  # ignore NaN
        for i, lbl in enumerate(labels):
            stars = sig_stars.get(lbl) or sig_stars.get(df["label"].tolist()[i])
            if stars and stars not in ("—", "baseline"):
                ax.text(x_max * 1.01, i, stars, va="center", ha="left",
                        fontsize=9, color="black", fontweight="bold")
        # Nudge right limit to make room
        ax.set_xlim(right=x_max * 1.12)
        ax.text(x_max * 1.01, len(labels) - 0.5,
                "* p<0.05  ** p<0.01  *** p<0.001\nvs. basic transfer (bootstrap, n=2000)",
                fontsize=6.5, va="top", ha="left", color=PALETTE["slate"], style="italic")

    # Legend patches
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=_APPROACH_COLOR["dummy"],               label="Dummy floor"),
        Patch(facecolor=_APPROACH_COLOR["neural pipeline"],     label="Neural pipeline"),
        Patch(facecolor=_APPROACH_COLOR["tabular"],             label="Pure tabular"),
        Patch(facecolor=_APPROACH_COLOR["tabular + SSL features"], label="Tabular + SSL features"),
        Patch(facecolor=_APPROACH_COLOR["best"],                label="Best"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    return fig, ax


def plot_approach_comparison(
    compare_df: pd.DataFrame,
    *,
    title: str | None = None,
):
    """Horizontal bars for a small approach-comparison DataFrame.

    Expected columns: model (str), approach_type (str), r2 (float).
    Optional: notes (str) — rendered as small text on bars.
    """
    df = compare_df.copy()
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    # Sort best→worst for display (bottom = best in horizontal barh)
    df = df.sort_values("r2", ascending=True)

    colors = [_APPROACH_COLOR.get(ap, _APPROACH_COLOR["tabular"])
              for ap in df["approach_type"].tolist()]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(df))))
    y_pos = range(len(df))
    ax.barh(list(y_pos), df["r2"].tolist(), color=colors, edgecolor="white")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["model"].tolist())
    ax.set_xlabel("R²  (recording-disjoint final holdout)")
    ax.set_title(title or "Approach comparison")

    # Vertical lines at 0 and best tabular
    ax.axvline(0, color="black", linewidth=0.8)
    tabular_mask = df["approach_type"].str.contains("tabular", case=False, na=False)
    if tabular_mask.any():
        best_tab = df.loc[tabular_mask, "r2"].max()
        ax.axvline(best_tab, linestyle=":", linewidth=1.2, color=_APPROACH_COLOR["tabular"],
                   label=f"Best tabular ({best_tab:.3f})")
        ax.legend(fontsize=8, framealpha=0.85)

    # Notes as small text
    if "notes" in df.columns:
        for i, (_, row) in enumerate(df.iterrows()):
            note = str(row.get("notes", ""))
            if note and note != "nan":
                ax.text(max(df["r2"]) * 0.02, i, note, va="center",
                        fontsize=7, color="white", alpha=0.9)

    fig.tight_layout()
    return fig, ax


def plot_family_r2_heatmap(
    summary_df: pd.DataFrame,
    *,
    title: str | None = None,
    vmin: float = -0.2,
    vmax: float = 0.8,
):
    """Heatmap: rows = paired families, columns = model labels.

    Returns (fig, ax, pivot_df).
    """
    import numpy as np

    col = "label" if "label" in summary_df.columns else "variant_id"
    group_col = "study_set" if "study_set" in summary_df.columns else summary_df.columns[0]

    pivot = (
        summary_df
        .pivot_table(index=group_col, columns=col, values="r2", aggfunc="mean")
        .rename(columns=LABEL_SHORT)
    )

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(pivot.columns)), max(4, 0.55 * len(pivot))))
    try:
        import matplotlib.colors as mcolors
        cmap = plt.get_cmap("RdYlGn")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.8, label="R²  (per-family holdout)")
        ax.grid(False)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=9)

        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="black")
        ax.set_title(title or "R² by family and model stage")
    except Exception:
        pass

    fig.tight_layout()
    return fig, ax, pivot.reset_index()


def plot_fpos_fmiss_ladder(
    fpos_bench: pd.DataFrame,
    fmiss_bench: pd.DataFrame,
    *,
    title: str | None = None,
):
    """Side-by-side horizontal bars on the same x-axis scale.

    Blue = fpos, orange = fmiss. Teal highlight for each target's best.
    """
    def _prep(df):
        d = df.copy()
        d["r2"] = pd.to_numeric(d["r2"], errors="coerce")
        return d

    fp = _prep(fpos_bench)
    fm = _prep(fmiss_bench)

    fp_best_val = fp["r2"].max()
    fm_best_val = fm["r2"].max()
    x_max = max(fp_best_val, fm_best_val) * 1.12

    def _colors(df, base_color, best_val):
        colors = []
        for _, row in df.iterrows():
            if abs(row["r2"] - best_val) < 1e-6:
                colors.append(_APPROACH_COLOR["best"])
            else:
                colors.append(base_color)
        return colors

    fp_colors = _colors(fp, PALETTE["blue"], fp_best_val)
    fm_colors = _colors(fm, PALETTE["orange"], fm_best_val)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, 0.45 * max(len(fp), len(fm)))))

    fp_labels = _shorten(fp["label"].tolist())
    fm_labels = _shorten(fm["label"].tolist())

    ax1.barh(fp_labels, fp["r2"].tolist(), color=fp_colors, edgecolor="white")
    ax1.set_xlim(0, x_max)
    ax1.set_xlabel("R²  (recording-disjoint holdout)")
    ax1.set_title(f"fpos  (best: {fp_best_val:.3f})")

    ax2.barh(fm_labels, fm["r2"].tolist(), color=fm_colors, edgecolor="white")
    ax2.set_xlim(0, x_max)
    ax2.set_xlabel("R²  (recording-disjoint holdout)")
    ax2.set_title(f"fmiss  (best: {fm_best_val:.3f})")

    fig.suptitle(title or "fpos vs fmiss: parallel progressions", fontsize=13)
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_domain_auc_annotated(
    domain_summary: pd.DataFrame,
    *,
    title: str | None = None,
):
    """Bar chart of domain classifier AUC with reference lines and danger zone.

    Replaces plot_domain_shift_separability.
    """
    table = domain_summary.copy()
    table["label"] = table["comparison"] + "\n" + table["feature_set"]
    aucs = pd.to_numeric(table["cv_auc_mean"], errors="coerce").tolist()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [PALETTE["blue"] if a < 0.90 else PALETTE["red"] for a in aucs]
    ax.bar(table["label"], aucs, color=colors, edgecolor="white")

    # Reference lines
    ax.axhline(0.5,  linestyle=":",  linewidth=1.2, color=PALETTE["slate"], label="Random (0.50)")
    ax.axhline(0.75, linestyle="--", linewidth=1.2, color=PALETTE["orange"], label="Strong (0.75)")
    ax.axhline(0.90, linestyle="-",  linewidth=1.2, color=PALETTE["red"],   label="Very strong (0.90)")

    # Danger zone shading above 0.90
    ax.axhspan(0.90, 1.02, alpha=0.08, color=PALETTE["red"], label="Transfer unsafe zone")
    ax.text(len(aucs) - 0.5, 0.955, "Transfer unsafe", color=PALETTE["red"],
            ha="right", fontsize=8, style="italic")

    # Value labels on bars
    for i, v in enumerate(aucs):
        if not (v != v):  # not NaN
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("CV AUC")
    ax.set_xlabel("Domain classifier AUC (hybrid vs paired, 5-fold CV)")
    ax.set_title(title or "Hybrid vs paired separability")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return fig, ax
