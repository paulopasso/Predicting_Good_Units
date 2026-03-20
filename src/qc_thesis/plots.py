from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .paths import THESIS_ROOT


PALETTE = {
    "blue": "#2957a4",
    "teal": "#1f8a70",
    "orange": "#d98e04",
    "red": "#b33a3a",
    "slate": "#5b6675",
}


LABEL_SHORT = {
    "Dummy paired mean": "Dummy",
    "Paired-only raw": "Paired raw",
    "Hybrid-only XGBoost": "Hybrid only",
    "Calibrated hybrid XGBoost": "Hybrid calibrated",
    "Hybrid+paired basic XGBoost": "Hybrid+paired",
    "Hybrid stack residual": "Hybrid residual",
    "Paired-only context": "Paired context",
    "Context-first source stack": "Context stack",
    "Best SSL neural": "SSL",
    "Best MMD neural": "MMD",
    "Pre-waveform unlabeled structure": "Pseudo+anchor",
    "Family-balanced robustness": "Family-balanced",
    "Waveform winner": "Waveform winner",
    "Calibrated hybrid": "Hybrid calibrated",
    "Source-reweighted full context": "Source reweighted",
    "Embed-nommd full context": "Embed full ctx",
    "Reduced-latent winner": "Reduced latent",
    "Transplanted `fpos` waveform recipe": "Transplanted fpos",
}


def apply_thesis_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
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
    fig.savefig(path, bbox_inches="tight")
    return path


def save_table(df: pd.DataFrame, output_dir: Path, stem: str) -> Path:
    path = output_dir / f"{stem}.csv"
    df.to_csv(path, index=False)
    return path


def _shorten(labels: list[str]) -> list[str]:
    return [LABEL_SHORT.get(x, x) for x in labels]


def plot_benchmark_metric(table: pd.DataFrame, metric: str = "r2", title: str | None = None):
    df = table.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    ascending = metric.lower() == "mae"
    df = df.sort_values(metric, ascending=ascending)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df))))
    color = PALETTE["blue"] if metric.lower() != "mae" else PALETTE["orange"]
    ax.barh(_shorten(df["label"].tolist()), df[metric], color=color)
    ax.set_xlabel(metric.upper())
    ax.set_title(title or f"Benchmark comparison by {metric.upper()}")
    fig.tight_layout()
    return fig, ax


def plot_group_metric(summary_df: pd.DataFrame, group_col: str, metric: str = "r2", title: str | None = None):
    column_col = "label" if "label" in summary_df.columns else "variant_id"
    pivot = summary_df.pivot_table(index=group_col, columns=column_col, values=metric, aggfunc="mean").sort_index()
    pivot = pivot.rename(columns=LABEL_SHORT)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(pivot))))
    pivot.plot(kind="barh", ax=ax, color=[PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["red"], PALETTE["slate"]][: len(pivot.columns)])
    ax.set_xlabel(metric.upper())
    ax.set_ylabel(group_col.replace("_", " ").title())
    ax.set_title(title or f"{metric.upper()} by {group_col}")
    fig.tight_layout()
    return fig, ax, pivot.reset_index()


def plot_domain_shift_separability(df: pd.DataFrame):
    table = df.copy()
    table["label"] = table["comparison"] + " | " + table["feature_set"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table["label"], table["cv_auc_mean"], color=[PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["red"]][: len(table)])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("CV AUC")
    ax.set_title("Hybrid vs paired separability")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig, ax


def plot_top_feature_shift(df: pd.DataFrame, top_n: int = 15):
    plot_df = df.sort_values("abs_smd", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["abs_smd"], color=PALETTE["orange"])
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_title("Top shifted features")
    fig.tight_layout()
    return fig, ax


def plot_pca_projection(df: pd.DataFrame, color_col: str = "dataset_type"):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["red"], PALETTE["slate"]]
    for (key, part), color in zip(df.groupby(color_col, dropna=False), colors * 10):
        ax.scatter(part["pc1"], part["pc2"], s=12, alpha=0.7, label=str(key), color=color)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_pca_projection_3d(df: pd.DataFrame, color_col: str = "dataset_type"):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["red"], PALETTE["slate"]]
    for (key, part), color in zip(df.groupby(color_col, dropna=False), colors * 10):
        ax.scatter(part["pc1"], part["pc2"], part["pc3"], s=10, alpha=0.65, label=str(key), color=color)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA projection")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_label_budget(df: pd.DataFrame, target: str = "fpos"):
    plot_df = df[df["target"] == target].copy()
    plot_df["budget_rows"] = pd.to_numeric(plot_df["budget_rows"], errors="coerce")
    plot_df["mean_r2"] = pd.to_numeric(plot_df["mean_r2"], errors="coerce")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["red"]]
    for (label, part), color in zip(plot_df.groupby("variant_id"), colors * 10):
        part = part.sort_values("budget_rows")
        ax.plot(part["budget_rows"], part["mean_r2"], marker="o", label=LABEL_SHORT.get(label, label), color=color)
    ax.set_xlabel("Paired adaptation rows")
    ax.set_ylabel("Mean R²")
    ax.set_title(f"Label-budget curve: {target}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_fpos_shap_top(df: pd.DataFrame, top_n: int = 15):
    plot_df = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color=PALETTE["red"])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("Top SHAP features for best `fpos` model")
    fig.tight_layout()
    return fig, ax


def plot_target_asymmetry(joined: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    plot_df = joined.sort_values("study_set")
    axes[0].barh(plot_df["study_set"], plot_df["fpos_r2"], color=PALETTE["blue"])
    axes[0].set_title("Best `fpos` R² by family")
    axes[0].set_xlabel("R²")
    axes[1].barh(plot_df["study_set"], plot_df["fmiss_r2"], color=PALETTE["orange"])
    axes[1].set_title("Best `fmiss` R² by family")
    axes[1].set_xlabel("R²")
    fig.suptitle("Target asymmetry across paired families")
    fig.tight_layout()
    return fig, axes


def plot_prediction_scatter(predictions: pd.DataFrame, *, title: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(predictions["true"], predictions["pred"], s=18, alpha=0.65, color=PALETTE["teal"])
    lo = min(predictions["true"].min(), predictions["pred"].min())
    hi = max(predictions["true"].max(), predictions["pred"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=PALETTE["slate"])
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
