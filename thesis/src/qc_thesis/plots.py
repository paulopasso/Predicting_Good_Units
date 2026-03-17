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
