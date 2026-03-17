from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 3D PCA scatter plot from pca_projection_3d.csv")
    parser.add_argument("csv_path", help="Path to pca_projection_3d.csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults next to the CSV as pca_projection_3d.png",
    )
    return parser.parse_args()


def render_plot(csv_path: Path, output_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"No rows found in {csv_path}")
    required = {"pc1", "pc2", "pc3", "dataset_type", "study_set"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    plot_df = df.copy()
    plot_df["domain_label"] = plot_df["dataset_type"].where(
        plot_df["dataset_type"] == "hybrid",
        plot_df["study_set"],
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for label, grp in plot_df.groupby("domain_label", dropna=False):
        alpha = 0.14 if label == "hybrid" else 0.55
        size = 10 if label == "hybrid" else 20
        ax.scatter(
            grp["pc1"],
            grp["pc2"],
            grp["pc3"],
            s=size,
            alpha=alpha,
            label=str(label),
            depthshade=False,
        )

    ax.set_title("3D PCA Projection: Hybrid vs Paired (Unit Features)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = _parse_args()
    csv_path = Path(args.csv_path).resolve()
    output_path = Path(args.output).resolve() if args.output else csv_path.with_suffix(".png")
    render_plot(csv_path, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
