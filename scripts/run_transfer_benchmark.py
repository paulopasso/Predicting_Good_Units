from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid→paired fpos/fmiss transfer benchmark")
    parser.add_argument(
        "--parquet",
        default="data/raw/33000_ROWS.parquet",
        help="Path to the QC parquet artifact",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/runs/fpos_fmiss_transfer_benchmark/outputs",
        help="Directory for CSV artifacts",
    )
    parser.add_argument(
        "--model-selection-splits",
        type=int,
        default=5,
        help="Number of grouped paired validation splits before the final holdout fit",
    )
    parser.add_argument(
        "--final-holdout-rows",
        type=int,
        default=100,
        help="Target number of matched paired rows to reserve for the final holdout",
    )
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="Skip the neural transfer family even if torch is installed",
    )
    parser.add_argument(
        "--lightgbm-only",
        action="store_true",
        help="Restrict paired_only baselines to LightGBM for faster smoke runs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print split-level benchmark progress and ETA",
    )
    parser.add_argument(
        "--neural-verbose",
        action="store_true",
        help="Print per-epoch neural transfer losses and ETA",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from qc_framework import QCTransferBenchmark, TransferBenchmarkConfig

    model_families = (
        "dummy",
        "paired_only",
        "hybrid_only",
        "hybrid_calibrated",
        "hybrid_stack",
        "hybrid_plus_paired",
    )
    if not args.skip_neural:
        model_families = model_families + ("neural_transfer",)

    config = TransferBenchmarkConfig(
        parquet_path=repo_root / args.parquet,
        paired_final_holdout_rows=args.final_holdout_rows,
        model_selection_splits=args.model_selection_splits,
        model_families=model_families,
        paired_models=("lightgbm",) if args.lightgbm_only else ("lightgbm", "catboost"),
        verbose=args.verbose,
        neural_verbose=args.neural_verbose,
    )
    runner = QCTransferBenchmark(config)
    artifacts = runner.run()

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in artifacts.items():
        if hasattr(obj, "to_csv"):
            obj.to_csv(output_dir / f"{name}.csv", index=False)

    winner = artifacts["winner_summary"]
    print(f"Saved outputs to: {output_dir}")
    if not winner.empty:
        row = winner.iloc[0]
        print(
            "Winner:",
            row["candidate_id"],
            f"(primary_avg_mae={row['primary_avg_mae']:.4f}, eligible_default={bool(row['eligible_default'])})",
        )
    else:
        print("Winner: unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
