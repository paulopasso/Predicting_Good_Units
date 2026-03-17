from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from .base_stack_runner import build_base_stack_state, build_embedding_universe
from .fpos_signal_embedding_stack import FPosSignalEmbeddingStackConfig
from .signal_embedding import fit_signal_embeddings
from .stack_recipe_runner import StackVariantSpec, run_stack_recipe


@dataclass
class FMissWaveformTransferConfig(FPosSignalEmbeddingStackConfig):
    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fmiss_waveform_transfer_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fpos-winner waveform recipe on fmiss")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FMissWaveformTransferConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _run_with_config(config: FMissWaveformTransferConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_base_stack_state(target="fmiss", config=config)
    universe_df = build_embedding_universe(
        split=state.split,
        max_rows=config.universe_max_rows,
        random_state=config.random_state,
    )

    _log(config, "[embed] training waveform autoencoder")
    wf_bundle = fit_signal_embeddings(
        train_signal=state.target_train_df.loc[:, state.prepared.wf_feature_cols],
        train_universe_signal=universe_df.loc[:, state.prepared.wf_feature_cols],
        unlabeled_signal=state.unlabeled_df.loc[:, state.prepared.wf_feature_cols],
        test_signal=state.target_test_df.loc[:, state.prepared.wf_feature_cols],
        prefix="wf",
        config=config,
    )

    variants = [
        StackVariantSpec(
            variant_id="fmiss_reference_anchor_stack_xgboost",
            stack=state.base_stack,
            notes="fpos winner recipe applied to fmiss",
        ),
        StackVariantSpec(
            variant_id="fmiss_wf_embed_anchor_stack_xgboost",
            stack=state.base_stack.apply(wf_bundle.as_augmenter()),
            notes="fpos winner recipe applied to fmiss",
        ),
    ]
    return run_stack_recipe(
        config=config,
        output_dir=output_dir,
        state=state,
        target="fmiss",
        family="fmiss_waveform_transfer",
        source_target="fmiss_extended",
        variants=variants,
        extra_outputs={
            "feature_learning_history.csv": pd.DataFrame(
                [{"wf_train_recon_loss": wf_bundle.history["train_recon_loss"]}]
            )
        },
        log=lambda message: _log(config, message),
    )


def run_with_config(config: FMissWaveformTransferConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FMissWaveformTransferConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )
    if args.smoke:
        cli_config.max_pseudo_multiplier = 1.0
        cli_config.max_pseudo_per_recording = 24
        cli_config.max_pseudo_per_cluster = 64
        cli_config.ae_epochs = 4
        cli_config.conv_embed_dim = 6
        cli_config.universe_max_rows = 4000
    _run_with_config(cli_config)
