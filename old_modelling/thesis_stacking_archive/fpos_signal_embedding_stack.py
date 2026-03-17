from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qc_thesis.modeling.core import FrameAugmenter, StackSet

from .base_stack_runner import build_base_stack_state, build_embedding_universe
from .signal_embedding import fit_signal_embeddings
from .stack_recipe_runner import StackVariantSpec, run_stack_recipe


@dataclass
class FPosSignalEmbeddingStackConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("artifacts/modeling/stacking")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    max_pseudo_multiplier: float = 1.5
    max_pseudo_per_recording: int = 48
    max_pseudo_per_cluster: int = 128
    max_pseudo_per_labeled_ratio: float = 1.5
    cluster_trust_power: float = 1.35
    study_trust_weight: float = 0.30
    min_trust: float = 0.25
    conv_embed_dim: int = 8
    conv_hidden_channels: int = 12
    ae_epochs: int = 28
    batch_size: int = 256
    learning_rate: float = 2e-3
    universe_max_rows: int | None = 18000
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_signal_embedding_stack_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fpos signal embedding stack benchmark")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet")
    parser.add_argument("--output-root", default="artifacts/modeling/stacking")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _log(config: FPosSignalEmbeddingStackConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _combined_stack(base_stack: StackSet, *feature_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> StackSet:
    train_frames = [bundle[0] for bundle in feature_frames]
    unlabeled_frames = [bundle[1] for bundle in feature_frames]
    test_frames = [bundle[2] for bundle in feature_frames]
    return base_stack.apply(
        FrameAugmenter(
            train_features=pd.concat(train_frames, axis=1),
            unlabeled_features=pd.concat(unlabeled_frames, axis=1),
            test_features=pd.concat(test_frames, axis=1),
        )
    )


def _run_with_config(config: FPosSignalEmbeddingStackConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_base_stack_state(target="fpos", config=config)
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
    _log(config, "[embed] training acg autoencoder")
    acg_bundle = fit_signal_embeddings(
        train_signal=state.target_train_df.loc[:, state.prepared.acg_feature_cols],
        train_universe_signal=universe_df.loc[:, state.prepared.acg_feature_cols],
        unlabeled_signal=state.unlabeled_df.loc[:, state.prepared.acg_feature_cols],
        test_signal=state.target_test_df.loc[:, state.prepared.acg_feature_cols],
        prefix="acg",
        config=config,
    )

    variants = [
        StackVariantSpec(
            variant_id="reference_anchor_stack_xgboost",
            stack=state.base_stack,
            notes="conv embedding stack variant",
        ),
        StackVariantSpec(
            variant_id="wf_embed_anchor_stack_xgboost",
            stack=state.base_stack.apply(wf_bundle.as_augmenter()),
            notes="conv embedding stack variant",
        ),
        StackVariantSpec(
            variant_id="acg_embed_anchor_stack_xgboost",
            stack=state.base_stack.apply(acg_bundle.as_augmenter()),
            notes="conv embedding stack variant",
        ),
        StackVariantSpec(
            variant_id="wf_acg_embed_anchor_stack_xgboost",
            stack=_combined_stack(
                state.base_stack,
                (wf_bundle.train, wf_bundle.unlabeled, wf_bundle.test),
                (acg_bundle.train, acg_bundle.unlabeled, acg_bundle.test),
            ),
            notes="conv embedding stack variant",
        ),
    ]
    return run_stack_recipe(
        config=config,
        output_dir=output_dir,
        state=state,
        target="fpos",
        family="signal_embedding_stack",
        source_target="fpos",
        variants=variants,
        extra_outputs={
            "embedding_history.csv": pd.DataFrame({"wf": wf_bundle.history, "acg": acg_bundle.history})
        },
        log=lambda message: _log(config, message),
    )


def run_with_config(config: FPosSignalEmbeddingStackConfig) -> Path:
    return _run_with_config(config)


if __name__ == "__main__":
    args = _parse_args()
    cli_config = FPosSignalEmbeddingStackConfig(
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
