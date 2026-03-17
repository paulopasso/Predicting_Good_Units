from __future__ import annotations

from pathlib import Path
from typing import Any

from qc_thesis.paths import THESIS_ROOT

from .config import MLStatisticsWorkbenchConfig
from .fmiss_waveform_transfer import FMissWaveformTransferConfig, run_with_config as run_fmiss_waveform_transfer
from .fpos_backend_sweep import FPosBackendSweepConfig, run_with_config as run_fpos_backend_sweep
from .fpos_family_robustness import FPosFamilyRobustnessConfig, run_with_config as run_fpos_family_robustness
from .fpos_signal_embedding_stack import FPosSignalEmbeddingStackConfig, run_with_config as run_fpos_signal_embedding_stack
from .run import run_with_config as run_ml_statistics_workbench


class ContextStackFamily:
    implementation_module = "qc_thesis.modeling.stacking.run"

    def run(self, recipe: Any, output_dir: Path, *, smoke: bool = False, parquet_path: Path | None = None) -> Path:
        config = MLStatisticsWorkbenchConfig(
            parquet_path=parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet"),
            output_root=output_dir.parent,
            run_name=output_dir.name,
            targets=(recipe.target,),
            save_checkpoints=False,
            verbose=False,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.model_selection_splits = 2
            config.lightgbm_estimators = 32
            config.xgboost_estimators = 48
            config.mmd_source_train_epochs = 2
            config.latent_candidate_topk = (4,)
            config.cluster_candidate_k = (2,)
        return run_ml_statistics_workbench(config)


class AnchorStackFamily:
    implementation_module = "qc_thesis.modeling.stacking.fpos_backend_sweep"

    def run(self, recipe: Any, output_dir: Path, *, smoke: bool = False, parquet_path: Path | None = None) -> Path:
        config = FPosBackendSweepConfig(
            parquet_path=parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet"),
            output_root=output_dir.parent,
            run_name=output_dir.name,
            verbose=False,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.max_pseudo_per_recording = 12
            config.max_pseudo_per_cluster = 32
            config.max_pseudo_per_labeled_ratio = 1.0
        return run_fpos_backend_sweep(config)


class WaveformStackFamily:
    implementation_module = "qc_thesis.modeling.stacking.fpos_signal_embedding_stack"

    def run(self, recipe: Any, output_dir: Path, *, smoke: bool = False, parquet_path: Path | None = None) -> Path:
        config = FPosSignalEmbeddingStackConfig(
            parquet_path=parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet"),
            output_root=output_dir.parent,
            run_name=output_dir.name,
            verbose=False,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.ae_epochs = 2
            config.conv_embed_dim = 4
            config.conv_hidden_channels = 6
            config.batch_size = 128
            config.universe_max_rows = 1200
            config.max_pseudo_per_recording = 12
            config.max_pseudo_per_cluster = 32
        return run_fpos_signal_embedding_stack(config)


class RobustnessStackFamily:
    implementation_module = "qc_thesis.modeling.stacking.fpos_family_robustness"

    def run(self, recipe: Any, output_dir: Path, *, smoke: bool = False, parquet_path: Path | None = None) -> Path:
        config = FPosFamilyRobustnessConfig(
            parquet_path=parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet"),
            output_root=output_dir.parent,
            run_name=output_dir.name,
            verbose=False,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.max_pseudo_per_recording = 12
            config.max_pseudo_per_cluster = 32
            config.max_pseudo_per_labeled_ratio = 1.0
        return run_fpos_family_robustness(config)


class WaveformTransferFamily:
    implementation_module = "qc_thesis.modeling.stacking.fmiss_waveform_transfer"

    def run(self, recipe: Any, output_dir: Path, *, smoke: bool = False, parquet_path: Path | None = None) -> Path:
        config = FMissWaveformTransferConfig(
            parquet_path=parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet"),
            output_root=output_dir.parent,
            run_name=output_dir.name,
            verbose=False,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.ae_epochs = 2
            config.conv_embed_dim = 4
            config.conv_hidden_channels = 6
            config.batch_size = 128
            config.universe_max_rows = 1200
            config.max_pseudo_per_recording = 12
            config.max_pseudo_per_cluster = 32
        return run_fmiss_waveform_transfer(config)


STACKING_FAMILIES = {
    "context_stack": ContextStackFamily(),
    "anchor_stack": AnchorStackFamily(),
    "waveform_stack": WaveformStackFamily(),
    "robustness_stack": RobustnessStackFamily(),
    "waveform_transfer": WaveformTransferFamily(),
}
