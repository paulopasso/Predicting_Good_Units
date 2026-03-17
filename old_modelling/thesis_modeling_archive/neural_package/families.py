from __future__ import annotations

from pathlib import Path
from typing import Any

from qc_thesis.paths import THESIS_ROOT

from .mmd.config import MMDDomainAdaptationConfig
from .mmd.run_experiment import run_with_config as run_mmd_with_config
from .ssl.config import SSLDomainAdaptationConfig
from .ssl.run_experiment import run_with_config as run_ssl_with_config


class SSLFamily:
    implementation_module = "qc_thesis.modeling.neural.ssl.run_experiment"

    def run(self, recipe: Any, output_dir: Path, *, smoke: bool = False, parquet_path: Path | None = None) -> Path:
        config = SSLDomainAdaptationConfig(
            parquet_path=parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet"),
            output_root=output_dir.parent,
            run_name=output_dir.name,
            targets=(recipe.target,),
            save_checkpoints=False,
            run_stress_test=False,
            verbose=False,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.ssl_pretrain_epochs = 2
            config.source_warmup_epochs = 1
            config.source_train_epochs = 2
            config.finetune_epochs = 2
            config.batch_size = 64
            config.latent_dim = 16
            config.hidden_dim = 32
            config.context_hidden_dim = 16
        return run_ssl_with_config(config)


class MMDFamily:
    implementation_module = "qc_thesis.modeling.neural.mmd.run_experiment"

    def run(self, recipe: Any, output_dir: Path, *, smoke: bool = False, parquet_path: Path | None = None) -> Path:
        config = MMDDomainAdaptationConfig(
            parquet_path=parquet_path or (THESIS_ROOT / "data/raw/33000_ROWS.parquet"),
            output_root=output_dir.parent,
            run_name=output_dir.name,
            targets=(recipe.target,),
            save_checkpoints=False,
            verbose=False,
        )
        if smoke:
            config.paired_final_holdout_rows = 24
            config.lambda_grid = (0.0, 0.05)
            config.source_train_epochs = 2
            config.finetune_epochs = 2
            config.batch_size = 64
            config.latent_dim = 16
            config.hidden_dim = 32
        return run_mmd_with_config(config)


NEURAL_FAMILIES = {
    "ssl_neural": SSLFamily(),
    "mmd_neural": MMDFamily(),
}
