from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class MMDDomainAdaptationConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("crash_tests/mmd_domain_adaptation/outputs")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    targets: tuple[str, ...] = ("fpos", "fmiss")
    source_fmiss_target: str = "fmiss_extended"
    lambda_grid: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.25)
    mmd_kernel: str = "rbf_multi_scale"
    mmd_bandwidth_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    run_finetune_stage: bool = True
    strict_protocol_only: bool = True
    source_val_fraction: float = 0.15
    paired_finetune_val_splits: int = 3
    paired_finetune_val_fraction: float = 0.40
    source_train_epochs: int = 25
    finetune_epochs: int = 12
    batch_size: int = 128
    latent_dim: int = 64
    hidden_dim: int = 96
    optimizer_name: str = "adamw"
    supervised_loss: str = "smooth_l1"
    source_lr: float = 2e-3
    finetune_lr: float = 8e-4
    weight_decay: float = 1e-4
    patience: int = 5
    lightgbm_estimators: int = 250
    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 31
    save_checkpoints: bool = True
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"mmd_domain_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(payload["parquet_path"])
        payload["output_root"] = str(payload["output_root"])
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload
