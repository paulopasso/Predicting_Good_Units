from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SSLDomainAdaptationConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("artifacts/modeling/neural/ssl")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    ssl_objective: str = "masked_denoising_autoencoder"
    targets: tuple[str, ...] = ("fpos", "fmiss")
    source_fmiss_target: str = "fmiss_extended"
    run_unit_only_arm: bool = True
    run_contextual_arm: bool = True
    context_feature_blocks: tuple[str, ...] = ("recording_relative", "recording_context")
    allow_test_time_unlabeled_context: bool = True
    main_protocol: str = "recording_disjoint"
    stress_protocol: str = "leave_one_study_out"
    ssl_val_recording_frac: float = 0.20
    source_val_fraction: float = 0.15
    paired_finetune_val_splits: int = 3
    paired_finetune_val_fraction: float = 0.40
    ssl_pretrain_epochs: int = 20
    source_warmup_epochs: int = 3
    source_train_epochs: int = 18
    finetune_epochs: int = 12
    batch_size: int = 128
    latent_dim: int = 64
    hidden_dim: int = 96
    context_hidden_dim: int = 48
    ssl_optimizer: str = "adamw"
    supervised_optimizer: str = "adamw"
    supervised_loss: str = "smooth_l1"
    ssl_mask_prob: float = 0.15
    ssl_noise_std: float = 0.05
    ssl_lr: float = 5e-3
    source_lr: float = 2e-3
    finetune_lr: float = 8e-4
    weight_decay: float = 1e-4
    patience: int = 5
    lightgbm_estimators: int = 250
    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 31
    save_checkpoints: bool = True
    run_stress_test: bool = True
    max_stress_studies: int | None = None
    verbose: bool = True

    def target_context_columns(self, target: str, context_feature_cols_by_target: dict[str, list[str]]) -> list[str]:
        return list(context_feature_cols_by_target.get(target, ()))

    def target_source_lr(self, target: str) -> float:
        if target == "fpos":
            return self.source_lr * 0.75
        return self.source_lr

    def target_finetune_lr(self, target: str) -> float:
        if target == "fpos":
            return self.finetune_lr * 0.75
        return self.finetune_lr

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"ssl_context_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(payload["parquet_path"])
        payload["output_root"] = str(payload["output_root"])
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload
