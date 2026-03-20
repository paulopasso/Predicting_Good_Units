from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class FeatureSpec:
    mode: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    excluded_cols: dict[str, str] = field(default_factory=dict)
    soft_excluded_cols: set[str] = field(default_factory=set)

    @property
    def all_features(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols


@dataclass
class TransferBenchmarkConfig:
    parquet_path: str | Path
    primary_targets: tuple[str, ...] = ("fpos", "fmiss")
    secondary_targets: tuple[str, ...] = ("accuracy",)
    source_fmiss_target: str = "fmiss_extended"
    final_eval_fmiss_target: str = "fmiss"
    paired_final_holdout_rows: int = 100
    allow_unlabeled_paired: bool = True
    feature_views: tuple[str, ...] = (
        "unit_raw",
        "norm_swap",
        "shape_only",
        "recording_relative",
        "recording_context",
        "study_identity",
    )
    model_families: tuple[str, ...] = (
        "dummy",
        "paired_only",
        "hybrid_only",
        "hybrid_calibrated",
        "hybrid_stack",
        "hybrid_plus_paired",
        "context_tabular",
        "context_transfer",
    )
    feature_view_sets: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("shape_only", ("shape_only",)),
        ("unit_raw", ("unit_raw",)),
        (
            "contextual_full",
            ("norm_swap", "recording_relative", "recording_context", "study_identity"),
        ),
    )
    protocol_modes: tuple[str, ...] = ("inductive", "contextual")
    model_selection_splits: int = 5
    model_selection_test_size: float = 0.40
    random_state: int = 42
    lightgbm_estimators: int = 250
    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 31
    xgboost_estimators: int = 250
    xgboost_learning_rate: float = 0.05
    xgboost_max_depth: int = 6
    boosting_backends: tuple[str, ...] = ("lightgbm", "catboost", "xgboost")
    paired_models: tuple[str, ...] = ()
    paired_weight_grid: tuple[float, ...] = (10.0, 25.0, 50.0)
    neural_pretrain_epochs: int = 35
    neural_finetune_epochs: int = 20
    neural_batch_size: int = 128
    neural_lr: float = 1e-3
    neural_domain_loss_weight: float = 0.15
    neural_target_loss_weight: float = 2.0
    target_transform: str = "logit"
    deploy_best_per_target: bool = True
    use_recording_context: bool = True
    include_ceiling_diagnostics: bool = True
    verbose: bool = False
    neural_verbose: bool = False

    def all_targets(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for target in list(self.primary_targets) + list(self.secondary_targets):
            if target not in seen:
                ordered.append(target)
                seen.add(target)
        return tuple(ordered)

    def source_target_for(self, eval_target: str) -> str:
        if eval_target == self.final_eval_fmiss_target:
            return self.source_fmiss_target
        return eval_target

    def effective_backends(self) -> tuple[str, ...]:
        return self.paired_models or self.boosting_backends


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


@dataclass
class MMDDomainAdaptationConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("artifacts/modeling/neural/mmd")
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


__all__ = [
    "FeatureSpec",
    "MMDDomainAdaptationConfig",
    "SSLDomainAdaptationConfig",
    "TransferBenchmarkConfig",
]
