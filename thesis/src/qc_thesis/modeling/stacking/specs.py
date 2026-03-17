from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class MLStatisticsWorkbenchConfig:
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
    targets: tuple[str, ...] = ("fpos", "fmiss")
    source_fmiss_target: str = "fmiss_extended"
    target_transform: str = "logit"
    use_target_transform: bool = True

    tree_backends: tuple[str, ...] = ("lightgbm", "xgboost")
    lightgbm_estimators: int = 250
    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 31
    xgboost_estimators: int = 350
    xgboost_learning_rate: float = 0.04
    xgboost_max_depth: int = 4
    xgboost_min_child_weight: float = 4.0
    xgboost_subsample: float = 0.8
    xgboost_colsample_bytree: float = 0.8

    domain_classifier_c: float = 1.0
    importance_weight_clip_min: float = 0.2
    importance_weight_clip_max: float = 5.0
    use_domain_reweighted_source: bool = True
    use_overlap_weighted_source: bool = True
    overlap_support_quantiles: tuple[float, float] = (0.02, 0.98)
    overlap_outside_support_scale: float = 0.25
    use_cluster_aware_variants: bool = True
    use_study_aware_variants: bool = True
    use_study_bias_correction_variants: bool = True
    use_subgroup_balanced_variants: bool = True
    study_bias_shrinkage: float = 8.0
    cluster_candidate_k: tuple[int, ...] = (2, 3, 4)
    cluster_assignment_temperature: float = 1.0
    cluster_min_target_rows: int = 50
    subgroup_balance_power: float = 1.0

    selection_metric: str = "r2"
    mae_guardrail_pct: float = 0.03
    model_selection_splits: int = 3
    model_selection_test_size: float = 0.40
    latent_selection_mode: str = "paired_cv_topk"
    latent_candidate_topk: tuple[int, ...] = (4, 8, 16)
    latent_summary_quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    min_r2_improvement: float = 0.0
    use_shift_filtered_variants: bool = True
    filtered_shift_top_n_unit: int = 12
    filtered_shift_top_n_full_context: int = 24
    filtered_shift_min_abs_smd: float = 1.0
    filtered_preserve_shape_features: bool = True
    use_target_protected_shift_variants: bool = True
    target_protected_top_n_unit: int = 10
    target_protected_top_n_full_context: int = 18
    target_protected_min_abs_smd: float = 1.0
    target_protected_feature_top_k: int = 8
    target_protected_min_abs_corr: float = 0.08
    reference_variant_by_target: dict[str, str] = field(
        default_factory=lambda: {
            "fpos": "embed_nommd_fullctx_lightgbm_filtered",
            "fmiss": "source_reweighted_fullctx_lightgbm",
        }
    )

    mmd_batch_size: int = 128
    mmd_latent_dim: int = 64
    mmd_hidden_dim: int = 96
    mmd_source_train_epochs: int = 25
    mmd_source_lr: float = 2e-3
    mmd_weight_decay: float = 1e-4
    mmd_patience: int = 5

    save_checkpoints: bool = True
    verbose: bool = True

    def resolved_target_transform(self) -> str:
        return self.target_transform if self.use_target_transform else "identity"

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"ml_stats_workbench_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_target_transform"] = self.resolved_target_transform()
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


@dataclass
class PseudoLabelManifoldConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("artifacts/modeling/stacking")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    top_shift_features: int = 40
    min_abs_smd: float = 0.8
    quantile_points: int = 129
    manifold_pca_components: int = 10
    manifold_neighbors: int = 16
    manifold_clusters: int = 4
    min_labeled_cluster_rows: int = 5
    support_quantile: float = 0.60
    agreement_quantile: float = 0.55
    cluster_coverage_quantile: float = 0.35
    max_pseudo_multiplier: float = 1.5
    max_pseudo_per_recording: int = 48
    max_pseudo_per_cluster: int = 128
    max_pseudo_per_labeled_ratio: float = 1.5
    pseudo_weight_scale: float = 0.25
    pseudo_weight_floor: float = 0.08
    teacher_transport_blend: float = 1.0
    teacher_transport_noise: float = 0.0
    selection_metric: str = "r2"
    mae_guardrail_pct: float = 0.03
    min_r2_improvement: float = 0.0
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"pseudo_label_manifold_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload


@dataclass
class TeacherBundle:
    variant_id: str
    family: str
    notes: str
    backend: str
    source_target: str | None
    train_frame: pd.DataFrame
    unlabeled_frame: pd.DataFrame
    test_frame: pd.DataFrame


@dataclass
class SupportArtifacts:
    cluster_model: Any
    projector: Any
    labeled_cluster_counts: dict[int, int]
    labeled_cluster_share: dict[int, float]
    labeled_support_radius: float
    support_distance_quantile: float
    agreement_quantile: float
    cluster_coverage_quantile: float
    pca_explained_variance: list[float]


@dataclass
class FPosClusterTrustPseudoConfig:
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
    selection_metric: str = "r2"
    mae_guardrail_pct: float = 0.03
    min_r2_improvement: float = 0.0
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"fpos_cluster_trust_pseudo_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload
