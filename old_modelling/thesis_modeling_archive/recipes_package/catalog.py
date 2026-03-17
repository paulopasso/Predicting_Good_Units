from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    label: str
    target: str
    protocol: str
    ablation_group: str
    feature_view: str
    model_family: str
    stage: str
    runner_key: str
    result_key: str
    implementation_module: str
    result_filters: tuple[tuple[str, str], ...] = ()
    notes: str = ""
    main_text: bool = True


FPOS_RECIPE_GROUPS = {
    "source_supervision": [
        "fpos_dummy",
        "fpos_paired_raw",
        "fpos_hybrid_only",
        "fpos_hybrid_calibrated",
        "fpos_hybrid_plus_paired",
        "fpos_hybrid_residual",
    ],
    "context": [
        "fpos_paired_raw",
        "fpos_paired_context",
        "fpos_context_stack",
    ],
    "unlabeled_paired_structure": [
        "fpos_context_stack",
        "fpos_pre_waveform_unlabeled",
        "fpos_family_balanced",
    ],
    "waveform_representation": [
        "fpos_pre_waveform_unlabeled",
        "fpos_family_balanced",
        "fpos_waveform_winner",
    ],
    "robustness": [
        "fpos_pre_waveform_unlabeled",
        "fpos_family_balanced",
        "fpos_waveform_winner",
    ],
}


FMISS_RECIPE_GROUPS = {
    "source_target_choice": [
        "fmiss_dummy",
        "fmiss_paired_raw",
        "fmiss_hybrid_only",
        "fmiss_hybrid_calibrated",
        "fmiss_hybrid_residual",
    ],
    "context": [
        "fmiss_paired_raw",
        "fmiss_paired_context",
        "fmiss_source_reweighted",
        "fmiss_embed_nommd",
        "fmiss_reduced_latent",
    ],
    "transfer_family": [
        "fmiss_hybrid_only",
        "fmiss_hybrid_calibrated",
        "fmiss_hybrid_residual",
        "fmiss_reduced_latent",
    ],
    "unlabeled_paired_use": [
        "fmiss_source_reweighted",
        "fmiss_reduced_latent",
        "fmiss_transplanted_fpos",
    ],
    "representation_filtering": [
        "fmiss_embed_nommd",
        "fmiss_reduced_latent",
        "fmiss_transplanted_fpos",
    ],
}


def get_recipe_specs() -> list[RecipeSpec]:
    return [
        RecipeSpec("fpos_dummy", "Dummy paired mean", "fpos", "recording_disjoint_main", "source_supervision", "none", "dummy", "baseline", "legacy_transfer", "inductive|dummy|paired_mean|none", "qc_thesis.modeling.transfer.transfer", (("phase", "final_holdout"),), notes="Sanity floor"),
        RecipeSpec("fpos_paired_raw", "Paired-only raw", "fpos", "recording_disjoint_main", "source_supervision", "unit_raw", "paired_only", "baseline", "legacy_transfer", "inductive|paired_only|lightgbm|unit_raw", "qc_thesis.modeling.transfer.transfer", (("phase", "final_holdout"),), notes="Small paired baseline"),
        RecipeSpec("fpos_hybrid_only", "Hybrid-only XGBoost", "fpos", "recording_disjoint_main", "source_supervision", "unit_raw", "hybrid_only", "baseline", "legacy_transfer", "inductive|hybrid_only|xgboost|unit_raw", "qc_thesis.modeling.transfer.transfer", (("phase", "final_holdout"),), notes="Pure source failure baseline"),
        RecipeSpec("fpos_hybrid_calibrated", "Calibrated hybrid XGBoost", "fpos", "recording_disjoint_main", "source_supervision", "unit_raw", "hybrid_calibrated", "baseline", "legacy_transfer", "inductive|hybrid_calibrated|xgboost+isotonic|unit_raw", "qc_thesis.modeling.transfer.transfer", (("phase", "final_holdout"),), notes="Simple calibrated transfer"),
        RecipeSpec("fpos_hybrid_plus_paired", "Hybrid+paired basic XGBoost", "fpos", "recording_disjoint_main", "source_supervision", "unit_raw", "hybrid_plus_paired", "baseline", "legacy_transfer", "inductive|hybrid_plus_paired|xgboost_w25|unit_raw", "qc_thesis.modeling.transfer.transfer", (("phase", "final_holdout"),), notes="Basic transfer baseline"),
        RecipeSpec("fpos_hybrid_residual", "Hybrid stack residual", "fpos", "recording_disjoint_main", "source_supervision", "unit_raw", "hybrid_stack", "baseline", "legacy_transfer", "inductive|hybrid_stack|lightgbm_residual|unit_raw", "qc_thesis.modeling.transfer.transfer", (("phase", "final_holdout"),), notes="First nonlinear stack"),
        RecipeSpec("fpos_paired_context", "Paired-only context", "fpos", "recording_disjoint_main", "context", "full_context", "context_first", "main_text", "context_stack", "contextual_paired_only_lightgbm", "qc_thesis.modeling.stacking.recipes", (("protocol", "main"),), notes="Context-only target baseline"),
        RecipeSpec("fpos_context_stack", "Context-first source stack", "fpos", "recording_disjoint_main", "context", "full_context", "context_first", "main_text", "context_stack", "embed_nommd_fullctx_lightgbm", "qc_thesis.modeling.stacking.recipes", (("protocol", "main"),), notes="Best pre-unlabeled context-first stack"),
        RecipeSpec("fpos_ssl", "Best SSL neural", "fpos", "recording_disjoint_main", "context", "contextual", "ssl_neural", "main_text", "ssl_neural", "ssl_fpos_contextual", "qc_thesis.modeling.neural.ssl.run_experiment", (("protocol", "main"),), notes="Best SSL comparison"),
        RecipeSpec("fpos_mmd", "Best MMD neural", "fpos", "recording_disjoint_main", "context", "contextual", "mmd_neural", "main_text", "mmd_neural", "neural_mmd_finetuned", "qc_thesis.modeling.neural.mmd.run_experiment", (("protocol", "main"),), notes="Best MMD comparison"),
        RecipeSpec("fpos_pre_waveform_unlabeled", "Pre-waveform unlabeled structure", "fpos", "recording_disjoint_main", "unlabeled_paired_structure", "full_context", "pseudo_anchor", "main_text", "anchor_stack", "source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default", "qc_thesis.modeling.stacking.recipes", (("protocol", "main"),), notes="Strongest positive result before waveform embedding"),
        RecipeSpec("fpos_family_balanced", "Family-balanced robustness", "fpos", "recording_disjoint_main", "robustness", "full_context", "robustness_weighted", "main_text", "robustness_stack", "family_balanced_strong_xgboost", "qc_thesis.modeling.stacking.recipes", (("protocol", "main"),), notes="Robustness tradeoff variant"),
        RecipeSpec("fpos_waveform_winner", "Waveform winner", "fpos", "recording_disjoint_main", "waveform_representation", "waveform_embedding", "signal_embedding", "main_text", "waveform_stack", "wf_embed_anchor_stack_xgboost", "qc_thesis.modeling.stacking.recipes", (("protocol", "main"),), notes="Final fpos winner"),
        RecipeSpec("fmiss_dummy", "Dummy paired mean", "fmiss", "recording_disjoint_main", "source_target_choice", "none", "dummy", "baseline", "legacy_transfer", "inductive|dummy|paired_mean|none", "qc_thesis.modeling.transfer.transfer", (("phase", "final_holdout"),), notes="Sanity floor"),
        RecipeSpec("fmiss_paired_raw", "Paired-only raw", "fmiss", "recording_disjoint_main", "source_target_choice", "shape_only", "paired_only", "baseline", "legacy_transfer", "inductive|paired_only|xgboost|shape_only", "qc_thesis.modeling.transfer.transfer", (("phase", "final_holdout"),), notes="Strict paired-only baseline"),
        RecipeSpec("fmiss_hybrid_only", "Hybrid-only XGBoost", "fmiss", "recording_disjoint_main", "source_target_choice", "unit_raw", "hybrid_only", "baseline", "legacy_transfer", "inductive|hybrid_only|xgboost|unit_raw", "qc_thesis.modeling.transfer.transfer", (("phase", "final_holdout"),), notes="Pure source failure baseline"),
        RecipeSpec("fmiss_hybrid_calibrated", "Calibrated hybrid", "fmiss", "recording_disjoint_main", "source_target_choice", "unit_raw", "hybrid_calibrated", "baseline", "legacy_transfer", "inductive|hybrid_calibrated|lightgbm+linear|unit_raw", "qc_thesis.modeling.transfer.transfer", (("phase", "final_holdout"),), notes="Simple calibrated transfer"),
        RecipeSpec("fmiss_hybrid_residual", "Hybrid stack residual", "fmiss", "recording_disjoint_main", "transfer_family", "unit_raw", "hybrid_stack", "baseline", "legacy_transfer", "inductive|hybrid_stack|lightgbm_residual|unit_raw", "qc_thesis.modeling.transfer.transfer", (("phase", "final_holdout"),), notes="First positive transfer baseline"),
        RecipeSpec("fmiss_ssl", "Best SSL neural", "fmiss", "recording_disjoint_main", "transfer_family", "contextual", "ssl_neural", "main_text", "ssl_neural", "ssl_fmiss_contextual", "qc_thesis.modeling.neural.ssl.run_experiment", (("protocol", "main"),), notes="Best SSL comparison"),
        RecipeSpec("fmiss_mmd", "Best MMD neural", "fmiss", "recording_disjoint_main", "transfer_family", "contextual", "mmd_neural", "main_text", "mmd_neural", "neural_mmd_finetuned", "qc_thesis.modeling.neural.mmd.run_experiment", (("protocol", "main"),), notes="Best MMD comparison"),
        RecipeSpec("fmiss_paired_context", "Paired-only context", "fmiss", "recording_disjoint_main", "context", "full_context", "context_first", "main_text", "context_stack", "contextual_paired_only_lightgbm", "qc_thesis.modeling.stacking.recipes", (("protocol", "main"),), notes="Context-only target baseline"),
        RecipeSpec("fmiss_source_reweighted", "Source-reweighted full context", "fmiss", "recording_disjoint_main", "context", "full_context", "context_first", "main_text", "context_stack", "source_reweighted_fullctx_lightgbm", "qc_thesis.modeling.stacking.recipes", (("protocol", "main"),), notes="Source-weighted context-first stack"),
        RecipeSpec("fmiss_embed_nommd", "Embed-nommd full context", "fmiss", "recording_disjoint_main", "representation_filtering", "full_context", "context_first", "main_text", "context_stack", "embed_nommd_fullctx_lightgbm", "qc_thesis.modeling.stacking.recipes", (("protocol", "main"),), notes="Full latent context-first stack"),
        RecipeSpec("fmiss_reduced_latent", "Reduced-latent winner", "fmiss", "recording_disjoint_main", "representation_filtering", "reduced_latent", "context_first", "main_text", "context_stack", "source_plus_reduced_latent_fullctx_lightgbm", "qc_thesis.modeling.stacking.recipes", (("protocol", "main"),), notes="Final fmiss winner"),
        RecipeSpec("fmiss_transplanted_fpos", "Transplanted fpos waveform recipe", "fmiss", "recording_disjoint_main", "unlabeled_paired_use", "waveform_embedding", "transplant", "main_text", "waveform_transfer", "fmiss_wf_embed_anchor_stack_xgboost", "qc_thesis.modeling.stacking.recipes", (("protocol", "main"),), notes="Failed fpos transplant"),
    ]


def get_recipe(recipe_id: str) -> RecipeSpec:
    for spec in get_recipe_specs():
        if spec.recipe_id == recipe_id:
            return spec
    raise KeyError(recipe_id)


def get_recipes(*, target: str | None = None, protocol: str | None = None, main_text: bool | None = None) -> list[RecipeSpec]:
    specs = get_recipe_specs()
    if target is not None:
        specs = [spec for spec in specs if spec.target == target]
    if protocol is not None:
        specs = [spec for spec in specs if spec.protocol == protocol]
    if main_text is not None:
        specs = [spec for spec in specs if spec.main_text == main_text]
    return specs


def get_recipe_groups(target: str) -> dict[str, list[str]]:
    if target == "fpos":
        return FPOS_RECIPE_GROUPS
    if target == "fmiss":
        return FMISS_RECIPE_GROUPS
    raise KeyError(target)


def get_main_text_recipe_ids(target: str) -> list[str]:
    return [spec.recipe_id for spec in get_recipes(target=target, protocol="recording_disjoint_main", main_text=True)]
