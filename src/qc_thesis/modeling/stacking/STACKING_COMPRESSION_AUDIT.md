# Stacking Compression Audit

Goal: reduce the active stacking implementation to about 6 files while keeping the full method.

## Common Wrapper Pattern To Eliminate
These functions appear in most files and should not survive as separate per-file logic:
- `_parse_args`
- `_log`
- `run_with_config`
- large `_run_with_config` bodies that differ only in stack augmentation and variant selection

These should collapse into one generic runner plus recipe specs.

## Current Essential Concepts
The folder currently mixes six real concerns:
1. configuration
2. split and pseudo-label preparation
3. stack assembly
4. feature augmentation
5. backend fitting and evaluation
6. recipe selection and CLI execution

The compression target should preserve those six concerns, not the current file boundaries.

## File-Level And Function-Level Audit

### `config.py`
- `MLStatisticsWorkbenchConfig`: keep, but move into the final shared config/spec file.

### `methods.py`
- Keep and split:
- `fit_tree_regressor`, `predict_tree_regressor`: backend helpers
- `compute_importance_weights`, `compute_overlap_importance_weights`, `compute_soft_group_balance_weights`: weighting/backends
- `build_stack_frame`, `build_stack_set`: stack foundation
- `make_study_indicator_frame`, `fit_study_bias_correction`, `leave_one_out_study_bias_adjustment`, `apply_study_bias_correction`: augmentation/backends
- `fit_latent_cluster_model`, `latent_cluster_responsibilities`: latent utilities
- `mask_feature_columns`: stack foundation
- `select_shift_filtered_features`, `select_shift_protected_features`: split/filter foundation
- `fit_quantile_transport_model`, `apply_quantile_transport_model`: augmentation/backends
- `fit_kmeans_subgroup_model`, `assign_kmeans_subgroups`: augmentation
- `select_target_transport_features`: split/filter foundation
- `fit_pca_support_projector`, `apply_pca_support_projector`: augmentation
- `make_local_regression_mixup`: augmentation
- `make_latent_summary_frame`, `rank_latent_dims_by_correlation`: augmentation
- `pick_r2_first_recommendation`: evaluation/reporting
- `prepare_unit_input`, `train_zero_shot_mmd_source`, `predict_embeddings_from_state`: augmentation/backends
- `make_metric_row`, `make_prediction_frame`: generic runner
- Conclusion: this file is a utility warehouse. It must be split across `foundation`, `augmenters`, `backends`, and `run`.

### `pseudo_label_manifold.py`
- Keep and split:
- `_build_filtered_drop_cols`: split/filter foundation
- `_teacher_frames_for_target`: split/pseudo foundation
- `_build_support_artifacts`: split/pseudo foundation
- `_manifold_support_table`: split/pseudo foundation
- `_teacher_predictions`: split/pseudo foundation
- `_select_pseudo_labels`: split/pseudo foundation
- `_assemble_student_training_data`, `_score_student_cv`, `_fit_student_holdout`: generic runner or backends
- `_summarize_clusters`: reporting
- `_build_readme`, `_run_with_config`, `run_with_config`, `_parse_args`, `_log`: delete after consolidation
- Conclusion: this is one of the true core files, but only about half the functions are essential.

### `run_experiment.py`
- Keep and split:
- `_build_filtered_feature_manifests`: split/filter foundation
- `_score_candidate_cv`, `_fit_predict_holdout`: generic runner/backends
- `_build_study_holdout_summary`: reporting
- `_run_target`: currently the main monolith; break into recipe specs + generic runner calls
- Delete or centralize:
- `_parse_args`, `_build_config`, `_log`, `_frame_with_columns`, `_grouped_oof_splits`, `_numeric_frame`, `_concat_frames`, `_topk_choice`, `_build_target_readme`, `_build_readme`, `main`
- Conclusion: this is the single biggest compression target.

### `base_stack_runner.py`
- Keep:
- `BaseStackState`
- `build_base_stack_state`
- `build_embedding_universe`
- `assemble_training_data`
- `evaluate_stack_variant`
- Move:
- `_base_configs`, `_source_variant_id` can stay internal or merge into `foundation`
- Conclusion: this is already close to the final shared stack foundation file.

### `stack_recipe_runner.py`
- Keep:
- `StackVariantSpec`
- `_config_payload`
- `run_stack_recipe`
- Conclusion: this is already the right shape for the final generic runner.

### `signal_embedding.py`
- Keep:
- `SignalEmbeddingBundle`
- `fit_signal_embeddings`
- Internal-only:
- `SignalEmbeddingConfig`, `_ensure_torch`, `_fit_signal_autoencoder`, `_encode_signal`
- Conclusion: this should become part of the final augmentation file.

### `families.py`
- Keep in spirit, but shrink:
- `ContextStackFamily`, `AnchorStackFamily`, `WaveformStackFamily`, `RobustnessStackFamily`, `WaveformTransferFamily`
- Conclusion: family dispatch should likely move into the final `recipes.py` or `run.py`.

### `fpos_signal_embedding_stack.py`
- Keep:
- `FPosSignalEmbeddingStackConfig`
- `_combined_stack`
- Move into generic recipe spec:
- `_run_with_config`
- Delete:
- `_parse_args`, `_log`, `run_with_config`
- Conclusion: recipe-spec file only.

### `fmiss_waveform_transfer.py`
- Keep:
- `FMissWaveformTransferConfig`
- Move into generic recipe spec:
- `_run_with_config`
- Delete:
- `_parse_args`, `_log`, `run_with_config`
- Conclusion: recipe-spec file only.

### `fpos_backend_sweep.py`
- Keep:
- `_fit_backend_predict`
- `_score_backend_cv`, `_fit_backend_holdout` if not fully subsumed by generic runner
- Move or delete:
- `_grouped_oof_splits` duplicated elsewhere
- `_run_with_config`, `run_with_config`, config/CLI wrappers
- Conclusion: this should collapse into backend definitions plus one recipe spec.

### `fpos_anchor_stack.py`
- Keep:
- `_fit_anchor_scores`
- Move or delete:
- `_grouped_oof_splits` duplicated
- `_run_with_config`, config/CLI wrappers
- Conclusion: should become one backend helper plus maybe one recipe entry.

### `fpos_cluster_trust_pseudo.py`
- Keep:
- `_trust_from_group_stats`
- `_apply_trust_selection`
- Move:
- `_build_readme` to reporting if still needed
- Delete:
- `_run_with_config`, config/CLI wrappers
- Conclusion: this is core pseudo-label trust logic and belongs in `foundation`.

### `fpos_family_robustness.py`
- Keep:
- `_family_weights`
- `_per_family_summary`
- `_pick_robust_variant`
- Maybe keep:
- `_cv_variant`, `_fit_holdout_variant` if they are still distinct after generic runner cleanup
- Move or delete:
- `_assemble_training_data` duplicated
- `_prepare_frames`, `_run_with_config`, config/CLI wrappers
- Conclusion: robustness should become an augmenter/weighting option, not a separate pipeline file.

### `fpos_expert_stack.py`
- Keep only if truly needed:
- `_make_meta_frame`
- `_fit_meta_predict`
- `_score_meta_cv`
- Delete or absorb:
- `_run_with_config`, config/CLI wrappers
- Conclusion: likely becomes one optional meta-backend recipe, not its own file.

### `fpos_leave_one_family_out.py`
- Keep only if thesis tables still need it:
- `_prepare_family_frames`
- `_fit_holdout`
- Delete or absorb:
- `_assemble_training_data` duplicated
- `_run_with_config`, config/CLI wrappers
- Conclusion: likely appendix/reporting, not part of the six-file core.

### `fpos_residual_correction.py`
- Keep only if distinct:
- `_winner_oof_and_holdout`
- `_build_residual_frame`
- `_fit_residual_model`
- Move or delete:
- `_prepare_frames`, `_run_with_config`, config/CLI wrappers
- Conclusion: should become one residual-backend recipe.

### `fpos_signal_embedding_bayes_opt.py`
- Keep only if tuning remains in-scope:
- `_fit_custom_xgboost_predict`
- `_cv_and_holdout_custom`
- `_sample_trial_config`
- Delete or archive:
- `_run_with_config`, config/CLI wrappers
- Conclusion: tuning should not live in the core six-file runtime.

### `fpos_tree_interaction_stack.py`
- Keep only if distinct:
- `_interaction_feature_subset`
- `_fit_leaf_encoder`
- `_transform_leaf_features`
- `_augment_with_interactions`
- `_cv_and_holdout_tree_interaction`
- Delete or absorb:
- `_run_with_config`, config/CLI wrappers
- Conclusion: this belongs inside a generic augmenter file.

### `fpos_waveform_enrichment_stack.py`
- Keep:
- `_waveform_array`
- `_normalize_waveforms`
- `_dct_frame`
- `_segment_pool_frame`
- `_derivative_shape_frame`
- `_waveform_enrichment_frames`
- Delete or absorb:
- `_run_with_config`, config/CLI wrappers
- Conclusion: these are pure augmenters and should move into `augmenters.py`.

### `fpos_waveform_multiscale_stack.py`
- Keep:
- `_fit_multiscale_autoencoder`
- `_encode_multiscale`
- `_multiscale_waveform_embeddings`
- Delete or absorb:
- `_run_with_config`, config/CLI wrappers
- Conclusion: should merge into `augmenters.py`.

### `synthetic_augmentation.py`
- Keep only if thesis needs it:
- `_augment_subset`
- `_prepare_variants`
- `_metric_row` can merge into generic runner utilities
- `_build_filtered_drop_cols` duplicated
- Delete or absorb:
- `_build_readme`, `_run_with_config`, config/CLI wrappers
- Conclusion: appendix augmenter, not core runtime.

### `label_budget.py`
- Keep only if thesis needs the analysis:
- `_grouped_budget_subset`
- `_aggregate_budget_curve`
- `_choose_budget_recommendations`
- `_metric_row`
- `_build_filtered_drop_cols` duplicated
- Delete or absorb:
- `_run_with_config`, config/CLI wrappers
- Conclusion: reporting/analysis, not core runtime.

### `paired_conditioned_hybrid.py`
- Keep only if this remains a core method:
- `_fit_global_transport_candidates`
- `_fit_cluster_translated_source`
- `_select_transport_variants`
- `_source_prediction`
- Move or delete:
- `_build_filtered_drop_cols` duplicated
- `_score_candidate_cv`, `_fit_predict_holdout` duplicated runner logic
- `_run_with_config`, config/CLI wrappers
- Conclusion: this is a major special case and must either be folded into `backends.py` or archived outside the core six files.

### `fpos_manifold_context.py`
- Keep selectively:
- `_fit_pca_features`, `_transform_pca_features`
- `_distance_weighted_knn_predict`, `_oof_knn_predictions`
- `_fit_tree_oof_and_eval`, `_fit_predict`
- `_fit_error_model`, `_pseudo_weight_from_error`, `_curriculum_refine_pseudo`
- Delete or absorb:
- `_assemble_training_data` duplicated
- `_build_readme`, `_run_with_config`, config/CLI wrappers
- Conclusion: this is an experimental mega-file. It should either be broken into augmenters + backends or archived as appendix-only.

### `fpos_paper_inspired_transfer_benchmark.py`
- Keep only if thesis explicitly uses it:
- `_ridge_adapter_score`
- `_fewshot_family_score`
- `_labeled_neighborhood_features`
- `_unlabeled_manifold_features`
- `_compute_fold_features`
- Delete or archive:
- `_run_with_config`, config/CLI wrappers
- Conclusion: appendix or archive, not core six-file runtime.

### `explain_fpos_best_model.py`
- Keep only if explainability is still required:
- `_fit_xgb`, `_predict_xgb`, `_refit_current_winner`, `_calibration_bins`, `_fit_residual_probe`, `run_report`
- Archive:
- CLI/report wrappers if moved outside the core runtime
- Conclusion: clearly not core stack runtime.

## Compression Buckets

### Core Functions That Must Survive
- `MLStatisticsWorkbenchConfig`
- `build_stack_frame`
- `build_stack_set`
- `mask_feature_columns`
- `_build_filtered_drop_cols`
- `_teacher_frames_for_target`
- `_build_support_artifacts`
- `_manifold_support_table`
- `_teacher_predictions`
- `_select_pseudo_labels`
- `_trust_from_group_stats`
- `_apply_trust_selection`
- `_fit_anchor_scores`
- `_fit_backend_predict`
- `build_base_stack_state`
- `assemble_training_data`
- `evaluate_stack_variant`
- `fit_signal_embeddings`
- `run_stack_recipe`

### Repeated Logic That Should Disappear As Separate Functions
- every per-file `_parse_args`
- every per-file `_log`
- every per-file `run_with_config`
- every per-file giant `_run_with_config`
- duplicated `_grouped_oof_splits`
- duplicated `_assemble_training_data`
- duplicated `_build_filtered_drop_cols`
- duplicated holdout/CV loops
- duplicated metric/prediction CSV writing

### Files That Look Appendix-Only Or Archive-Ready
- `explain_fpos_best_model.py`
- `fpos_paper_inspired_transfer_benchmark.py`
- `fpos_signal_embedding_bayes_opt.py`
- `label_budget.py`
- `synthetic_augmentation.py`
- `fpos_leave_one_family_out.py`
- possibly `fpos_manifold_context.py`

## Recommended Six-File Target
1. `stacking/specs.py`
   - all dataclasses, recipe specs, family dispatch
2. `stacking/foundation.py`
   - split prep, filtering, teacher frames, pseudo labels, trust, stack assembly
3. `stacking/augmenters.py`
   - waveform, multiscale, enrichment, tree interactions, synthetic transforms
4. `stacking/backends.py`
   - anchor, xgboost/lightgbm backends, residual/meta variants, weighting helpers
5. `stacking/recipes.py`
   - all `fpos` and `fmiss` recipes as declarative specs
6. `stacking/run.py`
   - generic runner, outputs, CLI entrypoint

## Immediate Refactor Order
1. Move the remaining duplicated split/pseudo/trust logic out of `pseudo_label_manifold.py`, `fpos_cluster_trust_pseudo.py`, `fpos_backend_sweep.py`, `fpos_anchor_stack.py`, and `run_experiment.py` into `foundation.py`.
2. Move all augmentation-only code out of waveform, multiscale, tree interaction, and synthetic files into `augmenters.py`.
3. Move backend-only code out of anchor, backend sweep, residual, expert, xgboost tuning, and bagging files into `backends.py`.
4. Replace every stack file whose only remaining job is “pick a variant and run it” with declarative entries in `recipes.py`.
5. Keep appendix-only analysis files outside the core runtime, or archive them.
