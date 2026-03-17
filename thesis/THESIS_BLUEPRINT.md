# Thesis Submission Blueprint

This document freezes the thesis-facing scientific story, benchmark hierarchy, ablation design, artifact requirements, notebook narrative, and repository structure for the submission-ready research package.

The structure described here is now implemented inside `thesis/`. Later passes should refine content, not invent new structure.

It complements:

- [THESIS_OUTLINE.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/THESIS_OUTLINE.md)
- [REPO_EXPERIMENT_HANDOFF.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/REPO_EXPERIMENT_HANDOFF.md)
- [EXPERIMENT_SUMMARY.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/EXPERIMENT_SUMMARY.md)

The thesis package now includes thesis-local runtime code and curated provenance in:

- [modeling/MODEL_CODE_INDEX.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/MODEL_CODE_INDEX.md)
- [src/qc_thesis](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis)

## 1. Thesis Scope

### Working title

- `Learning Spike-Sorting QC Under Severe Domain Shift: From Hybrid Labels to Real Paired Recordings`

### Central claim

- robust spike-sorting QC transfer from hybrid to paired recordings requires all of the following:
  - strict leakage-safe evaluation
  - explicit diagnosis of hybrid-to-paired domain shift
  - context-aware tabular modeling
  - target-specific modeling for `fpos` and `fmiss`
  - conservative structural use of unlabeled paired units

### Core scientific problem

- real paired labels are scarce
- hybrid labels are abundant but distributionally different
- the paired target domain is heterogeneous across recording families
- model gains must be demonstrated without cross-recording leakage

### Main benchmark strategy

- primary benchmark:
  - recording-disjoint by `recording_key`
- robustness benchmark:
  - leave-one-family-out by `study_set`
- reporting rule:
  - every thesis-facing benchmark family must record both the recording-disjoint result and the leave-one-family-out result
  - optimization stays tied to the recording-disjoint benchmark unless explicitly stated otherwise
- mandatory additional view:
  - per-recording evaluation for every main model wave

### Target structure

- `fpos` is the main positive result and gets the richer model-progression chapter
- `fmiss` is the harder asymmetric target and gets a target-specific chapter
- `accuracy` is secondary and can appear as a short methodological note or appendix-only result

## 2. Definitions And Scientific Assumptions

### Leakage boundary

- the hard leakage boundary is `recording_key`
- no holdout recordings may be used in:
  - scaling
  - adaptation
  - pseudo-label selection
  - feature filtering
  - calibration
  - model selection

### Why context is not leakage

Context features are valid in the main benchmark because they are computed from the same recording being scored, without using target labels or holdout-derived fitting.

Examples of valid context features:

- recording aggregates:
  - recording mean / std / iqr of amplitude, template norm, confusability, drift
- recording-relative ranks:
  - `amp_std_recording_pct`
  - `amp_bin_cv_mean_recording_pct`
  - `si_amplitude_cv_median_recording_pct`
- recording deltas:
  - `wf_energy_recording_delta`
  - `si_amplitude_cv_median_recording_delta`
- source-derived contextual features:
  - `source_pred`
  - latent summaries
  - anchor score
  - support / trust features

Important distinction:

- context is not leakage under the recording-disjoint protocol
- context can still create shortcut dependence on recording or family structure
- leave-one-family-out is therefore a robustness stress test, not the primary benchmark

### Dataset assumptions

- paired recordings define the real target domain of interest
- hybrid labels are useful but biased by covariate shift
- unlabeled paired units are valuable because they carry target-domain structure
- `fpos` and `fmiss` are related but not equivalent transfer problems

## 3. Main-Text Benchmark Inventory

All main-text benchmark rows must report:

- MAE
- RMSE
- `R²`
- bias
- calibration slope
- protocol label:
  - `recording_disjoint_main`
- paired robustness companion:
  - linked `family_held_out` summary for the same model family

### 3.1 `fpos` main-text ladder

The thesis should show the following `fpos` progression in order.

| Role | Concrete row / run | Why it is shown |
| --- | --- | --- |
| 1. Dummy floor | `inductive|dummy|paired_mean|none` from [legacy_final_holdout_results.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/legacy_final_holdout_results.csv) | sanity floor |
| 2. Paired-only raw baseline | `inductive|paired_only|lightgbm|unit_raw` from [legacy_final_holdout_results.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/legacy_final_holdout_results.csv) | target-only small-data baseline |
| 3. Hybrid-only raw XGBoost baseline | `inductive|hybrid_only|xgboost|unit_raw` from [legacy_final_holdout_results.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/legacy_final_holdout_results.csv) | pure source failure baseline |
| 4. Calibrated hybrid raw XGBoost baseline | `inductive|hybrid_calibrated|xgboost+isotonic|unit_raw` from [legacy_final_holdout_results.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/legacy_final_holdout_results.csv) | simple transfer calibration baseline |
| 5. Basic hybrid+paired raw transfer baseline | `inductive|hybrid_plus_paired|xgboost_w25|unit_raw` from [legacy_final_holdout_results.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/legacy_final_holdout_results.csv) | simplest hybrid-to-paired transfer baseline |
| 6. First nonlinear stacked transfer baseline | `inductive|hybrid_stack|lightgbm_residual|unit_raw` from [legacy_final_holdout_results.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/legacy_final_holdout_results.csv) | first explicit correction stack |
| 7. Paired-only context baseline | `contextual_paired_only_lightgbm` from [workbench_r2_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_r2_metrics.csv) | clean context-only target baseline |
| 8. Strongest pre-unlabeled-structure context-first source stack | `embed_nommd_fullctx_lightgbm` from [workbench_r2_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_r2_metrics.csv) | best context-first stack before manifold / pseudo / anchor methods |
| 9. Best SSL neural run | `ssl_fpos_contextual` from [ssl_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/ssl_metrics.csv) | representative deep SSL comparison |
| 10. Best MMD neural run | `neural_mmd_finetuned` from [mmd_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/mmd_metrics.csv) | representative neural alignment comparison |
| 11. Best pre-waveform unlabeled-structure model | `source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default` from [fpos_backend_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_backend_metrics.csv) | strongest positive result before signal embedding |
| 12. Family-balanced robustness variant | `family_balanced_strong_xgboost` from [fpos_family_robustness_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_family_robustness_metrics.csv) | robustness tradeoff against the global winner |
| 13. Final waveform-embedding winner | `wf_embed_anchor_stack_xgboost` from [fpos_signal_embedding_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_signal_embedding_metrics.csv) | final `fpos` main result |

### 3.2 `fmiss` main-text ladder

The thesis should show the following `fmiss` progression in order.

| Role | Concrete row / run | Why it is shown |
| --- | --- | --- |
| 1. Dummy floor | `inductive|dummy|paired_mean|none` from [legacy_final_holdout_results.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/legacy_final_holdout_results.csv) or equivalent summary table | sanity floor |
| 2. Paired-only baseline | `inductive|paired_only|xgboost|shape_only` from [legacy_final_holdout_results.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/legacy_final_holdout_results.csv) | strict paired-only low-capacity baseline |
| 3. Hybrid-only baseline | `inductive|hybrid_only|xgboost|unit_raw` from [legacy_final_holdout_results.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/legacy_final_holdout_results.csv) | source-only failure control |
| 4. Calibrated hybrid baseline | `inductive|hybrid_calibrated|lightgbm+linear|unit_raw` from [legacy_final_holdout_results.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/legacy_final_holdout_results.csv) | simple calibrated transfer baseline |
| 5. Simple hybrid+paired baseline | `inductive|hybrid_stack|lightgbm_residual|unit_raw` from [legacy_final_holdout_results.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/legacy_final_holdout_results.csv), with `inductive|hybrid_plus_paired|lightgbm_w10|unit_raw` reported alongside it in the appendix table | first positive transfer baseline |
| 6. Best SSL run | `ssl_fmiss_contextual` from [ssl_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/ssl_metrics.csv) | neural SSL negative result |
| 7. Best MMD run | `neural_mmd_finetuned` from [mmd_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/mmd_metrics.csv) | neural MMD negative result |
| 8. Context-only paired baseline | `contextual_paired_only_lightgbm` from [workbench_r2_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_r2_metrics.csv) | main target-only context baseline |
| 9. Source-reweighted full-context LightGBM | `source_reweighted_fullctx_lightgbm` from [workbench_r2_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_r2_metrics.csv) | source-weighted context-first candidate |
| 10. Embed-nommd full-context LightGBM | `embed_nommd_fullctx_lightgbm` from [workbench_r2_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_r2_metrics.csv) | best full-latent context stack |
| 11. Reduced-latent full-context LightGBM winner | `source_plus_reduced_latent_fullctx_lightgbm` from [workbench_r2_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_r2_metrics.csv) | final `fmiss` main result |
| 12. Failed transplant of the `fpos` waveform recipe | `fmiss_wf_embed_anchor_stack_xgboost` from [fmiss_waveform_transfer_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fmiss_waveform_transfer_metrics.csv) | direct proof that `fpos` tricks did not transfer |

### 3.3 Robustness benchmark to show in main text

Use leave-one-family-out as a separate robustness table, not the primary benchmark.

Required `fpos` rows:

- `reference_anchor_stack_xgboost`
- `wf_embed_anchor_stack_xgboost`
- optional robustness tradeoff row:
  - `family_balanced_strong_xgboost`

Required metrics:

- macro `R²`
- weighted `R²`
- worst-family `R²`
- best-family `R²`
- macro MAE

Primary source:

- [fpos_leave_one_family_aggregate.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_leave_one_family_aggregate.csv)

## 4. Ablation Study Design

Each ablation block must have:

- one main metric table
- one per-family summary table
- one per-recording summary table
- one short interpretation paragraph covering:
  - what changed
  - what improved
  - what broke

### 4.1 `fpos` ablations

#### A. Source supervision

- paired-only
- hybrid-only
- hybrid+paired basic transfer
- calibrated hybrid
- stacked source prior

Scientific question:

- how much of the final gain comes from source supervision vs paired-only fitting

#### B. Context

- unit-only
- recording-relative only
- full context
- context minus recording-relative block
- context minus source-derived block

Scientific question:

- which parts of context drive gains and which parts mainly create shortcut dependence

#### C. Unlabeled paired structure

- no unlabeled use
- trust-aware pseudo only
- anchor score only
- pseudo + anchor
- family-balanced + pseudo + anchor

Scientific question:

- how much value comes from structural use of unlabeled paired units

#### D. Waveform representation

- no learned waveform embedding
- waveform embedding
- ACG embedding
- waveform + ACG embedding
- stronger learned waveform variants
- handcrafted waveform enrichment comparison

Scientific question:

- whether local signal representation adds information beyond handcrafted QC summaries

#### E. Robustness

- no family balancing
- family-balanced
- family-balanced plus waveform
- family-balanced plus waveform under leave-one-family-out

Scientific question:

- what is the tradeoff between peak score and family robustness

### 4.2 `fmiss` ablations

#### A. Source target choice

- strict `fmiss`
- `fmiss_extended` on source side
- paired-only strict-target baseline

Scientific question:

- whether source-side label extension is necessary for `fmiss`

#### B. Context

- paired-only no context
- paired-only full context
- source summary + context
- reduced latent + context

Scientific question:

- whether `fmiss` gains are mainly due to richer context or richer transfer summaries

#### C. Transfer family

- hybrid-only
- calibrated hybrid
- hybrid+paired basic transfer
- context-first LightGBM variants

Scientific question:

- where the real `fmiss` gains start and which simple transfer baselines remain weak

#### D. Unlabeled paired use

- no unlabeled paired use
- conservative source reweighting
- failed waveform/anchor transplant
- any pseudo-style `fmiss` variant kept as a negative control

Scientific question:

- why unlabeled-paired structure that helps `fpos` does not cleanly help `fmiss`

#### E. Filtering and representation

- no filtering
- global filtering
- protected target-specific filtering if available
- latent summary vs full latent

Scientific question:

- which representation and filtering choices help `fmiss` rather than `fpos`

## 5. Per-Recording Evaluation Requirement

Every main-text model wave must expose:

- per-recording MAE
- per-recording `R²`
- per-recording bias
- per-recording row count
- best recordings
- worst recordings
- a comparison to the previous wave

This is mandatory for:

- `fpos`
- `fmiss`
- all thesis-facing benchmark families, even when the family-held-out result is not the optimization metric

Required figure family:

- wave-by-wave recording distribution plot
- separate figure for `fpos`
- separate figure for `fmiss`

This figure family must answer:

- are gains broad or concentrated
- are gains coming from a small number of recordings
- are family effects actually recording effects

## 6. Figures And Tables To Produce

### 6.1 Before the benchmark tables

- dataset lineage diagram:
  - SpikeForest / Colab extraction to final parquet
- split diagram:
  - recording-disjoint main benchmark
  - family-held-out stress test
- hybrid vs paired separability:
  - unit-only
  - full context
- PCA 2D
- PCA 3D
- paired-family colored PCA
- top shifted features bar plot
- feature-group shift summary

### 6.2 Main-text result figures

- `fpos` benchmark ladder table
- `fmiss` benchmark ladder table
- neural vs tabular summary figure
- leave-one-family-out robustness table
- per-family heatmap of main models
- per-recording distribution figure for each main model wave
- calibration plots for final winners

### 6.3 Interpretability figures

- SHAP bar for best `fpos`
- SHAP beeswarm for best `fpos`
- top dependence plots for:
  - `source_pred`
  - one amplitude-instability feature
  - one waveform feature
- `fpos` vs `fmiss` feature-usage confrontation figure
- residual by family
- residual by recording

### 6.4 Limitation figures

- label-budget curve for paired labels
- leave-one-family-out degradation figure
- calibration ceiling / residual limit figure
- missing-signal summary figure

## 7. Notebook Narrative

The submission package should eventually contain these numbered notebooks.

### `01_data_extraction_and_audit`

- Colab extraction workflow
- data provenance
- extraction audit
- dataset integrity checks

Primary current source:

- [ntb_data_extraction_colab.ipynb](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data_pipeline/extraction/ntb_data_extraction_colab.ipynb)

### `02_domain_shift_hybrid_vs_paired`

- separability
- PCA
- top shifted features
- context vs no-context domain shift

Primary current source:

- [02_domain_shift_hybrid_vs_paired.ipynb](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/notebooks/02_domain_shift_hybrid_vs_paired.ipynb)

### `03_paired_family_structure`

- family heterogeneity
- paired-family PCA
- manifold observations
- leave-one-family-out rationale

Primary current sources:

- [03_paired_family_structure.ipynb](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/notebooks/03_paired_family_structure.ipynb)
- leave-one-family-out frozen summaries:
  - [fpos_leave_one_family_aggregate.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_leave_one_family_aggregate.csv)

### `04_fpos_model_progression`

- `fpos` benchmark ladder
- `fpos` ablations
- per-recording wave comparisons
- final `fpos` winner

### `05_fmiss_model_progression`

- `fmiss` benchmark ladder
- `fmiss` ablations
- target asymmetry versus `fpos`
- final `fmiss` winner

### `06_fpos_vs_fmiss_xai`

- SHAP comparison
- feature-block comparison
- manifold difference
- subgroup error comparison

### `07_limitations_and_data_budget`

- label-budget curve
- family-unseen stress test
- calibration limits
- missing-signal analysis

### `08_appendix_additional_ablations`

- synthetic
- transport
- study-aware
- cluster-aware
- tree interaction
- Bayesian optimization
- paper-inspired variants

## 8. Main Text Versus Appendix Split

### Main text

- dataset / split diagram
- domain-shift separability
- PCA with family structure
- `fpos` benchmark ladder
- `fmiss` benchmark ladder
- leave-one-family-out robustness table
- `fpos` SHAP and failure analysis
- label-budget / limitations figure

### Appendix

- long-tail ablations
- synthetic generation experiments
- transport experiments
- study-aware and cluster-aware branches
- detailed backend sweeps
- Bayesian optimization runs
- tree-interaction negative result
- paper-inspired transfer add-ons
- additional per-sorter tables

## 9. Artifact And Reproducibility Rules

Every frozen thesis-cited run must include:

- `config.json`
- `split_manifest.json`
- `metrics.csv`
- `predictions.csv`
- `per_family_summary.csv`
- `per_recording_summary.csv`
- figure exports
- script or notebook provenance
- data version reference
- benchmark protocol tag

Every thesis-facing benchmark family must therefore produce two linked artifacts:

- `recording_disjoint_main`
- `family_held_out`

Current scaffold files created for this contract:

- [REPRODUCIBILITY_SPEC.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/reproducibility/REPRODUCIBILITY_SPEC.md)
- [RUN_CHECKLIST.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/reproducibility/RUN_CHECKLIST.md)
- [ENVIRONMENT_TEMPLATE.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/reproducibility/ENVIRONMENT_TEMPLATE.md)
- [current_environment_snapshot.json](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/reproducibility/current_environment_snapshot.json)
- [export_repro_snapshot.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/scripts/export_repro_snapshot.py)

If a current run does not yet export `per_family_summary.csv` or `per_recording_summary.csv`, the thesis implementation pass must add them before citing the run in the final package.

Logical classification of frozen runs:

- `main_text_models`
- `appendix_models`

Shared low-level utilities should remain common and not be duplicated across those buckets.

## 10. Storytelling Order

The writing flow should be:

1. extraction and audit
2. understanding the domain mismatch
3. defining safe evaluation
4. comparing model families
5. exploiting unlabeled target structure
6. showing why `fpos` and `fmiss` diverge
7. interpreting the winner
8. limitations and next gains

Important framing rules:

- `fpos` is the strongest positive result
- `fmiss` is the harder counterexample
- negative results are part of the scientific contribution
- family-unseen evaluation is a robustness chapter, not the headline benchmark
- even so, family-held-out results must be reported throughout the benchmark package so robustness is visible for every main model family

## 11. Future Thesis-Ready Repository Structure

This is the target structure for a later cleanup pass. It is not implemented yet.

```text
thesis/
  README.md
  THESIS_BLUEPRINT.md
  MANUSCRIPT_PLAN.md
  FIGURE_INDEX.md
  TABLE_INDEX.md
  main_text/
  appendix/
  figures/
  tables/
  notes/

data_pipeline/
  extraction/
  audit/
  docs/

analysis_pipeline/
  domain_shift/
  spikeforest_audit/
  dataset_understanding/

modeling/
  framework/
  targets/
  registries/
  configs/

notebooks/
  01_data_extraction_and_audit.ipynb
  02_domain_shift_hybrid_vs_paired.ipynb
  03_paired_family_structure.ipynb
  04_fpos_model_progression.ipynb
  05_fmiss_model_progression.ipynb
  06_fpos_vs_fmiss_xai.ipynb
  07_limitations_and_data_budget.ipynb
  08_appendix_additional_ablations.ipynb

artifacts/
  frozen_runs/
  figures/
  tables/

src/
tests/
```

### Current repo to future structure mapping

- `data_extraction_colab/`
  - future `data_pipeline/extraction/`
- `analysis/`
  - future legacy-material archive plus early baseline reference data
- `crash_tests/domain_shift_audit/`
  - future `analysis_pipeline/domain_shift/`
- `crash_tests/spikeforest_audit/`
  - future `analysis_pipeline/spikeforest_audit/`
- `crash_tests/ml_statistics_workbench/`
  - future `modeling/` core benchmark framework and target registries
- `crash_tests/self_supervised_domain_adaptation/`
  - future appendix neural-comparison area
- `crash_tests/mmd_domain_adaptation/`
  - future appendix neural-comparison area

## 12. Run Taxonomy And Clean Extensibility Rules

Future thesis-packaging work should organize experiments by scientific wave, not by ad hoc script names.

### 12.1 Scientific waves

Use this wave taxonomy for documentation, notebooks, and artifact registries:

1. `wave_00_baseline_raw_transfer`
- dummy
- paired-only
- hybrid-only
- calibrated hybrid
- basic hybrid+paired transfer

2. `wave_01_context_first_tabular`
- paired-only context
- context-first source stacks
- latent and reduced-latent variants

3. `wave_02_neural_transfer`
- SSL
- MMD
- any direct neural transfer comparisons

4. `wave_03_domain_structure_and_subgroups`
- study-aware
- cluster-aware
- subgroup balancing
- family-aware stress tests

5. `wave_04_unlabeled_paired_exploitation`
- pseudo-label manifold
- trust-aware pseudo
- anchor stack
- source reweighting from target structure

6. `wave_05_signal_representation`
- waveform embedding
- ACG embedding
- waveform + ACG
- stronger learned waveform variants
- waveform enrichment negatives

7. `wave_06_robustness_and_limitations`
- leave-one-family-out
- label-budget curves
- calibration limits
- missing-signal analysis

8. `wave_07_appendix_ablations`
- synthetic
- transport
- tree interactions
- Bayesian optimization
- paper-inspired add-ons

### 12.2 Required structure inside each wave

Each wave should eventually have:

- one short `README.md` or registry entry stating the question being answered
- one canonical summary table
- one per-family summary
- one per-recording summary
- one notebook that explains the wave scientifically
- one frozen list of which variants are:
  - `main_text_models`
  - `appendix_models`

### 12.3 Code organization rules

To keep code clean and extensible:

- shared data loading, splitting, scoring, and artifact writing must stay in reusable modules
- target-specific logic must live behind configuration objects, not duplicated scripts
- new variants should add:
  - a config / registry entry
  - a short scientific note
  - a standardized output schema
- notebooks should follow scientific questions, not individual minor ablations
- main-text and appendix variants should be distinguished logically in registries, not by scattering code into unrelated folders

### 12.4 Output schema contract

Every future thesis-facing run should write the same core outputs:

- `config.json`
- `split_manifest.json`
- `metrics.csv`
- `predictions.csv`
- `per_family_summary.csv`
- `per_recording_summary.csv`
- optional:
  - `variant_cv_summary.csv`
  - `feature_importance.csv`
  - `teacher_agreement_summary.csv`
  - `family_robustness_summary.csv`

## 13. What Must Be Saved

For every thesis-cited model family, save:

- frozen metrics
- frozen predictions
- frozen split manifest
- per-family summary
- per-recording summary
- final figure exports
- exact provenance of script / notebook used

For every notebook, save:

- notebook path
- run-name used
- referenced frozen run folder
- expected output figures and tables

## 14. How To Improve Performance Next Without Changing The Scientific Benchmark

The next credible performance gains should come from new information, not from redefining the benchmark.

Priority directions:

- richer spatial-footprint features
- richer template morphology
- better drift / time-evolution features
- improved waveform representation
- stronger target-specific `fmiss` features
- robustness-aware model selection
- preserving no-leakage constraints

Current interpretation:

- `fpos` probably still has headroom from richer template and geometry information
- `fmiss` likely needs target-specific feature work more than `fpos`-style unlabeled-target tricks
- family-unseen stress tests should remain fixed while feature quality improves

## 15. Acceptance Checklist

The thesis blueprint is only complete if all of the following are frozen:

- all main-text benchmark rows for `fpos`
- all main-text benchmark rows for `fmiss`
- both target-specific ablation structures
- per-recording evaluation requirement
- main-text versus appendix split
- notebook sequence
- future thesis-ready repository structure
- storytelling order
- artifact schema

If any of these remain undefined, the implementation pass is not allowed to invent structure.
