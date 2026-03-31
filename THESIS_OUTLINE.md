# Thesis Outline And Publication Positioning

This file is the compact chapter-level outline.

The authoritative thesis submission specification is now:

- [THESIS_BLUEPRINT.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/THESIS_BLUEPRINT.md)

Use the blueprint for:

- exact benchmark rows
- ablation structure
- notebook sequence
- main-text versus appendix split
- future repository structure

## Working Thesis Direction

- topic:
  - leakage-safe domain adaptation for spike-sorting QC prediction from hybrid to paired recordings
- main prediction tasks:
  - `fpos`
  - `fmiss`
- secondary task:
  - `accuracy`
- central claim:
  - severe hybrid-to-paired domain shift makes naive transfer unreliable
  - context-aware tabular models outperform generic neural domain-adaptation methods in this low-label tabular setting
  - unlabeled paired units provide real value for `fpos` when used structurally and conservatively

## Proposed Thesis Structure

### 1. Introduction

- spike sorting as a core bottleneck in electrophysiology workflows
- need for unit-quality prediction without access to ground truth
- mismatch between abundant synthetic / hybrid labels and scarce real paired labels
- domain shift as the central methodological problem
- research objective:
  - build leakage-safe transfer methods from hybrid to paired data
- high-level thesis contribution:
  - benchmark construction
  - domain-shift analysis
  - negative results on generic deep transfer
  - positive results from context-first tabular transfer and unlabeled paired structure

### 2. Background

- spike sorting and sorter-dependent errors
- definitions:
  - `fpos`
  - `fmiss`
  - `accuracy`
- hybrid recordings vs paired recordings
- why paired recordings are scarce but scientifically important
- domain adaptation concepts:
  - covariate shift
  - subpopulation / subgroup shift
  - semi-supervised domain adaptation
  - pseudo-labeling
  - test-time / target-aware adaptation
- why tabular scientific datasets differ from image benchmarks

### 3. Data And Extraction

- primary curated dataset:
  - `data/raw/33000_ROWS.parquet`
- dataset summary:
  - total rows: `33,206`
  - unique recordings: `61`
  - unique sorters: `9`
  - no duplicate rows
  - no duplicate `row_uid`
  - missing-label pattern concentrated in `fmiss`
- effective modeling subsets:
  - hybrid labeled source: about `24k`
  - paired matched labeled units: `207`
  - paired unmatched unlabeled units: `8,775`
- extraction workflow:
  - prepared in one Colab notebook:
    - [ntb_data_extraction_colab.ipynb](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data_pipeline/extraction/ntb_data_extraction_colab.ipynb)
- extraction implementation details:
  - Google Colab CPU workflow
  - Drive-mounted persistence
  - kachery-cloud metadata/bootstrap handling
  - checkpoint / resume support
  - paired coverage audit
  - paired recovery manifest generation
- extraction runtime details documented in notebook:
  - runtime estimate about `90 min`
  - peak RAM about `8 GB`

### 4. Feature Representation

- waveform-bin features:
  - `wf_bin_*`
- autocorrelogram features:
  - `acg_*`
- scalar QC features:
  - amplitude, SNR, ISI, drift, rate, confusability, isolation-related quantities
- expanded feature families used later:
  - recording-context aggregates
  - recording-relative features
  - study indicators
  - source-derived predictions
  - latent summaries
  - pseudo-label / support / trust features
- important data property:
  - same feature family can help one target and harm another
  - especially true for `fpos` vs `fmiss`

### 5. Experimental Protocol

- hard leakage boundary:
  - `recording_key`
- strict paired holdout:
  - about `100` matched paired rows
- paired non-test labeled pool:
  - about `107` matched rows
- non-test paired unlabeled pool after holdout exclusion:
  - about `4,963` rows
- random seed:
  - `42`
- all serious model comparisons use:
  - recording-disjoint train/test split
  - grouped CV inside non-test paired pool
- every thesis-facing benchmark family should also have a linked leave-one-family-out robustness report, even when optimization remains recording-disjoint
- no holdout recordings used in:
  - scaling
  - adaptation
  - pseudo-label selection
  - feature filtering
  - model selection
- deployment rule:
  - `fpos` and `fmiss` chosen as separate final models

### 6. Reproducibility

- code base versioned in repo
- per-run outputs saved with:
  - `config.json`
  - `split_manifest.json`
  - `metrics.csv`
  - `predictions.csv`
- project dependencies tracked in:
  - [pyproject.toml](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/pyproject.toml)
- relevant optional dependencies:
  - `lightgbm`
  - `catboost`
  - `xgboost`
  - `torch`
  - `matplotlib`
  - `seaborn`
  - `pyarrow`
- crash-test history maintained in:
  - [EXPERIMENT_SUMMARY.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/EXPERIMENT_SUMMARY.md)
- regression tests were added for new modules
- focused regression coverage was extended throughout the crash-test additions
- exact passing-count snapshot depends on the active local branch state and should be frozen before submission
- hardware / runtime details that can be stated from repo:
  - extraction designed for Colab CPU
  - neural code prefers CUDA or CPU, not MPS
  - transfer code limits Torch threads to `1`
- hardware detail still missing and should be added before submission:
  - exact local workstation CPU / RAM / GPU for crash-test runs

### 7. Methodology Chapter Layout

#### 7.1 Baseline Benchmark

- paired-only baseline
- hybrid-only baseline
- calibrated hybrid transfer
- hybrid + paired stack
- early tabular baseline using LightGBM
- role:
  - establish leakage-safe reference point

#### 7.2 Neural SSL Domain Adaptation

- unit encoder pretrained on unlabeled paired units
- supervised hybrid stage
- paired fine-tuning stage
- contextual neural branch explored
- main reason for inclusion:
  - test whether representation learning can close source-target gap

#### 7.3 MMD Domain Adaptation

- source supervision plus latent MMD alignment
- zero-shot and fine-tuned comparisons
- role:
  - test classical neural alignment under severe domain shift

#### 7.4 Context-First Tabular Workbench

- source-derived predictions stacked on paired context
- latent summaries
- reduced latent dimensions
- domain-reweighted source prediction
- LightGBM / XGBoost / CatBoost comparisons
- role:
  - focus only on methods that remained competitive after earlier failures

#### 7.5 Domain-Shift Audit

- domain classifier:
  - hybrid vs paired
- PCA projections
- feature-group shift summaries
- study-level and subgroup diagnostics
- role:
  - explain why transfer fails or succeeds

#### 7.6 SpikeForest Audit

- verify whether more paired data existed inside the current extraction pipeline
- paired coverage by study, recording, sorter output
- role:
  - separate missing data from modeling limitations

#### 7.7 Cluster / Subgroup Methods

- cluster-aware source prediction
- study-aware features
- study-bias correction
- subgroup balancing
- role:
  - test whether paired heterogeneity can be modeled explicitly

#### 7.8 Filtering / Shift-Aware Feature Control

- global drift filtering
- target-protected filtering
- `fpos`-specific helpful filtering
- `fmiss`-specific failure of overly broad filtering
- role:
  - reduce harmful source-target mismatch without dropping predictive structure

#### 7.9 Synthetic / Transport Methods

- synthetic paired augmentation
- paired-conditioned hybrid transport
- global quantile transport
- cluster-conditioned transport
- target-manifold transport
- role:
  - test whether hybrid features can be shifted closer to paired support

#### 7.10 Pseudo-Label / Manifold Methods

- real unlabeled paired `x`
- teacher ensemble
- manifold support estimation
- cluster-aware quotas
- trust-aware pseudo selection
- anchor false-positive regime scoring
- backend sweep across tree models
- role:
  - exploit the `8,775` unlabeled paired units without label leakage

#### 7.11 Family-Robust `fpos` Training

- family-balanced weighting on matched paired rows
- worst-family evaluation and comparison against global-`R²` selection
- robustness vs average-performance tradeoff
- role:
  - test whether the `fpos` model can be made less family-sensitive without leakage

#### 7.12 Signal-Embedding Enrichment

- compact 1D convolutional autoencoders on:
  - `wf_bin_*`
  - `acg_*`
- embeddings learned only from non-holdout rows
- embeddings stacked into the final XGBoost pipeline
- role:
  - recover local waveform / ACG structure not captured by handcrafted summaries

#### 7.13 Paper-Inspired Small-Data Transfer Variants

- adapter-style transfer on frozen waveform embeddings
- few-shot same-family scoring
- labeled-neighborhood support features
- unlabeled-manifold pseudo-features
- role:
  - test literature-inspired low-data transfer ideas against the current winning `fpos` pipeline

### 8. Results Chapter Layout

#### 8.1 Main Empirical Findings

- hybrid-only transfer insufficient
- context-aware tabular transfer beats generic neural transfer
- unlabeled paired structure materially helps `fpos`
- `fmiss` remains harder and more regime-sensitive

#### 8.2 Best Results

- best strict-holdout `fpos`:
  - `wf_embed_anchor_stack_xgboost`
  - frozen artifact:
    - [fpos_signal_embedding_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_signal_embedding_metrics.csv)
  - MAE `0.1140`
  - `R² 0.5479`
- previous strong `fpos` winner before waveform enrichment:
  - `source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default`
  - frozen artifact:
    - [fpos_backend_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_backend_metrics.csv)
  - MAE `0.1169`
  - `R² 0.5331`
- best strict-holdout `fmiss` by `R²`:
  - `source_plus_reduced_latent_fullctx_lightgbm`
  - frozen artifact:
    - [workbench_r2_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_r2_metrics.csv)
  - MAE `0.1562`
  - `R² 0.4850`
- best `fmiss` by MAE among strongest runs:
  - `source_reweighted_fullctx_lightgbm`
  - frozen artifact:
    - [workbench_shift_subgroup_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_shift_subgroup_metrics.csv)
  - MAE `0.1533`
  - `R² 0.4758`

#### 8.3 Domain-Shift Results

- domain classifier AUC, hybrid vs paired:
  - unit features: about `0.976`
  - full context: about `0.9997`
- implication:
  - source and target domains are almost perfectly separable with existing features
- paired domain heterogeneity:
  - `PAIRED_ENGLISH` spans multiple subregions
  - study label alone too coarse to explain all structure

#### 8.4 Negative Results

- SSL neural transfer not competitive
- MMD improved neural path but not enough
- broad contextual neural models too data-hungry
- direct unmatched-pool label use invalid for matched-unit `fpos`
- unmatched-pool label use harmful for `accuracy`
- naive synthetic paired generation insufficient
- free-form meta-stacking not beneficial
- seed bagging not better than current `fpos` winner

### 9. Discussion

- why context-aware boosting dominated
- why source-derived signals help only after target-context anchoring
- why `fpos` responds to unlabeled paired structure
- why `fmiss` remains resistant
- why domain shift is not just “hybrid vs paired” but also subgroup / family / recording mixture
- why matching feature marginals is not enough to recover paired conditional target structure

### 10. Limitations

- small matched paired sample size
- only about `107` non-test matched rows available for model selection / fine-tuning
- many exploratory crash-test iterations against one final holdout
- risk of adaptive overfitting to the holdout through repeated experimentation
- extraction coverage gaps remained in SpikeForest pipeline
- exact local hardware snapshot not fully frozen in repo
- some ideas only evaluated in smoke mode
- confidence intervals / repeated holdouts not yet finalized for the best models
- strongest result currently clearer for `fpos` than for `fmiss`

### 11. Future Work

- freeze final benchmark and rerun top models under repeated grouped resampling
- per-study and worst-group reporting
- richer raw/template features beyond current parquet summaries
- more matched paired data if obtainable
- stronger `fmiss`-specific protected-feature strategies
- more principled uncertainty modeling for pseudo labels

## Literature Review: What To Cover

### A. Spike Sorting And Ground-Truth QC

- SpikeForest benchmark paper
- papers on paired recordings as ground truth
- hybrid ground-truth generation papers
- cross-recording variability in sorter performance
- papers discussing mismatch between synthetic / hybrid and real recordings

### B. Covariate Shift And Importance Weighting

- Shimodaira
- Sugiyama
- KLIEP and direct density-ratio estimation
- why relevant:
  - extreme hybrid vs paired feature shift
  - source reweighting was one of the most useful directions in this project

### C. Dataset Shift Diagnostics

- dataset shift taxonomy
- covariate shift vs label shift vs concept drift
- domain classifier diagnostics
- hidden subpopulation shift
- why relevant:
  - this thesis relies heavily on explicit shift measurement rather than assuming transfer methods work

### D. Subpopulation Shift / Robustness

- WILDS
- subgroup robustness
- worst-group performance
- latent-domain / heterogeneous domain adaptation
- why relevant:
  - paired target domain is heterogeneous
  - study label alone insufficient

### E. Domain Adaptation For Tabular Data

- tabular domain adaptation
- test-time adaptation for tabular data
- domain generalization in structured/tabular settings
- why relevant:
  - best models in this thesis are tabular, not end-to-end deep vision models

### F. Representation Learning For Tabular Data

- deep tabular models
- SAINT / TabNet / related work
- hybrid embedding + boosting pipelines
- why relevant:
  - thesis shows neural representations alone were weaker
  - but source-derived embeddings and summaries were still useful as features

### G. Semi-Supervised Learning / Pseudo-Labeling

- classic self-training
- confidence-based pseudo-label selection
- uncertainty-aware pseudo labeling
- support-constrained pseudo labeling
- why relevant:
  - unlabeled paired pool was useful only under careful trust and support control

### H. Domain Adaptation With Unlabeled Target Data

- MMD
- CORAL
- source-free / semi-supervised adaptation
- why relevant:
  - several such methods were tested directly in this thesis
  - many became informative negative results

### I. Negative Results And Scientific Reporting

- papers / discussions on benchmark leakage
- adaptive overfitting to validation / holdout
- importance of strong baselines and negative results
- why relevant:
  - thesis contains many failed methods that are still scientifically important

## Literature Review: Core Papers / Sources To Include

- SpikeForest paper
- paired / hybrid spike-sorting ground-truth papers
- Shimodaira on covariate shift
- Sugiyama on importance weighting and KLIEP
- WILDS benchmark
- Sagawa et al. on worst-group robustness
- Quionero-Candela et al. on dataset shift
- Subbaswamy and Saria on failures under hidden shift
- strong tabular-learning papers showing boosting remains hard to beat

## Publication Positioning

### What Looks Publishable Now

- strong domain-shift audit with near-perfect hybrid-vs-paired separability
- leakage-safe benchmark construction for this QC task
- clear negative result:
  - generic neural adaptation did not beat context-aware tabular models
- positive result:
  - unlabeled paired units improve `fpos` when exploited structurally
- practical modeling lesson:
  - context-aware tree models plus conservative pseudo-labeling outperform more generic deep transfer here

### Best Publication Angle

- primary paper angle:
  - domain-shift analysis and leakage-safe transfer for spike-sorting QC
- strongest empirical story:
  - `fpos`
- secondary story:
  - `fmiss` as a harder and still unresolved target
- useful framing:
  - “negative results on generic deep domain adaptation, positive results on structure-aware tabular transfer”
  - “waveform representation enrichment plus conservative target-aware boosting beats generic deep adaptation in low-label spike QC transfer”

### Realistic Venues

- neuroscience / neuroinformatics methods venue
- machine-learning-for-science workshop
- domain adaptation / trustworthy ML workshop
- tabular ML workshop
- applied ML for scientific data venue

### NeurIPS Workshop Assessment

- yes:
  - a NeurIPS workshop submission is plausible
- no:
  - this does not currently look like a strong NeurIPS main-conference paper
- strongest NeurIPS-workshop-style framing:
  - tabular domain adaptation under severe shift
  - scientific ML / ML for neuroscience
  - robust / subgroup-aware adaptation
  - negative results on deep transfer with strong tabular baselines
- why a workshop is realistic:
  - clean methodological problem
  - strong diagnostics
  - useful positive and negative results
  - domain-specific but generalizable lesson

### What Would Strengthen A Workshop Submission

- repeated-split confidence intervals for top models
- one frozen final benchmark table
- worst-group / per-study reporting
- stronger ablation table for the winning `fpos` method
- final hardware/environment appendix
- sharper narrative centered on:
  - severe domain shift
  - why deep adaptation failed
  - why structure-aware boosting + unlabeled paired geometry worked

## Files To Cite In Thesis Appendices

- extraction notebook:
  - [ntb_data_extraction_colab.ipynb](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data_pipeline/extraction/ntb_data_extraction_colab.ipynb)
- experiment log:
  - [EXPERIMENT_SUMMARY.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/EXPERIMENT_SUMMARY.md)
- best `fpos` run:
  - [fpos_signal_embedding_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_signal_embedding_metrics.csv)
- previous `fpos` winner / pseudo-label benchmark:
  - [fpos_backend_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_backend_metrics.csv)
- best `fmiss` run:
  - [workbench_r2_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_r2_metrics.csv)
- domain-shift audit:
  - [domain_shift](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/domain_shift)
- SpikeForest audit:
  - [spikeforest_audit](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/spikeforest_audit)

## Final Advice For Writing

- write the thesis around one frozen benchmark
- distinguish exploratory crash tests from final confirmatory comparisons
- make `fpos` the main success case
- present `fmiss` honestly as the harder target
- do not hide negative results
- explicitly discuss adaptive holdout risk
- emphasize that the strongest contribution is methodological rigor plus a clear empirical lesson, not just one number
