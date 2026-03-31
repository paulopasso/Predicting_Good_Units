# Reproducibility Specification

## Global defaults

- canonical random seed: `42`
- primary optimization benchmark:
  - recording-disjoint by `recording_key`
- mandatory stress benchmark:
  - leave-one-family-out by `study_set`
- all thesis-cited benchmarks must record both:
  - the primary recording-disjoint result
  - the corresponding family-held-out robustness result

## Required environment capture

Every frozen thesis run must record:

- git commit hash
- current branch name
- Python version
- OS / machine description
- CPU description
- RAM size
- GPU description if used
- package lock / installed dependency snapshot
- key environment variables affecting training
- local source roots used during execution:
  - `thesis/src`
  - `thesis/src/qc_thesis/modeling/provenance`
  - `thesis/analysis_pipeline/packages` when audit code is executed

## Required artifact set

Each frozen run must include:

- `config.json`
- `split_manifest.json`
- `metrics.csv`
- `predictions.csv`
- `per_family_summary.csv`
- `per_recording_summary.csv`
- figure outputs used in the thesis
- provenance:
  - script path or notebook path
  - run name
  - data version path

## Required split metadata

Every benchmark must state:

- optimization protocol:
  - `recording_disjoint_main`
- stress protocol:
  - `family_held_out`
- target
- number of source rows
- number of paired labeled rows
- number of paired unlabeled rows
- test row count
- held-out recording keys
- held-out study sets if applicable

## Required metric fields

For every top-level benchmark row:

- MAE
- RMSE
- `R²`
- bias
- calibration slope
- calibration intercept

For every per-family summary:

- family
- row count
- MAE
- `R²`
- bias

For every per-recording summary:

- recording key
- family
- row count
- MAE
- `R²`
- bias

## Determinism notes

- use seed `42` unless a benchmark explicitly studies seed variation
- when stochastic embeddings are used, save:
  - embedding seed
  - model seed
  - CV seed
- if exact determinism is not possible, note the stochastic component explicitly in the run README or config

## Required model-code packaging

The thesis package must include the source code used to build the benchmarked models, not only frozen outputs.

Required local code roots:

- [src/qc_thesis](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis)
- [src/qc_thesis/modeling](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling)
- [src/qc_thesis/modeling/provenance](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/provenance)
- [analysis_pipeline/packages](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/analysis_pipeline/packages)

Reviewer-facing code map:

- [src/qc_thesis/modeling/MODEL_CODE_INDEX.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/MODEL_CODE_INDEX.md)
