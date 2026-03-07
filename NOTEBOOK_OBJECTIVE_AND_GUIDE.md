# Spike QC Notebook: Objective and File Guide

This repo currently centers on one main notebook supposed to run on colab:

- `Spike_UNIT_Quality.ipynb`

## Objective

The notebook builds a research-grade pipeline to predict spike-sorting quality **without requiring ground truth at inference time**.

It trains models to predict:

- `fmiss` (missed spike fraction proxy)
- `fpos` (false positive fraction proxy)

from waveform/statistical/relational features extracted from sorted units.

## High-Level Workflow

1. Install and validate runtime dependencies (Colab-safe, ABI checks included).
2. Mount Google Drive and configure persistent paths.
3. Enumerate SpikeForest outputs from SHA1 manifests (with kachery-cloud auth handling and fallback).
4. Extract unit-level features for training data (checkpointed per recording/sorter).
5. Audit dataset and leakage risks.
6. Train group-aware XGBoost models with leakage controls.
7. Evaluate diagnostics (optional SHAP).
8. Build test-set features from `sim_hybrid` precomputed sorter output.
9. Evaluate test metrics and export artifacts.

## Core Design Choices

- **Leakage control**:
  - Grouped CV on composite recording identity
  - Fold-wise imputation (fit on train fold only)
  - Inner grouped split for early stopping
- **Reproducibility**:
  - Seeded randomness
  - Version/runtime audit
  - Persistent manifests for source and model artifacts
- **Crash-safe execution**:
  - Feature extraction written in small parts to Drive
  - Resume states for feature extraction and CV folds
  - Re-run skips completed chunks by default

## Data and Storage Layout

Configured under:

- `/content/drive/MyDrive/Thesis/prototype`

Main subfolders:

- `artifacts/features` (train/test parquet)
- `artifacts/models`
- `artifacts/results`
- `artifacts/manifests`
- `artifacts/audits`
- `checkpoints/` (part files + resume state)
- `logs/`

Raw datasets are referenced from:

- `/content/drive/MyDrive/Thesis/Data`

## Feature Families in the Notebook

- **Block A**: waveform morphology
- **Block B**: amplitude distribution + temporal drift/evolution
- **Block C**: spike train and ACG-based features
- **Block E**: recording-level context
- **Block F**: relational/template-confusability features
- **SpikeInterface metrics**: unit quality metrics with safe fallback when optional extensions are unavailable

## External Services and APIs

- **SpikeForest** for benchmark recordings/sorting outputs
- **kachery-cloud** for content-addressed metadata/assets
- **SpikeInterface** for extractor interoperability and quality metrics

Important API detail:

- Enumeration uses the same SHA1 URIs shown in `spikeforest/examples`:
  - default sorting outputs + recordings
  - hybrid janelia sorting outputs + recordings
  - synth monotrode sorting outputs + recordings
- `SFSortingOutput` provides `study_name`, `recording_name`, `sorter_name`.
- `study_set_name` is resolved by joining against the paired recordings manifest.
- Phase 0 stores `sorting_object` directly in manifest rows and Phase 1 loads sortings from that object, avoiding brittle object re-discovery.

## Practical Run Modes

- **Fast/resume mode**: reuse existing train/test features and CV artifacts.
- **Rebuild mode**: set `FORCE_REBUILD_*` / `FORCE_RETRAIN_MODELS` in config.

## Known Failure Points and Current Handling

- **NumPy ABI mismatch** (`numpy.dtype size changed`):
  - Notebook enforces `numpy<2` and runs ABI smoke checks/repair paths.
- **SpikeForest auth (`Client keys not found`)**:
  - Best-effort key bootstrap
  - Persistent `KACHERY_CLOUD_DIR`
  - Graceful fallback to cached manifest if needed
- **Dependency resolver conflicts**:
  - Staged install strategy with hard checks for critical packages (`numcodecs`, `zarr`) and compatibility checks for the rest.
- **SHAP import instability**:
  - SHAP is optional; diagnostics are skipped if unavailable.

## Expected Outputs

- `features_train.parquet`
- `features_test.parquet`
- fold models + CV metrics
- OOF predictions
- test predictions + test metrics
- leakage/data audit JSONs
- run/source/model manifests

## If You Need to Extend the Notebook

- Keep new features independent of ground-truth labels at inference.
- Preserve group-aware splits and fold-local preprocessing.
- Keep extraction idempotent/checkpointed (write per-item partial outputs first).
- Update manifests when changing schema or dependencies.
