# Predicting Good Units

This repository is the working codebase for leakage-safe prediction of spike-sorting quality metrics under hybrid-to-paired domain shift.

The public documentation surface is intentionally small:

- this file for navigation
- [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md) for the main findings and why they matter
- folder-level README files for the parts of the repo that people actually edit

## Read These First

1. [README.md](README.md)
2. [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)
3. [notebooks/README.md](notebooks/README.md)
4. [src/qc_thesis/README.md](src/qc_thesis/README.md)
5. [src/qc_thesis/modeling/README.md](src/qc_thesis/modeling/README.md)

## Repo Layout

- [src/qc_thesis](src/qc_thesis)
  Reusable Python package. This is the main implementation layer.

- [notebooks](notebooks)
  Thesis story notebooks. These should call into `qc_thesis`.

- [data_pipeline](data_pipeline)
  SpikeForest extraction and corpus-building workflow.

- [scripts](scripts)
  Small operational utilities such as notebook execution and registry validation.

- [data](data)
  Public derived dataset bundle plus frozen inputs used by notebooks. These files were extracted from SpikeForest outputs; upstream project: [SpikeForest](https://github.com/flatironinstitute/spikeforest).

- [reproducibility](reproducibility)
  Minimal reproducibility notes and environment snapshot support.

- [artifacts](artifacts), [figures](figures), [tables](tables)
  Mostly generated outputs.

Archived exploratory audit code lives under `legacy_architecture/` and is not part of the main workflow.

## Where To Make Changes

- modeling logic:
  [src/qc_thesis/modeling](src/qc_thesis/modeling)

- notebook narrative and plots:
  [notebooks](notebooks)

- SpikeForest extraction:
  [data_pipeline/extraction/ntb_data_extraction_colab.ipynb](data_pipeline/extraction/ntb_data_extraction_colab.ipynb)

- helper commands and repo maintenance:
  [scripts](scripts)

## Minimal Commands

```bash
conda env create -f reproducibility/environment.yml
conda activate spike-qc-local
python scripts/validate_model_registry.py
python scripts/execute_notebooks.py --only 04_fpos_model_progression.ipynb
```
