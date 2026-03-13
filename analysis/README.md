# Analysis

This directory contains exploratory and modeling work that is intentionally separate from
the main production notebook.

Structure:

- `notebooks/`: reusable analysis notebooks that work on any compatible QC parquet
- `runs/<run_name>/outputs/`: local CSV/JSON artifacts produced by notebook runs

The main pipeline entry point remains `Spike_UNIT_Quality.ipynb` in the repository root.

Current reusable notebooks:

- `qc_dataset_review.ipynb`: inspect train/test data, targets, feature families, and leakage warnings
- `qc_modeling_benchmark.ipynb`: run leak-aware modeling benchmarks, transfer tests, and SHAP analysis
