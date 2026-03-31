# `qc_thesis` Package Guide

`qc_thesis` is the reusable framework layer behind the thesis notebooks.

If you are changing logic rather than prose, this is usually the place to start.

## What Lives Here

- [data.py](data.py)
  Dataset loading and dataset-bundle helpers.

- [features.py](features.py)
  Feature groups, feature selection helpers, and context-feature construction.

- [protocols.py](protocols.py)
  Split logic such as recording-disjoint and leave-one-family-out protocols.

- [domain.py](domain.py)
  Domain-shift bundles and feature-group summaries used by the domain-shift notebooks.

- [limitations.py](limitations.py)
  Helpers for label-budget and robustness analysis.

- [plots.py](plots.py)
  Shared plotting utilities for the thesis notebooks.

- [story.py](story.py)
  Notebook-facing table builders and summary helpers.

- [xai.py](xai.py)
  Explainability helpers and `fpos`/`fmiss` asymmetry tables.

- [modeling](modeling)
  Experiment recipes, runners, model registry, and stack-family logic.

## Notebook-Facing API

The notebooks generally import from the top-level package:

```python
from qc_thesis import *
```

The most important notebook-facing entrypoints are:

- dataset helpers such as `build_dataset_bundle()` and `load_raw_dataset()`
- split helpers such as `recording_disjoint_split()` and `leave_one_family_out()`
- plotting helpers such as `plot_benchmark_metric()` and `plot_pca_projection()`
- recipe helpers such as `list_recipes()`, `build_recipe_inventory()`, `run_recipe()`, and `load_or_run_experiment()`

## Practical Rule

- put reusable logic in `qc_thesis`
- keep notebooks thin
- treat `figures/`, `tables/`, and most of `artifacts/` as outputs, not implementation
