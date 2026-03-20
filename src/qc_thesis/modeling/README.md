# Modeling Code

`qc_thesis.modeling` is the thesis-local research package for runnable model code.

Structure:

- `core`
  - shared stack primitives and feature-shift helpers
- `transfer`
  - hybrid-to-paired tabular transfer benchmarks
- `stacking`
  - context, pseudo-label, anchor, waveform, and robustness stack families
- `neural`
  - SSL and MMD runners
- `recipes`
  - recipe catalog, runner registry, and notebook-facing APIs

Notebook-facing entrypoints:

- `list_recipes()`
- `run_recipe(recipe_id, output_dir=None, force=False)`
- `load_recipe_result(recipe_id)`
- `build_benchmark_table(...)`

The thesis package no longer treats historical file paths as a primary interface.
The source of truth is the runnable recipe registry under `qc_thesis.modeling.recipes`.
