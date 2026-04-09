# Scripts

This directory contains small maintenance and execution utilities.

These scripts are not the main modeling framework; they exist to run, validate, or regenerate parts of the repo.

## Stable Tracked Scripts

- [audit_thesis_package.py](audit_thesis_package.py)
  Basic repo/package audit, including notebook bootstrap checks.

- [execute_notebooks.py](execute_notebooks.py)
  Executes notebooks from `notebooks/` and writes outputs to `artifacts/executed_notebooks/`.

- [export_repro_snapshot.py](export_repro_snapshot.py)
  Writes the current environment snapshot into `reproducibility/`.

- [rebuild_story_notebooks.py](rebuild_story_notebooks.py)
  Regenerates the story notebooks from a notebook-construction script.

- [validate_model_registry.py](validate_model_registry.py)
  Checks that the recipe registry is internally consistent.

## Thesis-Finalization Helpers

- `run_full_seed_robustness.py`
  Runs the full `42-61` recording-disjoint robustness sweep and writes seed-specific outputs plus aggregate intermediates under `artifacts/`.

- `generate_missing_figures.py`
  Regenerates summary figures that are computed directly from saved tables and seed artifacts rather than from the notebook execution path alone.

- `regenerate_retained_benchmark_figures.py`
  Refreshes the retained benchmark figure set directly from tracked tables and aggregate seed outputs.

- `improve_figures_nb01.py`, `improve_figures_nb02.py`, `improve_figures_nb04.py`, `improve_figures_nb05.py`, `improve_figures_nb06.py`, `improve_figures_nb07.py`
  Thesis-facing figure cleanup scripts for the notebook chapters that needed more polished publication-style plots than the raw notebook defaults.

- `make_thesis_highlight_figures.py`
  Rebuilds the thesis highlight/synthesis figures from frozen summary tables.

These helpers are part of the final thesis packaging workflow, not ad hoc local scratch files. Logs and partial sweep outputs in `artifacts/` are ignored; the scripts that generate the final tracked figures and tables are not.

## Practical Rule

- if the code is a reusable library API, it belongs in [../src/qc_thesis](../src/qc_thesis)
- if it is a small operational helper, it belongs here
