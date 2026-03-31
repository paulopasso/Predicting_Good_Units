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

## Practical Rule

- if the code is a reusable library API, it belongs in [../src/qc_thesis](../src/qc_thesis)
- if it is a small operational helper, it belongs here
