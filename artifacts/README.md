# Thesis Artifacts

This directory is for generated outputs.

## Main Contents

- [executed_notebooks](executed_notebooks)
  Executed copies of the story notebooks.

- `runs*`
  Experiment outputs from recipe runs and seed sweeps.

- `what_if*`
  Scenario or diagnostic run outputs.

- `ablation_checks/`
  Smaller focused checks and comparison outputs.

## Important Rule

Treat this directory as output storage, not as the main implementation layer.

If you want to change how an artifact is produced, edit:

- the relevant notebook in [../notebooks](../notebooks)
- the supporting code in [../src/qc_thesis](../src/qc_thesis)
- or the runner script in [../scripts](../scripts)
