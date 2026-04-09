# Reproducibility

This directory keeps the minimal reproducibility metadata that remains visible in the repo.

## Main Files

- [environment.yml](environment.yml)
- [current_environment_snapshot.json](current_environment_snapshot.json)

## Helper Script

- [../scripts/export_repro_snapshot.py](../scripts/export_repro_snapshot.py)

## Intended Use

- record the environment used for a thesis-facing run
- make it obvious which pieces of metadata are expected to change run-to-run
- give a concrete conda environment file for recreating the project stack

## Current Limits

This repo is closer to reproducible than before, but it is still not perfectly one-command reproducible.

- `reproducibility/environment.yml` covers the local modeling and notebook environment, not the exact extraction runtime used to build the SpikeForest training parquet.
- the SpikeForest extraction was run in Google Colab and depends on Colab-specific setup such as runtime resets, Drive mounting, and notebook-first execution order
- rebuilding the training corpus from scratch still depends on external SpikeForest/kachery-cloud access and credentials
- the repo preserves extracted data and generated outputs, but it does not yet provide one fully validated command that recreates every table and figure from a fresh machine
- some code paths still distinguish between the local notebook/modeling environment and the Colab/training environment

## What Still Needs To Happen

The repo will be thesis-ready reproducible when these four conditions are all true:

1. the `42-61` recording-disjoint robustness sweep has finished and the aggregate summaries are regenerated from the completed seed set
2. the story notebooks are rebuilt from source and re-executed into `artifacts/executed_notebooks/`
3. the tracked `figures/` and `tables/` are regenerated from that final sweep, not from a partial intermediate state
4. one clean environment run reproduces the same thesis-facing outputs using only tracked commands and tracked data

## Final Freeze Checklist

After the seed sweep is complete, the final verification path should be:

```bash
conda env create -f reproducibility/environment.yml
conda activate spike-qc-local
python scripts/validate_model_registry.py
PYTHONPATH=src python scripts/run_full_seed_robustness.py --seeds 42-61
python scripts/rebuild_story_notebooks.py
python scripts/execute_notebooks.py
python scripts/generate_missing_figures.py
python scripts/export_repro_snapshot.py
python scripts/audit_thesis_package.py --check-cache
```

That is the missing "single documented path" for step 4. The important point is not that the whole repo becomes one command; it is that the thesis-facing outputs can be reproduced from a fresh environment using one explicit, documented sequence.

What still goes beyond the current audit:

- confirm that all seeds `42-61` completed before accepting the aggregate claims
- confirm that the tracked `figures/`, `tables/`, and `artifacts/executed_notebooks/` were regenerated after the completed sweep
- optionally add one manifest or hash check for the key thesis-facing outputs if a stricter freeze is needed

## Typical Flow

```bash
conda env create -f reproducibility/environment.yml
conda activate spike-qc-local
python scripts/validate_model_registry.py
python scripts/execute_notebooks.py --only 04_fpos_model_progression.ipynb
```
