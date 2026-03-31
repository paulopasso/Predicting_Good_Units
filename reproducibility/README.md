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

## Typical Flow

```bash
conda env create -f reproducibility/environment.yml
conda activate spike-qc-local
python scripts/execute_notebooks.py --only 04_fpos_model_progression.ipynb
```
