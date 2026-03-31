# Thesis Data

This directory contains the public dataset inputs used by the notebooks and framework.

## Provenance

These files are derived from the SpikeForest extraction workflow used for this project.

- upstream project: [SpikeForest](https://github.com/flatironinstitute/spikeforest)
- extraction notebook in this repo: [../data_pipeline/extraction/ntb_data_extraction_colab.ipynb](../data_pipeline/extraction/ntb_data_extraction_colab.ipynb)
- canonical extraction runtime: Google Colab standard CPU runtime

## Main Contents

- `raw/33000_ROWS.parquet`
  The main local modeling table.

- `dataset_summary.json`
  A compact summary of the dataset.

- `frozen_inputs/`
  CSV and manifest assets that are intentionally fixed for certain notebooks or benchmark summaries.

## Rule Of Thumb

- use `raw/` when you want the underlying dataset
- use `frozen_inputs/` when a notebook intentionally references a frozen benchmark export
- do not hand-edit generated CSVs under `frozen_inputs/` unless you are explicitly refreshing those frozen inputs
