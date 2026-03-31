# Data Pipeline

This directory documents how the training corpus was created before the modeling stage.

## Main File

- [extraction/ntb_data_extraction_colab.ipynb](extraction/ntb_data_extraction_colab.ipynb)

This notebook is the extraction-side source of truth for:

- enumerating SpikeForest study sets and outputs
- extracting per-unit features for the training corpus
- writing checkpointed extraction outputs
- resuming long-running extraction work

## Extraction Provenance

- the SpikeForest training extraction was run through [extraction/ntb_data_extraction_colab.ipynb](extraction/ntb_data_extraction_colab.ipynb)
- that workflow was executed in Google Colab on the standard CPU runtime
- in other words, the canonical extraction environment was Colab-first, not the local conda environment used for the thesis notebooks and modeling package
- the extraction notebook itself is the operational source of truth for installs, runtime restarts, Drive mounting, and checkpoint resume behavior

## What Belongs Here

- extraction logic
- extraction-specific provenance notes
- long-running Colab workflow details

## What Does Not Belong Here

- thesis result plots and benchmark interpretation
- reusable modeling logic
- final notebook-facing experiment registry code

Those belong in [../notebooks](../notebooks) and [../src/qc_thesis](../src/qc_thesis).
