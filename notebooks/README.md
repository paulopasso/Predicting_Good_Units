# Scientific Notebooks

These notebooks are thesis-facing scientific walkthroughs.

They are now live notebooks backed by the thesis-only package in:

- [src/qc_thesis](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis)
- exploratory data in:
  - [data/raw/33000_ROWS.parquet](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/raw/33000_ROWS.parquet)
- model lineage in:
  - [src/qc_thesis/modeling/MODEL_CODE_INDEX.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/MODEL_CODE_INDEX.md)

They are defined by:

- [THESIS_BLUEPRINT.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/THESIS_BLUEPRINT.md)

Notebook rule:

- notebooks should call the framework
- notebooks may load frozen recipe artifacts through `load_or_run_experiment`
- notebooks should not be thin CSV viewers
- the runtime implementation must stay inside [src/qc_thesis](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis)
- the modeling runtime registry and experiment loaders live in [src/qc_thesis/modeling](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling)
- provenance files under [src/qc_thesis/modeling/provenance](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/provenance) are explanatory references, not notebook runtime dependencies
