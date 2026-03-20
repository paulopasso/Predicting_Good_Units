# Thesis Package

This directory is the thesis-facing research repository for the project.

It is intended to behave like the clean standalone package that would be handed to a thesis reviewer or publisher.

It contains:

- documentation and manuscript planning
- thesis-local data extraction code
- thesis-local analysis and audit code
- thesis-local experiment framework code
- curated historical provenance code for the benchmarked model families
- frozen thesis inputs
- exploratory notebooks
- exported manuscript figures and tables
- executed notebook artifacts
- reproducibility metadata

Authoritative specification:

- [THESIS_BLUEPRINT.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/THESIS_BLUEPRINT.md)

Core companion files:

- [MANUSCRIPT_PLAN.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/MANUSCRIPT_PLAN.md)
- [FIGURE_INDEX.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/FIGURE_INDEX.md)
- [TABLE_INDEX.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/TABLE_INDEX.md)
- [reproducibility/README.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/reproducibility/README.md)
- [data_pipeline/README.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data_pipeline/README.md)
- [analysis_pipeline/README.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/analysis_pipeline/README.md)
- [src/qc_thesis/modeling/README.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/README.md)
- [data/README.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/README.md)
- [notebooks/README.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/notebooks/README.md)

Framework-first rule:

- runtime experiment and plotting logic lives only in [src/qc_thesis](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis)
- notebooks call that framework
- [src/qc_thesis/modeling](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling) is the modeling namespace:
  - clean runtime recipe/experiment APIs
  - curated provenance files that explain how the historical model families were originally built
