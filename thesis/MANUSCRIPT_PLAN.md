# Manuscript Plan

## Purpose

This file translates the thesis blueprint into a writing order.

## Chapter flow

1. Introduction
- problem: QC prediction under hybrid-to-paired shift
- motivation for strict no-leakage evaluation
- thesis contributions

2. Background
- spike sorting
- paired vs hybrid recordings
- domain adaptation for tabular scientific data

3. Data and extraction
- Colab extraction workflow
- data provenance
- audit and coverage checks

4. Benchmark design
- primary benchmark: recording-disjoint
- stress test: leave-one-family-out
- artifact schema and reproducibility rules

5. Domain-shift analysis
- hybrid vs paired separability
- PCA / 3D PCA
- family heterogeneity

6. Modeling results for `fpos`
- benchmark ladder
- ablations
- robustness and recording-level views
- final winner

7. Modeling results for `fmiss`
- benchmark ladder
- ablations
- target asymmetry vs `fpos`
- final winner

8. Interpretability and failure analysis
- SHAP
- calibration
- per-family and per-recording residuals
- missing-signal analysis

9. Limitations and future gains
- label-budget curve
- family-unseen degradation
- likely next feature directions

## Writing rules

- main text includes the key negative results
- appendix receives the long tail of ablations
- `fpos` is the main positive result
- `fmiss` is the harder counterexample
- family-unseen is reported for every thesis-facing benchmark family even when optimization is recording-disjoint

## Draft chapter files

- [main_text/03_domain_shift_results.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/main_text/03_domain_shift_results.md)
- [main_text/04_fpos_results.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/main_text/04_fpos_results.md)
- [main_text/05_fmiss_results.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/main_text/05_fmiss_results.md)
- [main_text/06_interpretability_and_limitations.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/main_text/06_interpretability_and_limitations.md)
