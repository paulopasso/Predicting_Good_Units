# Repository And Experiment Handoff

This file is the short operational guide for resuming work inside the thesis package.

It complements:

- [THESIS_BLUEPRINT.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/THESIS_BLUEPRINT.md)
- [THESIS_OUTLINE.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/THESIS_OUTLINE.md)
- [EXPERIMENT_SUMMARY.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/EXPERIMENT_SUMMARY.md)

## 1. What Is Authoritative

Open these first:

- [README.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/README.md)
- [THESIS_BLUEPRINT.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/THESIS_BLUEPRINT.md)
- [THESIS_OUTLINE.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/THESIS_OUTLINE.md)
- [EXPERIMENT_SUMMARY.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/EXPERIMENT_SUMMARY.md)
- [src/qc_thesis](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis)

Framework rule:

- runtime experiment logic lives in [src/qc_thesis](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis)
- exploratory notebooks live in [notebooks](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/notebooks)
- historical implementation lineage lives in [src/qc_thesis/modeling/provenance](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/provenance)

## 2. Core Thesis Structure

Main thesis-local areas:

- [data](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data)
  - thesis-local raw parquet and frozen inputs
- [data_pipeline](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data_pipeline)
  - extraction notebook and provenance docs
- [analysis_pipeline](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/analysis_pipeline)
  - domain-shift and SpikeForest audit code
- [src/qc_thesis](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis)
  - clean runtime framework
- [src/qc_thesis/modeling](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling)
  - curated provenance for cited model families
- [notebooks](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/notebooks)
  - scientific narrative notebooks
- [figures](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/figures)
  - manuscript-ready figure exports
- [tables](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/tables)
  - manuscript-ready table exports
- [tests](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/tests)
  - thesis-local validation

## 3. Data Context

Primary curated table:

- [33000_ROWS.parquet](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/raw/33000_ROWS.parquet)

Current working understanding:

- total rows: about `33,206`
- hybrid labeled source: about `24k`
- paired matched labeled units: `207`
- paired unmatched unlabeled units: `8,775`
- strict paired test holdout: about `100` matched rows
- paired non-test labeled pool: about `107` matched rows
- non-test unlabeled paired pool after holdout exclusion: about `4,963`

Hard leakage boundary:

- `recording_key`

Thesis benchmark rule:

- every thesis-facing benchmark family should record both:
  - `recording_disjoint_main`
  - `family_held_out`

## 4. Current Best Results

Best `fpos`:

- variant: `wf_embed_anchor_stack_xgboost`
- metrics source:
  - [fpos_signal_embedding_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_signal_embedding_metrics.csv)
- performance:
  - MAE `0.1140`
  - `R² 0.5479`

Previous important `fpos` winner:

- variant: `source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default`
- metrics source:
  - [fpos_backend_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_backend_metrics.csv)
- performance:
  - MAE `0.1169`
  - `R² 0.5331`

Best `fmiss` by `R²`:

- variant: `source_plus_reduced_latent_fullctx_lightgbm`
- metrics source:
  - [workbench_r2_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_r2_metrics.csv)
- performance:
  - MAE `0.1562`
  - `R² 0.4850`

Best `fmiss` by MAE among strong runs:

- variant: `source_reweighted_fullctx_lightgbm`
- metrics source:
  - [workbench_shift_subgroup_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_shift_subgroup_metrics.csv)
- performance:
  - MAE `0.1533`
  - `R² 0.4758`

## 5. Important Scientific Conclusions

What clearly helped:

- context-aware tabular modeling
- source-derived signals stacked on paired context
- trust-aware pseudo use of unlabeled paired units
- family-balanced paired weighting for robustness
- waveform representation learning as features

What clearly did not become the winning path:

- generic SSL domain adaptation
- MMD neural alignment
- broad contextual neural models
- naive synthetic paired generation
- explicit depth-3 tree interaction features on top of the waveform winner
- several paper-inspired low-data add-ons beyond the waveform winner

Interpretation:

- hybrid vs paired domain shift is extreme
- paired target domain is heterogeneous
- `fpos` benefits from unlabeled paired structure much more than `fmiss`
- tree models remain the best final predictors in this setting

## 6. Important Supporting Audits

Domain shift:

- [domain_shift](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/domain_shift)

SpikeForest coverage:

- [spikeforest_audit](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/spikeforest_audit)

## 7. Important Provenance Files

Most important model-lineage files:

- [fpos_signal_embedding_stack.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/provenance/ml_statistics_workbench/fpos_signal_embedding_stack.py)
- [fpos_family_robustness.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/provenance/ml_statistics_workbench/fpos_family_robustness.py)
- [fpos_backend_sweep.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/provenance/ml_statistics_workbench/fpos_backend_sweep.py)
- [fpos_anchor_stack.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/provenance/ml_statistics_workbench/fpos_anchor_stack.py)
- [pseudo_label_manifold.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/provenance/ml_statistics_workbench/pseudo_label_manifold.py)
- [run_experiment.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/provenance/ml_statistics_workbench/run_experiment.py)
- [transfer.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/provenance/legacy_transfer/transfer.py)

Use [MODEL_CODE_INDEX.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/MODEL_CODE_INDEX.md) for the full map.

## 8. Reproducibility Workflow

Current thesis-local execution expectations:

- use `PYTHONPATH=thesis/src`
- keep `recording_key` as the hard leakage boundary
- treat `family_held_out` as the mandatory robustness companion
- update [EXPERIMENT_SUMMARY.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/EXPERIMENT_SUMMARY.md) after meaningful thesis-local changes

Examples:

```bash
PYTHONPATH=thesis/src python -m pytest thesis/tests/test_qc_thesis.py thesis/tests/test_model_snapshot.py -q
PYTHONPATH=thesis/src python thesis/scripts/execute_notebooks.py --only 04_fpos_model_progression.ipynb
python thesis/scripts/validate_model_provenance.py
```

## 9. Restart Order

1. read [EXPERIMENT_SUMMARY.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/EXPERIMENT_SUMMARY.md)
2. inspect the best `fpos` frozen artifacts:
   - [fpos_signal_embedding_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/fpos_signal_embedding_metrics.csv)
3. inspect the best `fmiss` frozen artifacts:
   - [workbench_r2_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_r2_metrics.csv)
   - [workbench_shift_subgroup_metrics.csv](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs/benchmarks/workbench_shift_subgroup_metrics.csv)
4. inspect the framework:
   - [src/qc_thesis](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis)
5. only then add new thesis-local branches
