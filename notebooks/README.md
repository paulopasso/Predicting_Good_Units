# Scientific Notebooks

These notebooks are the thesis narrative layer.

They should explain the dataset, the benchmark, and the main results, while relying on reusable code from [src/qc_thesis](../src/qc_thesis).

## Notebook Sequence

- [01_data_extraction_and_audit.ipynb](01_data_extraction_and_audit.ipynb)
  Dataset composition, subset definitions, and split protocol setup.

- [02_domain_shift_hybrid_vs_paired.ipynb](02_domain_shift_hybrid_vs_paired.ipynb)
  Hybrid-vs-paired domain shift and PCA/separability diagnostics.

- [03_paired_family_structure.ipynb](03_paired_family_structure.ipynb)
  Heterogeneity inside the paired domain and family-level structure.

- [04_fpos_model_progression.ipynb](04_fpos_model_progression.ipynb)
  `fpos` benchmark ladder and ablations.

- [05_fmiss_model_progression.ipynb](05_fmiss_model_progression.ipynb)
  `fmiss` benchmark ladder and ablations.

- [06_fpos_vs_fmiss_xai.ipynb](06_fpos_vs_fmiss_xai.ipynb)
  XAI, family behavior, and target asymmetry.

- [07_limitations_and_data_budget.ipynb](07_limitations_and_data_budget.ipynb)
  Label-budget and robustness analysis.

- [08_appendix_additional_ablations.ipynb](08_appendix_additional_ablations.ipynb)
  Appendix-level supplementary analyses.

## Practical Rules

- notebook cells should call the package API in [src/qc_thesis](../src/qc_thesis)
- long-lived logic belongs in Python modules, not inline notebook cells
- output figures go to `figures/<notebook_stem>/`
- output tables go to `tables/<notebook_stem>/`
- executed copies go to `artifacts/executed_notebooks/`

## Related Docs

- [../README.md](../README.md)
- [../EXPERIMENT_SUMMARY.md](../EXPERIMENT_SUMMARY.md)
- [../src/qc_thesis/README.md](../src/qc_thesis/README.md)
- [../src/qc_thesis/modeling/README.md](../src/qc_thesis/modeling/README.md)
