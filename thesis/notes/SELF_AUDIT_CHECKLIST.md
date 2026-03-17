# Self Audit Checklist

This file is the independent cleanup and packaging check performed after reorganizing the thesis package. There is no separate multi-agent execution tool in this environment, so this checklist is the explicit second-pass audit.

## Structural checks

- [x] Thesis-facing notebooks live under [thesis/notebooks](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/notebooks)
- [x] Thesis-facing Python code lives under [thesis/src/qc_thesis](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis)
- [x] Modeling runtime and provenance now share one namespace under [thesis/src/qc_thesis/modeling](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling)
- [x] Duplicate top-level modeling shims were removed; runtime modeling imports now resolve directly through [thesis/src/qc_thesis/modeling](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling)
- [x] Thesis-facing scripts live under [thesis/scripts](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/scripts)
- [x] Thesis-local model provenance layer lives under [thesis/src/qc_thesis/modeling/provenance](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/src/qc_thesis/modeling/provenance)
- [x] Thesis-local analysis and audit code lives under [thesis/analysis_pipeline](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/analysis_pipeline)
- [x] Thesis-local extraction code lives under [thesis/data_pipeline](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data_pipeline)
- [x] Thesis-local raw parquet lives under [thesis/data/raw](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/raw)
- [x] Frozen thesis inputs live under [thesis/data/frozen_inputs](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/data/frozen_inputs)
- [x] Reproducibility files live under [thesis/reproducibility](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/reproducibility)
- [x] Thesis docs now live inside `thesis/`

## Execution checks

- [x] Environment snapshot regenerated from [export_repro_snapshot.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/scripts/export_repro_snapshot.py)
- [x] Thesis notebooks executed from [execute_notebooks.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/scripts/execute_notebooks.py)
- [x] Thesis-local model provenance check passes via [validate_model_provenance.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/scripts/validate_model_provenance.py)
- [x] Thesis package audit passes via [audit_thesis_package.py](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/scripts/audit_thesis_package.py)
- [x] Executed notebook copies written to [artifacts/executed_notebooks](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/artifacts/executed_notebooks)
- [x] Figures exported under [thesis/figures](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/figures)
- [x] Tables exported under [thesis/tables](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/tables)
- [x] Current notebooks are framework-driven rather than static CSV display notebooks

## Reproducibility checks

- [x] Thesis-local `pyproject.toml` exists
- [x] Thesis-local package metadata declares `src`-based packaging in [thesis/pyproject.toml](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/pyproject.toml)
- [x] Thesis-local tests exist in [thesis/tests](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/tests)
- [x] Notebook ids normalized for cleaner execution
- [x] Thesis-local `.gitignore` exists for cache residue
- [x] Cache directories removed from the committed thesis tree
- [x] Notebook bootstrap now finds the thesis root instead of assuming `Path.cwd() / "src"`
- [x] Core thesis docs no longer depend on non-thesis runtime paths
- [x] Broad mirrored model snapshot was replaced by a curated provenance layer

## Manuscript checks

- [x] Blueprint exists in [thesis/THESIS_BLUEPRINT.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/THESIS_BLUEPRINT.md)
- [x] Outline exists in [thesis/THESIS_OUTLINE.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/THESIS_OUTLINE.md)
- [x] Handoff exists in [thesis/REPO_EXPERIMENT_HANDOFF.md](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/REPO_EXPERIMENT_HANDOFF.md)
- [x] Main-text chapter drafts exist in [thesis/main_text](/Users/paulruiz/Documents/Predicting_Good_Units/thesis/main_text)
- [x] Figure and table indices updated

## Remaining work

- [ ] Tighten figure typography and final manuscript layout
- [ ] Add dataset lineage and split diagrams as explicit figure assets
- [ ] Freeze final main-text figure subset
- [ ] Write introduction, background, and methods prose
