# Self-Supervised Domain Adaptation

This crash test prototypes a two-stage transfer pipeline for paired-unit QC:

- self-supervised pretraining on unlabeled paired units from non-test recordings
- supervised source training on hybrid labels
- paired fine-tuning with separate downstream models for `fpos` and `fmiss`
- optional contextual fusion using handcrafted recording-relative and recording-context features

The implementation is intentionally isolated from `src/` so the experiment can evolve without changing the main transfer benchmark.

## Runner

```bash
python crash_tests/self_supervised_domain_adaptation/run_experiment.py
```

Smoke test:

```bash
python crash_tests/self_supervised_domain_adaptation/run_experiment.py --smoke --skip-stress
```

## Outputs

Each run writes a new folder under `crash_tests/self_supervised_domain_adaptation/outputs/` with:

- `config.json`
- `split_manifest.json`
- `metrics.csv`
- `predictions.csv`
- `baseline_comparison.csv`
- `stress_test_results.csv`
- `training_history.csv`
- `README_run.md`

Use `run_experiment.ipynb` to launch a run interactively from Jupyter with the
full config visible in the notebook and live training logs in the output cell.

Use `analyze_results.ipynb` to inspect the latest saved run interactively after
training finishes.
