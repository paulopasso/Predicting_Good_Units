# MMD Domain Adaptation

This crash test prototypes a strict recording-disjoint MMD transfer pipeline for
paired-unit QC:

- source supervision on hybrid labels
- unsupervised latent alignment against unmatched paired units with MMD
- optional fine-tuning on labeled paired non-test rows
- strict holdout evaluation on paired recordings never seen in any training stage

The implementation is isolated from `src/` so the experiment can evolve without
changing the main benchmark.

## Runner

```bash
python crash_tests/mmd_domain_adaptation/run_experiment.py
```

Smoke test:

```bash
python crash_tests/mmd_domain_adaptation/run_experiment.py --smoke
```

Skip paired fine-tuning:

```bash
python crash_tests/mmd_domain_adaptation/run_experiment.py --skip-finetune
```

## Outputs

Each run writes a new folder under `crash_tests/mmd_domain_adaptation/outputs/`
with:

- `config.json`
- `split_manifest.json`
- `metrics.csv`
- `predictions.csv`
- `baseline_comparison.csv`
- `lambda_sweep.csv`
- `training_history.csv`
- `README_run.md`

Use `run_experiment.ipynb` to launch a run interactively from Jupyter with live
training logs in the notebook output cell.

Use `analyze_results.ipynb` to inspect the latest saved run after training
finishes.
