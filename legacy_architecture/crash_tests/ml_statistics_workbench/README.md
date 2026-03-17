# ML / Statistics Workbench

Strict recording-disjoint experiment harness for trying high-value transfer ideas without changing the main pipeline.

Current focused candidate families:

- `contextual_paired_only`
- `embed_nommd_fullctx`
- `source_reweighted_fullctx`
- `latent_summary_fullctx`
- `source_plus_reduced_latent_fullctx`
- `cluster_source_fullctx`

All runs:

- hold out paired recordings at the `recording_key` level
- keep `fpos` and `fmiss` as separate target-specific models
- use `fmiss_extended` only on the hybrid source side
- select winners with `R²` first and MAE guardrails
- optionally add filtered LightGBM variants that drop the most shifted training-side non-shape features
- optionally add cluster-aware source stacks built from target-side latent clusters
- write artifacts under `crash_tests/ml_statistics_workbench/outputs/<run_name>/`

Runner:

```bash
PYTHONPATH=src:. python crash_tests/ml_statistics_workbench/run_experiment.py --run-name first_pass
```
