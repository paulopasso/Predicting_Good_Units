# SpikeForest Audit

Completeness audit for the paired SpikeForest extraction already present in this repo.

The audit:

- parses `SPIKEFOREST_STUDY_SETS` from the extraction notebook
- compares those requested paired studies against the current parquet
- writes per-study and per-recording coverage manifests
- flags likely missing study/recording/sorter items using study-level sorter coverage heuristics

Runner:

```bash
PYTHONPATH=src:. python crash_tests/spikeforest_audit/run_experiment.py --run-name first_audit
```

Outputs are written under:

`crash_tests/spikeforest_audit/outputs/<run_name>/`
