# Domain Shift Audit

Notebook-backed audit for understanding how the hybrid and paired domains differ in the current feature space.

It writes:

- `feature_shift_summary.csv`
- `feature_group_summary.csv`
- `domain_classifier_summary.csv`
- `domain_classifier_coefficients.csv`
- `study_shift_summary.csv`
- `pca_projection.csv`
- `pca_projection_3d.csv`

Runner:

```bash
PYTHONPATH=src:. python crash_tests/domain_shift_audit/run_report.py --run-name first_pass
```
