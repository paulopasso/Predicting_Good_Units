# Spike QC Pipeline Redesign Plan

## Progress Saved
- [x] Commit 1: `6fc3d97` docs update for SHA1 workflow
- [x] Commit 2: `8608d2b` notebook hardening for runtime diagnostics and extraction stability
- [x] Branch pushed: `origin/prototype`

## Execution Checklist (with estimates)

| Status | Task | Estimate | Output |
|---|---|---:|---|
| [ ] | 1. Freeze data source contracts (all SHA1 manifests and expected schemas) | 2h | `data_contracts.json` |
| [ ] | 2. Build full metadata census for all recordings (study set, channels, fs, duration, true units) | 4h | `recording_census.parquet` |
| [ ] | 3. Build full sorting-output census (sorter, return code, timed out, sorting object format) | 4h | `sorting_output_census.parquet` |
| [ ] | 4. Ground-truth audit per recording (extractor load, unit-id dtype, spike train integrity) | 5h | `ground_truth_audit.parquet` |
| [ ] | 5. Recording-type taxonomy (monotrode, tetrode, siprobe, mea, channel bins, drift/no-drift) | 3h | `recording_profiles.parquet` |
| [ ] | 6. Define feature profile matrix by recording type (required vs optional features) | 4h | `feature_profile_matrix.yaml` |
| [ ] | 7. Implement validation-first preflight gate (block bad rows before extraction) | 6h | `preflight_report.json` |
| [ ] | 8. Implement extractor adapters per profile and GT format | 10h | Updated notebook Phase 1 + tests |
| [ ] | 9. Run dry extraction benchmark on 1 sample per profile, then full run | 8h | `extraction_benchmark.md` |
| [ ] | 10. Final reproducibility and leakage audit pass | 4h | `final_validation_report.md` |

## Technical Redesign Rules
- Always validate schema first, then extract.
- Never run feature extraction on a row that fails preflight.
- Canonicalize all unit IDs to integer `NumpySorting` before comparison.
- Separate transient infra failures from deterministic data/schema failures.
- Track every skip reason with normalized error category.
- Make feature extraction profile-aware (do not assume same channels/geometry across datasets).

## Ground Truth Handling Strategy
- Validate GT extractor loadability for each recording before any sorter comparison.
- Record GT unit-id dtype before and after canonicalization.
- Enforce one canonical comparison path: `GT -> int`, `SortingOut -> int`, then compare.
- Mark rows as `gt_invalid` only when GT itself is missing/corrupt, not when sorter output fails.

## Timing and Milestones
- Milestone A (Tasks 1-4): 15h, complete full data understanding and GT contract.
- Milestone B (Tasks 5-8): 23h, implement profile-based extraction engine.
- Milestone C (Tasks 9-10): 12h, verify reliability and research validity.
- Total estimated effort: 50h.

## Immediate Next Step
- Run Tasks 1-3 first in one pass and lock manifests/census before changing extraction logic again.
