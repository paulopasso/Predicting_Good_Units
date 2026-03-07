# Spike QC Pipeline Redesign Plan

## Progress Saved
- [x] Commit 1: `6fc3d97` docs update for SHA1 workflow
- [x] Commit 2: `8608d2b` notebook hardening for runtime diagnostics and extraction stability
- [x] Branch pushed: `origin/prototype`

## Execution Checklist (with estimates)

| Status | Task | Estimate | Output |
|---|---|---:|---|
| [ ] | 1. Freeze data source contracts (all SHA1 manifests and expected schemas) | 4h | `data_contracts.json` |
| [ ] | 2. Build full metadata census for all recordings (study set, channels, fs, duration, true units) | 10h | `recording_census.parquet` |
| [ ] | 3. Build full sorting-output census (sorter, return code, timed out, sorting object format) | 8h | `sorting_output_census.parquet` |
| [ ] | 4. Ground-truth audit per recording (extractor load, unit-id dtype, spike train integrity) | 12h | `ground_truth_audit.parquet` |
| [ ] | 5. Recording-type taxonomy (monotrode, tetrode, siprobe, mea, channel bins, drift/no-drift) | 6h | `recording_profiles.parquet` |
| [ ] | 6. Define feature profile matrix by recording type (required vs optional features) | 8h | `feature_profile_matrix.yaml` |
| [ ] | 7. Implement validation-first preflight gate (block bad rows before extraction) | 12h | `preflight_report.json` |
| [ ] | 8. Implement extractor adapters per profile and GT format | 20h | Updated notebook Phase 1 + tests |
| [ ] | 9. Run dry extraction benchmark on 1 sample per profile, then full run | 16h | `extraction_benchmark.md` |
| [ ] | 10. Final reproducibility and leakage audit pass | 8h | `final_validation_report.md` |

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
- Milestone A (Tasks 1-4): 34h, complete full data understanding and GT contract.
- Milestone B (Tasks 5-8): 46h, implement profile-based extraction engine.
- Milestone C (Tasks 9-10): 24h, verify reliability and research validity.
- Total estimated effort: 104h.

## Self-Check Loop (Mandatory)
- Run preflight and fail-fast on bad rows before every extraction run.
- Compare new skip-category histogram against previous run; require net reduction before proceeding.
- For each patch, re-run a deterministic smoke subset and verify:
  - no new error categories introduced,
  - unit-id dtype canonicalization remains integer on both GT and output sortings,
  - reproducibility manifests are updated.
- Only then launch full extraction.

## Immediate Next Step
- Run Tasks 1-3 first in one pass and lock manifests/census before changing extraction logic again.
