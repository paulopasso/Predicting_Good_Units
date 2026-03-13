# SpikeForest and SpikeInterface Notes

This file records the concrete schema and comparison semantics the notebook relies on.

## SpikeForest Payload Shapes

Current package-style SpikeForest records are split into two payload types.

Recordings payload:

- top-level list key: `recordings`
- fields used by the notebook:
  - `studySetName`
  - `studyName`
  - `name`
  - `sampleRateHz`
  - `numChannels`
  - `durationSec`
  - `numTrueUnits`
  - `recordingObject`
  - `sortingTrueObject`

Sorting outputs payload:

- top-level list key: `sortingOutputs`
- fields used by the notebook:
  - `studyName`
  - `recordingName`
  - `sorterName`
  - `cpuTimeSec`
  - `returnCode`
  - `timedOut`
  - `sortingObject`

Important distinction:

- some older SpikeForest dev scripts use flatter records with `recordingUri`, `sortingTrueUri`, or `firings`
- those are not the primary schema used by the notebook

## Janelia Source Pair

The current notebook enumerates `HYBRID_JANELIA` from:

- recordings: `sha1://43298d72b2d0860ae45fc9b0864137a976cb76e8?hybrid-janelia-spikeforest-recordings.json`
- sorting outputs: `sha1://9259d3ec1d981560e35c2ca41e59c39f2af3d37e?label=spikeforest-sorting-outputs.json`

The selected-manifest rows show the expected Janelia sortings payload shape:

```json
{
  "study_set": "HYBRID_JANELIA",
  "study_name": "hybrid_static_tetrode",
  "recording_name": "rec_4c_600s_11",
  "sorter_name": "IronClust",
  "sorting_object": {
    "firings": "sha1://.../firings.mda",
    "samplerate": 30000
  }
}
```

This means the manifest join itself is structurally correct for Janelia. The comparison bug happens later.

## SpikeInterface Comparison Semantics

The critical distinction in SpikeInterface is:

- `cmp.get_performance(method="by_unit")` is GT-indexed
- `cmp.count_score` is GT-indexed
- `cmp.best_match_21` maps tested units to GT units
- `cmp.hungarian_match_21` maps tested units to GT units under Hungarian matching
- `cmp.match_event_count` is indexed as `[gt_unit_id, tested_unit_id]`
- `cmp.event_counts1` counts GT spikes
- `cmp.event_counts2` counts tested spikes

The old bug came from treating the GT-indexed performance table as if it were indexed by tested unit IDs.

## Correct Target Construction

For each tested unit `u2`:

1. find `u1 = cmp.best_match_21[u2]`
2. if `u1` exists:
   - `tp = cmp.match_event_count.at[u1, u2]`
   - `num_gt = cmp.event_counts1.at[u1]`
   - `num_tested = cmp.event_counts2.at[u2]`
   - `fn = num_gt - tp`
   - `fp = num_tested - tp`
   - `recall = tp / num_gt`
   - `precision = tp / num_tested`
   - `fmiss = fn / num_gt`
   - `fpos = fp / num_tested`
   - `accuracy = tp / (tp + fn + fp)`
3. if `u1` does not exist:
   - `tp = 0`
   - `fp = num_tested`
   - `accuracy = 0.0`
   - `fpos = 1.0`
   - `fmiss = NaN`

This is the logic implemented in the notebook now.

## Match Status Fields

The notebook also stores:

- `matched_gt_unit_id`
- `hungarian_gt_unit_id`
- `match_status`
- `agreement_score`
- `tp`, `fn`, `fp`
- `num_gt`, `num_tested`
- `recall`, `precision`

`match_status` is derived from tested-unit classification helpers:

- `get_well_detected_units()`
- `get_false_positive_units()`
- `get_redundant_units()`
- `get_overmerged_units()`

with a fallback of `matched_best` when a best GT match exists but no stronger class applies.

## Why the Previous Dataset Was Degenerate

The previous Phase 1 run wrote rows with:

- `fmiss = 1.0`
- `fpos = 1.0`
- `accuracy = 0.0`

for every saved unit.

That happened because:

1. many comparisons failed with dtype-mismatch errors
2. even when a comparison ran, the GT-indexed performance table was being joined against tested unit IDs
3. the self-check was not strict enough to reject constant/degenerate targets early

## New Audit Artifacts

The notebook now writes:

- `artifacts/audits/spikeforest_schema_audit.json`
- `artifacts/audits/comparison_debug_summary.json`
- `artifacts/audits/comparison_debug/*.json`
- `artifacts/audits/comparison_debug/*.csv`

These are meant to answer three questions directly:

1. what did the upstream SpikeForest record look like
2. what did SpikeInterface compute internally
3. what target row was finally attached to each tested unit
