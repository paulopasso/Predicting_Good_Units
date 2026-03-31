# SpikeForest Audit Summary

## Requested Paired Studies
- requested_paired_studies: `PAIRED_BOYDEN, PAIRED_CRCNS_HC1, PAIRED_ENGLISH, PAIRED_KAMPFF, PAIRED_MEA64C_YGER`
- present_requested_studies: `5`
- requested_studies_missing: `0`

## Recording Coverage
- paired_recordings_in_parquet: `31`
- paired_recordings_with_no_matched_units: `2`
- recordings_with_missing_common_sorters: `16`

## Missing Item Summary
- total_missing_items: `24`

## Notes
- `missing_sorter_output_common` is a heuristic based on study-level sorter coverage inside the current parquet.
- Missing common sorters are likely the most actionable recovery candidates without re-querying SpikeForest manifests.
