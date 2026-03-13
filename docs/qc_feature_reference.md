# QC Feature Reference

This document explains the features produced by the QC extraction pipeline in
[notebooks/Spike_UNIT_Quality.ipynb](/Users/paulruiz/Documents/Predicting_Good_Units/notebooks/Spike_UNIT_Quality.ipynb).

The descriptions below are based on the current notebook logic and the matching
helper code in [scripts/refactor_colab_phase01.py](/Users/paulruiz/Documents/Predicting_Good_Units/scripts/refactor_colab_phase01.py).

## 1. Row-Level Extraction Overview

Each output row corresponds to one tested sorter unit.

At a high level, the notebook does this:

1. Load one recording and one sorter output.
2. Compare the sorter output against ground truth to build supervision targets.
3. Build a mean multichannel waveform template for each unit.
4. Extract custom features from:
   - waveform morphology
   - amplitude distribution over time
   - spike-train timing and ACG
   - within-recording temporal stability
   - recording-level context
   - pairwise template similarity to other units
5. Compute additional SpikeInterface quality metrics when available.
6. Assemble metadata, targets, and features into one row.

## 2. Important Extraction Constants

Current notebook constants:

- `MAX_SPIKES_TEMPLATE = 500`
- `MAX_SPIKES_ACG = 10000`
- `MAX_SPIKES_AMP = 2000`
- `ACG_MAX_LAG_MS = 50`
- `ACG_BIN_SIZE_MS = 2.5`
- `COSINE_THRESHOLD = 0.7`
- `AMPLITUDE_THRESHOLD = 0.25`
- `TEMPLATE_WINDOW_MS = 1.5`
- `AMP_WINDOW_SAMPLES = 15`

These constants matter because several features depend on the template window,
ACG window, subsampling cap, and cosine threshold.

## 3. Metadata Columns

These columns identify where the unit came from. They are useful for grouping,
auditing, and leakage checks, but they are usually **not** model features.

| Column | Meaning |
| --- | --- |
| `study_set` | Top-level dataset family, such as `HYBRID_*`, `PAIRED_*`, or `SYNTH_*`. |
| `study_name` | Specific study name inside the study set. |
| `recording_name` | Recording identifier. |
| `sorter_name` | Name of the spike sorter that produced the tested unit. |
| `unit_id` | Original unit ID from the sorter output. |
| `group_key` | Recording-level grouping key used for split logic. |
| `row_uid` | Unique row identifier: study set + study + recording + sorter + unit. |

## 4. Supervision Targets And Match Columns

These columns are built from the ground-truth comparison object.

### Match identity

| Column | Meaning |
| --- | --- |
| `matched_gt_unit_id` | Best matching GT unit for the tested unit, if any. |
| `hungarian_gt_unit_id` | GT unit assigned by the Hungarian matching logic, if any. |
| `match_status` | Categorical status for the tested unit. |
| `agreement_score` | Agreement score from the comparison object for the best GT match. |

### Count-based target components

| Column | Computation |
| --- | --- |
| `tp` | Number of matched spikes between the tested unit and its best GT unit. |
| `fn` | `num_gt - tp` when a GT match exists, else missing. |
| `fp` | `num_tested - tp` when a GT match exists; if no GT match, all tested spikes count as false positives. |
| `num_gt` | Total spikes in the matched GT unit, if a GT match exists. |
| `num_tested` | Total spikes in the tested sorter unit. |

### Derived targets

| Column | Computation |
| --- | --- |
| `recall` | `tp / num_gt` when `num_gt > 0`. |
| `precision` | `tp / num_tested` when `num_tested > 0`. |
| `fmiss` | `fn / num_gt` when a GT match exists and `num_gt > 0`; otherwise missing. |
| `fpos` | `fp / num_tested` when `num_tested > 0`. If no GT match, this becomes `1.0` for nonempty units. |
| `accuracy` | `tp / (tp + fn + fp)` when the denominator is positive. If no GT match and the unit is nonempty, this becomes `0.0`. |

### Why `fmiss` can be missing

`fmiss` only makes sense when a tested unit is attached to a specific GT unit,
because it needs a valid GT denominator (`num_gt`).

If a tested unit has no matched GT unit:

- `fmiss = NaN`
- `fpos` can still be defined
- `accuracy` can still be defined

### `match_status`

Status is assigned in this priority order:

1. `well_detected`
2. `false_positive`
3. `redundant`
4. `overmerged`
5. `matched_best`
6. `unclassified`

## 5. How Templates And Amplitudes Are Built

### Mean template

For each unit:

1. Get the unit spike train.
2. If the unit has fewer than 5 spikes, skip template extraction.
3. Deterministically subsample up to `MAX_SPIKES_TEMPLATE`.
4. Around each sampled spike, extract a multichannel snippet using a symmetric
   template window of `TEMPLATE_WINDOW_MS`.
5. Average all valid snippets.

The result is a template with shape:

- `(n_channels, n_samples)`

This template is the source for the waveform and cosine features.

### Unit amplitudes

Amplitude features are computed from sampled spikes on the peak channel:

1. Find the template peak channel as the channel with the largest peak-to-peak value.
2. Sample up to `MAX_SPIKES_AMP` spikes.
3. Around each sampled spike, extract a small snippet of width
   `2 * AMP_WINDOW_SAMPLES`.
4. Store the maximum absolute value on the peak channel for that snippet.
5. Store a normalized spike time fraction in `[0, 1]`.

These amplitude samples drive Block B and part of Block D.

## 6. Block A: Waveform Morphology

Block A is built from the **peak-channel waveform** of the mean template and from
channel geometry.

Definitions used internally:

- `ptp_per_ch = np.ptp(mean_template, axis=1)`
- `peak_ch = argmax(ptp_per_ch)`
- `wf = mean_template[peak_ch]`
- `trough_idx = argmin(wf)`
- `peak_idx = argmax(wf)`
- `ptp_uv = ptp_per_ch[peak_ch]`
- `wf_std = std(wf)`
- `ms_per_sample = 1000 / fs`
- `wf_norm = wf / max(abs(wf))`

### Basic waveform shape and scale

| Feature | Computation | Notes |
| --- | --- | --- |
| `snr` | `peak_to_trough_uv / std(wf)` | In the current notebook this is identical to `wf_ptp_ratio`. |
| `peak_to_trough_uv` | Peak-to-peak amplitude on the peak channel. | Scale feature. |
| `half_width_ms` | Number of samples where `wf < trough/2`, multiplied by `ms_per_sample`. | This is a legacy global width, not a strict contiguous half-width around the trough. |
| `repolarization_slope` | `(wf[trough_idx + 1] - wf[trough_idx]) / ms_per_sample` | One-step post-trough slope. |
| `pre_trough_peak_uv` | Largest positive value before the trough, clipped at zero. | Measures pre-trough positive lobe. |
| `post_trough_peak_uv` | Largest positive value after the trough, clipped at zero. | Measures rebound peak. |
| `trough_time_ms` | `trough_idx * ms_per_sample` | Time of the trough relative to the **start** of the template window. |
| `wf_energy` | `sum(wf ** 2)` | Peak-channel waveform energy. |
| `wf_ptp_ratio` | `peak_to_trough_uv / std(wf)` | Exact duplicate of `snr` in the current implementation. |
| `template_norm` | `||wf||_2` | Peak-channel L2 norm. |

### Asymmetry and lobe ratios

| Feature | Computation | Notes |
| --- | --- | --- |
| `wf_asymmetry` | `(post_peak_uv - pre_peak_uv) / (post_peak_uv + pre_peak_uv + 1e-8)` | Bounded in `[-1, 1]`. Positive values mean a stronger post-trough positive lobe. |
| `wf_prepeak_ratio` | `pre_peak_uv / abs(trough)` | Pre-trough positive lobe relative to trough magnitude. |
| `wf_rebound_ratio` | `post_peak_uv / abs(trough)` | Post-trough rebound relative to trough magnitude. |

### Timing features

| Feature | Computation | Notes |
| --- | --- | --- |
| `wf_peak_trough_ms` | `abs(peak_idx - trough_idx) * ms_per_sample` | Uses the global waveform maximum, which may be before or after the trough. |
| `wf_prepeak_to_trough_ms` | `(trough_idx - pre_peak_idx) * ms_per_sample` | Missing if no positive pre-trough peak exists. |
| `wf_trough_to_peak_ms` | `(post_peak_idx - trough_idx) * ms_per_sample` | Missing if no positive post-trough peak exists. |
| `wf_zero_cross_pre_ms` | Time from the trough back to the last nonnegative sample before the trough. | Missing if no nonnegative sample exists before the trough. |
| `wf_zero_cross_post_ms` | Time from the trough forward to the first nonnegative sample after the trough. | Missing if no nonnegative sample exists after the trough. |

### Width, area, and shape complexity

| Feature | Computation | Notes |
| --- | --- | --- |
| `wf_trough_width_25_ms` | Contiguous width around the trough where `wf <= 0.25 * trough_value`. | Since trough is negative, this tracks deep negative width. |
| `wf_trough_width_50_ms` | Contiguous width around the trough where `wf <= 0.50 * trough_value`. | |
| `wf_trough_width_75_ms` | Contiguous width around the trough where `wf <= 0.75 * trough_value`. | |
| `wf_neg_area_uv_ms` | `sum(max(-wf, 0)) * ms_per_sample` | Area of the negative lobe. |
| `wf_pos_area_uv_ms` | `sum(max(wf, 0)) * ms_per_sample` | Area of the positive lobe(s). |
| `wf_pos_neg_area_ratio` | `wf_pos_area_uv_ms / (wf_neg_area_uv_ms + 1e-8)` | Positive-to-negative area ratio. |
| `wf_line_length_norm` | `sum(abs(diff(wf_norm)))` | Total normalized line length. |
| `wf_curvature_norm` | `sum(abs(diff(wf_norm, n=2)))` | Total normalized second-difference magnitude. |
| `wf_trough_sharpness` | `abs(wf_norm[i-1] - 2*wf_norm[i] + wf_norm[i+1])` at the trough | A local discrete curvature measure. |
| `wf_pre_trough_slope` | `(wf[trough_idx] - wf[trough_idx - 1]) / ms_per_sample` | One-step slope into the trough. |
| `wf_post_peak_slope` | `(wf[post_peak_idx + 1] - wf[post_peak_idx]) / ms_per_sample` | One-step slope after the rebound peak. |

### Spatial footprint features

Active channels are defined by:

- `ptp_per_ch > AMPLITUDE_THRESHOLD * max(ptp_per_ch)`

With the current notebook, `AMPLITUDE_THRESHOLD = 0.25`.

Only the second channel-location coordinate is used as depth.

| Feature | Computation | Notes |
| --- | --- | --- |
| `n_active_channels` | Number of channels passing the active-channel threshold. | Spatial footprint size. |
| `spread_um` | `max(depths) - min(depths)` across active channels. | Unweighted depth spread. |
| `center_of_mass_um` | Weighted average depth using channel PTP as weights. | Spatial center of mass. |
| `spread_weighted_um` | Weighted standard deviation of depth around the weighted center of mass. | Weighted spatial spread. |
| `peak_channel_depth_um` | Depth coordinate of the peak channel. | Depends on channel geometry. |
| `n_channels` | Number of channels in the template. | Usually probe channel count. |

### Waveform-bin features

`wf_bin_00` through `wf_bin_29` are computed the same way:

1. Take the peak-channel waveform.
2. Normalize it by `max(abs(wf))`.
3. Express time relative to the trough.
4. Interpolate the normalized waveform onto a fixed grid of 30 equally spaced bins.

These bins are intended to capture **waveform shape**, not raw amplitude.

## 7. Block B: Amplitude Statistics

Block B summarizes the sampled single-spike amplitudes over time.

Definitions used internally:

- `amps`: sampled peak-channel amplitudes
- `t_vec`: normalized spike-time fractions in `[0, 1]`

If fewer than 5 amplitudes are available, the whole block becomes missing.

### Distribution features

| Feature | Computation |
| --- | --- |
| `amp_mean` | Mean of sampled amplitudes. |
| `amp_std` | Standard deviation of sampled amplitudes. |
| `amp_cv` | `amp_std / abs(amp_mean)` |
| `amp_p5` | 5th percentile of amplitudes. |
| `amp_p50` | Median amplitude. |
| `amp_p95` | 95th percentile of amplitudes. |
| `amp_iqr` | `p75 - p25` |
| `amp_skew` | `scipy.stats.skew(amps)` |
| `amp_kurtosis` | `scipy.stats.kurtosis(amps)` |
| `amp_bimodality` | Pearson-style bimodality coefficient: `(skew^2 + 1) / adjusted_kurtosis_term` |
| `amp_outlier_pct` | Fraction of amplitudes more than `3 * std` away from the mean. |

### Global temporal trend features

| Feature | Computation |
| --- | --- |
| `amp_drift_slope` | Linear regression slope of amplitude vs normalized time. |
| `amp_drift_r2` | `r^2` from the same linear regression. |
| `amp_trend_spearman` | Spearman correlation between amplitude and normalized time. |

### Early / middle / late features

Time is split into thirds:

- early: `t <= 1/3`
- middle: `1/3 < t <= 2/3`
- late: `t > 2/3`

| Feature | Computation |
| --- | --- |
| `amp_early_p50` | Median amplitude in the early third. |
| `amp_mid_p50` | Median amplitude in the middle third. |
| `amp_late_p50` | Median amplitude in the late third. |
| `amp_early_late_ratio` | `amp_late_p50 / amp_early_p50` |

### Half-wise and decile-wise stability

| Feature | Computation |
| --- | --- |
| `amp_first_half_slope` | Linear regression slope of amplitude vs time for `t <= 0.5`. |
| `amp_second_half_slope` | Linear regression slope of amplitude vs time for `t > 0.5`. |
| `amp_bin_cv_mean` | Mean coefficient of variation across 10 time bins. |
| `amp_bin_cv_std` | Standard deviation of the CV across the same 10 bins. |
| `amp_bin_spread_ratio` | `(max(bin_median) - min(bin_median)) / abs(global_mean)` across the 10 bins. |

## 8. Block C: Spike Timing And ACG

Block C is built from the spike train itself.

If a unit has fewer than 5 spikes, the custom Block C extraction returns no values
and the row keeps the default missing values.

Definitions used internally:

- `st`: sorted spike train in samples
- `isis`: inter-spike intervals in milliseconds
- `fr = len(st) / duration_sec`

### Scalar timing features

| Feature | Computation |
| --- | --- |
| `firing_rate_hz` | `n_spikes / duration_sec` |
| `isi_mean_ms` | Mean ISI in milliseconds. |
| `isi_std_ms` | Standard deviation of ISIs in milliseconds. |
| `isi_cv` | `isi_std_ms / isi_mean_ms` |
| `isi_skew` | `scipy.stats.skew(isis)` when enough ISIs exist. |
| `presence_ratio` | Fraction of 1-second windows containing at least one spike. |
| `isi_violation_rate` | Observed refractory violations divided by expected violations under the notebook formula. |
| `burst_index` | Fraction of ISIs below 10 ms. |
| `spike_count` | Number of spikes in the unit spike train. |

### Refractory violation formula

The notebook uses:

- refractory period: `rp_ms = 2.0`
- observed violations: `n_viol = count(isi < 2 ms)`
- expected violations:
  - `n_exp = 2 * rp_sec * n_spikes^2 / (2 * duration_sec)`
- final rate:
  - `isi_violation_rate = n_viol / (n_exp + 1e-8)`

### ACG bins

The notebook computes a normalized autocorrelogram with:

- max lag: `50 ms`
- bin size: `2.5 ms`
- zero-lag self-pairs removed
- spike subsampling capped at `MAX_SPIKES_ACG`

This produces 40 bins:

- `acg_00` to `acg_39`

Each `acg_xx` value is the normalized count in one lag bin across the
symmetric ACG window.

## 9. Block D: Temporal Stability

Block D compares the early and late parts of the recording.

If a unit has fewer than 5 spikes, the whole block becomes missing.

The recording is split into thirds:

- early: first third
- middle: middle third
- late: last third

### Features

| Feature | Computation | Notes |
| --- | --- | --- |
| `d_rate_change_ratio` | `late_rate / early_rate` | Ratio of firing rate in the last third vs first third. |
| `d_amp_change_ratio` | `median(late_amplitude) / median(early_amplitude)` | Uses the sampled amplitudes if they exist. |
| `d_waveform_corr_early_late` | Pearson correlation between flattened early and late templates. | Measures waveform stability over time. |
| `d_peak_channel_switch` | `1.0` if the early and late peak channels differ, else `0.0` | Detects a spatial shift of the dominant channel. |

## 10. Block E: Recording Context

Block E is computed once per recording and broadcast to all rows in that recording.

Noise is estimated from the first up to 10 seconds of data using:

- `median(abs(trace)) / 0.6745` per channel

### Features

| Feature | Computation |
| --- | --- |
| `rec_noise_mean_uv` | Mean noise estimate across channels. |
| `rec_noise_std_uv` | Standard deviation of the channelwise noise estimates. |
| `rec_duration_sec` | Recording duration in seconds. |
| `rec_n_channels` | Number of recording channels. |
| `rec_n_units` | Number of tested units for that recording/sorter output. |
| `rec_sampling_rate` | Recording sampling frequency. |

## 11. Block F: Pairwise Relational Features

Block F compares each unit template to other unit templates from the same
recording and sorter output.

If fewer than two templates are available, default values are used.

### Two cosine definitions

#### Full-template cosine

Before cosine similarity:

1. Normalize each full template by its own maximum absolute amplitude.
2. Flatten across channels and time.
3. Compute cosine similarity.

This is intended to measure **shape similarity**, not raw amplitude similarity.

#### Peak-waveform cosine

Before cosine similarity:

1. Take the peak-channel waveform from each template.
2. Normalize each waveform by its own maximum absolute amplitude.
3. Flatten and compute cosine similarity.

This is a narrower single-channel shape similarity measure.

### Features

| Feature | Computation |
| --- | --- |
| `max_cosine` | Maximum full-template cosine to any other unit. |
| `mean_cosine_top3` | Mean of the 3 highest full-template cosine values. |
| `mean_cosine_top5` | Mean of the 5 highest full-template cosine values. |
| `median_cosine` | Median full-template cosine across all neighbors. |
| `std_cosine` | Standard deviation of full-template cosine across all neighbors. |
| `cosine_gap_top2` | Difference between the top-1 and top-2 full-template cosine values. |
| `n_confusable` | Number of neighboring units with full-template cosine above `COSINE_THRESHOLD` (`0.7`). |
| `isolation_score` | `1 - max_cosine` |
| `max_waveform_cosine` | Maximum peak-waveform cosine to any other unit. |
| `mean_waveform_cosine_top3` | Mean of the top 3 peak-waveform cosine values. |
| `median_waveform_cosine` | Median peak-waveform cosine across all neighbors. |
| `std_waveform_cosine` | Standard deviation of peak-waveform cosine across all neighbors. |
| `n_waveform_confusable` | Number of neighboring units with peak-waveform cosine above `0.7`. |
| `min_neighbor_dist_um` | Minimum Euclidean distance between the unit peak channel and any neighbor peak channel. |
| `mean_neighbor_dist_um` | Mean Euclidean distance between the unit peak channel and all neighbor peak channels. |

## 12. SpikeInterface Metrics (`si_*`)

The notebook also computes a set of SpikeInterface metrics when the required
extensions and analyzer steps are available.

Important implementation detail:

- these features are requested by the notebook
- the exact internal formulas come from the installed SpikeInterface version
- if a metric cannot be computed, it remains missing

### Analyzer extensions requested before quality metrics

The notebook attempts to compute:

- `random_spikes`
- `waveforms`
- `templates`
- `noise_levels`
- `spike_amplitudes`
- `spike_locations` with `method="center_of_mass"`

### Direct SpikeInterface metrics

| Feature | Notebook call |
| --- | --- |
| `si_num_spikes` | `sqm.compute_num_spikes(...)` |
| `si_firing_rate_hz` | `sqm.compute_firing_rates(...)` |
| `si_presence_ratio` | `sqm.compute_presence_ratios(..., bin_duration_s=60.0, mean_fr_ratio_thresh=0.0)` |
| `si_isi_violations_ratio` | `sqm.compute_isi_violations(..., isi_threshold_ms=2.0, min_isi_ms=0.0)` |
| `si_isi_violations_count` | Same call as above, count output field. |

### Extended SpikeInterface quality metrics

These are requested through `sqm.compute_quality_metrics(...)` when available.

| Feature | Source quality-metric column |
| --- | --- |
| `si_snr` | `snr` |
| `si_firing_range` | `firing_range` |
| `si_amplitude_median` | `amplitude_median` |
| `si_amplitude_cutoff` | `amplitude_cutoff` |
| `si_amplitude_cv_median` | `amplitude_cv_median` |
| `si_amplitude_cv_range` | `amplitude_cv_range` |
| `si_sd_ratio` | `sd_ratio` |
| `si_drift_ptp` | `drift_ptp` |
| `si_drift_std` | `drift_std` |
| `si_drift_mad` | `drift_mad` |

## 13. Common Redundancies And Caveats

The current notebook intentionally keeps a broad feature set, so some columns are
redundant or strongly overlapping.

### Exact or near-exact duplicates

- `snr` and `wf_ptp_ratio` are the same formula in the current code.
- `isolation_score = 1 - max_cosine`

### Strongly related amplitude features

- `peak_to_trough_uv`
- `wf_energy`
- `template_norm`

These are not identical, but they all measure waveform size and often move together.

### Important interpretation caveats

- `trough_time_ms` depends on template alignment and template window placement.
- `half_width_ms` is a legacy count below half trough amplitude, not a strict local half-width.
- `wf_bin_00..29` are normalized-shape features, not raw amplitude features.
- `fmiss` is missing for unmatched units by design.
- Some SpikeInterface metrics may be missing if the installed environment does not
  support all extensions or quality metrics.

## 14. Practical Split Between Feature Types

A useful practical view is:

- **Targets**: `accuracy`, `fpos`, `fmiss`
- **IDs / metadata**: `study_set`, `study_name`, `recording_name`, `sorter_name`, `unit_id`, `row_uid`, `group_key`
- **Custom unit features**: Blocks A, B, C, D, F
- **Recording context**: Block E
- **Library quality metrics**: `si_*`

For conservative modeling, the safest default is usually:

- keep the custom unit features
- keep selected `si_*` features
- decide deliberately whether to keep `rec_*` context
- exclude dataset names, dataset type, and sorter identity as model inputs
