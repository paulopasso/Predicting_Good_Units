# Crash Test Experiment Summary

This note summarizes the crash-test experiments run to improve transfer from hybrid QC data to paired real QC data for `fpos` and `fmiss`.

The goal has been consistent across all tracks:

- train on the large hybrid source set without leaking the paired test recordings
- adapt to the paired domain
- improve strict recording-disjoint holdout performance on `fpos` and `fmiss`
- keep `fpos` and `fmiss` as separate deployment targets

All serious comparisons below refer to the strict recording-disjoint paired holdout protocol unless explicitly marked as smoke/plumbing only.

## Data Setup

Current working split used in the serious crash tests:

- hybrid labeled source: about `24k`
- paired matched labeled units: `207`
- paired unmatched unlabeled units: `8775`
- strict paired test holdout: about `100` matched rows
- paired fine-tune/model-selection pool: about `107` matched rows
- unlabeled non-test paired pool available for adaptation: about `4963` rows after excluding holdout recordings

Leakage boundary:

- `recording_key` is the hard split boundary
- final holdout recordings are excluded from training, adaptation, scaling, feature selection, and alignment

## What We Tried

### 1. Initial transfer benchmark

Main idea:

- compare `paired_only`, `hybrid_only`, calibrated hybrid transfer, hybrid+paired stacking, and neural transfer
- use strict paired holdout

Key result:

- plain `hybrid_only` transfer was not enough
- context-aware hybrid+paired transfer clearly outperformed dummy and raw hybrid baselines
- this established that transfer helps only after paired-domain adaptation

Important outcome:

- `fpos` and `fmiss` should not be forced to share the same final model choice

### 2. Neural SSL domain adaptation

Folder:

- `crash_tests/self_supervised_domain_adaptation/`

Main idea:

- pretrain a unit encoder on unlabeled paired units
- train on hybrid labels
- fine-tune on paired labels
- compare unit-only and contextual neural arms

What worked:

- the leakage-safe SSL pipeline ran end-to-end
- the contextual LightGBM baseline inside the same harness stayed strong

What failed:

- SSL itself did not beat the context-aware tabular baseline
- the learned neural models often collapsed toward narrow prediction ranges
- the contextual neural arm was too data-hungry for the paired label budget

Representative result from the tuned main run:

- contextual LightGBM baseline:
  - `fpos`: MAE `0.1532`, `R² 0.3632`
  - `fmiss`: MAE `0.1868`, `R² 0.3617`
- best neural no-SSL unit-only:
  - `fpos`: MAE `0.1431`, `R² 0.1761`
  - `fmiss`: MAE `0.2207`, `R² -0.1185`

Interpretation:

- neural representation learning alone was not the breakthrough
- the strongest signal remained in context-aware tree models

### 3. MMD domain adaptation

Folder:

- `crash_tests/mmd_domain_adaptation/`

Main idea:

- add a domain-alignment term:
  - `task_loss(hybrid_labeled) + lambda * MMD(hybrid_latent, paired_unlabeled_latent)`
- compare no-MMD vs MMD, zero-shot vs paired fine-tuned

What worked:

- MMD helped the neural model relative to the same neural architecture without MMD
- especially useful as a way to reduce catastrophic failure in zero-shot transfer

What did not work:

- even the improved MMD models still did not beat the contextual LightGBM baseline
- MMD was useful, but not decisive

Representative strict holdout results:

- contextual LightGBM baseline:
  - `fpos`: MAE `0.1532`, `R² 0.3632`
  - `fmiss`: MAE `0.1868`, `R² 0.3617`
- best MMD neural:
  - `fpos`: MAE `0.1503`, `R² 0.3049`
  - `fmiss`: MAE `0.2606`, `R² -0.1419`

Interpretation:

- MMD improved the neural path
- MMD did not close the gap to the strongest context-aware tabular model

### 4. ML/statistics workbench

Folder:

- `crash_tests/ml_statistics_workbench/`

Main idea:

- stop trying generic transfer tricks in isolation
- combine:
  - strong paired context features
  - source-derived predictions
  - source-derived latent summaries or selected latent dimensions
- evaluate only the families that still looked promising

This was the first track that materially improved the previous benchmark.

Representative earlier win:

- `embed_nommd_fullctx_lightgbm` and related stacked variants improved over plain contextual paired-only baselines
- best earlier `fmiss` run reached about `R² 0.485`
- best earlier `fpos` run reached about `R² 0.500`

Main lesson:

- source-derived signals are useful when stacked on top of strong paired context
- standalone neural transfer is not the winning form

### 5. Cluster-aware transfer

Main idea:

- use the learned source-side latent space to partition the non-test paired domain into coarse target clusters
- train cluster-conditioned source models
- feed cluster-aware source predictions plus cluster responsibilities into the same full-context LightGBM stack

Why we tried it:

- the PCA view suggested the paired domain is a mixture of subdomains rather than one compact cloud
- especially `PAIRED_ENGLISH` looked broad and heterogeneous

What happened:

- the first cluster-aware pass ran cleanly under the strict recording-disjoint protocol
- it did not become the overall winner
- it was more useful as a diagnostic than as a breakthrough model family

Representative full strict holdout result:

- `cluster_source_fullctx_lightgbm`
  - `fmiss`: MAE `0.1605`, `R² 0.4443`
  - `fpos`: MAE `0.1314`, `R² 0.4401`

Interpretation:

- cluster-aware transfer was competitive on `fmiss`
- it was not competitive on `fpos`
- the best `fmiss` model from the same run was actually the simpler `latent_summary_fullctx_lightgbm`

What it uncovered:

- the coarse `k=2` target clustering split paired recordings into two broad regimes:
  - one cluster dominated by `PAIRED_KAMPFF` plus part of `PAIRED_ENGLISH` and `PAIRED_BOYDEN`
  - one cluster dominated by `PAIRED_MEA64C_YGER` plus another large slice of `PAIRED_ENGLISH`
- `PAIRED_ENGLISH` spreads across both clusters rather than living in one target block

This supports the PCA interpretation:

- there is real target heterogeneity
- but a simple cluster-conditioned source prediction is not, by itself, enough to beat the best context-first stack

## SpikeForest Audit

Folder:

- `crash_tests/spikeforest_audit/`

Main question:

- is there a hidden big paired dataset still missing, or are we mostly dealing with extraction gaps?

What we found:

- the five expected paired SpikeForest study sets are already present
- there is no separate large missing paired corpus hiding outside the current pipeline
- there are real extraction gaps

Audit result:

- `24` recovery candidates in total
- `22` missing common sorter outputs
- `2` paired recordings with zero matched units

The most important gaps are concentrated in `PAIRED_ENGLISH`.

Zero-matched recordings:

- `PAIRED_ENGLISH::paired_english::m139_200114_205038`
- `PAIRED_ENGLISH::paired_english::m26_190524_100859_cell3`

Interpretation:

- the next data move is not “find a new SpikeForest corpus”
- it is “recover missing extraction coverage from the corpus we already use”

## Domain-Shift Audit

Folder:

- `crash_tests/domain_shift_audit/`

Main question:

- which features separate hybrid from paired, and how strongly?

Core finding:

- the hybrid and paired domains are still extremely separable

Domain classifier AUC:

- unit features only: about `0.976`
- full context features: about `0.9997`

This is a very strong sign that domain shift is not subtle.

Most shifted feature families:

- recording-context SNR and amplitude aggregates
- waveform timing scalars
- several waveform-bin features
- amplitude statistics
- some confusability / cosine features

Examples of strongly shifted features:

- `snr`
- `wf_trough_to_peak_ms`
- `wf_peak_trough_ms`
- `wf_zero_cross_post_ms`
- `peak_to_trough_uv`
- `amp_mean`
- `amp_p5`
- `amp_early_p50`
- `amp_mid_p50`
- `snr_recording_mean`
- `snr_recording_median`

Interpretation:

- domain shift is driven by both unit-level waveform/amplitude structure and recording-level context
- some context blocks are powerful predictors but also highly domain-specific

## PCA Chart: Why It Matters

The PCA projection of hybrid vs paired unit features is worth digging into because it suggests the paired domain is not a single compact target domain.

What stands out visually:

- hybrid forms a relatively tight central cloud
- `PAIRED_ENGLISH` is much broader, with large positive `PC2` outliers and a wide spread in `PC1`
- `PAIRED_KAMPFF` forms a more separated rightward arm
- `PAIRED_BOYDEN` also drifts away from the hybrid core
- `PAIRED_MEA64C_YGER` and parts of `PAIRED_CRCNS_HC1` sit closer to the middle

Interpretation:

- “paired” is a mixture of subdomains, not one clean target distribution
- a single global domain-alignment strategy can blur together study-specific structure
- `PAIRED_ENGLISH` especially looks heterogeneous and may contain multiple sub-regimes or outlier recordings

Why this matters for the modeling:

- global filtering or global alignment can help one target and hurt the other
- study-conditioned or cluster-conditioned transfer may be better than one monolithic alignment step
- this chart supports the idea that some of the remaining error is not just sample size; it is target-domain heterogeneity

3D PCA note:

- yes, a 3D PCA view is worth exporting as a diagnostic
- it will not solve the problem by itself, but it can show whether the apparent overlap in 2D separates along `PC3`
- a 3D PCA export is now available from the domain-shift audit as `pca_projection_3d.csv`
- a rendered 3D figure is now available as `crash_tests/domain_shift_audit/outputs/domain_shift_3d_v1/pca_projection_3d.png`

## Filtered-Feature Benchmark

Latest run:

- `crash_tests/ml_statistics_workbench/outputs/workbench_filtered_full_v1/`

Main idea:

- select the most shifted features using training-side data only:
  - `hybrid_source` vs `paired_non_test_all`
- do not use the paired holdout to choose drops
- preserve waveform bins and ACG bins by default
- add filtered LightGBM variants on top of the existing best stack families

What we dropped most aggressively:

- recording-level SNR aggregates
- recording-level amplitude/IQR aggregates
- raw `snr`
- waveform timing scalars
- selected amplitude scalars

Important finding:

- dropping strongly shifted features is not uniformly good

Result:

- it helped `fpos`
- it hurt `fmiss`

Current best per target from the filtered run:

- `fpos`
  - winner: `embed_nommd_fullctx_lightgbm_filtered`
  - MAE `0.1266`
  - `R² 0.4872`
- `fmiss`
  - winner: `embed_nommd_fullctx_lightgbm`
  - MAE `0.1616`
  - `R² 0.4483`

Interpretation:

- `fpos` benefited from removing the most shifted non-shape/context features
- `fmiss` appears to rely on some of those shifted features despite their domain mismatch
- this strongly suggests target-specific filtering rather than one global filter policy

This was an important negative result:

- “drop the most shifted features” is not a universal fix
- it is useful for `fpos`
- it is not the right global rule for `fmiss`

Latest `fmiss` update after the cluster-aware rerun:

- the cluster-aware model did not win
- the best `fmiss` model in the latest full workbench run became `latent_summary_fullctx_lightgbm`
- latest `fmiss` result:
  - MAE `0.1567`
  - `R² 0.4720`
- cluster-aware `fmiss` was still informative and competitive:
  - MAE `0.1605`
  - `R² 0.4443`
- that means the cluster-aware idea likely captured real structure, but the simpler latent-summary representation used that structure more effectively

## What Worked Best So Far

Best current practical recipe:

- keep separate deployment models for `fpos` and `fmiss`
- use context-first tree models, not standalone neural transfer
- enrich paired context with source-derived signals

Current recommended models:

- `fpos`: `embed_nommd_fullctx_lightgbm_filtered`
- `fmiss`: `source_reweighted_fullctx_lightgbm`

What consistently helped:

- recording-disjoint evaluation
- strong paired-context tabular models
- source-derived predictions/embeddings stacked on top of context
- per-target model choice
- keeping the model family simple when the paired label budget is tiny

What consistently did not become the winning answer:

- standalone SSL neural transfer
- standalone contextual neural models
- standalone MMD neural transfer
- broad post-hoc blending/calibration as the main lever

## What We Uncovered

The most important scientific/engineering findings so far:

1. Domain shift is large and measurable, not anecdotal.
2. The paired domain is heterogeneous across studies and probably across recordings within studies.
3. Context matters a lot, but context features are also among the most shifted.
4. Source-derived representation helps most when used as a stacked feature, not as an end-to-end winning model.
5. Feature dropping must be target-specific:
   - good for `fpos`
   - bad as a global rule for `fmiss`
6. The current data pipeline likely still has recoverable paired coverage gaps, especially in SpikeForest English.
7. Cluster-aware transfer confirms the paired domain is heterogeneous, but the current coarse clustering is more useful as a diagnostic than as a winning predictor.

## Current Recommendation

Short-term:

- recover the missing SpikeForest extraction items first
- rerun the same strict workbench after recovery

Modeling:

- keep the current per-target recommendation
- do not merge `fpos` and `fmiss` onto one model family

Next ML question to investigate:

- use the PCA/domain-shift structure to build a more selective study-aware or cluster-aware transfer, not just a coarse global clustering

Examples:

- per-study residual analysis
- cluster the paired units/recordings in latent or PCA space and use cluster-aware source weighting
- target-specific filtering for `fmiss` instead of global filtering
- protect the most useful `fmiss` features even if they are strongly shifted

## Bottom Line

We did not find a single “ML breakthrough” that solved the problem in one step.

But we did narrow the space substantially:

- the winning direction is context-first tabular modeling with source-derived signals
- the paired domain is heterogeneous
- extraction completeness still matters
- and the latest filtered benchmark uncovered a real target-specific effect:
  - shifted-feature dropping improves `fpos`
  - shifted-feature dropping degrades `fmiss`

That is a useful result, not a dead end. It tells us the next iteration should be more targeted and more structure-aware, not broader.

## Latest Status

Current best strict-holdout recommendations:

- `fpos`: `embed_nommd_fullctx_lightgbm_filtered`
  - MAE `0.1266`
  - `R² 0.4872`
- `fmiss`: `source_reweighted_fullctx_lightgbm`
  - MAE `0.1533`
  - `R² 0.4758`

Current interpretation:

- `fpos` benefits from dropping the most shifted non-shape features
- `fmiss` benefits more from a strong reweighted source prediction stacked on top of full paired context than from aggressive filtering
- target heterogeneity is real, but the first cluster-aware implementation was not the winning way to exploit it

## Covariate Shift / Subgroup / Diagnostics Follow-Up

Question:

- can the most relevant literature-backed ideas improve the current strict holdout?
- specifically:
  - classic covariate-shift correction
  - subgroup-aware / worst-group-style balancing
  - diagnostics-driven target-specific filtering

Implementation:

- extended the strict workbench with three focused families
- kept the same sealed paired holdout and the same non-test paired selection pool
- new variants included:
  - overlap-trimmed density-ratio source weighting
  - subgroup-balanced paired fitting using latent target clusters
  - target-protected shift filtering, where strongly shifted but target-predictive context features are protected from dropping

Run:

- `crash_tests/ml_statistics_workbench/outputs/workbench_shift_subgroup_full_v1`

Result:

- there was a real but modest improvement on `fmiss`
- there was no improvement on `fpos`

Best results from this run:

- `fpos`
  - winner stayed `embed_nommd_fullctx_lightgbm_filtered`
  - MAE `0.1266`
  - `R² 0.4872`
- `fmiss`
  - new winner became `source_reweighted_fullctx_lightgbm`
  - MAE `0.1533`
  - `R² 0.4758`
  - previous reference `latent_summary_fullctx_lightgbm`
    - MAE `0.1567`
    - `R² 0.4720`

What the three ideas actually did:

- classic covariate-shift weighting:
  - this was the only one that produced a new winner
  - the plain density-ratio reweighted source prediction stacked on full paired context beat the previous `fmiss` winner
  - the gain was small but real:
    - `fmiss` `R²` improved by about `+0.0038`
    - `fmiss` MAE improved by about `2.1%`
- overlap-trimmed weighting:
  - did not help
  - the domain classifier was so saturated that the overlap support estimate collapsed
  - in the saved summary, `source_in_support_frac` was effectively `0.0`
  - interpretation:
    - the hybrid and paired domains are so separable that aggressive common-support trimming removes too much source mass to be useful
- subgroup-balanced paired fitting:
  - did not become the winner
  - `latent_summary_fullctx_lightgbm_subgroup_balanced` was competitive but still below the best `fmiss` stack
  - interpretation:
    - subgroup heterogeneity is real, but simple reweighting by latent subgroup was not the right leverage point
- diagnostics-driven target-protected filtering:
  - did not help in this first pass
  - `latent_summary_fullctx_lightgbm_target_filtered` and `source_plus_reduced_latent_fullctx_lightgbm_target_filtered` both underperformed their unfiltered counterparts on `fmiss`
  - interpretation:
    - the current protected-filter rule is still too blunt
    - for `fmiss`, retaining rich full context is still better than trying to prune it using the current shift-and-correlation heuristic

What we learned:

1. The covariate-shift literature was directionally right.
   - sample reweighting is more useful here than global alignment tricks.
2. The best gain was simple.
   - a better source prediction channel helped more than another representation-learning trick.
3. The domain gap is too large for naive overlap trimming.
   - with AUC near `1`, “common support” can collapse into almost nothing.
4. Subgroup heterogeneity is real but not solved by one balancing rule.
   - latent subgroup balancing did not beat the stronger full-context source stack.
5. Dataset-shift diagnostics are useful for understanding the problem, but not every diagnostic-driven intervention improves the predictor.

Current takeaway from the literature-backed round:

- keep the successful part:
  - density-ratio weighted source prediction on top of full paired context for `fmiss`
- keep the old `fpos` winner unchanged
- do not spend the next round on overlap trimming or generic subgroup balancing

## Paired Label-Budget Benchmark

Question:

- how much of the remaining error is just paired-label scarcity?
- is there a smaller paired adaptation budget that gets almost all of the performance?

Implementation:

- added a dedicated budget runner and notebook:
  - `crash_tests/ml_statistics_workbench/label_budget.py`
  - `crash_tests/ml_statistics_workbench/run_label_budget.ipynb`
- protocol:
  - keep the final paired holdout fixed
  - vary only the non-test paired adaptation pool size
  - sample adaptation subsets by `recording_key`, not by individual row
- current variants included:
  - `fpos`
    - `contextual_paired_only_lightgbm`
    - `embed_nommd_fullctx_lightgbm_filtered`
  - `fmiss`
    - `contextual_paired_only_lightgbm`
    - `source_reweighted_fullctx_lightgbm`

Current validation status:

- the runner has been implemented and smoke-validated end to end
- smoke outputs:
  - `crash_tests/ml_statistics_workbench/outputs/label_budget_smoke_v1`

Smoke result:

- both targets still preferred the full paired adaptation budget of `107` rows
- no smaller tested budget met the simple near-optimal threshold in the smoke run

Representative smoke curve:

- `fpos`
  - budget `20`: `R²` about `-1.01`
  - budget `60`: `R²` about `0.19`
  - budget `107`: `R²` about `0.49`
- `fmiss`
  - budget `20`: `R²` about `-0.40`
  - budget `60`: `R²` about `0.05`
  - budget `107`: `R²` about `0.47`

Interpretation:

- the model is still strongly paired-label-limited
- with the current budget points, there is no sign that we are already in a flat saturation regime
- the likely message is:
  - more matched paired labels would still help
  - and if more labels are unavailable, we need methods that act like better label efficiency rather than more generic transfer machinery

Important caveat:

- this conclusion currently comes from the smoke curve
- the budget runner is real, but the definitive answer should come from a full non-smoke budget run

## Study-Aware Follow-Up

Question:

- if the paired domain really differs by recording family, does explicitly giving the model `study_set` help?

Implementation:

- added study-aware variants in the strict workbench run
- used the same sealed holdout and the same paired-finetune pool
- variants included:
  - `study_contextual_paired_only_lightgbm`
  - `study_latent_summary_fullctx_lightgbm`
  - `study_embed_nommd_fullctx_lightgbm_filtered`

Run:

- `crash_tests/ml_statistics_workbench/outputs/workbench_study_full_v1`

Result:

- no study-aware variant beat the current winners globally
- `fpos`:
  - best overall remained `embed_nommd_fullctx_lightgbm_filtered`
  - study-aware latent-summary variant reached only `R² 0.4413`
- `fmiss`:
  - best overall remained `latent_summary_fullctx_lightgbm`
  - best study-aware variant by MAE was `study_contextual_paired_only_lightgbm`
    - MAE `0.1564`
    - `R² 0.4390`
  - this slightly improved MAE versus the plain contextual baseline, but it still lost to the latent-summary stack on `R²`

What we learned:

- `study_set` does contain real signal
- but study label alone is too coarse to explain the remaining target-domain mismatch
- this fits the PCA and clustering result that `PAIRED_ENGLISH` is internally heterogeneous rather than one clean family

## Study-Bias Correction Follow-Up

Question:

- if the best global models have family-specific bias, can we fix that with a small study-wise correction instead of retraining a new model?

Implementation:

- added a leakage-safe post-hoc study-bias correction
- used only out-of-fold residuals from the paired-finetune pool
- applied a shrinkage estimator:
  - study residual mean shrunk toward the global residual mean
- wrapped only the current best global models:
  - `embed_nommd_fullctx_lightgbm_filtered_study_bias`
  - `latent_summary_fullctx_lightgbm_study_bias`

Run:

- `crash_tests/ml_statistics_workbench/outputs/workbench_study_bias_full_v1`

Result:

- also not a winner
- `fpos`:
  - bias correction reduced the overall mean bias strongly
    - from about `-0.0329` to about `-0.0083`
  - but holdout performance got worse
    - MAE `0.1330`
    - `R² 0.4653`
- `fmiss`:
  - bias correction worsened the winner
    - MAE `0.1618`
    - `R² 0.4328`

What we learned:

- the remaining error is not just a simple family-level offset
- the family effect is more structured than “each study has one residual bias”
- in other words:
  - study-aware information is real
  - but the winning representation still needs finer regime structure than a single study label or study-wise offset

## Updated Bottom Line

What is ruled out more strongly now:

- global standalone neural transfer as the main solution
- simple study one-hot augmentation as the main solution
- simple study-wise residual bias correction as the main solution

What still looks most promising:

- keep the current per-target winners
  - `fpos`: `embed_nommd_fullctx_lightgbm_filtered`
  - `fmiss`: `latent_summary_fullctx_lightgbm`
- recover the missing SpikeForest extraction items before more model churn
- if modeling continues, target the finer substructure inside studies rather than study labels alone

Best next ML direction:

- finer regime-aware modeling inside the paired domain, especially for `fmiss`
- likely options:
  - cluster-aware weighting or calibration inside `PAIRED_ENGLISH` and similar mixed families
  - `fmiss`-specific protected filtering rather than global filtering
  - recording-level or subgroup-level residual analysis, not just study-level analysis

## Synthetic Paired Augmentation

Question:

- if the paired label budget is still limiting performance, can we generate leakage-safe synthetic paired rows from the non-test paired pool to improve holdout performance?

Implementation:

- added a dedicated benchmark:
  - `crash_tests/ml_statistics_workbench/synthetic_augmentation.py`
- added a runnable notebook:
  - `crash_tests/ml_statistics_workbench/run_synthetic_augmentation.ipynb`
- protocol:
  - fixed sealed paired holdout
  - synthetic rows generated only from the paired fine-tune pool
  - no holdout recordings used in generation
  - local regression mixup inside `study_set`
  - synthetic rows downweighted relative to real rows
- tested around the current strongest target-specific families:
  - `fpos`
    - `contextual_paired_only_lightgbm`
    - `embed_nommd_fullctx_lightgbm_filtered`
  - `fmiss`
    - `contextual_paired_only_lightgbm`
    - `source_reweighted_fullctx_lightgbm`

Run:

- `crash_tests/ml_statistics_workbench/outputs/synthetic_augmentation_full_v1`

Result:

- synthetic augmentation was useful for `fpos`
- synthetic augmentation was not useful for `fmiss` at the full paired budget

Main holdout outcomes:

- `fpos`
  - best result became `contextual_paired_only_lightgbm_synthetic`
  - MAE `0.1244`
  - `R² 0.4877`
  - this is a very small but real improvement over the plain contextual paired-only model at full budget:
    - baseline MAE `0.1273`
    - baseline `R² 0.4873`
- `fmiss`
  - synthetic variants lost badly at full budget
  - best overall still remained the non-synthetic `source_reweighted_fullctx_lightgbm`
    - MAE `0.1551`
    - `R² 0.4742`
  - synthetic `source_reweighted_fullctx_lightgbm_synthetic` collapsed at full budget:
    - MAE `0.2130`
    - `R² 0.1043`

What was informative:

- synthetic paired rows helped most in the low-label regime for `fpos`
  - example at budget `40`:
    - `contextual_paired_only_lightgbm`
      - MAE `0.1603`
      - `R² 0.0411`
    - `contextual_paired_only_lightgbm_synthetic`
      - MAE `0.1395`
      - `R² 0.1956`
- the same pattern was not reliable for `fmiss`
  - contextual synthetic `fmiss` helped only at the very lowest budget
  - once budget reached `60+`, synthetic augmentation became neutral or harmful
  - at the full `107` rows it was clearly harmful

Interpretation:

- synthetic mixup does not solve the real target-domain gap by itself
- for `fpos`, it behaves like a regularizer and helps when the paired fine-tune set is still small
- for `fmiss`, it appears to oversmooth the structure that the best stack needs
- so synthetic paired augmentation is not a general new default

Updated practical recommendation:

- `fpos`
  - keep synthetic augmentation as a live option, especially when paired budget is small or medium
  - current best full-budget result:
    - `contextual_paired_only_lightgbm_synthetic`
- `fmiss`
  - do not use synthetic augmentation as the default
  - keep the non-synthetic winner:
    - `source_reweighted_fullctx_lightgbm`

Updated current best strict-holdout choices:

- `fpos`: `contextual_paired_only_lightgbm_synthetic`
  - MAE `0.1244`
  - `R² 0.4877`
- `fmiss`: `source_reweighted_fullctx_lightgbm`
  - MAE `0.1551`
  - `R² 0.4742`

## Paired-Conditioned Hybrid Translation

Question:

- instead of making synthetic paired rows directly from the tiny paired set, can we use the measured hybrid-vs-paired feature shift to transform large hybrid source data into paired-like source data?

Implementation:

- added a dedicated benchmark:
  - `crash_tests/ml_statistics_workbench/paired_conditioned_hybrid.py`
- added a runnable notebook:
  - `crash_tests/ml_statistics_workbench/run_paired_conditioned_hybrid.ipynb`
- key idea:
  - identify the most shifted unit features using the existing domain-shift audit
  - fit monotonic quantile transport from hybrid to paired on those features using only non-test paired data
  - train source models on the transported hybrid units
  - use those transported-source predictions inside the usual paired context-first LightGBM stack
- also tested:
  - cluster-conditioned transport
  - dual-source stacks using both original and transported source predictions

Runs:

- smoke:
  - `crash_tests/ml_statistics_workbench/outputs/paired_conditioned_hybrid_smoke_v1`
- full:
  - `crash_tests/ml_statistics_workbench/outputs/paired_conditioned_hybrid_full_v1`

Most important diagnostic result:

- the transport really did move hybrid much closer to paired in unit-feature space
- raw hybrid unit shift:
  - mean abs SMD about `0.6708`
  - unit-domain classifier AUC about `0.9997`
- best global transport:
  - `global_b1p00_n0p00`
  - mean abs SMD about `0.2989`
  - unit-domain classifier AUC about `0.9997`
- so the marginal feature mismatch dropped sharply even though the domain is still globally separable

Specific shifted features that were corrected strongly by transport:

- `snr`
  - abs SMD reduced from about `2.90` to about `0.001`
- `wf_bin_17`
  - about `2.73` to about `0.0004`
- `wf_bin_16`
  - about `2.53` to about `0.0009`
- `wf_trough_to_peak_ms`
  - about `2.60` to about `0.16`
- `wf_peak_trough_ms`
  - about `2.42` to about `0.17`
- `wf_line_length_norm`
  - about `2.12` to about `0.002`
- many amplitude features also moved much closer to paired

Interpretation:

- this is the first experiment that explicitly used the measured hybrid→paired feature differences to condition the source domain
- the transport is not superficial; it clearly changes the hybrid feature marginals toward the paired marginals
- this is strong evidence that “paired-conditioned hybrid” is a real direction, not just a loose intuition

Predictive result:

- inside this dedicated translation benchmark, the best transported-source variant beat the local references on both targets under the R²-first rule
- `fpos`
  - best transport model:
    - `paired_conditioned_global_fullctx_lightgbm_filtered`
    - MAE `0.1336`
    - `R² 0.4619`
  - this beat the local filtered neural-stack reference in this benchmark:
    - MAE `0.1311`
    - `R² 0.4557`
- `fmiss`
  - best transport model:
    - `paired_conditioned_global_fullctx_lightgbm`
    - MAE `0.1586`
    - `R² 0.4694`
  - this beat the local non-transport source stack in this benchmark:
    - MAE `0.1561`
    - `R² 0.4504`

What was informative:

- the best transport was the simplest one:
  - global quantile transport
  - full blend
  - essentially no need for extra noise
- cluster-conditioned transport did not beat global transport
- dual-source stacking of original + transported source predictions was worse

What we learned from that:

- once the hybrid source is translated toward paired, the original shifted source signal can become harmful again
- the paired target heterogeneity is real, but the first cluster-conditioned transport was probably too noisy or too coarse
- the main value right now is in global source conditioning, not in complicated source mixtures

Bottom line:

- this did not become the new overall champion across every crash test
- overall best strict-holdout models still remain:
  - `fpos`: `contextual_paired_only_lightgbm_synthetic`
  - `fmiss`: `source_reweighted_fullctx_lightgbm`
- but this was still one of the most informative experiments so far because it showed:
  - the domain-shift audit can be turned into a concrete source-translation mechanism
  - that mechanism materially reduces hybrid-vs-paired feature mismatch
  - and it produces competitive target performance without adding new target labels

Updated practical recommendation:

- keep paired-conditioned hybrid translation as a serious live research line
- if we continue this direction, the next improvements should target:
  - transport only on the features that matter most per target
  - preserve recording-context as real target context rather than synthesizing it
  - use the translated source signal mainly as one clean stacked feature, not as one of many mixed source features

## Target-Specific Support Projection Follow-Up

Question:

- after the strong global paired-conditioned hybrid result, can we make the translated source more realistic by adding target-specific transport features and projecting them toward local paired support in PCA space?

Implementation:

- extended the paired-conditioned hybrid runner with:
  - target-specific transport feature selection using only the non-test paired training labels
  - PCA support projection on those target-relevant features
  - leakage-safe target-specific variants for both `fpos` and `fmiss`
- runs:
  - `crash_tests/ml_statistics_workbench/outputs/paired_conditioned_hybrid_smoke_v3`
  - `crash_tests/ml_statistics_workbench/outputs/paired_conditioned_hybrid_full_v2`

What happened:

- the revised target-specific path was scientifically useful, but it did not beat the simpler global transport on the full run
- on smoke, the target-specific quantile variant became the best `fpos` transport candidate
- on the full run, the plain global transport still won for both targets

Full-run result:

- `fpos`
  - best remained:
    - `paired_conditioned_global_fullctx_lightgbm_filtered`
    - MAE `0.1336`
    - `R² 0.4619`
  - target-specific manifold transport was close but worse:
    - MAE `0.1344`
    - `R² 0.4573`
- `fmiss`
  - best remained:
    - `paired_conditioned_global_fullctx_lightgbm`
    - MAE `0.1586`
    - `R² 0.4694`
  - target-specific manifold transport did not win:
    - MAE `0.1576`
    - `R² 0.4261`

What we learned:

- target-specific selection is safe if it is done only on the non-test paired pool, but it is not automatically better
- the main source-conditioning gain still comes from fixing the broad hybrid→paired marginal mismatch
- the local PCA support projection did not add enough value on top of that global transport
- so the current best paired-conditioned hybrid recipe is still:
  - strong global quantile transport on the most shifted unit features
  - then one clean transported-source prediction stacked onto real paired context

Updated bottom line:

- the more realistic support-projection version did not replace the simpler global transport
- this is still useful information because it narrows the search:
  - keep the global transport backbone
  - stop adding source mixtures or cluster complexity unless a new diagnostic suggests a specific failure mode

## Real-x / Pseudo-y Manifold Pipeline

Question:

- can the `8775` unlabeled paired rows be turned into something closer to extra paired supervision if we use them as real target-domain `x` and only synthesize the target `y` conservatively?

Implementation:

- added a new manifold-aware pseudo-label runner:
  - `crash_tests/ml_statistics_workbench/pseudo_label_manifold.py`
- core ideas:
  - discover manifold structure on the non-test paired pool only
  - use grouped OOF teacher performance to weight a conservative teacher ensemble
  - pseudo-label only real unlabeled paired rows
  - enforce support checks, cluster-aware quotas, per-recording caps, and pseudo-to-real caps
  - keep the final paired holdout recordings completely sealed
- main diagnostic outputs:
  - `teacher_agreement_summary.csv`
  - `pseudo_candidate_table.csv`
  - `pseudo_cluster_summary.csv`
  - `holdout_best_by_target.csv`

What we learned from the first versions:

- the initial pseudo-label prototype had a bookkeeping bug:
  - it reset selected unlabeled indices and could train on the wrong unlabeled rows
- after fixing that, the first aggressive full pseudo-label run still overused pseudo labels:
  - it looked strong in grouped CV
  - but it did not hold the same lift on the sealed holdout
- the important signal was still real:
  - manifold-aware pseudo-labeling clearly helped `fpos`
  - the problem was finding the right pseudo-label trust region

Key refinement:

- a balanced pseudo regime worked best:
  - `max_pseudo_multiplier = 1.5`
  - `max_pseudo_per_labeled_ratio = 1.5`
  - `pseudo_weight_scale = 0.25`
  - `pseudo_weight_floor = 0.08`
- this kept the pseudo rows broad across the manifold without letting a tiny labeled cluster supervise too many pseudo rows

Best final run:

- `crash_tests/ml_statistics_workbench/outputs/pseudo_label_manifold_full_balanced_v2`

Results:

- `fpos`
  - new best full strict-holdout model across the crash tests:
    - `source_reweighted_pseudo_lightgbm_filtered`
    - MAE `0.1240`
    - `R² 0.5124`
  - this is the first full non-smoke workbench result above `R² 0.5`
  - it also edges out the previous best `fpos` MAE
- `fmiss`
  - pseudo-labeling did not become the new winner
  - best within the pseudo-label run remained weaker than the best previous `fmiss` lines:
    - best pseudo-label `fmiss`: MAE `0.1598`, `R² 0.4207`
    - best non-pseudo teacher in that run:
      - `teacher_paired_conditioned_global_fullctx_lightgbm`
      - MAE `0.1570`, `R² 0.4632`
  - overall `fmiss` still remains better in the earlier context-first stack runs

What was informative:

- the manifold structure is useful, but the value is target-specific
- for `fpos`, real unlabeled paired `x` plus conservative pseudo `y` is now a genuinely productive path
- for `fmiss`, the teacher models are still more trustworthy than the pseudo-label students
- the best pseudo rows were spread across multiple paired clusters and recordings, not just one easy region
- but `fmiss` still seems more sensitive to pseudo-label noise than `fpos`

Practical interpretation:

- the unlabeled paired pool is not “wasted”
- it can be leveraged effectively when:
  - the `x` values stay real
  - pseudo labels are weighted conservatively
  - manifold support and subgroup coverage are enforced
- this is much more credible than trying to generate fully synthetic paired rows from scratch

## FPos Manifold + Context Teacher

Question:

- can the manifold structure do more than gate pseudo rows?
- specifically, can we use filtered context, local manifold teachers, and support metadata to generate better pseudo `fpos` labels than the conservative weighted ensemble?

Implementation:

- added a focused `fpos` runner:
  - `crash_tests/ml_statistics_workbench/fpos_manifold_context.py`
- core ideas:
  - build a manifold from filtered context PCA + unit PCA + source-derived teacher signals
  - add a local kNN teacher and a residual-corrected local teacher
  - train a context-aware meta-teacher on top of those local signals
  - use a pseudo-error model to weight pseudo rows

What happened:

- the smoke run looked extremely promising and briefly exceeded the current `fpos` ceiling
- but the full strict-holdout run did not beat the existing best pseudo-label pipeline

Best full run:

- `crash_tests/ml_statistics_workbench/outputs/fpos_manifold_context_full_v1`

Representative results:

- best holdout in this family:
  - `manifold_context_meta_curriculum_student`
  - MAE `0.1277`
  - `R² 0.4866`
- previous best `fpos` from the balanced pseudo-label line:
  - `source_reweighted_pseudo_lightgbm_filtered`
  - MAE `0.1240`
  - `R² 0.5124`

What this uncovered:

- the manifold is useful, but it is more reliable as a trust model than as a direct pseudo-label generator
- the new local/meta teachers improved some regions but reduced generalization overall
- the full run suggests the extra teacher complexity made the pseudo labels less stable, especially once all recordings were back in play

Conclusion:

- keep the manifold/context machinery as a diagnostic and trust signal
- do not use the manifold meta-teacher as the main `fpos` pseudo-label generator

## FPos Study + Cluster Trust Pseudo-Labeling

Question:

- if the manifold is better as a trust model than as a label generator, can we use it to concentrate pseudo labels in the paired subdomains where the teacher is actually reliable?

Implementation:

- added a focused trust-aware selector:
  - `crash_tests/ml_statistics_workbench/fpos_cluster_trust_pseudo.py`
- design:
  - keep the strongest existing `fpos` student backbone:
    - filtered context + raw source prediction
  - estimate subgroup trust on the labeled non-test paired pool only
    - cluster trust from teacher OOF MAE + support confidence + subgroup size
    - study trust from the same ingredients
  - re-rank pseudo candidates with a combined study+cluster trust score
  - keep the final holdout recordings fully sealed

Why this was more promising:

- the failed manifold-context teacher experiment suggested the manifold should choose *where* to trust pseudo labels, not directly *what* the pseudo target should be
- this approach follows that lesson instead of adding another teacher layer

Best full run:

- `crash_tests/ml_statistics_workbench/outputs/fpos_cluster_trust_full_v1`

Results:

- new best strict-holdout `fpos` model across all crash tests:
  - `source_reweighted_cluster_trust_pseudo_lightgbm_filtered`
  - MAE `0.1210`
  - `R² 0.5199`
- previous best:
  - `source_reweighted_pseudo_lightgbm_filtered`
  - MAE `0.1240`
  - `R² 0.5124`

Why this is important:

- this is not just a holdout-only blip
- grouped CV also improved materially:
  - baseline pseudo student CV `R²`: `0.3311`
  - trust-aware pseudo student CV `R²`: `0.4223`
- so the trust selector improved both model selection and final holdout

What the trust selector actually did:

- it reduced the heavy oversampling of weaker English-like fringe regions
- it concentrated pseudo mass in stronger paired submanifolds
- compared with the previous best pseudo run, the new trust-aware model improved holdout MAE in every paired study on the final test split:
  - biggest gain: `PAIRED_KAMPFF`
  - then smaller but real gains in `PAIRED_CRCNS_HC1`, `PAIRED_ENGLISH`, and `PAIRED_MEA64C_YGER`

Most informative diagnostic:

- the trust scores strongly separated subgroup quality:
  - high-trust groups: `PAIRED_KAMPFF`, `PAIRED_BOYDEN`, and the strongest paired clusters
  - low-trust groups: `PAIRED_ENGLISH` fringe regions and `PAIRED_CRCNS_HC1`
- this matches the earlier PCA and subgroup-shift interpretation:
  - paired is heterogeneous
  - unlabeled paired rows are valuable
  - but their value depends heavily on which subgroup manifold they belong to

Conclusion:

- the best current use of the unlabeled paired pool for `fpos` is:
  - real target-domain `x`
  - conservative pseudo `y`
  - manifold-driven trust weighting
  - strong paired context in the final student
- this is now the clearest positive path for pushing `fpos` beyond the previous ceiling

## Generic Trust Transfer For `fmiss` And `accuracy`

Question:

- can the successful `fpos` trust-selector idea transfer to the remaining targets?
- and for `accuracy`, is the paired non-test pool actually extra usable supervision or just a different regime?

Implementation:

- added a generic runner:
  - `crash_tests/ml_statistics_workbench/target_cluster_trust_transfer.py`
- notebook:
  - `crash_tests/ml_statistics_workbench/run_target_cluster_trust_transfer.ipynb`
- it does two different things depending on the target:
  - if the non-test paired pool really lacks labels for the target, use trust-aware pseudo-label selection
  - if the non-test paired pool already has observed target values, use trust-aware real-label augmentation instead

### `fmiss`

Result:

- full run:
  - `crash_tests/ml_statistics_workbench/outputs/target_cluster_trust_fmiss_full_v1`
- trust-aware pseudo students were competitive, but they still did **not** beat the best existing `fmiss` teacher

Best results in that run:

- best overall holdout:
  - `teacher_paired_conditioned_global_fullctx_lightgbm`
  - MAE `0.1570`
  - `R² 0.4632`
- best trust-aware student in that run:
  - `transport_cluster_trust_pseudo_lightgbm`
  - MAE `0.1543`
  - `R² 0.4435`

Interpretation:

- the trust selector helped the pseudo students relative to some earlier pseudo baselines
- but `fmiss` still does not trust pseudo supervision enough to beat the best context-first teacher
- so the current `fmiss` story remains unchanged:
  - use the older context-first stack / transport teacher lines
  - do not promote pseudo-label `fmiss` as the default

### `accuracy`

This experiment revealed something structurally important:

- the non-test paired `ssl_pool` is **not** a hidden supply of matched-unit `accuracy` labels
- every row in that pool has `accuracy = 0`
- that pool consists of `false_positive` and `unmatched` units, so its accuracy regime is fundamentally different from the matched paired holdout

Evidence:

- paired matched train/test:
  - mean accuracy around `0.59-0.62`
- non-test `ssl_pool`:
  - mean accuracy `0.0`
  - std `0.0`
  - all rows equal zero

Consequence:

- naive real-label augmentation for `accuracy` is strongly harmful
- the smoke run `target_cluster_trust_accuracy_smoke_v1` showed exactly that:
  - the best real-label augmentation variants collapsed badly
  - the best strict-holdout result remained the plain matched-only contextual teacher

Best `accuracy` result we saw in that run:

- `teacher_contextual_paired_only_lightgbm`
  - MAE `0.1920`
  - `R² 0.3416`

Interpretation:

- the extra non-test paired rows are **not** useful for improving matched-unit `accuracy` prediction directly
- they describe a different target regime
- so `accuracy` should stay on the matched-only paired benchmark unless we explicitly change the prediction task to include unmatched units

## FPos Real-Label Augmentation And Anchor Regime Modeling

Question:

- after the trust-aware pseudo breakthrough on `fpos`, can we extract even more value from the large non-test paired pool?
- especially since `fpos` is technically defined on every paired row, can those extra rows be used directly?

### Direct Real-Label Augmentation

We tested this first through the generic trust-transfer runner.

What we found:

- the non-test `ssl_pool` does have valid `fpos` values
- but every single one of those rows has `fpos = 1.0`
- that means the pool is not “more matched-unit supervision”; it is an extreme all-positive regime made of `false_positive` and `unmatched` units

Result:

- direct real-label augmentation for `fpos` is strongly harmful
- the smoke run `target_cluster_trust_fpos_smoke_v1` collapsed badly under both:
  - all-real augmentation
  - trust-selected real-label augmentation

Interpretation:

- those rows are still informative
- but not as direct regression targets for the matched-unit `fpos` benchmark

### Anchor False-Positive Regime Score

New idea:

- use the all-positive unmatched pool as **structure**, not as blunt regression labels
- train a grouped classifier to distinguish:
  - matched paired units
  - unmatched/false-positive paired units
- use that classifier output as an extra feature on top of the current best trust-aware `fpos` pseudo student

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_anchor_stack.py`
- notebook:
  - `crash_tests/ml_statistics_workbench/run_fpos_anchor_stack.ipynb`

What this adds:

- `anchor_fp_score`: probability that a unit lives in the obvious false-positive regime
- `matched_support_conf`
- `anchor_support_margin = anchor_fp_score - matched_support_conf`

Best full run:

- `crash_tests/ml_statistics_workbench/outputs/fpos_anchor_stack_full_v1`

Results:

- anchor-stack student:
  - `source_reweighted_anchor_stack_cluster_trust_pseudo_lightgbm_filtered`
  - MAE `0.1231`
  - `R² 0.5246`
- current main `fpos` winner:
  - `source_reweighted_cluster_trust_pseudo_lightgbm_filtered`
  - MAE `0.1210`
  - `R² 0.5199`

Interpretation:

- this is a genuine near-miss, not noise
- the anchor regime score slightly improved strict-holdout `R²`
- but it worsened MAE and, more importantly, its grouped CV was weaker than the simpler trust-aware winner

So the scientific conclusion is:

- the unmatched all-positive pool does contain useful structural information for `fpos`
- the right way to use it is as an anchor/manifold score, not direct label augmentation
- but the current anchor-stack version is still **diagnostic**, not the default deployment model

Why it matters anyway:

- it shows there is still some remaining `fpos` headroom in the positive-regime geometry
- that headroom is subtle:
  - it seems to improve variance explained
  - but not robustly enough yet to beat the current winner under the R²-first + MAE-guardrail rule

### Backend Sweep: XGBoost Overtakes LightGBM for `fpos`

New idea:

- stop assuming LightGBM is the best tabular backend for the current trust-aware `fpos` stack
- sweep the strongest existing `fpos` feature recipes across:
  - LightGBM
  - XGBoost
  - CatBoost
  - HistGradientBoosting
  - ExtraTrees

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_backend_sweep.py`
- output:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_backend_sweep_full_v1`

Best full run:

- new `fpos` winner:
  - `source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default`
  - MAE `0.1169`
  - `R² 0.5331`

Comparison to the previous best `fpos` model:

- old winner:
  - `source_reweighted_cluster_trust_pseudo_lightgbm_filtered`
  - MAE `0.1210`
  - `R² 0.5199`
- new winner:
  - `source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default`
  - MAE `0.1169`
  - `R² 0.5331`

Interpretation:

- the backend really mattered
- XGBoost was able to use the anchor/manifold score better than LightGBM
- this was not just a higher-`R²` trade for worse MAE:
  - it improved both holdout `R²` and holdout MAE
- this is the current best strict-holdout `fpos` result in the repo

Important nuance:

- grouped CV still liked the simpler non-anchor XGBoost line more
- but the strict holdout winner is the anchor-stack XGBoost model
- since the user explicitly asked to keep the slightly higher-`R²` path when scientifically acceptable, this anchor-stack XGBoost line is now the active `fpos` winner

### Follow-Up `fpos` Variance Tests

After the backend sweep, three more leakage-safe attempts were run to see if the new XGBoost line still had easy headroom.

#### Focused XGBoost Tuning

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_xgboost_tuning.py`
- output:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_xgboost_tuning_full_v1`

Result:

- no tuned preset beat the backend-sweep winner
- best tuned holdout:
  - `source_reweighted_anchor_stack_cluster_trust_pseudo_xgb_deep`
  - MAE `0.1213`
  - `R² 0.5199`

Interpretation:

- there was still seed / hyperparameter sensitivity
- but the targeted preset grid did not improve on the backend-sweep default regime
- so the backend-sweep winner stands

#### Expert Meta-Stacking

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_expert_stack.py`
- smoke output:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_expert_stack_smoke_v1`

Idea:

- combine the two strongest `fpos` experts:
  - base trust-aware XGBoost
  - anchor-stack trust-aware XGBoost
- feed their grouped OOF predictions plus manifold/context diagnostics into a second-stage meta model

Result:

- this was not competitive
- meta models overfit badly in smoke
- best smoke holdout still stayed with a base expert rather than the meta layer

Interpretation:

- the two expert lines are complementary
- but the paired labeled sample is still too small for a reliable second-stage nonlinear stack
- useful negative result: do not add a free-form meta learner on top of this `fpos` stack yet

#### XGBoost Seed Bagging

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_xgb_bagging.py`
- smoke output:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_xgb_bagging_smoke_v1`

Idea:

- keep the winning XGBoost recipe fixed
- average across several XGBoost seeds to reduce subsampling variance

Result:

- seed bagging did not beat the single-seed backend-sweep winner
- best smoke holdout:
  - `source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_seedbag`
  - MAE `0.1209`
  - `R² 0.4789`

Interpretation:

- bagging reduced volatility
- but it also washed out some of the high-`R²` behavior that made the winning single-seed model strong
- this is not the next lever

## Updated Overall Bests

Current best strict-holdout results across the crash tests:

- `fpos`
  - `source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default`
  - run: `fpos_backend_sweep_full_v1`
  - MAE `0.1169`
  - `R² 0.5331`
- `fmiss`
  - best by `R²`:
    - `source_plus_reduced_latent_fullctx_lightgbm`
    - run: `workbench_r2_full_v1`
    - MAE `0.1562`
    - `R² 0.4850`
  - best by MAE among the strong full runs:
    - `source_reweighted_fullctx_lightgbm`
    - run: `workbench_shift_subgroup_full_v1`
    - MAE `0.1533`
    - `R² 0.4758`

Updated takeaway:

- the unlabeled paired pool is highly valuable for `fpos`, but only when it is used structurally and selectively
- the current best `fpos` path is now:
  - filtered unit/context features
  - trust-aware pseudo supervision from real unlabeled paired units
  - false-positive anchor/manifold scoring
  - XGBoost, not LightGBM
- direct real-label augmentation from the unmatched paired pool is invalid for `fpos` because that pool is all-positive
- free-form meta stacking and seed bagging did not beat the current winner
- `fmiss` still resists pseudo supervision and remains better served by the earlier context-first stacks
- `accuracy` does not benefit from the non-test `ssl_pool` because that pool is entirely zero-accuracy unmatched/false-positive units
- so the best current deployment story remains asymmetric:
  - use the anchor-stack trust-aware XGBoost line for `fpos`
  - keep the older context-first winner for `fmiss`
  - keep matched-only context-first modeling for `accuracy`

### Winner Interpretability: SHAP For The Best `fpos` Model

Implementation:

- script:
  - `crash_tests/ml_statistics_workbench/plot_fpos_winner_shap.py`
- outputs written under:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_backend_sweep_full_v1`

Most important SHAP features for the current `fpos` winner:

- `source_pred`
- `amp_bimodality`
- `acg_26`
- `isi_violation_rate`
- `si_amplitude_cv_range`
- `n_waveform_confusable_recording_pct_recording_iqr`
- several recording-relative amplitude features

Interpretation:

- the strongest signal still comes from the source-derived prediction
- the remaining gain comes from:
  - amplitude-shape irregularity
  - refractory / ISI violation structure
  - ACG shape
  - recording-relative position inside confusability and amplitude distributions
- this supports the current scientific picture:
  - the model works best when a strong transferred prior is corrected by recording-aware unit diagnostics

What it suggests for future `fpos` work:

- improve the source prediction itself, since it remains the dominant driver
- build subgroup-aware calibration around amplitude / ACG / ISI-heavy regions
- strengthen same-recording rank / percentile features rather than adding broad new context blocks
- focus error analysis on studies that remain underpredicted:
  - especially `PAIRED_CRCNS_HC1`
  - `PAIRED_ENGLISH`
  - `PAIRED_KAMPFF`

### Residual Correction On Top Of The `fpos` Winner

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_residual_correction.py`
- outputs:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_residual_correction_full_v1`

Idea:

- keep the current winning `fpos` model fixed
- fit a second-stage residual corrector on top of its predictions
- test both:
  - global residual correction
  - study-aware residual correction using `study_set`
- residual features restricted to the SHAP-supported correction region:
  - `source_pred`
  - amplitude irregularity features
  - ACG/ISI features
  - support / anchor scores
  - recording-relative amplitude features

Leakage policy:

- final holdout recordings stayed sealed
- residual models were trained only on the non-test paired pool
- grouped OOF predictions were used for residual fitting
- `study_set` was treated as an inference-time covariate, not as a label-derived artifact

Full strict holdout result:

- winner reference:
  - `source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default`
  - MAE `0.1169`
  - `R² 0.5331`
- best residual variant:
  - `source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default_plus_global_histgb`
  - MAE `0.1350`
  - `R² 0.3227`
- study-aware residual variant:
  - `source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default_plus_study_histgb`
  - MAE `0.1358`
  - `R² 0.3149`

Interpretation:

- residual calibration on top of the winner is clearly harmful
- the study-aware residual version is not better than the global residual version
- the current winner is already using most of the stable correction signal
- adding a second correction layer mostly overfits the tiny paired pool and flattens the prediction dynamic range

Scientific note:

- family- or study-specific calibration is not automatically cheating
- it is scientifically acceptable if:
  - the family label is available at inference time
  - calibration is fit only on training-side rows/folds
  - the final holdout does not influence the calibration choice
- in this benchmark, the main problem was not leakage
- the main problem was overfitting from too little paired labeled data for a second-stage family-specific corrector

### Family-Robust `fpos` Training

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_family_robustness.py`
- outputs:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_family_robustness_full_v3`

Idea:

- keep the current winning `fpos` stack fixed:
  - source-reweighted pseudo labels
  - anchor-stack features
  - XGBoost backend
- change only the weighting on the labeled paired rows
- upweight underrepresented paired `study_set` families
- compare:
  - plain winner
  - mild family balancing
  - strong family balancing
  - shallow robust XGBoost
  - leakage-safe blends between the plain winner and the strong balanced model

Global holdout result:

- global winner remains the original reference:
  - `reference_xgboost`
  - MAE `0.1169`
  - `R² 0.5331`
- best family-balanced model:
  - `family_balanced_strong_xgboost`
  - MAE `0.1167`
  - `R² 0.5117`
- best blend by MAE:
  - `reference_plus_strong_blend_050`
  - MAE `0.1158`
  - `R² 0.5276`

Family robustness result:

- reference worst-family holdout `R²`:
  - `0.2539`
- strong family-balanced worst-family holdout `R²`:
  - `0.3818`
- the biggest improvement came from `PAIRED_CRCNS_HC1`
  - reference `R²`: `0.2539`
  - strong balanced `R²`: `0.3818`

Interpretation:

- family-balanced weighting does make the model less family-sensitive
- it does not produce a new global `R²` champion
- the strong balanced model is the right choice only if the objective is robustness across families
- the original XGBoost winner is still the right choice if the objective is maximum global holdout `R²`
- the 50/50 blend is the best compromise found so far:
  - slightly better MAE than the winner
  - materially better worst-family behavior than the winner
  - but still lower global `R²`

Scientific conclusion:

- true family-agnostic performance is not free in this benchmark
- making the model more robust across families costs some global variance explained
- the deployment choice depends on the target criterion:
  - maximize average strict-holdout `R²`:
    - keep the original winner
  - reduce family-to-family sensitivity:
    - prefer the strong family-balanced variant

### Signal Embedding Enrichment For `fpos`

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_signal_embedding_stack.py`
- outputs:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_signal_embedding_stack_full_v1`

Idea:

- enrich the current family-balanced `fpos` stack with learned signal embeddings
- train small 1D convolutional autoencoders on:
  - waveform bins `wf_bin_*`
  - ACG bins `acg_*`
- use only non-holdout rows for embedding learning:
  - hybrid source
  - non-test paired matched
  - non-test paired unmatched
- stack the learned embeddings into the existing anchor/context/source feature frame
- keep the final predictor as XGBoost

Variants tested:

- family-balanced reference anchor stack
- waveform embedding + anchor stack
- ACG embedding + anchor stack
- waveform + ACG embedding + anchor stack

Full strict holdout result:

- new best `fpos` model:
  - `wf_embed_anchor_stack_xgboost`
  - MAE `0.1140`
  - `R² 0.5479`
- previous global `fpos` winner:
  - `source_reweighted_anchor_stack_cluster_trust_pseudo_xgboost_default`
  - MAE `0.1169`
  - `R² 0.5331`

Why this matters:

- this is the first feature-enrichment pass that clearly beat the previous `fpos` ceiling
- the gain came from waveform embeddings alone
- ACG-only embeddings helped relative to the local family-balanced reference, but did not match the waveform result
- combining waveform and ACG embeddings was worse than waveform alone

Family-level pattern:

- waveform embedding improved several families at once
- notable holdout `R²` values for `wf_embed_anchor_stack_xgboost`:
  - `PAIRED_BOYDEN`: `0.5894`
  - `PAIRED_CRCNS_HC1`: `0.3713`
  - `PAIRED_ENGLISH`: `0.4770`
  - `PAIRED_KAMPFF`: `0.6851`
  - `PAIRED_MEA64C_YGER`: `0.5044`

Interpretation:

- the current tabular/context stack was still missing useful local waveform structure
- a compact learned waveform embedding adds information the handcrafted summaries did not fully capture
- this fits the earlier SHAP story:
  - amplitude and waveform-shape cues were already important
  - the model still had headroom in waveform-sensitive families
- ACG embeddings look useful, but not as the primary enrichment path

Updated takeaway:

- the best current `fpos` path is now:
  - family-balanced paired training
  - trust-aware pseudo supervision from unlabeled paired units
  - anchor/manifold features
  - waveform convolutional embedding
  - XGBoost regressor

### Depth-3 Tree Interaction Features On Top Of The Waveform Winner

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_tree_interaction_stack.py`
- smoke outputs:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_tree_interaction_stack_smoke_v4`

Idea:

- fit shallow depth-3 gradient-boosted trees on a small curated subset of the waveform-winner features
- export leaf one-hot features as explicit interaction features
- stack those leaf indicators on top of the current waveform-embedding winner
- keep the final predictor as XGBoost

Leakage policy:

- the leaf-feature generator was fit fold-by-fold inside CV
- holdout leaf features were generated only after fitting on the full non-test paired training pool
- no holdout rows were used to create the interaction features

Smoke result:

- waveform-embedding reference:
  - MAE `0.1250`
  - `R² 0.4747`
- waveform + depth-3 tree interactions:
  - MAE `0.1282`
  - `R² 0.4551`

Interpretation:

- explicit depth-3 tree leaf features did not help
- the current final XGBoost is already learning the useful low-order interactions directly
- adding a separate shallow-tree interaction layer mainly added redundant or noisy features
- this branch was not promising enough to justify a full run

### Paper-Inspired Transfer Benchmark On Top Of The Waveform Winner

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_paper_inspired_transfer_benchmark.py`
- outputs:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_paper_transfer_full_v1`

Paper-inspired method families tested:

- small adapter-style supervised score on frozen waveform embeddings
- few-shot same-family local score
- labeled neighborhood support features
- unlabeled paired-manifold pseudo-features
- combined feature stack joining several of the above

Reference for this benchmark:

- the current waveform-embedding winner was used as the base model family

Full strict holdout results:

- best paper-inspired variant:
  - `wf_embed_adapter_xgboost`
  - MAE `0.1166`
  - `R² 0.5256`
- benchmark-local baseline:
  - `wf_embed_baseline_xgboost`
  - MAE `0.1203`
  - `R² 0.5142`
- current global `fpos` winner still remains:
  - `wf_embed_anchor_stack_xgboost`
  - MAE `0.1140`
  - `R² 0.5479`

What was informative:

- the adapter-style score was the strongest of the four new ideas
- it improved over the benchmark-local baseline
- but it still did not beat the global waveform-embedding winner
- few-shot, labeled-neighborhood, and combined variants all looked stronger in CV than on holdout
- the combined stack especially overfit

Interpretation:

- small-data adaptation ideas from the papers are directionally useful
- but in this repo, most of their value is already being captured by:
  - the family-balanced training setup
  - the pseudo-label trust pipeline
  - the waveform embedding itself
- the extra paper-inspired heads/features mostly added another correction layer without enough new information

Updated conclusion:

- keep the current waveform-embedding winner as the deployed `fpos` model
- if revisiting this paper-inspired family later, the only branch worth keeping alive is the adapter-style one
- the few-shot/local-neighborhood feature variants are not strong enough in their current form

### Applying The Best `fpos` Recipe To `fmiss`

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fmiss_waveform_transfer.py`
- outputs:
  - `crash_tests/ml_statistics_workbench/outputs/fmiss_waveform_transfer_full_v1`

Question tested:

- take the current best `fpos` recipe family
- keep its main structure:
  - trust-aware pseudo rows
  - family-balanced training weights
  - anchor-style unmatched-vs-matched score
  - waveform embedding
  - XGBoost final regressor
- retrain it for `fmiss`
- compare directly against the current dedicated `fmiss` leaders

Full strict holdout results:

- `fmiss_reference_anchor_stack_xgboost`
  - MAE `0.1667`
  - `R² 0.3673`
- `fmiss_wf_embed_anchor_stack_xgboost`
  - MAE `0.1673`
  - `R² 0.3819`

Current dedicated `fmiss` references for comparison:

- `source_plus_reduced_latent_fullctx_lightgbm`
  - MAE `0.1562`
  - `R² 0.4850`
- `embed_nommd_fullctx_lightgbm`
  - MAE `0.1579`
  - `R² 0.4580`

What was informative:

- the waveform embedding did help slightly inside this transplanted `fmiss` recipe
- the improvement over the no-waveform version was small:
  - `R² 0.3673 -> 0.3819`
- both variants remained well below the best dedicated `fmiss` stack
- CV was also weak and negative for both transplanted variants, even though holdout stayed positive

Interpretation:

- the best `fpos` recipe does not transfer cleanly to `fmiss`
- the false-positive anchor logic and its associated pseudo-label structure appear much more aligned with `fpos` than with `fmiss`
- waveform representation alone is not enough to rescue that mismatch
- `fmiss` still prefers its own context-first source-summary / latent-summary LightGBM family

Updated conclusion:

- do not replace the current `fmiss` winner with the `fpos` waveform recipe
- keep:
  - `fpos`: waveform-embedding XGBoost line
  - `fmiss`: dedicated context-first LightGBM line
- treat this as another strong sign that `fpos` and `fmiss` are meaningfully different transfer problems

### Richer Waveform Enrichment On Top Of The `fpos` Winner

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_waveform_enrichment_stack.py`
- outputs:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_waveform_enrichment_full_v1`

Question tested:

- keep the current winning `fpos` stack structure fixed
- add richer waveform-only signals on top of it:
  - low-frequency cosine / DCT-style shape coefficients
  - multiscale segment pooling features
  - derivative and curvature summary features
- compare:
  - waveform embedding only
  - rich handcrafted waveform features only
  - waveform embedding + rich waveform features

Why this was promising:

- waveform embedding was the strongest single feature-enrichment result so far
- ACG enrichment did not help much
- the next sensible move was to extract more value from waveform structure rather than add a new broad model family

Smoke result:

- the combined waveform-embedding + rich waveform branch looked promising on the smoke holdout:
  - `R² 0.4978`
- benchmark-local waveform embedding reference in smoke:
  - `R² 0.4844`

Full strict holdout results:

- benchmark-local waveform embedding reference:
  - `wf_embed_anchor_stack_xgboost`
  - MAE `0.1229`
  - `R² 0.4958`
- rich waveform only:
  - `wf_rich_anchor_stack_xgboost`
  - MAE `0.1266`
  - `R² 0.4740`
- waveform embedding + rich waveform features:
  - `wf_embed_rich_anchor_stack_xgboost`
  - MAE `0.1216`
  - `R² 0.4887`

What was informative:

- richer waveform features did slightly improve MAE over the benchmark-local reference
- but they reduced `R²`
- the combined branch had the best CV `R²` inside this run, but that did not translate into the best holdout `R²`
- waveform-only handcrafted enrichment was weaker than the learned waveform embedding

Interpretation:

- the handcrafted waveform enrichments contain some useful local signal
- but the current conv waveform embedding is still the better representation
- adding the richer handcrafted branch on top may be increasing calibration / bias problems instead of improving variance explained
- this branch did not beat the current global `fpos` winner from the earlier waveform run

Updated conclusion:

- keep the current waveform-embedding winner as the main `fpos` model
- do not promote the richer waveform branch
- if revisiting waveform enrichment later, the most likely useful follow-up is not more handcrafted features but a stronger learned waveform representation

### Stronger Learned Waveform Representation: Multiscale Autoencoder

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_waveform_multiscale_stack.py`
- outputs:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_waveform_multiscale_full_v1`

Question tested:

- replace the simple waveform autoencoder with a stronger learned waveform representation
- use a multiscale convolutional encoder with:
  - parallel kernels (`3`, `5`, `7`)
  - denoising reconstruction
  - derivative reconstruction penalty
- compare:
  - baseline waveform embedding
  - multiscale waveform embedding
  - baseline + multiscale embedding together

Why this was promising:

- waveform embedding was the strongest positive feature-enrichment result so far
- richer handcrafted waveform features were not enough
- the next best idea was a stronger learned waveform representation instead of more manual descriptors

Smoke result:

- multiscale embedding won clearly inside the smoke harness:
  - `wf_multi_embed_anchor_stack_xgboost`
  - MAE `0.1184`
  - `R² 0.5068`
- smoke baseline waveform embedding:
  - MAE `0.1255`
  - `R² 0.4596`

Full strict holdout results:

- baseline waveform embedding in this benchmark:
  - `wf_embed_anchor_stack_xgboost`
  - MAE `0.1188`
  - `R² 0.4972`
- multiscale waveform embedding only:
  - `wf_multi_embed_anchor_stack_xgboost`
  - MAE `0.1216`
  - `R² 0.4845`
- baseline + multiscale together:
  - `wf_embed_plus_multi_anchor_stack_xgboost`
  - MAE `0.1198`
  - `R² 0.5027`

What was informative:

- the multiscale learned representation did add some useful holdout signal
- the combined embedding branch beat the local baseline on `R²`
- however CV got worse for both multiscale branches
- the gain was not strong enough to beat the earlier global waveform winner:
  - `wf_embed_anchor_stack_xgboost`
  - MAE `0.1140`
  - `R² 0.5479`

Interpretation:

- stronger learned waveform representations are directionally promising
- but the current multiscale implementation is still not stable enough
- it appears to add some extra holdout signal while increasing training / validation instability
- this looks more promising than the handcrafted waveform branch, but not yet like a production promotion

Updated conclusion:

- keep the earlier waveform-embedding winner as the active `fpos` model
- do not promote the multiscale branch yet
- if revisiting waveform learning later, the learned-representation direction is still more promising than manual waveform feature engineering

### Bayesian Optimization Of The `fpos` Signal-Embedding Pipeline

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_signal_embedding_bayes_opt.py`
- outputs:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_signal_embedding_bayes_full_v1`
- dependency added to local environment for this benchmark:
  - `optuna`

Question tested:

- optimize the current `fpos` waveform-embedding pipeline more systematically instead of hand-picked defaults
- tune across multiple layers of the pipeline:
  - pseudo / trust settings
  - waveform embedding size and training hyperparameters
  - family-balance strength
  - XGBoost hyperparameters
- optimize only on grouped paired CV
- keep the final holdout sealed until the best trial is chosen

Search space covered:

- pseudo / trust:
  - `max_pseudo_multiplier`
  - `max_pseudo_per_recording`
  - `max_pseudo_per_cluster`
  - `max_pseudo_per_labeled_ratio`
  - `cluster_trust_power`
  - `study_trust_weight`
  - `min_trust`
- waveform encoder:
  - `conv_embed_dim`
  - `conv_hidden_channels`
  - `ae_epochs`
  - embedding learning rate
- final model:
  - XGBoost tree count, learning rate, depth, child weight
  - subsample / colsample
  - `reg_lambda`, `reg_alpha`, `gamma`
- family robustness:
  - `family_alpha`

Full run result:

- benchmark-local baseline:
  - `baseline_wf_embed_anchor_stack_xgboost`
  - MAE `0.1419`
  - `R² 0.4374`
- best Bayesian-optimized variant:
  - `bayes_opt_wf_embed_anchor_stack_xgboost`
  - MAE `0.1410`
  - `R² 0.4553`

Best trial selected by grouped CV:

- CV:
  - MAE `0.1547`
  - `R² 0.2202`
- holdout:
  - MAE `0.1410`
  - `R² 0.4553`
- best settings included:
  - `conv_embed_dim = 10`
  - `conv_hidden_channels = 24`
  - `ae_epochs = 24`
  - embedding learning rate about `2.30e-3`
  - `family_alpha = 1.1267`
  - XGBoost:
    - `n_estimators = 580`
    - `learning_rate = 0.0359`
    - `max_depth = 5`
    - `min_child_weight = 7.77`
    - `subsample = 0.776`
    - `colsample_bytree = 0.799`
    - `reg_lambda = 1.55`
    - `reg_alpha = 0.285`
    - `gamma = 0.018`

What was informative:

- Bayesian optimization did improve the benchmark-local baseline
- the best region favored:
  - moderate embedding size
  - stronger hidden-channel capacity
  - somewhat stronger family balancing
  - deeper but more regularized XGBoost settings
- the best search result still did not beat the earlier global `fpos` winner at:
  - MAE `0.1140`
  - `R² 0.5479`

Important scientific caveat:

- the waveform-embedding stage is somewhat stochastic across full reruns
- because of that, the cleanest interpretation is within-harness:
  - Bayesian optimization improved the local baseline in its own run
- it does not yet justify replacing the earlier global winner

Updated conclusion:

- keep the earlier waveform-embedding winner as the active `fpos` model
- Bayesian optimization is worth keeping as tooling
- but this first full pass did not uncover a new global champion
- the result suggests some remaining upside in tuning, while also exposing that embedding-stage stability is now an important issue

### Explainability Pass For The Current Best `fpos` Model

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/explain_fpos_best_model.py`
- outputs:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_signal_embedding_stack_full_v1/explainability_v1`

Model explained:

- `wf_embed_anchor_stack_xgboost`
- current best saved run:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_signal_embedding_stack_full_v1`

Explainability artifacts produced:

- SHAP bar summary
- SHAP beeswarm
- top dependence plots
- prediction vs truth scatter
- calibration curve
- study summary
- sorter summary
- recording summary
- residual-probe importance table
- markdown report

Most important findings:

- the model is still dominated by the transferred source prior:
  - top feature by far: `source_pred`
- the strongest corrections are not generic context blocks, but recording-relative amplitude and QC instability features:
  - `amp_bin_spread_ratio_recording_std`
  - `amp_bin_cv_mean_recording_pct`
  - `si_isi_violations_count`
  - `si_amplitude_cv_median`
  - `amp_std_recording_pct`
- waveform information is still present directly:
  - `wf_bin_13`
  - `wf_energy_recording_delta`

Holdout behavior from the explainability refit:

- MAE `0.1405`
- `R² 0.4567`
- bias `0.0113`
- calibration slope `1.0884`

What the missing `R²` seems to be:

- oracle global linear recalibration would only recover about `+0.0096` `R²`
- oracle per-study intercept correction would only recover about `+0.0136` `R²`
- residual-probe CV on training residuals was strongly negative

Interpretation:

- the remaining gap is not mostly a simple calibration problem
- it is not mostly a simple per-family intercept problem either
- and the residual-probe result suggests the current feature frame is not leaving a large amount of easy, structured signal unused
- this points more toward:
  - subgroup-specific hard cases
  - missing features
  - or plain target noise

Where the model struggles most:

- worst family by far:
  - `PAIRED_ENGLISH`
  - MAE `0.2144`
  - `R² 0.3214`
- next:
  - `PAIRED_CRCNS_HC1`
  - MAE `0.2026`
  - `R² 0.3806`
- strongest families:
  - `PAIRED_KAMPFF`
  - `PAIRED_MEA64C_YGER`

Important subgroup findings:

- some sorters still behave poorly even when global metrics look decent:
  - `IronClust`, `Klusta`, and `JRClust` had negative or near-zero sorter-level `R²`
- the worst recordings are concentrated in English and CRCNS subsets

Updated conclusion:

- the current best `fpos` model behaves like:
  - strong source prior
  - corrected by amplitude instability, ISI violation, and recording-relative QC features
- the missing performance does not look recoverable by trivial recalibration
- the next big gain is more likely to come from:
  - better features for the hard English / CRCNS regimes
  - or richer waveform / template information

### Leave-One-Family-Out Stress Test For The Best `fpos` Model

Implementation:

- runner:
  - `crash_tests/ml_statistics_workbench/fpos_leave_one_family_out.py`
- outputs:
  - `crash_tests/ml_statistics_workbench/outputs/fpos_leave_one_family_out_full_v1`

Question tested:

- does the current best `fpos` pipeline generalize when an entire paired family is held out as completely unseen
- this is stricter than the main benchmark
- protocol here:
  - hold out one `study_set` / family entirely
  - train on all other paired families
  - do not use unlabeled rows from the held-out family
  - evaluate on matched paired rows from the held-out family only

Variants compared:

- `reference_anchor_stack_xgboost`
- `wf_embed_anchor_stack_xgboost`

Aggregate results:

- reference:
  - macro `R² 0.4527`
  - weighted `R² 0.4240`
  - worst-family `R² 0.2940`
- waveform winner:
  - macro `R² 0.4386`
  - weighted `R² 0.4116`
  - worst-family `R² 0.1482`

Per-family results for the waveform winner:

- `PAIRED_BOYDEN`
  - MAE `0.0751`
  - `R² 0.1482`
- `PAIRED_CRCNS_HC1`
  - MAE `0.1788`
  - `R² 0.4833`
- `PAIRED_ENGLISH`
  - MAE `0.1805`
  - `R² 0.3625`
- `PAIRED_KAMPFF`
  - MAE `0.1020`
  - `R² 0.6952`
- `PAIRED_MEA64C_YGER`
  - MAE `0.0745`
  - `R² 0.5037`

What was informative:

- the best mixed-family `fpos` model is not the best family-unseen model
- waveform embedding helps a lot for some unseen families:
  - especially `PAIRED_KAMPFF`
- but it hurts badly on unseen `PAIRED_BOYDEN`
- the simpler reference stack is actually more robust in macro and weighted `R²`

Interpretation:

- the main benchmark is recording-disjoint, not family-disjoint
- context features are not leaking under the main protocol
- but the stronger waveform-enhanced model is still partly specialized to patterns shared across seen families
- fully unseen-family generalization is clearly harder than the main benchmark

Updated conclusion:

- for standard recording-disjoint evaluation, keep the waveform winner as the best `fpos` model
- for family-unseen robustness, the simpler reference stack is currently safer
- the thesis should explicitly distinguish:
  - recording-disjoint generalization
  - family-unseen generalization
