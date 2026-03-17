#!/usr/bin/env python3
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


THESIS_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = THESIS_ROOT / "notebooks"


def md(text: str):
    return new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return new_code_cell(textwrap.dedent(text).strip() + "\n")


def setup_code(extra: str):
    return code(SETUP.rstrip() + "\n\n" + textwrap.dedent(extra).strip())


SETUP = """
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

def _find_thesis_root():
    cwd = Path.cwd().resolve()
    direct = [cwd, *cwd.parents]
    nested = [candidate / "thesis" for candidate in direct]
    for candidate in [*direct, *nested]:
        if (candidate / "THESIS_BLUEPRINT.md").exists() and (candidate / "src" / "qc_thesis" / "__init__.py").exists():
            return candidate
    raise FileNotFoundError("Could not find thesis root from notebook session")

ROOT = _find_thesis_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qc_thesis import *

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)
apply_thesis_style()
"""


def notebook_01():
    stem = "01_data_extraction_and_audit"
    return new_notebook(cells=[
        md("""
        # 01 Data Extraction And Audit

        This notebook opens the thesis package the way a reviewer should: first by understanding where the dataset came from, what was packaged locally, and how the final modeling subsets were defined.

        **Questions answered here**
        - What data are inside the thesis package?
        - How were hybrid, paired-labeled, and paired-unlabeled subsets defined?
        - Why is `recording_key` the leakage boundary?
        - What did the extraction and audit phase actually establish before modeling began?
        """),
        setup_code(f"""
        fig_dir, table_dir = notebook_output_dirs("{stem}")
        dataset_summary = load_dataset_summary()
        computed_summary = compute_dataset_summary()
        partition_table = build_dataset_partition_table()
        target_missingness = build_target_missingness_table()
        family_coverage = build_family_coverage_table()
        feature_inventory = build_feature_inventory_table()
        manifest = json.loads((ROOT / "data/frozen_inputs/manifest.json").read_text())
        """),
        md("""
        ## 1. Packaged assets

        The thesis package contains both the exploratory raw parquet and the generated benchmark assets. The point is not only to show the final tables, but to show the entire route from extraction to evaluation.
        """),
        code("""
        display(pd.DataFrame(dataset_summary.items(), columns=["packaged_item", "value"]))
        display(pd.DataFrame(computed_summary.items(), columns=["recomputed_item", "value"]))
        display(pd.DataFrame(manifest.items(), columns=["artifact_group", "path"]))
        save_table(pd.DataFrame(dataset_summary.items(), columns=["item", "value"]), table_dir, "packaged_dataset_summary")
        save_table(pd.DataFrame(computed_summary.items(), columns=["item", "value"]), table_dir, "recomputed_dataset_summary")
        """),
        md("""
        ## 2. Subset definitions

        The thesis uses three operational subsets:

        - **hybrid labeled**: large source domain with labels
        - **paired labeled**: small matched target-domain supervision
        - **paired unlabeled**: real target-domain support used only structurally

        This table is the foundation for every later benchmark and ablation.
        """),
        code("""
        display(partition_table)
        display(target_missingness)
        save_table(partition_table, table_dir, "dataset_partition_table")
        save_table(target_missingness, table_dir, "target_missingness")
        """),
        md("""
        ## 3. Family and feature coverage

        Before modeling, the important question was whether the paired domain behaved like one domain or a mixture of regimes. Family coverage and feature inventory already hint that the answer is “mixture”.
        """),
        code("""
        display(family_coverage)
        display(feature_inventory)
        save_table(family_coverage, table_dir, "paired_family_coverage")
        save_table(feature_inventory, table_dir, "feature_inventory")
        """),
        md("""
        ## 4. Why the split protocol is strict

        The benchmark is optimized on **recording-disjoint** transfer. A second stress protocol leaves an entire paired family unseen. This notebook makes that separation explicit before any performance claims are shown.
        """),
        code("""
        bundle = build_dataset_bundle(with_context=False)
        split = recording_disjoint_split(bundle.paired_labeled_df, test_rows_target=100, random_state=42)
        lofo_rows = []
        for family_name in sorted(bundle.paired_labeled_df["study_set"].dropna().unique()):
            family_split = leave_one_family_out(bundle.paired_labeled_df, family=family_name)
            lofo_rows.append({
                "held_out_family": family_name,
                "train_rows": len(family_split.train_df),
                "test_rows": len(family_split.test_df),
                "train_recordings": family_split.train_df["recording_key"].nunique(),
                "test_recordings": family_split.test_df["recording_key"].nunique(),
            })
        lofo_table = pd.DataFrame(lofo_rows).sort_values("held_out_family").reset_index(drop=True)
        protocol_summary = pd.DataFrame([
            {
                "protocol": "recording_disjoint_main",
                "train_rows": len(split.train_df),
                "test_rows": len(split.test_df),
                "train_recordings": split.train_df["recording_key"].nunique(),
                "test_recordings": split.test_df["recording_key"].nunique(),
            }
        ])
        display(protocol_summary)
        display(lofo_table)
        save_table(protocol_summary, table_dir, "recording_disjoint_protocol_summary")
        save_table(lofo_table, table_dir, "family_held_out_protocol_summary")
        """),
        code("""
        display(Markdown(
            f\"\"\"
        ## Key takeaways

        - The thesis package contains **{partition_table.loc[partition_table['subset'] == 'full', 'rows'].iloc[0]:,} rows** in total.
        - Only **{partition_table.loc[partition_table['subset'] == 'paired_labeled', 'rows'].iloc[0]} paired labeled rows** are available for direct supervised adaptation.
        - The benchmark therefore depends on transfer and conservative use of **{partition_table.loc[partition_table['subset'] == 'paired_unlabeled', 'rows'].iloc[0]:,} paired unlabeled rows**.
        - `fmiss` has the strongest missing-label burden, which already suggests that `fmiss` will be scientifically harder than `fpos`.
        \"\"\"
        ))
        """),
    ])


def notebook_02():
    stem = "02_domain_shift_hybrid_vs_paired"
    return new_notebook(cells=[
        md("""
        # 02 Domain Shift: Hybrid Vs Paired

        This notebook shows why naive transfer was never going to be enough. The central question is not whether the model is powerful, but whether the source and target domains are even close enough for direct transfer to work.

        **Questions answered here**
        - How separable are hybrid and paired units?
        - Does context make prediction easier while also making domains more separable?
        - Which feature groups carry the strongest hybrid-to-paired shift?
        """),
        setup_code(f"""
        fig_dir, table_dir = notebook_output_dirs("{stem}")
        live_bundle = get_domain_shift_bundle()
        frozen_domain_summary = pd.read_csv(ROOT / "data/frozen_inputs/domain_shift/domain_classifier_summary.csv")
        frozen_feature_group = pd.read_csv(ROOT / "data/frozen_inputs/domain_shift/feature_group_summary.csv")
        """),
        md("""
        ## 1. Domain separability

        The benchmark result that matters here is simple: if a lightweight classifier can tell hybrid from paired almost perfectly, then naive transfer is structurally unsafe. Context usually helps the prediction task, but it also increases domain identifiability.
        """),
        code("""
        display(frozen_domain_summary)
        save_table(frozen_domain_summary, table_dir, "domain_classifier_summary")
        fig, _ = plot_domain_shift_separability(live_bundle.domain_summary)
        save_figure(fig, fig_dir, "domain_separability")
        fig
        """),
        md("""
        ## 2. Which feature groups shift the most?

        We want to know whether the shift lives only in broad recording context, or whether unit-level morphology is also shifted. The answer is: both matter, but context blocks and waveform-related features are especially strong.
        """),
        code("""
        display(frozen_feature_group.head(12))
        display(live_bundle.feature_shift.head(20))
        save_table(frozen_feature_group, table_dir, "feature_group_shift_summary")
        save_table(live_bundle.feature_shift.head(50), table_dir, "feature_shift_summary_top50")
        fig, _ = plot_top_feature_shift(live_bundle.feature_shift, top_n=20)
        save_figure(fig, fig_dir, "top_feature_shift")
        fig
        """),
        md("""
        ## 3. PCA structure

        PCA does not prove transfer failure by itself, but it makes the geometry visible. Hybrid and paired units occupy different regions, and the paired families are not collapsed onto one compact manifold.
        """),
        code("""
        fig, _ = plot_pca_projection(live_bundle.pca_2d, color_col="dataset_type")
        save_figure(fig, fig_dir, "pca_2d_by_domain")
        fig
        """),
        code("""
        paired_only = live_bundle.pca_2d[live_bundle.pca_2d["dataset_type"] == "paired"].copy()
        centroids = (
            paired_only.groupby("study_set", as_index=False)
            .agg(pc1_mean=("pc1", "mean"), pc2_mean=("pc2", "mean"), n_rows=("recording_key", "size"), n_recordings=("recording_key", "nunique"))
            .sort_values("study_set")
            .reset_index(drop=True)
        )
        display(centroids)
        save_table(centroids, table_dir, "paired_family_pca_centroids")
        fig, _ = plot_pca_projection(paired_only, color_col="study_set")
        save_figure(fig, fig_dir, "pca_2d_paired_by_family")
        fig
        """),
        code("""
        fig, _ = plot_pca_projection_3d(live_bundle.pca_3d, color_col="dataset_type")
        save_figure(fig, fig_dir, "pca_3d_by_domain")
        fig
        """),
        code("""
        unit_auc = float(frozen_domain_summary.query("feature_set == 'unit'").iloc[0]["cv_auc_mean"])
        ctx_auc = float(frozen_domain_summary.query("feature_set == 'full_context'").iloc[0]["cv_auc_mean"])
        top_group = frozen_feature_group.sort_values("mean_abs_smd", ascending=False).iloc[0]["feature_group"]
        display(Markdown(
            f\"\"\"
        ## Key takeaways

        - Hybrid vs paired separation is already very high on unit features (**AUC {unit_auc:.3f}**).
        - With full context, the domain classifier becomes essentially perfect (**AUC {ctx_auc:.3f}**).
        - The strongest shifted feature block in the frozen audit is **{top_group}**, which helps explain why context is both useful and potentially shortcut-prone.
        - This notebook motivates the rest of the thesis: the task is not “train a stronger regressor”, it is “transfer under severe dataset shift”.
        \"\"\"
        ))
        """),
    ])


def notebook_03():
    stem = "03_paired_family_structure"
    return new_notebook(cells=[
        md("""
        # 03 Paired Family Structure

        Once the thesis established that hybrid and paired domains are far apart, the next question became: is the paired target domain itself homogeneous? The answer is no, and this notebook explains why that matters.

        **Questions answered here**
        - How different are the paired families from one another?
        - What happens when one family is fully unseen during training?
        - Why do some families transfer well while others break?
        """),
        setup_code(f"""
        fig_dir, table_dir = notebook_output_dirs("{stem}")
        domain_bundle = get_domain_shift_bundle()
        limitations = get_limitations_bundle()
        family_coverage = build_family_coverage_table()
        """),
        md("""
        ## 1. Family coverage inside the paired domain

        The paired benchmark is not just small; it is uneven. Some families contribute many more units or recordings than others, which already creates a robustness challenge before any model is trained.
        """),
        code("""
        display(family_coverage)
        save_table(family_coverage, table_dir, "paired_family_coverage")
        """),
        md("""
        ## 2. Geometry of the paired families

        The paired-only PCA shows whether families occupy distinct regions or overlap cleanly. The important point is not perfect separation, but visible heterogeneity.
        """),
        code("""
        paired_only = domain_bundle.pca_2d[domain_bundle.pca_2d["dataset_type"] == "paired"].copy()
        save_table(paired_only, table_dir, "paired_pca_points")
        fig, _ = plot_pca_projection(paired_only, color_col="study_set")
        save_figure(fig, fig_dir, "paired_family_pca_2d")
        fig
        """),
        md("""
        ## 3. Leave-one-family-out stress test

        The main benchmark is still recording-disjoint. But this stress test asks a harsher question: what happens when an entire paired family is unseen? The answer shows which families behave like bridge domains and which behave like island domains.
        """),
        code("""
        display(limitations.lofo_aggregate)
        display(limitations.lofo_family)
        save_table(limitations.lofo_aggregate, table_dir, "leave_one_family_aggregate")
        save_table(limitations.lofo_family, table_dir, "leave_one_family_by_family")
        fig, _, _ = plot_group_metric(limitations.lofo_family, group_col="held_out_family", metric="r2", title="Leave-one-family-out `fpos` R² by held-out family")
        save_figure(fig, fig_dir, "leave_one_family_r2_by_family")
        fig
        """),
        md("""
        ## 4. Why some families improve and others crash

        The interesting behavior in the leave-one-family-out benchmark is that some held-out families remain strong while others collapse. That happens because each held-out family defines a different transfer problem: some are reconstructible from the remaining families, others are not.
        """),
        code("""
        pivot = limitations.lofo_family.pivot_table(index="held_out_family", columns="variant_id", values="r2")
        pivot["waveform_minus_reference"] = pivot["wf_embed_anchor_stack_xgboost"] - pivot["reference_anchor_stack_xgboost"]
        gap_table = pivot.reset_index().sort_values("waveform_minus_reference")
        display(gap_table)
        save_table(gap_table, table_dir, "leave_one_family_gap_table")
        """),
        code("""
        display(Markdown(
            \"\"\"
        ## Key takeaways

        - The paired domain is not one clean target domain; it is a mixture of families with different transfer behavior.
        - Leave-one-family-out is therefore a real robustness test, not just a stricter version of the main benchmark.
        - The waveform-enhanced `fpos` winner is not uniformly best under unseen-family transfer, which is why the thesis reports both optimization and robustness protocols.
        \"\"\"
        ))
        """),
    ])


def notebook_04():
    stem = "04_fpos_model_progression"
    return new_notebook(cells=[
        md("""
        # 04 `fpos` Model Progression

        This notebook is the main positive result of the thesis. It reconstructs the `fpos` path from weak transfer baselines to the final waveform-enhanced, trust-aware, family-balanced winner.

        **Questions answered here**
        - How much does each modeling wave improve over the simple transfer baseline?
        - Which ablation blocks mattered most?
        - Were the gains broad across recordings and families, or concentrated in a few easy regimes?
        """),
        setup_code(f"""
        fig_dir, table_dir = notebook_output_dirs("{stem}")
        benchmark = build_benchmark_progress_table("fpos")
        ablation_winners = build_ablation_winner_table("fpos")
        wave_ids = ["fpos_paired_context", "fpos_context_stack", "fpos_pre_waveform_unlabeled", "fpos_family_balanced", "fpos_waveform_winner"]
        waves = build_main_wave_summary("fpos", wave_ids)
        winner_diag = build_prediction_diagnostics("fpos_waveform_winner", "fpos")
        limitations = get_limitations_bundle()
        """),
        md("""
        ## 1. Benchmark ladder

        The right baseline for the thesis is not the dummy model; it is the **basic hybrid+paired XGBoost transfer baseline**. Everything after that should be interpreted as an incremental answer to a specific scientific problem.
        """),
        code("""
        display(benchmark[["label", "mae", "rmse", "r2", "bias", "calibration_slope", "delta_r2_vs_basic_transfer", "delta_mae_vs_basic_transfer", "notes"]])
        save_table(benchmark, table_dir, "fpos_benchmark_ladder")
        fig, _ = plot_benchmark_metric(benchmark, metric="r2", title="`fpos` benchmark ladder by R²")
        save_figure(fig, fig_dir, "fpos_benchmark_r2")
        fig
        """),
        code("""
        fig, _ = plot_benchmark_metric(benchmark, metric="mae", title="`fpos` benchmark ladder by MAE")
        save_figure(fig, fig_dir, "fpos_benchmark_mae")
        fig
        """),
        code("""
        winner_row = benchmark.sort_values("r2", ascending=False).iloc[0]
        simple_row = benchmark.loc[benchmark["recipe_id"] == "fpos_hybrid_plus_paired"].iloc[0]
        display(Markdown(
            f\"\"\"
        ## Main progression result

        - The simple hybrid+paired baseline reaches **R² {simple_row['r2']:.4f}**.
        - The final waveform winner reaches **R² {winner_row['r2']:.4f}**.
        - That is a gain of **{winner_row['r2'] - simple_row['r2']:+.4f} R²** over the basic transfer recipe.
        - The thesis should therefore present `fpos` as a sequence of answers: context, unlabeled target structure, robustness weighting, and waveform representation.
        \"\"\"
        ))
        """),
        md("""
        ## 2. Which ablation blocks mattered?

        Each ablation group corresponds to one scientific question. The goal is not to show every run equally, but to make the causal story of the winner understandable.
        """),
        code("""
        display(ablation_winners)
        save_table(ablation_winners, table_dir, "fpos_ablation_winners")
        """),
        code("""
        for group_name in ["source_supervision", "context", "unlabeled_paired_structure", "waveform_representation", "robustness"]:
            bundle = build_ablation_bundle("fpos", group_name, "recording_disjoint_main")
            display(Markdown(f"### {group_name.replace('_', ' ').title()}"))
            display(bundle["benchmark"][["label", "mae", "r2", "notes"]])
            save_table(bundle["benchmark"], table_dir, f"fpos_ablation_{group_name}")
        """),
        md("""
        ## 3. Family and recording behavior of the main waves

        The thesis should not claim improvement unless it can show that gains are broad. These summaries make it clear whether a new wave improved many recordings or only a handful.
        """),
        code("""
        display(waves["family"])
        display(waves["recording"].head(20))
        save_table(waves["family"], table_dir, "fpos_family_wave_summary")
        save_table(waves["recording"], table_dir, "fpos_recording_wave_summary")
        fig, _, family_pivot = plot_group_metric(waves["family"], group_col="study_set", metric="r2", title="`fpos` family R² by model wave")
        save_figure(fig, fig_dir, "fpos_family_r2_by_wave")
        save_table(family_pivot, table_dir, "fpos_family_r2_pivot")
        fig
        """),
        code("""
        recording_mean = (
            waves["recording"]
            .groupby(["label", "recording_key"], as_index=False)["r2"]
            .mean()
        )
        top_rec, bottom_rec = build_extreme_groups_table(recording_mean[recording_mean["label"] == "Waveform winner"], group_col="recording_key", metric="r2", top_n=12)
        display(top_rec)
        display(bottom_rec)
        save_table(top_rec, table_dir, "fpos_top_recordings")
        save_table(bottom_rec, table_dir, "fpos_worst_recordings")
        fig, _, rec_pivot = plot_group_metric(waves["recording"], group_col="recording_key", metric="r2", title="`fpos` recording R² by model wave")
        save_figure(fig, fig_dir, "fpos_top_recording_r2_by_wave")
        save_table(rec_pivot, table_dir, "fpos_recording_r2_pivot")
        fig
        """),
        md("""
        ## 4. Diagnostics for the final winner

        The final model is not only a score. We also want a structural read: does it track the target at all, and what robustness cost does it pay under family-unseen transfer?
        """),
        code("""
        winner_predictions = winner_diag["predictions"]
        display(winner_predictions.head(20))
        fig, _ = plot_prediction_scatter(winner_predictions, title="Best `fpos` model: observed vs predicted")
        save_figure(fig, fig_dir, "fpos_winner_prediction_scatter")
        fig
        """),
        code("""
        lofo_companion = limitations.lofo_aggregate[limitations.lofo_aggregate["variant_id"].isin(["reference_anchor_stack_xgboost", "family_balanced_strong_xgboost", "wf_embed_anchor_stack_xgboost"])].copy()
        display(lofo_companion)
        save_table(lofo_companion, table_dir, "fpos_family_held_out_companion")
        """),
        code("""
        display(Markdown(
            f\"\"\"
        ## Key takeaways

        - `fpos` improves in a staged way: **context -> unlabeled paired structure -> robustness weighting -> waveform representation**.
        - The final winner has mean absolute error **{winner_diag['mae']:.4f}** on the saved holdout predictions and a true/predicted correlation of **{winner_diag['correlation']:.3f}**.
        - The family-held-out companion table is essential: the waveform winner is the headline model for the main benchmark, but not the safest unseen-family model.
        \"\"\"
        ))
        """),
    ])


def notebook_05():
    stem = "05_fmiss_model_progression"
    return new_notebook(cells=[
        md("""
        # 05 `fmiss` Model Progression

        This notebook is the counterexample chapter. It shows that the strategy that wins on `fpos` does not simply transfer to `fmiss`, and that the best `fmiss` path stayed more context-first and more conservative.

        **Questions answered here**
        - Which `fmiss` modeling path actually worked?
        - Why did many `fpos`-inspired tricks fail to transfer?
        - How broad were the winning `fmiss` gains across families and recordings?
        """),
        setup_code(f"""
        fig_dir, table_dir = notebook_output_dirs("{stem}")
        benchmark = build_benchmark_progress_table("fmiss")
        ablation_winners = build_ablation_winner_table("fmiss")
        wave_ids = ["fmiss_paired_context", "fmiss_source_reweighted", "fmiss_embed_nommd", "fmiss_reduced_latent", "fmiss_transplanted_fpos"]
        waves = build_main_wave_summary("fmiss", wave_ids)
        winner_diag = build_prediction_diagnostics("fmiss_reduced_latent", "fmiss")
        transplant_diag = build_prediction_diagnostics("fmiss_transplanted_fpos", "fmiss")
        """),
        md("""
        ## 1. Benchmark ladder

        The `fmiss` ladder should be read differently from `fpos`. The winning path is not “add every clever adaptation trick”; it is “keep the backbone stable, use context well, and avoid aggressive unlabeled transfer”.
        """),
        code("""
        display(benchmark[["label", "mae", "rmse", "r2", "bias", "calibration_slope", "delta_r2_vs_basic_transfer", "delta_mae_vs_basic_transfer", "notes"]])
        save_table(benchmark, table_dir, "fmiss_benchmark_ladder")
        fig, _ = plot_benchmark_metric(benchmark, metric="r2", title="`fmiss` benchmark ladder by R²")
        save_figure(fig, fig_dir, "fmiss_benchmark_r2")
        fig
        """),
        code("""
        fig, _ = plot_benchmark_metric(benchmark, metric="mae", title="`fmiss` benchmark ladder by MAE")
        save_figure(fig, fig_dir, "fmiss_benchmark_mae")
        fig
        """),
        code("""
        winner_row = benchmark.sort_values("r2", ascending=False).iloc[0]
        transplant_row = benchmark.loc[benchmark["recipe_id"] == "fmiss_transplanted_fpos"].iloc[0]
        display(Markdown(
            f\"\"\"
        ## Main `fmiss` result

        - The best `fmiss` row is **{winner_row['label']}** with **R² {winner_row['r2']:.4f}**.
        - The direct transplant of the best `fpos` recipe reaches only **R² {transplant_row['r2']:.4f}**.
        - This is direct evidence that `fpos` and `fmiss` are distinct transfer problems, not just different targets on the same pipeline.
        \"\"\"
        ))
        """),
        md("""
        ## 2. `fmiss` ablation story

        The ablations here are organized around `fmiss`-specific scientific questions: source target choice, context, transfer family, unlabeled paired use, and representation/filtering.
        """),
        code("""
        display(ablation_winners)
        save_table(ablation_winners, table_dir, "fmiss_ablation_winners")
        for group_name in ["source_target_choice", "context", "transfer_family", "unlabeled_paired_use", "representation_filtering"]:
            bundle = build_ablation_bundle("fmiss", group_name, "recording_disjoint_main")
            display(Markdown(f"### {group_name.replace('_', ' ').title()}"))
            display(bundle["benchmark"][["label", "mae", "r2", "notes"]])
            save_table(bundle["benchmark"], table_dir, f"fmiss_ablation_{group_name}")
        """),
        md("""
        ## 3. Family and recording behavior

        A good `fmiss` story is not just the final best row. The thesis should show where the context-first winner helps and where the transplanted `fpos` recipe clearly misaligns.
        """),
        code("""
        display(waves["family"])
        display(waves["recording"].head(20))
        save_table(waves["family"], table_dir, "fmiss_family_wave_summary")
        save_table(waves["recording"], table_dir, "fmiss_recording_wave_summary")
        fig, _, family_pivot = plot_group_metric(waves["family"], group_col="study_set", metric="r2", title="`fmiss` family R² by model wave")
        save_figure(fig, fig_dir, "fmiss_family_r2_by_wave")
        save_table(family_pivot, table_dir, "fmiss_family_r2_pivot")
        fig
        """),
        code("""
        recording_mean = (
            waves["recording"]
            .groupby(["label", "recording_key"], as_index=False)["r2"]
            .mean()
        )
        top_rec, bottom_rec = build_extreme_groups_table(recording_mean[recording_mean["label"] == "Reduced-latent winner"], group_col="recording_key", metric="r2", top_n=12)
        display(top_rec)
        display(bottom_rec)
        save_table(top_rec, table_dir, "fmiss_top_recordings")
        save_table(bottom_rec, table_dir, "fmiss_worst_recordings")
        fig, _, rec_pivot = plot_group_metric(waves["recording"], group_col="recording_key", metric="r2", title="`fmiss` recording R² by model wave")
        save_figure(fig, fig_dir, "fmiss_top_recording_r2_by_wave")
        save_table(rec_pivot, table_dir, "fmiss_recording_r2_pivot")
        fig
        """),
        md("""
        ## 4. Dedicated winner vs transplanted `fpos` recipe

        This is the structural comparison that matters most for the asymmetry claim.
        """),
        code("""
        transplant_compare = pd.DataFrame([
            {"model": "Dedicated `fmiss` winner", "mae": winner_diag["mae"], "bias": winner_diag["bias"], "correlation": winner_diag["correlation"]},
            {"model": "Transplanted `fpos` recipe", "mae": transplant_diag["mae"], "bias": transplant_diag["bias"], "correlation": transplant_diag["correlation"]},
        ])
        display(transplant_compare)
        save_table(transplant_compare, table_dir, "fmiss_transplant_comparison")
        fig, _ = plot_prediction_scatter(winner_diag["predictions"], title="Best `fmiss` model: observed vs predicted")
        save_figure(fig, fig_dir, "fmiss_winner_prediction_scatter")
        fig
        """),
        code("""
        display(Markdown(
            f\"\"\"
        ## Key takeaways

        - The best `fmiss` path remained **context-first** and relatively conservative.
        - The dedicated winner reaches correlation **{winner_diag['correlation']:.3f}** on the saved holdout predictions.
        - The transplanted `fpos` recipe is informative precisely because it fails: it demonstrates that unlabeled-paired + waveform tricks were not universally transferable.
        \"\"\"
        ))
        """),
    ])


def notebook_06():
    stem = "06_fpos_vs_fmiss_xai"
    return new_notebook(cells=[
        md("""
        # 06 `fpos` Vs `fmiss`: XAI And Target Asymmetry

        This notebook confronts the two targets directly. The goal is not only to show that their winners differ, but to show why the thesis treats them as different scientific problems.

        **Questions answered here**
        - Which features dominate the best `fpos` model?
        - How do family-level behaviors differ between the best `fpos` and best `fmiss` lines?
        - Where is the strongest evidence that the two targets should not share one final recipe?
        """),
        setup_code(f"""
        fig_dir, table_dir = notebook_output_dirs("{stem}")
        shap_df = load_fpos_shap_importance()
        shap_group_df = build_shap_group_summary()
        fpos_family = build_family_summary("fpos_waveform_winner", "fpos", "recording_disjoint_main")
        fmiss_family = build_family_summary("fmiss_reduced_latent", "fmiss", "recording_disjoint_main")
        asymmetry = build_target_asymmetry_table()
        fpos_progress = build_benchmark_progress_table("fpos")
        fmiss_progress = build_benchmark_progress_table("fmiss")
        """),
        md("""
        ## 1. What drives the best `fpos` model?

        The SHAP summary is useful because it tells us whether the winner is a black box or a structured correction model. The answer from earlier analysis was: a strong transferred prior plus waveform/amplitude/ISI corrections.
        """),
        code("""
        display(shap_df.head(20))
        display(shap_group_df)
        save_table(shap_df, table_dir, "fpos_shap_importance")
        save_table(shap_group_df, table_dir, "fpos_shap_group_summary")
        fig, _ = plot_fpos_shap_top(shap_df, top_n=15)
        save_figure(fig, fig_dir, "fpos_shap_top")
        fig
        """),
        md("""
        ## 2. Family behavior side by side

        The family tables below make the asymmetry concrete. The question is not only which model is better overall, but which families reward or punish each target differently.
        """),
        code("""
        display(fpos_family)
        display(fmiss_family)
        display(asymmetry)
        save_table(fpos_family, table_dir, "fpos_family_summary")
        save_table(fmiss_family, table_dir, "fmiss_family_summary")
        save_table(asymmetry, table_dir, "fpos_vs_fmiss_family_comparison")
        """),
        code("""
        fig, _, _ = plot_group_metric(fpos_family, group_col="study_set", metric="r2", title="Best `fpos` family behavior")
        save_figure(fig, fig_dir, "fpos_family_behavior")
        fig
        """),
        code("""
        fig, _, _ = plot_group_metric(fmiss_family, group_col="study_set", metric="r2", title="Best `fmiss` family behavior")
        save_figure(fig, fig_dir, "fmiss_family_behavior")
        fig
        """),
        code("""
        asymmetry["r2_gap_fpos_minus_fmiss"] = asymmetry["fpos_r2"] - asymmetry["fmiss_r2"]
        display(asymmetry.sort_values("r2_gap_fpos_minus_fmiss", ascending=False))
        fig, _ = plot_target_asymmetry(asymmetry)
        save_figure(fig, fig_dir, "target_asymmetry_combined")
        fig
        """),
        md("""
        ## 3. Benchmark-level asymmetry

        This table compresses the whole thesis finding into one summary: `fpos` gained much more from the later target-aware structure than `fmiss` did.
        """),
        code("""
        basic_fpos = fpos_progress.loc[fpos_progress["recipe_id"] == "fpos_hybrid_plus_paired"].iloc[0]
        final_fpos = fpos_progress.sort_values("r2", ascending=False).iloc[0]
        basic_fmiss = fmiss_progress.loc[fmiss_progress["recipe_id"] == "fmiss_hybrid_residual"].iloc[0]
        final_fmiss = fmiss_progress.sort_values("r2", ascending=False).iloc[0]
        target_story = pd.DataFrame([
            {"target": "fpos", "basic_transfer_r2": basic_fpos["r2"], "final_r2": final_fpos["r2"], "gain_r2": final_fpos["r2"] - basic_fpos["r2"]},
            {"target": "fmiss", "basic_transfer_r2": basic_fmiss["r2"], "final_r2": final_fmiss["r2"], "gain_r2": final_fmiss["r2"] - basic_fmiss["r2"]},
        ])
        display(target_story)
        save_table(target_story, table_dir, "target_progress_summary")
        """),
        code("""
        display(Markdown(
            \"\"\"
        ## Key takeaways

        - The best `fpos` model is heavily shaped by transferred prior information plus waveform/amplitude corrections.
        - The best `fmiss` model is more conservative and context-first.
        - Family-level asymmetry is not noise; it is evidence that `fpos` and `fmiss` respond to different kinds of transfer signal.
        - This notebook is the strongest support for using separate deployment models for the two targets.
        \"\"\"
        ))
        """),
    ])


def notebook_07():
    stem = "07_limitations_and_data_budget"
    return new_notebook(cells=[
        md("""
        # 07 Limitations And Data Budget

        This notebook answers the final practical question: how much headroom is likely left, and where does the current thesis package still remain scientifically limited?

        **Questions answered here**
        - Are we still on the steep part of the paired-label curve?
        - How much performance is lost under family-unseen transfer?
        - Which limitations are methodological, and which are data-limited?
        """),
        setup_code(f"""
        fig_dir, table_dir = notebook_output_dirs("{stem}")
        bundle = get_limitations_bundle()
        fpos_progress = build_benchmark_progress_table("fpos")
        fmiss_progress = build_benchmark_progress_table("fmiss")
        """),
        md("""
        ## 1. Label-budget evidence

        The thesis should make the small-data regime visible. If the curve is still rising steeply, then the current benchmark is probably data-limited rather than fully saturated.
        """),
        code("""
        display(bundle.label_budget)
        save_table(bundle.label_budget, table_dir, "label_budget_summary")
        fig, _ = plot_label_budget(bundle.label_budget, target="fpos")
        save_figure(fig, fig_dir, "label_budget_fpos")
        fig
        """),
        code("""
        fig, _ = plot_label_budget(bundle.label_budget, target="fmiss")
        save_figure(fig, fig_dir, "label_budget_fmiss")
        fig
        """),
        md("""
        ## 2. Family-held-out robustness

        The main benchmark optimizes recording-disjoint transfer. This section reports the explicit penalty for requiring generalization to an unseen paired family.
        """),
        code("""
        display(bundle.lofo_aggregate)
        display(bundle.lofo_family)
        save_table(bundle.lofo_aggregate, table_dir, "leave_one_family_aggregate")
        save_table(bundle.lofo_family, table_dir, "leave_one_family_by_family")
        fig, _, _ = plot_group_metric(bundle.lofo_family, group_col="held_out_family", metric="r2", title="Leave-one-family-out `fpos` R²")
        save_figure(fig, fig_dir, "leave_one_family_r2")
        fig
        """),
        md("""
        ## 3. What the current results imply about headroom

        The table below puts the current best results next to their simpler transfer baselines. It is not a theoretical upper bound, but it does show that there is still room between “basic transfer” and “current best”.
        """),
        code("""
        basic_fpos = fpos_progress.loc[fpos_progress["recipe_id"] == "fpos_hybrid_plus_paired"].iloc[0]
        final_fpos = fpos_progress.sort_values("r2", ascending=False).iloc[0]
        basic_fmiss = fmiss_progress.loc[fmiss_progress["recipe_id"] == "fmiss_hybrid_residual"].iloc[0]
        final_fmiss = fmiss_progress.sort_values("r2", ascending=False).iloc[0]
        headroom = pd.DataFrame([
            {"target": "fpos", "basic_transfer_r2": basic_fpos["r2"], "current_best_r2": final_fpos["r2"], "gain_r2": final_fpos["r2"] - basic_fpos["r2"]},
            {"target": "fmiss", "basic_transfer_r2": basic_fmiss["r2"], "current_best_r2": final_fmiss["r2"], "gain_r2": final_fmiss["r2"] - basic_fmiss["r2"]},
        ])
        display(headroom)
        save_table(headroom, table_dir, "headroom_summary")
        """),
        code("""
        display(Markdown(
            \"\"\"
        ## Limitation summary

        - The paired labeled pool is still small enough that label-budget curves matter.
        - Leave-one-family-out performance remains clearly worse than the main benchmark, so family-unseen robustness is not solved.
        - The strongest positive result is currently clearer for `fpos` than for `fmiss`.
        - The most plausible next gains should come from richer spatial/template/drift information, not from more generic backend tinkering.
        \"\"\"
        ))
        """),
    ])


def notebook_08():
    stem = "08_appendix_additional_ablations"
    return new_notebook(cells=[
        md("""
        # 08 Appendix Additional Ablations

        This notebook collects the non-winning branches that still matter scientifically. They should stay out of the main text, but they are important for showing that the final thesis claims are not based on a shallow search.
        """),
        setup_code("""
        fig_dir, table_dir = notebook_output_dirs("{stem}")
        appendix_files = (
            build_recipe_inventory()
            .loc[:, ["runner_key", "implementation_module", "recipe_id", "result_key"]]
            .drop_duplicates()
            .sort_values(["runner_key", "recipe_id"])
            .reset_index(drop=True)
        )
        fpos_groups = get_recipe_groups("fpos")
        fmiss_groups = get_recipe_groups("fmiss")
        """.replace("{stem}", stem)),
        md("""
        ## 1. Research module inventory

        This is the live recipe-to-module inventory for the thesis modeling package.
        """),
        code("""
        display(appendix_files)
        save_table(appendix_files, table_dir, "research_module_inventory")
        """),
        md("""
        ## 2. Ablation grouping reference

        The appendix should still be organized by scientific question, not by raw script name. These group maps are the compact reference for the appendix chapter structure.
        """),
        code("""
        appendix_fpos = pd.DataFrame([(k, ', '.join(v)) for k, v in fpos_groups.items()], columns=["ablation_group", "recipe_ids"])
        appendix_fmiss = pd.DataFrame([(k, ', '.join(v)) for k, v in fmiss_groups.items()], columns=["ablation_group", "recipe_ids"])
        display(appendix_fpos)
        display(appendix_fmiss)
        save_table(appendix_fpos, table_dir, "appendix_fpos_recipe_groups")
        save_table(appendix_fmiss, table_dir, "appendix_fmiss_recipe_groups")
        """),
        md("""
        ## 3. What belongs in appendix rather than main text

        The scientific rule is simple:

        - main text keeps the strongest benchmark ladder, target asymmetry, and robustness story
        - appendix keeps negative branches, tuning branches, transport/synthetic branches, and exploratory dead ends that still matter for scientific completeness
        """),
        code("""
        display(Markdown(
            \"\"\"
        **Appendix families to emphasize**

        - synthetic augmentation and paired-conditioned transport
        - residual correction and manifold-context variants
        - waveform enrichment and multiscale waveform variants
        - Bayesian tuning and meta/bagging branches
        - tree interaction and paper-inspired transfer branches
        \"\"\"
        ))
        """),
    ])


NOTEBOOK_BUILDERS = {
    "01_data_extraction_and_audit.ipynb": notebook_01,
    "02_domain_shift_hybrid_vs_paired.ipynb": notebook_02,
    "03_paired_family_structure.ipynb": notebook_03,
    "04_fpos_model_progression.ipynb": notebook_04,
    "05_fmiss_model_progression.ipynb": notebook_05,
    "06_fpos_vs_fmiss_xai.ipynb": notebook_06,
    "07_limitations_and_data_budget.ipynb": notebook_07,
    "08_appendix_additional_ablations.ipynb": notebook_08,
}


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in NOTEBOOK_BUILDERS.items():
        nb = builder()
        out = NOTEBOOK_DIR / name
        nbformat.write(nb, out)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
