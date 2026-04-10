#!/usr/bin/env python3
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


THESIS_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = THESIS_ROOT / "notebooks"
SRC_DIR = THESIS_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qc_thesis.highlight_figures import build_highlight_figures
from qc_thesis.paths import PRE_WAVEFORM_LABEL, REDUCED_LATENT_CONTEXT_LABEL, WAVEFORM_AUGMENTED_LABEL


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
from IPython.display import Image, Markdown, display

def _find_thesis_root():
    cwd = Path.cwd().resolve()
    direct = [cwd, *cwd.parents]
    nested = [candidate / "thesis" for candidate in direct]
    for candidate in [*direct, *nested]:
        if (
            (candidate / "src" / "qc_thesis" / "__init__.py").exists()
            and (candidate / "README.md").exists()
            and (candidate / "notebooks").exists()
        ):
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


def show_saved_figure(path, caption=None):
    figure_path = Path(path)
    if not figure_path.is_absolute():
        figure_path = (ROOT / figure_path).resolve()
    if not figure_path.exists():
        display(Markdown(f"_Missing figure: `{figure_path}`_"))
        return
    if caption:
        display(Markdown(caption))
    display(Image(filename=str(figure_path)))
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
        recipe_registry = build_recipe_inventory()
        manifest = json.loads((ROOT / "data/frozen_inputs/manifest.json").read_text())
        """),
        md("""
        ## 1. Packaged assets

        The thesis package contains both the exploratory raw parquet and the generated benchmark assets. The point is not only to show the final tables, but to show the entire route from extraction to evaluation.
        """),
        code("""
        display(pd.DataFrame(dataset_summary.items(), columns=["packaged_item", "value"]))
        display(pd.DataFrame(computed_summary.items(), columns=["recomputed_item", "value"]))
        display(recipe_registry.loc[:, ["recipe_id", "target", "runner_key", "main_text"]])
        display(pd.DataFrame(manifest.items(), columns=["artifact_group", "path"]))
        save_table(pd.DataFrame(dataset_summary.items(), columns=["item", "value"]), table_dir, "packaged_dataset_summary")
        save_table(pd.DataFrame(computed_summary.items(), columns=["item", "value"]), table_dir, "recomputed_dataset_summary")
        save_table(recipe_registry, table_dir, "recipe_registry")
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
        md("""
        ---
        ### Thesis highlight — Domain shift story

        _Provenance._ This figure is a thesis synthesis artifact regenerated by `scripts/make_thesis_highlight_figures.py` from frozen summary tables. It deliberately combines three views that would otherwise live in separate exploratory figures:

        - domain-classifier AUC on unit vs full-context feature sets
        - full-context feature-group shift magnitudes
        - paired-family PCA centroids

        The point is not that one feature group alone “explains” the transfer gap. The narrower thesis-safe claim is that **the hybrid→paired gap is severe and the paired target domain is internally structured rather than one compact regime**.
        """),
        code("""
        show_saved_figure(
            ROOT / "figures" / "09_thesis_highlight_figures" / "domain_shift_story.png",
            "This synthesis figure is regenerated from frozen summary tables before the notebooks are rebuilt, so the notebook does not depend on a hand-edited PNG.",
        )
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
        code("""
        show_saved_figure(
            fig_dir / "recording_vs_family_disjoint_r2_gap.png",
            "This summary compares the best recording-disjoint `fpos` waveform-augmented stack with its leave-one-family-out performance, which is the cleanest way to show that family-unseen generalization is a stricter transfer problem than the main benchmark.",
        )
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
        - The waveform-augmented `fpos` stack is not uniformly best under unseen-family transfer, which is why the thesis reports both optimization and robustness protocols.
        \"\"\"
        ))
        """),
        md("""
        ---
        ### Thesis highlights — Family structure preference map and protocol tradeoff

        _Provenance._ These are thesis synthesis figures regenerated from frozen tables rather than notebook-local plotting code.

        - **Family structure preference map:** The left panel keeps the family-by-model preference heatmap. The right panel adds actual family structure variables from the local package, so the figure now shows not only _which_ family prefers which recipe, but also _how different regimes sit relative to one another_.
        - **Protocol tradeoff:** This figure is sourced directly from the curated `fpos_protocol_regime_preference.csv` table, so the notebook is displaying the same protocol comparison that the thesis cites.
        """),
        code("""
        show_saved_figure(
            ROOT / "figures" / "09_thesis_highlight_figures" / "family_structure_preference_map.png",
            "This synthesis figure combines family-level performance preference with structural context from the local package.",
        )
        show_saved_figure(
            ROOT / "figures" / "09_thesis_highlight_figures" / "protocol_tradeoff.png",
            "The protocol figure is sourced from the frozen curated comparison table rather than reconstructed ad hoc inside the notebook.",
        )
        """),
    ])


def notebook_04():
    stem = "04_fpos_model_progression"
    return new_notebook(cells=[
        md("""
        # 04 `fpos` Model Progression

        This notebook started as the main positive result of the thesis, but the **20-seed recording-disjoint sweep (`42` to `61`)** changes the tone. `fpos` still has several strong transfer recipes, yet the aggregate result is no longer “one universal best recipe”.

        Unless otherwise noted, the benchmark tables below report those 20 seeds and the diagnostics pool the saved holdout predictions across the same splits.

        **Questions answered here**
        - Which `fpos` recipes actually survive the 20-seed sweep?
        - Which split compositions and paired families favor which model?
        - Which ablation blocks mattered most?
        - Which structural differences across the paired datasets help explain those preferences?
        """),
        setup_code(f"""
        fig_dir, table_dir = notebook_output_dirs("{stem}")
        benchmark = build_benchmark_progress_table("fpos", scope="core")
        ablation_winners = build_ablation_winner_table("fpos")
        wave_ids = ["fpos_paired_context", "fpos_context_stack", "fpos_pre_waveform_unlabeled", "fpos_waveform_winner"]
        family_pref_ids = ["fpos_paired_raw", *wave_ids]
        seed_ids = ["fpos_paired_raw", *wave_ids]
        waves = build_main_wave_summary("fpos", wave_ids)
        seed_story = build_split_regime_bundle("fpos", seed_ids, split_profile_recipe_id="fpos_waveform_winner")
        family_preference = build_fpos_lofo_main_family_preference_table()
        family_context = build_family_structure_context_table("fpos", wave_ids)
        winner_diag = build_prediction_diagnostics("fpos_waveform_winner", "fpos")
        limitations = get_limitations_bundle()
        """),
        md("""
        ## 1. Benchmark ladder

        The main-text `fpos` ladder is intentionally compressed to the core comparison set: baseline floor, target-only reference, source-only transfer, context-first stacking, unlabeled structure, and waveform augmentation. It should be read as a methodological progression rather than as an exhaustive recipe catalog.
        """),
        code("""
        display(benchmark[["label", "mae", "rmse", "r2", "bias", "calibration_slope", "delta_r2_vs_basic_transfer", "delta_mae_vs_basic_transfer", "notes"]])
        save_table(benchmark, table_dir, "fpos_benchmark_ladder")
        fig, _ = plot_benchmark_metric(benchmark, metric="r2", title="`fpos` benchmark ladder by R² (20 recording-disjoint seeds)")
        save_figure(fig, fig_dir, "fpos_benchmark_r2")
        fig
        """),
        code("""
        fig, _ = plot_benchmark_metric(benchmark, metric="mae", title="`fpos` benchmark ladder by MAE (20 recording-disjoint seeds)")
        save_figure(fig, fig_dir, "fpos_benchmark_mae")
        fig
        """),
        code("""
        milestone = pd.DataFrame([
            {"stage": "1  sanity floor", "model": "Dummy paired mean", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fpos_dummy", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fpos_dummy", "mae"].iloc[0])},
            {"stage": "2  paired baseline", "model": "Paired-only raw", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fpos_paired_raw", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fpos_paired_raw", "mae"].iloc[0])},
            {"stage": "3  source transfer", "model": "Hybrid-only XGBoost", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fpos_hybrid_only", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fpos_hybrid_only", "mae"].iloc[0])},
            {"stage": "4  paired context", "model": "Paired-only context", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fpos_paired_context", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fpos_paired_context", "mae"].iloc[0])},
            {"stage": "5  context stacking", "model": "Context-first source stack", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fpos_context_stack", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fpos_context_stack", "mae"].iloc[0])},
            {"stage": "6  + unlabeled paired", "model": "Trust-filtered unlabeled stack", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fpos_pre_waveform_unlabeled", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fpos_pre_waveform_unlabeled", "mae"].iloc[0])},
            {"stage": "7  + waveform embedding", "model": "Waveform-augmented stack", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fpos_waveform_winner", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fpos_waveform_winner", "mae"].iloc[0])},
        ])
        display(milestone)
        save_table(milestone, table_dir, "fpos_milestone_trajectory")
        """),
        code("""
        show_saved_figure(
            fig_dir / "fpos_seed_stability.png",
            "These violin panels summarize the 20 recording-disjoint splits (`42` to `61`) for the retained compared `fpos` models after omitting the dummy and pure source-only baselines, so the thesis can separate one favorable split from the overall stability pattern.",
        )
        show_saved_figure(
            fig_dir / "fpos_fmiss_milestone_trajectory.png",
            "The milestone trajectory makes the asymmetry of the two targets visible: later transfer additions produce much larger gains for `fpos` than for `fmiss`.",
        )
        """),
        code("""
        top_mean_row = benchmark.sort_values(["r2", "mae"], ascending=[False, True]).iloc[0]
        source_only_row = benchmark.loc[benchmark["recipe_id"] == "fpos_hybrid_only"].iloc[0]
        best_rank_row = seed_story["summary"].iloc[0]
        waveform_rank_row = seed_story["summary"].loc[seed_story["summary"]["recipe_id"] == "fpos_waveform_winner"].iloc[0]
        display(Markdown(
            f\"\"\"
        ## Main 20-seed read

        - The source-only transfer baseline reaches **R² {source_only_row['r2']:.4f}**.
        - The best **mean R²** in the full ladder is **{top_mean_row['label']}** at **{top_mean_row['r2']:.4f}**.
        - Across the competitive late-wave recipes, the best **mean seed rank** is **{best_rank_row['label']}** (mean rank **{best_rank_row['mean_rank']:.2f}**, **{int(best_rank_row['winner_count'])}** seed wins).
        - The recipe named **`Waveform-augmented stack`** is still competitive (mean rank **{waveform_rank_row['mean_rank']:.2f}**, **{int(waveform_rank_row['winner_count'])}** seed wins), but the 20-seed sweep says the story is **regime-dependent rather than dominated by one universally best recipe**.
        \"\"\"
        ))
        """),
        md("""
        ## 2. What survived the 20-seed sweep?

        A useful robustness question is not only “who has the highest average `R²`?”, but also “who keeps showing up near the top when the holdout recordings change?”
        """),
        code("""
        display(seed_story["summary"])
        save_table(seed_story["summary"], table_dir, "fpos_seed_rank_summary")
        """),
        code("""
        import matplotlib.pyplot as plt

        rank_pivot = seed_story["rank_pivot"].copy()
        fig, ax = plt.subplots(figsize=(10, 3.8))
        im = ax.imshow(rank_pivot.to_numpy(dtype=float), cmap="YlGn_r", vmin=1, vmax=max(len(rank_pivot), 1), aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(rank_pivot.shape[1]))
        ax.set_xticklabels(rank_pivot.columns.astype(str).tolist(), rotation=0)
        ax.set_yticks(range(rank_pivot.shape[0]))
        ax.set_yticklabels(rank_pivot.index.tolist())
        ax.set_xlabel("Recording-disjoint seed")
        ax.set_title("Rank by seed across the five main compared `fpos` models\\n(20 recording-disjoint seeds, 42-61; 1 = best `R²`)", pad=10)
        for row_idx, row_label in enumerate(rank_pivot.index):
            for col_idx, col_label in enumerate(rank_pivot.columns):
                value = rank_pivot.loc[row_label, col_label]
                if pd.notna(value):
                    ax.text(col_idx, row_idx, int(value), ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Rank")
        fig.tight_layout()
        save_figure(fig, fig_dir, "fpos_seed_rank_heatmap")
        fig
        """),
        code("""
        display(seed_story["winner_table"][[
            "seed",
            "label",
            "r2",
            "mae",
            "true_mean",
            "true_std",
            "rows__PAIRED_BOYDEN",
            "rows__PAIRED_CRCNS_HC1",
            "rows__PAIRED_ENGLISH",
            "rows__PAIRED_KAMPFF",
            "rows__PAIRED_MEA64C_YGER",
        ]])
        display(seed_story["winner_profile"])
        save_table(seed_story["winner_table"], table_dir, "fpos_seed_winner_by_split")
        save_table(seed_story["winner_profile"], table_dir, "fpos_seed_winner_profile")
        """),
        code("""
        winner_profile = seed_story["winner_profile"].copy()
        share_cols = [col for col in winner_profile.columns if col.startswith("share__")]
        share_plot = winner_profile.loc[:, ["label", *share_cols]].copy()
        share_plot = share_plot.rename(columns=lambda col: col.replace("share__PAIRED_", "") if col.startswith("share__") else col)
        family_cols = [col for col in share_plot.columns if col != "label"]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        left = np.zeros(len(share_plot), dtype=float)
        palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]
        for color, col in zip(palette, family_cols):
            values = pd.to_numeric(share_plot[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ax.barh(share_plot["label"], values, left=left, label=col, color=color, alpha=0.9)
            left = left + values
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean family share in seeds won by the model")
        ax.set_title("Winning models prefer different holdout mixes", pad=10)
        ax.legend(title="Held-out family", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        save_figure(fig, fig_dir, "fpos_split_winner_family_mix")
        fig
        """),
        md("""
        ---
        ### Thesis highlights — Split regime map and raw-vs-waveform complementarity

        _Provenance._ These figures are thesis synthesis artifacts regenerated from frozen tables and aggregated seed metrics. They intentionally sit on top of the notebook analysis rather than replacing the exploratory cells above.

        - **Split regime map:** Seeds are ordered by clustered family-share composition rather than by one hand-picked family. The second panel now shows the leading-recipe `R²` together with target difficulty on the same seed order.
        - **Raw vs waveform complementarity:** This figure compares `Paired-only raw` against `Waveform-augmented stack` directly at the seed level. The claim is intentionally narrow: waveform gains are regime-dependent and modestly associated with slightly higher KAMPFF share, slightly lower ENGLISH share, and lower target `IQR` rather than with one absolute “waveform-friendly family”.
        """),
        code("""
        show_saved_figure(
            ROOT / "figures" / "09_thesis_highlight_figures" / "split_regime_map.png",
            "This synthesis figure keeps the family-composition story in notebook 04, but the ordering is now driven by clustered split composition rather than a single family share.",
        )
        show_saved_figure(
            ROOT / "figures" / "09_thesis_highlight_figures" / "raw_vs_waveform_complementarity.png",
            "This is the thesis-ready comparison between the simple paired model and the waveform-enhanced stack; it explains complementarity rather than claiming a universal waveform advantage.",
        )
        """),
        md("""
        ## 3. Which ablation blocks mattered?

        Each ablation group corresponds to one scientific question. The goal is not to show every run equally, but to make the causal story of the leading stack understandable.
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
        ## 4. Which families prefer which recipe?

        The stricter family-held-out view is where the generalization story becomes interpretable. This comparison averages `R²` across `20` leave-one-family-out seeds and is kept aligned with the five main compared `fpos` models, excluding only the dummy and pure source-only baselines.
        """),
        code("""
        display(family_preference["best"])
        display(family_preference["r2_pivot"])
        save_table(family_preference["best"], table_dir, "fpos_lofo_family_preference_best")
        save_table(family_preference["r2_pivot"].reset_index(), table_dir, "fpos_lofo_family_preference_r2")
        """),
        code("""
        import matplotlib.pyplot as plt

        family_r2 = family_preference["r2_pivot"].copy()
        fig, ax = plt.subplots(figsize=(8.4, 3.8))
        im = ax.imshow(family_r2.to_numpy(dtype=float), cmap="RdYlGn", vmin=-0.25, vmax=0.85, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(family_r2.shape[1]))
        ax.set_xticklabels(family_r2.columns.tolist(), rotation=24, ha="right")
        ax.set_yticks(range(family_r2.shape[0]))
        ax.set_yticklabels([name.replace("PAIRED_", "") for name in family_r2.index.tolist()])
        ax.set_title("Mean held-out-family `R²` across the five main compared `fpos` models\\n(20-seed leave-one-family-out aggregate)", pad=10)
        for row_idx, family_name in enumerate(family_r2.index):
            for col_idx, label in enumerate(family_r2.columns):
                value = family_r2.loc[family_name, label]
                if pd.notna(value):
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Mean R²")
        fig.tight_layout()
        save_figure(fig, fig_dir, "fpos_lofo_family_heatmap")
        fig
        """),
        code("""
        display(Markdown(
            \"\"\"
        **Online context note**

        The columns `channel_context`, `duration_context`, and `source_context` below paraphrase the paired-study descriptions from **SpikeForest / eLife 2020 Table 2**. The remaining columns are computed from the local thesis package and the pooled 20-seed outputs.
        \"\"\"
        ))
        display(family_context[[
            "study_set",
            "source_context",
            "channel_context",
            "duration_context",
            "spikeforest_recordings",
            "packaged_recordings",
            "packaged_labeled_rows",
            "packaged_unlabeled_rows",
            "packaged_recording_fraction_vs_spikeforest",
            "target_mean",
            "target_std",
            "paired_pca_distance_from_hybrid",
            "best_recording_disjoint_label",
            "best_recording_disjoint_r2",
            "best_lofo_label",
            "best_lofo_r2",
        ]])
        save_table(family_context, table_dir, "fpos_family_structure_context")
        """),
        md("""
        ## 5. Family and recording behavior of the main waves

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
        top_rec, bottom_rec = build_extreme_groups_table(recording_mean[recording_mean["label"] == WAVEFORM_AUGMENTED_LABEL], group_col="recording_key", metric="r2", top_n=12)
        focus_recordings = pd.concat([top_rec, bottom_rec], ignore_index=True)["recording_key"].drop_duplicates().tolist()
        recording_focus = waves["recording"][waves["recording"]["recording_key"].isin(focus_recordings)].copy()
        display(top_rec)
        display(bottom_rec)
        save_table(top_rec, table_dir, "fpos_top_recordings")
        save_table(bottom_rec, table_dir, "fpos_worst_recordings")
        fig, _, rec_pivot = plot_group_metric(recording_focus, group_col="recording_key", metric="r2", title="`fpos` recording R² by model wave\\n(top/bottom recordings from 20 recording-disjoint seeds)")
        save_figure(fig, fig_dir, "fpos_top_recording_r2_by_wave")
        save_table(rec_pivot, table_dir, "fpos_recording_r2_pivot")
        fig
        """),
        md("""
        ## 6. Split-specific robustness and diagnostics

        The last question is whether the recording-disjoint favorite remains the strongest choice when the split gets harsher. The leave-one-family-out companion table shows that it does not.
        """),
        code("""
        lofo_map = {
            "fpos_pre_waveform_unlabeled": "reference_anchor_stack_xgboost",
            "fpos_waveform_winner": "wf_embed_anchor_stack_xgboost",
        }
        protocol_compare = benchmark.loc[
            benchmark["recipe_id"].isin(["fpos_pre_waveform_unlabeled", "fpos_waveform_winner"]),
            ["recipe_id", "label", "r2", "mae"],
        ].rename(columns={"r2": "recording_disjoint_r2", "mae": "recording_disjoint_mae"})
        lofo_companion = limitations.lofo_aggregate.copy()
        lofo_companion["recipe_id"] = lofo_companion["variant_id"].map({value: key for key, value in lofo_map.items()})
        protocol_compare = protocol_compare.merge(
            lofo_companion.loc[:, ["recipe_id", "macro_r2", "macro_mae", "weighted_r2", "weighted_mae", "worst_family_r2"]],
            on="recipe_id",
            how="left",
        )
        protocol_compare["macro_r2_gap"] = protocol_compare["macro_r2"] - protocol_compare["recording_disjoint_r2"]
        protocol_compare["weighted_r2_gap"] = protocol_compare["weighted_r2"] - protocol_compare["recording_disjoint_r2"]
        display(protocol_compare.sort_values("recording_disjoint_r2", ascending=False))
        save_table(protocol_compare, table_dir, "fpos_protocol_regime_preference")

        waveform_protocol = pd.DataFrame([
            {
                "model": WAVEFORM_AUGMENTED_LABEL,
                "protocol": "recording_disjoint_main",
                "r2": float(protocol_compare.loc[protocol_compare["recipe_id"] == "fpos_waveform_winner", "recording_disjoint_r2"].iloc[0]),
                "mae": float(protocol_compare.loc[protocol_compare["recipe_id"] == "fpos_waveform_winner", "recording_disjoint_mae"].iloc[0]),
                "delta_r2_vs_recording": 0.0,
                "delta_mae_vs_recording": 0.0,
            },
            {
                "model": WAVEFORM_AUGMENTED_LABEL,
                "protocol": "family_disjoint_lofo",
                "r2": float(protocol_compare.loc[protocol_compare["recipe_id"] == "fpos_waveform_winner", "macro_r2"].iloc[0]),
                "mae": float(protocol_compare.loc[protocol_compare["recipe_id"] == "fpos_waveform_winner", "macro_mae"].iloc[0]),
                "delta_r2_vs_recording": float(protocol_compare.loc[protocol_compare["recipe_id"] == "fpos_waveform_winner", "macro_r2_gap"].iloc[0]),
                "delta_mae_vs_recording": float(protocol_compare.loc[protocol_compare["recipe_id"] == "fpos_waveform_winner", "macro_mae"].iloc[0] - protocol_compare.loc[protocol_compare["recipe_id"] == "fpos_waveform_winner", "recording_disjoint_mae"].iloc[0]),
            },
        ])
        save_table(waveform_protocol, table_dir, "fpos_winner_protocol_comparison")
        """),
        code("""
        import matplotlib.pyplot as plt

        protocol_plot = protocol_compare.sort_values("recording_disjoint_r2", ascending=False).reset_index(drop=True)
        x = np.arange(len(protocol_plot))
        width = 0.38
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.bar(x - width / 2, protocol_plot["recording_disjoint_r2"], width=width, label="Recording-disjoint mean R²", color="#2E6DA4")
        ax.bar(x + width / 2, protocol_plot["macro_r2"], width=width, label="Family-held-out macro R²", color="#CC6677")
        ax.set_xticks(x)
        ax.set_xticklabels(protocol_plot["label"], rotation=25, ha="right")
        ax.set_ylabel("R²")
        ax.set_title("Which `fpos` recipe is best depends on the split", pad=10)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, fig_dir, "fpos_protocol_regime_preference")
        fig
        """),
        code("""
        winner_predictions = winner_diag["predictions"]
        display(winner_predictions.head(20))
        fig, _ = plot_prediction_scatter(winner_predictions, title="Best `fpos` model: observed vs predicted")
        save_figure(fig, fig_dir, "fpos_winner_prediction_scatter")
        fig
        """),
        code("""
        lofo_companion = limitations.lofo_aggregate[limitations.lofo_aggregate["variant_id"].isin(["reference_anchor_stack_xgboost", "wf_embed_anchor_stack_xgboost"])].copy()
        display(lofo_companion)
        save_table(lofo_companion, table_dir, "fpos_family_held_out_companion")
        """),
        code("""
        best_rank_row = seed_story["summary"].iloc[0]
        display(Markdown(
            f\"\"\"
        ## Key takeaways

        - The 20-seed sweep does **not** support one universal `fpos` best recipe. The strongest late-wave recipe by mean seed rank is **{best_rank_row['label']}**, and several recipes lead on multiple individual seeds.
        - Family structure explains part of that instability: **English dominates local coverage, CRCNS_HC1 is tiny and high-variance, and the packaged subset is an uneven slice of the larger SpikeForest study sets**.
        - On the harsher family-held-out protocol, the retained comparison shifts slightly toward **`Waveform-augmented stack`**, even though **{best_rank_row['label']}** remains the stronger robust reference on the recording-disjoint benchmark.
        - The waveform-augmented recipe still tracks the target reasonably on pooled holdout predictions (**MAE {winner_diag['mae']:.4f}**, correlation **{winner_diag['correlation']:.3f}**), but it should now be presented as a **regime-specific headline model**, not as an unconditional best recipe.
        \"\"\"
        ))
        """),
    ])


def notebook_05():
    stem = "05_fmiss_model_progression"
    return new_notebook(cells=[
        md("""
        # 05 `fmiss` Model Progression

        This notebook is the counterexample chapter. It shows that the strategy that works best on `fpos` does not simply transfer to `fmiss`, and that the best `fmiss` path stayed more context-first and more conservative.

        Unless otherwise noted, the benchmark tables below report **20 recording-disjoint seeds (`42` to `61`)** and the diagnostics pool the saved holdout predictions across those splits.

        **Questions answered here**
        - Which `fmiss` modeling path actually worked?
        - Why did many `fpos`-inspired tricks fail to transfer?
        - How broad were the winning `fmiss` gains across families and recordings?
        """),
        setup_code(f"""
        fig_dir, table_dir = notebook_output_dirs("{stem}")
        benchmark = build_benchmark_progress_table("fmiss", scope="core")
        ablation_winners = build_ablation_winner_table("fmiss")
        seed_ids = ["fmiss_paired_raw", "fmiss_paired_context", "fmiss_reduced_latent"]
        seed_story = build_seed_rank_bundle("fmiss", seed_ids)
        wave_ids = ["fmiss_paired_raw", "fmiss_hybrid_only", "fmiss_paired_context", "fmiss_reduced_latent"]
        waves = build_main_wave_summary("fmiss", wave_ids)
        lofo_family = build_fmiss_lofo_main_family_preference_table()
        winner_diag = build_prediction_diagnostics("fmiss_reduced_latent", "fmiss")
        """),
        md("""
        ## 1. Benchmark ladder

        The main-text `fmiss` ladder is also compressed to the core narrative set. The point is not to show every intermediate variant, but to make the central result legible: `fmiss` rewards a more conservative context-first path and does not benefit from transplanting the strongest `fpos` waveform recipe.
        """),
        code("""
        display(benchmark[["label", "mae", "rmse", "r2", "bias", "calibration_slope", "delta_r2_vs_basic_transfer", "delta_mae_vs_basic_transfer", "notes"]])
        save_table(benchmark, table_dir, "fmiss_benchmark_ladder")
        fig, _ = plot_benchmark_metric(benchmark, metric="r2", title="`fmiss` benchmark ladder by R² (20 recording-disjoint seeds)")
        save_figure(fig, fig_dir, "fmiss_benchmark_r2")
        fig
        """),
        code("""
        fig, _ = plot_benchmark_metric(benchmark, metric="mae", title="`fmiss` benchmark ladder by MAE (20 recording-disjoint seeds)")
        save_figure(fig, fig_dir, "fmiss_benchmark_mae")
        fig
        """),
        code("""
        show_saved_figure(
            fig_dir / "fmiss_seed_stability.png",
            "The `fmiss` stability view shows why this target is the weaker transfer story: the central tendency stays much closer to break-even and the split-to-split variance is much larger.",
        )
        show_saved_figure(
            fig_dir / "fmiss_seed_rank_heatmap.png",
            "This rank heatmap complements the violin plot by showing which of the three main compared `fmiss` models actually wins each of the 20 recording-disjoint seeds.",
        )
        """),
        code("""
        display(seed_story["summary"])
        save_table(seed_story["summary"], table_dir, "fmiss_seed_rank_summary")
        save_table(seed_story["winner_rows"], table_dir, "fmiss_seed_winner_by_split")
        """),
        code("""
        import matplotlib.pyplot as plt

        rank_pivot = seed_story["rank_pivot"].copy()
        fig, ax = plt.subplots(figsize=(8.6, 2.9))
        im = ax.imshow(rank_pivot.to_numpy(dtype=float), cmap="YlGn_r", vmin=1, vmax=max(len(rank_pivot), 1), aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(rank_pivot.shape[1]))
        ax.set_xticklabels(rank_pivot.columns.astype(str).tolist(), rotation=0)
        ax.set_yticks(range(rank_pivot.shape[0]))
        ax.set_yticklabels(rank_pivot.index.tolist())
        ax.set_xlabel("Recording-disjoint seed")
        ax.set_title("Rank by seed across the three main compared `fmiss` models\\n(20 recording-disjoint seeds, 42-61; 1 = best `R²`)", pad=10)
        for row_idx, row_label in enumerate(rank_pivot.index):
            for col_idx, col_label in enumerate(rank_pivot.columns):
                value = rank_pivot.loc[row_label, col_label]
                if pd.notna(value):
                    ax.text(col_idx, row_idx, int(value), ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Rank")
        fig.tight_layout()
        save_figure(fig, fig_dir, "fmiss_seed_rank_heatmap")
        fig
        """),
        code("""
        winner_row = benchmark.sort_values("r2", ascending=False).iloc[0]
        paired_context_row = benchmark.loc[benchmark["recipe_id"] == "fmiss_paired_context"].iloc[0]
        hybrid_only_row = benchmark.loc[benchmark["recipe_id"] == "fmiss_hybrid_only"].iloc[0]
        display(Markdown(
            f\"\"\"
        ## Main `fmiss` result

        - The best `fmiss` row is **{winner_row['label']}** with **R² {winner_row['r2']:.4f}**.
        - Compared with the source-only transfer baseline, the retained best row improves from **R² {hybrid_only_row['r2']:.4f}** to **R² {winner_row['r2']:.4f}**.
        - Compared with the simpler paired-context model, the reduced-latent stack adds a smaller but still meaningful gain (**R² {paired_context_row['r2']:.4f}** to **{winner_row['r2']:.4f}**).
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
        for group_name in ["source_target_choice", "context", "transfer_family"]:
            bundle = build_ablation_bundle("fmiss", group_name, "recording_disjoint_main")
            display(Markdown(f"### {group_name.replace('_', ' ').title()}"))
            display(bundle["benchmark"][["label", "mae", "r2", "notes"]])
            save_table(bundle["benchmark"], table_dir, f"fmiss_ablation_{group_name}")
        """),
        md("""
        ## 3. Family-held-out behavior

        The stricter `fmiss` family story should be read under leave-one-family-out evaluation rather than ordinary recording-disjoint family averaging. This section keeps the comparison aligned with the retained main compared `fmiss` models: `Paired-only raw`, `Paired-only context`, and `Reduced-latent context`.
        """),
        code("""
display(lofo_family["best"])
display(lofo_family["r2_pivot"])
save_table(lofo_family["best"], table_dir, "fmiss_lofo_family_preference_best")
save_table(lofo_family["r2_pivot"].reset_index(), table_dir, "fmiss_lofo_family_preference_r2")
        """),
        code("""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        family_r2 = lofo_family["r2_pivot"].copy()
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        norm = mcolors.TwoSlopeNorm(vmin=-1.10, vcenter=0.0, vmax=0.85)
        im = ax.imshow(family_r2.to_numpy(dtype=float), cmap="RdYlGn", norm=norm, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(family_r2.shape[1]))
        ax.set_xticklabels(family_r2.columns.tolist(), rotation=24, ha="right")
        ax.set_yticks(range(family_r2.shape[0]))
        ax.set_yticklabels([name.replace("PAIRED_", "") for name in family_r2.index.tolist()])
        ax.set_title("Mean held-out-family `R²` across the main compared `fmiss` models\\n(20-seed leave-one-family-out aggregate)", pad=10)
        for row_idx, family_name in enumerate(family_r2.index):
            for col_idx, label in enumerate(family_r2.columns):
                value = family_r2.loc[family_name, label]
                if pd.notna(value):
                    text_color = "white" if abs(value) >= 0.72 else "black"
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Mean R²")
        fig.tight_layout()
        save_figure(fig, fig_dir, "fmiss_lofo_family_heatmap")
        fig
        """),
        md("""
        ### Additional recording-disjoint diagnostics

        The plots below remain useful as supporting diagnostics, but the thesis-facing family comparison above is the leave-one-family-out view.
        """),
        code("""
display(waves["family"])
display(waves["recording"].head(20))
save_table(waves["family"], table_dir, "fmiss_family_wave_summary")
save_table(waves["recording"], table_dir, "fmiss_recording_wave_summary")
fig, _, family_pivot = plot_group_metric(waves["family"], group_col="study_set", metric="r2", title="`fmiss` family R² by model wave\\n(paired-family summaries from 20 recording-disjoint seeds)")
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
        top_rec, bottom_rec = build_extreme_groups_table(recording_mean[recording_mean["label"] == REDUCED_LATENT_CONTEXT_LABEL], group_col="recording_key", metric="r2", top_n=12)
        focus_recordings = pd.concat([top_rec, bottom_rec], ignore_index=True)["recording_key"].drop_duplicates().tolist()
        recording_focus = waves["recording"][waves["recording"]["recording_key"].isin(focus_recordings)].copy()
        display(top_rec)
        display(bottom_rec)
        save_table(top_rec, table_dir, "fmiss_top_recordings")
        save_table(bottom_rec, table_dir, "fmiss_worst_recordings")
        fig, _, rec_pivot = plot_group_metric(recording_focus, group_col="recording_key", metric="r2", title="`fmiss` recording R² by model wave\\n(top/bottom recordings from 20 recording-disjoint seeds)")
        save_figure(fig, fig_dir, "fmiss_top_recording_r2_by_wave")
        save_table(rec_pivot, table_dir, "fmiss_recording_r2_pivot")
        fig
        """),
        md("""
        ## 4. Dedicated reduced-latent context stack diagnostics

        This is the structural comparison that matters most for the `fmiss` story: the final model still behaves like a conservative context-first correction model rather than a dramatic architecture shift.
        """),
        code("""
        context_compare = pd.DataFrame([
            {
                "model": "Paired-only context",
                "r2": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_paired_context", "r2"].iloc[0]),
                "mae": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_paired_context", "mae"].iloc[0]),
            },
            {
                "model": "Reduced-latent context",
                "r2": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_reduced_latent", "r2"].iloc[0]),
                "mae": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_reduced_latent", "mae"].iloc[0]),
            },
        ])
        display(context_compare)
        save_table(context_compare, table_dir, "fmiss_context_progress_comparison")
        fig, _ = plot_prediction_scatter(winner_diag["predictions"], title="Best `fmiss` model: observed vs predicted")
        save_figure(fig, fig_dir, "fmiss_winner_prediction_scatter")
        fig
        """),
        code("""
        milestone = pd.DataFrame([
            {"stage": "1  sanity floor", "model": "Dummy paired mean", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_dummy", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_dummy", "mae"].iloc[0])},
            {"stage": "2  paired baseline", "model": "Paired-only raw", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_paired_raw", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_paired_raw", "mae"].iloc[0])},
            {"stage": "3  source transfer", "model": "Hybrid-only XGBoost", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_hybrid_only", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_hybrid_only", "mae"].iloc[0])},
            {"stage": "4  paired context", "model": "Paired-only context", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_paired_context", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_paired_context", "mae"].iloc[0])},
            {"stage": "5  reduced latent", "model": "Reduced-latent context", "r2": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_reduced_latent", "r2"].iloc[0]), "mae": float(benchmark.loc[benchmark["recipe_id"] == "fmiss_reduced_latent", "mae"].iloc[0])},
        ])
        display(milestone)
        save_table(milestone, table_dir, "fmiss_milestone_trajectory")
        """),
        code("""
        display(Markdown(
            f\"\"\"
        ## Key takeaways

        - The best `fmiss` path remained **context-first** and relatively conservative.
        - The dedicated reduced-latent context stack reaches correlation **{winner_diag['correlation']:.3f}** on the pooled holdout predictions from the 20 recording-disjoint seeds.
        - The main retained gain is incremental rather than dramatic: reduced latent improves on the simpler paired-context model, but `fmiss` remains much harder than `fpos`.
        \"\"\"
        ))
        """),
    ])


def notebook_06():
    stem = "06_fpos_vs_fmiss_xai"
    return new_notebook(cells=[
        md("""
        # 06 `fpos` Vs `fmiss`: XAI And Target Asymmetry

        This notebook confronts the two targets directly. The goal is not only to show that their leading recipes differ, but to show why the thesis treats them as different scientific problems.

        **Questions answered here**
        - Which features dominate the best `fpos` model?
        - How do family-level behaviors differ between the best `fpos` and best `fmiss` lines?
        - Where is the strongest evidence that the two targets should not share one final recipe?
        """),
        setup_code(f"""
        fig_dir, table_dir = notebook_output_dirs("{stem}")
        shap_df = load_fpos_shap_importance()
        shap_group_df = build_shap_group_summary()
        fpos_family = (
            pd.read_csv(ROOT / "artifacts" / "aggregates" / "family_held_out" / "fpos" / "leave_one_family_main5_by_family.csv")
            .query("recipe_id == 'fpos_waveform_winner'")
            .rename(columns={"held_out_family": "study_set"})
            .reset_index(drop=True)
        )
        fmiss_family = (
            pd.read_csv(ROOT / "artifacts" / "aggregates" / "family_held_out" / "fmiss" / "leave_one_family_main3_by_family.csv")
            .query("recipe_id == 'fmiss_paired_raw'")
            .rename(columns={"held_out_family": "study_set"})
            .reset_index(drop=True)
        )
        asymmetry = build_target_asymmetry_table()
        fpos_progress = build_benchmark_progress_table("fpos", scope="core")
        fmiss_progress = build_benchmark_progress_table("fmiss", scope="core")
        """),
        md("""
        ## 1. What drives the best `fpos` model?

        The SHAP summary is useful because it tells us whether the lead `fpos` stack is a black box or a structured correction model. The answer from earlier analysis was: a strong transferred prior plus waveform/amplitude/ISI corrections.
        """),
        code("""
        display(shap_df.head(20))
        display(shap_group_df)
        save_table(shap_df, table_dir, "fpos_shap_importance")
        save_table(shap_group_df, table_dir, "fpos_shap_group_summary")
        fig, _ = plot_fpos_shap_top(shap_df, top_n=14)
        save_figure(fig, fig_dir, "fpos_shap_top")
        fig
        """),
        code("""
        show_saved_figure(
            fig_dir / "fpos_shap_group_summary.png",
            "The grouped SHAP summary collapses the feature ranking into thesis-level blocks, making it clearer that the best `fpos` model uses a transferred prior plus waveform, amplitude, and ISI corrections rather than one narrow shortcut.",
        )
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
        fig, _, _ = plot_group_metric(fpos_family, group_col="study_set", metric="r2", title="Held-out-family `fpos` behavior")
        save_figure(fig, fig_dir, "fpos_family_behavior")
        fig
        """),
        code("""
        fig, _, _ = plot_group_metric(fmiss_family, group_col="study_set", metric="r2", title="Held-out-family `fmiss` behavior")
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
        code("""
        show_saved_figure(
            fig_dir / "fpos_fmiss_family_r2_scatter.png",
            "This family-level scatter is the compact comparison plot for the thesis: each point is one paired family, with lead `fpos` and lead `fmiss` performance on the two axes.",
        )
        """),
        md("""
        ## 3. Benchmark-level asymmetry

        This table compresses the whole thesis finding into one summary: `fpos` gained much more from the later target-aware structure than `fmiss` did.
        """),
        code("""
        basic_fpos = fpos_progress.loc[fpos_progress["recipe_id"] == "fpos_hybrid_only"].iloc[0]
        final_fpos = fpos_progress.sort_values("r2", ascending=False).iloc[0]
        basic_fmiss = fmiss_progress.loc[fmiss_progress["recipe_id"] == "fmiss_hybrid_only"].iloc[0]
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
        fpos_progress = build_benchmark_progress_table("fpos", scope="core")
        fmiss_progress = build_benchmark_progress_table("fmiss", scope="core")
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
        code("""
        show_saved_figure(
            fig_dir / "label_budget_combined.png",
            "This label-budget figure isolates the `fpos` waveform-augmented stack on the fixed recording-disjoint split (seed 42), with `R²` on the left axis and `MAE` on the right; the `20` and `60` row points are two grouped-recording subset repeats.",
        )
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

        The table below puts the current best results next to the dummy floor. It is not a theoretical upper bound, but it does show how much usable signal the retained models recover above a non-informative predictor.
        """),
        code("""
        basic_fpos = fpos_progress.loc[fpos_progress["recipe_id"] == "fpos_dummy"].iloc[0]
        final_fpos = fpos_progress.sort_values("r2", ascending=False).iloc[0]
        basic_fmiss = fmiss_progress.loc[fmiss_progress["recipe_id"] == "fmiss_dummy"].iloc[0]
        final_fmiss = fmiss_progress.sort_values("r2", ascending=False).iloc[0]
        headroom = pd.DataFrame([
            {"target": "fpos", "basic_transfer_r2": basic_fpos["r2"], "current_best_r2": final_fpos["r2"], "gain_r2": final_fpos["r2"] - basic_fpos["r2"]},
            {"target": "fmiss", "basic_transfer_r2": basic_fmiss["r2"], "current_best_r2": final_fmiss["r2"], "gain_r2": final_fmiss["r2"] - basic_fmiss["r2"]},
        ])
        display(headroom)
        save_table(headroom, table_dir, "headroom_summary")
        """),
        code("""
        show_saved_figure(
            fig_dir / "headroom_summary.png",
            "This headroom summary shows how far each target moved from the dummy floor to the current best retained model, which is more interpretable than a raw bar chart of the final rows alone.",
        )
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
            .loc[lambda df: df["main_text"].fillna(False)]
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
    build_highlight_figures(THESIS_ROOT)
    for name, builder in NOTEBOOK_BUILDERS.items():
        nb = builder()
        out = NOTEBOOK_DIR / name
        nbformat.write(nb, out)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
