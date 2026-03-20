from __future__ import annotations

from pathlib import Path

import pandas as pd

from qc_thesis import (
    build_benchmark_table,
    build_dataset_bundle,
    build_dataset_partition_table,
    build_feature_inventory_table,
    build_recipe_inventory,
    build_target_missingness_table,
    get_domain_shift_bundle,
    get_limitations_bundle,
    get_main_text_recipe_ids,
    get_recipe,
    list_recipes,
    recording_disjoint_split,
    validate_disjoint_recordings,
    validate_recipe_registry,
)
from qc_thesis.modeling.recipes import RecipeResult


def test_dataset_bundle_and_recording_split_are_valid():
    bundle = build_dataset_bundle()
    assert not bundle.full_df.empty
    partition_table = build_dataset_partition_table()
    missingness_table = build_target_missingness_table()
    inventory_table = build_feature_inventory_table()
    assert not partition_table.empty
    assert not missingness_table.empty
    assert not inventory_table.empty
    split = recording_disjoint_split(bundle.paired_labeled_df, test_rows_target=100, random_state=42)
    assert validate_disjoint_recordings(split.train_df, split.test_df)
    assert split.test_df.shape[0] > 0


def test_domain_shift_bundle_is_not_empty():
    bundle = get_domain_shift_bundle()
    assert not bundle.domain_summary.empty
    assert not bundle.feature_shift.empty
    assert not bundle.pca_2d.empty


def test_recipe_inventory_and_groups_exist():
    inventory = build_recipe_inventory()
    assert not inventory.empty
    assert validate_recipe_registry() == []
    assert len(list_recipes(target="fpos")) > 0
    assert len(get_main_text_recipe_ids("fmiss")) > 0


def test_build_benchmark_table_uses_recipe_results(monkeypatch):
    recipe = get_recipe("fpos_dummy")

    fake_metrics = pd.DataFrame(
        [
            {
                "recipe_id": recipe.recipe_id,
                "label": recipe.label,
                "protocol": recipe.protocol,
                "mae": 0.1,
                "rmse": 0.2,
                "r2": 0.3,
                "bias": 0.0,
                "notes": recipe.notes,
            }
        ]
    )
    fake_result = RecipeResult(
        recipe=recipe,
        metrics=fake_metrics,
        predictions=None,
        per_family_summary=None,
        per_recording_summary=None,
        output_dir=Path("."),
    )

    monkeypatch.setattr(
        "qc_thesis.modeling.recipes.load_experiment",
        lambda recipe_id, target, protocol: fake_result,
    )
    table = build_benchmark_table("fpos", ["fpos_dummy"], "recording_disjoint_main")
    assert list(table["recipe_id"]) == ["fpos_dummy"]


def test_limitations_and_asymmetry_bundles_exist():
    bundle = get_limitations_bundle()
    assert not bundle.label_budget.empty
    assert not bundle.lofo_aggregate.empty
