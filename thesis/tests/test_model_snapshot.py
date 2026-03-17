from __future__ import annotations

from pathlib import Path

from qc_thesis import build_recipe_inventory, validate_recipe_registry


def test_key_modeling_files_exist():
    thesis_root = Path(__file__).resolve().parents[1]
    required = [
        thesis_root / "src/qc_thesis/modeling/MODEL_CODE_INDEX.md",
        thesis_root / "src/qc_thesis/modeling/specs.py",
        thesis_root / "src/qc_thesis/modeling/shared.py",
        thesis_root / "src/qc_thesis/modeling/backends.py",
        thesis_root / "src/qc_thesis/modeling/recipes.py",
        thesis_root / "src/qc_thesis/modeling/neural.py",
        thesis_root / "src/qc_thesis/modeling/transfer.py",
        thesis_root / "src/qc_thesis/modeling/stacking/specs.py",
        thesis_root / "src/qc_thesis/modeling/stacking/base_stack_runner.py",
        thesis_root / "src/qc_thesis/modeling/stacking/backends.py",
        thesis_root / "src/qc_thesis/modeling/stacking/augmenters.py",
        thesis_root / "src/qc_thesis/modeling/stacking/stack_recipe_runner.py",
        thesis_root / "src/qc_thesis/modeling/stacking/recipes.py",
        thesis_root / "scripts/validate_model_registry.py",
        thesis_root / "data_pipeline/extraction/ntb_data_extraction_colab.ipynb",
    ]
    for path in required:
        assert path.exists(), path


def test_recipe_inventory_and_registry_are_valid():
    inventory = build_recipe_inventory()
    assert not inventory.empty
    assert inventory["recipe_id"].is_unique
    assert validate_recipe_registry() == []
