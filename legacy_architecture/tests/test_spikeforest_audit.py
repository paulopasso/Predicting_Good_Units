from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from crash_tests.spikeforest_audit.run_experiment import (
    _missing_items,
    _recording_level_manifest,
    _study_level_manifest,
    parse_requested_study_sets,
)


def test_parse_requested_study_sets_from_notebook(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["hello"]},
            {
                "cell_type": "code",
                "source": [
                    "SPIKEFOREST_STUDY_SETS = [\n",
                    "    'PAIRED_FOO',\n",
                    "    'PAIRED_BAR',\n",
                    "    'HYBRID_JANELIA',\n",
                    "]\n",
                ],
            },
        ]
    }
    path = tmp_path / "demo.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    assert parse_requested_study_sets(path) == ["PAIRED_FOO", "PAIRED_BAR", "HYBRID_JANELIA"]


def test_audit_manifests_and_missing_items_flag_common_sorter_gap() -> None:
    df = pd.DataFrame(
        {
            "study_set": ["PAIRED_FOO"] * 6,
            "study_name": ["foo"] * 6,
            "recording_name": ["rec1", "rec1", "rec2", "rec2", "rec3", "rec3"],
            "recording_key": ["foo::rec1", "foo::rec1", "foo::rec2", "foo::rec2", "foo::rec3", "foo::rec3"],
            "row_uid": [f"u{i}" for i in range(6)],
            "sorter_name": ["s1", "s2", "s1", "s2", "s1", "s1"],
            "matched_gt_unit_id": ["a", None, "b", None, None, None],
        }
    )
    study_manifest = _study_level_manifest(df, ["PAIRED_FOO"])
    recording_manifest, _ = _recording_level_manifest(df, 0.5)
    missing = _missing_items(study_manifest=study_manifest, recording_manifest=recording_manifest)
    assert set(study_manifest["study_set"]) == {"PAIRED_FOO"}
    rec3 = recording_manifest[recording_manifest["recording_name"] == "rec3"].iloc[0]
    assert rec3["missing_common_sorter_count"] == 1
    assert "s2" in str(rec3["missing_common_sorters"])
    assert (missing["item_type"] == "missing_sorter_output_common").any()
