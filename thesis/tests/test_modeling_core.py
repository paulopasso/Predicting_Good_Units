from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd

from qc_thesis.data import RAW_PARQUET_PATH
from qc_thesis.modeling.shared import FrameAugmenter, StackSet, build_stack_frame


def _sample_recipe_parquet(tmp_path: Path) -> Path:
    raw = pd.read_parquet(RAW_PARQUET_PATH)
    is_paired = raw["study_set"].astype(str).str.startswith("PAIRED")
    hybrid = raw[~is_paired]
    paired = raw[is_paired]
    paired_labeled = paired[paired["match_status"].isin(["matched_best", "well_detected"])]
    paired_unlabeled = paired[~paired["match_status"].isin(["matched_best", "well_detected"])]
    sampled = pd.concat(
        [
            hybrid.sample(n=min(300, len(hybrid)), random_state=1),
            paired_labeled.sample(n=min(120, len(paired_labeled)), random_state=2),
            paired_unlabeled.sample(n=min(120, len(paired_unlabeled)), random_state=3),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["row_uid"])
    parquet_path = tmp_path / "sampled_recipe_input.parquet"
    sampled.to_parquet(parquet_path, index=False)
    return parquet_path


def _run_recipe_subprocess(recipe_id: str, output_dir: Path, parquet_path: Path) -> None:
    script = f"""
from pathlib import Path
import sys
sys.path.insert(0, 'thesis/src')
from qc_thesis import run_recipe
run_recipe(
    {recipe_id!r},
    output_dir=Path({str(output_dir)!r}),
    force=True,
    smoke=True,
    parquet_path=Path({str(parquet_path)!r}),
)
"""
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[2])


def test_stack_primitives_align_frames():
    train_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    unlabeled_df = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    test_df = pd.DataFrame({"a": [9, 10], "b": [11, 12]})

    base = StackSet.from_context_frames(
        train_df=train_df,
        unlabeled_df=unlabeled_df,
        test_df=test_df,
        context_cols=["a"],
        train_features={"pred": [0.1, 0.2]},
        unlabeled_features={"pred": [0.3, 0.4]},
        test_features={"pred": [0.5, 0.6]},
    )
    augmented = base.apply(
        FrameAugmenter(
            train_features={"latent": [[1.0, 2.0], [3.0, 4.0]]},
            unlabeled_features={"latent": [[5.0, 6.0], [7.0, 8.0]]},
            test_features={"latent": [[9.0, 10.0], [11.0, 12.0]]},
        )
    )

    direct = build_stack_frame(train_df, context_cols=["a"], extra_features={"pred": [0.1, 0.2]})

    assert list(base.train.columns) == ["a", "pred"]
    assert list(augmented.test.columns) == ["a", "pred", "latent_00", "latent_01"]
    assert direct.shape == (2, 2)


def test_transfer_recipe_smoke(tmp_path: Path):
    parquet_path = _sample_recipe_parquet(tmp_path)
    _run_recipe_subprocess("fpos_dummy", tmp_path / "transfer", parquet_path)
    assert (tmp_path / "transfer" / "run_manifest.json").exists()


def test_stack_recipe_smoke(tmp_path: Path):
    parquet_path = _sample_recipe_parquet(tmp_path)
    _run_recipe_subprocess("fpos_context_stack", tmp_path / "stack", parquet_path)
    assert (tmp_path / "stack" / "run_manifest.json").exists()


def test_waveform_stack_recipe_smoke(tmp_path: Path):
    parquet_path = _sample_recipe_parquet(tmp_path)
    _run_recipe_subprocess("fpos_waveform_winner", tmp_path / "waveform_stack", parquet_path)
    assert (tmp_path / "waveform_stack" / "run_manifest.json").exists()


def test_waveform_transfer_recipe_smoke(tmp_path: Path):
    parquet_path = _sample_recipe_parquet(tmp_path)
    _run_recipe_subprocess("fmiss_transplanted_fpos", tmp_path / "waveform_transfer", parquet_path)
    assert (tmp_path / "waveform_transfer" / "run_manifest.json").exists()


def test_neural_recipe_smoke(tmp_path: Path):
    parquet_path = _sample_recipe_parquet(tmp_path)
    _run_recipe_subprocess("fpos_ssl", tmp_path / "neural", parquet_path)
    assert (tmp_path / "neural" / "run_manifest.json").exists()
