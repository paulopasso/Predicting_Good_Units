from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crash_tests.self_supervised_domain_adaptation.config import SSLDomainAdaptationConfig
from crash_tests.self_supervised_domain_adaptation.dataset import (
    PreparedModelInput,
    prepare_feature_table_from_frame,
)
from crash_tests.self_supervised_domain_adaptation.split_utils import (
    build_stress_splits,
    make_main_split,
)
from crash_tests.self_supervised_domain_adaptation.train import (
    pretrain_ssl_autoencoder,
    run_supervised_experiment,
)


def _synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_id = 0
    hybrid_recordings = [f"HYBRID::study::rec_{idx}" for idx in range(6)]
    for rec_idx, recording_key in enumerate(hybrid_recordings):
        for unit_idx in range(6):
            rows.append(
                {
                    "row_uid": f"hyb_{row_id}",
                    "study_set": "HYBRID_JANELIA",
                    "study_name": "hybrid_study",
                    "recording_name": f"hybrid_rec_{rec_idx}",
                    "recording_key": recording_key,
                    "dataset_type": "hybrid",
                    "sorter_name": "ks4",
                    "unit_id": unit_idx,
                    "matched_gt_unit_id": str(unit_idx),
                    "fpos": float(0.1 + 0.01 * unit_idx),
                    "fmiss": float(0.2 + 0.01 * unit_idx),
                    "accuracy": float(0.6 - 0.01 * unit_idx),
                    "wf_bin_00": 0.1 * unit_idx,
                    "wf_bin_01": 0.2 * unit_idx,
                    "acg_00": 0.3 * unit_idx,
                    "acg_01": 0.4 * unit_idx,
                    "snr": float(2 + unit_idx),
                    "amp_mean": float(10 + unit_idx),
                    "template_norm": float(5 + unit_idx),
                }
            )
            row_id += 1

    paired_studies = {
        "PAIRED_ENGLISH": [f"PAIRED_ENGLISH::study::rec_{idx}" for idx in range(3)],
        "PAIRED_KAMPFF": [f"PAIRED_KAMPFF::study::rec_{idx}" for idx in range(2)],
    }
    for study_set, recordings in paired_studies.items():
        for rec_idx, recording_key in enumerate(recordings):
            for unit_idx in range(3):
                rows.append(
                    {
                        "row_uid": f"paired_match_{row_id}",
                        "study_set": study_set,
                        "study_name": "paired_study",
                        "recording_name": f"paired_rec_{rec_idx}",
                        "recording_key": recording_key,
                        "dataset_type": "paired",
                        "sorter_name": "ks4",
                        "unit_id": unit_idx,
                        "matched_gt_unit_id": str(unit_idx),
                        "fpos": float(0.05 + 0.02 * unit_idx),
                        "fmiss": float(0.15 + 0.02 * unit_idx),
                        "accuracy": float(0.75 - 0.02 * unit_idx),
                        "wf_bin_00": 0.1 * unit_idx,
                        "wf_bin_01": 0.2 * unit_idx,
                        "acg_00": 0.3 * unit_idx,
                        "acg_01": 0.4 * unit_idx,
                        "snr": float(3 + unit_idx),
                        "amp_mean": float(9 + unit_idx),
                        "template_norm": float(6 + unit_idx),
                    }
                )
                row_id += 1
            for unit_idx in range(5):
                rows.append(
                    {
                        "row_uid": f"paired_unmatch_{row_id}",
                        "study_set": study_set,
                        "study_name": "paired_study",
                        "recording_name": f"paired_rec_{rec_idx}",
                        "recording_key": recording_key,
                        "dataset_type": "paired",
                        "sorter_name": "ks4",
                        "unit_id": 100 + unit_idx,
                        "matched_gt_unit_id": None,
                        "fpos": float(0.2 + 0.01 * unit_idx),
                        "fmiss": np.nan,
                        "accuracy": float(0.4 - 0.01 * unit_idx),
                        "wf_bin_00": 0.05 * unit_idx,
                        "wf_bin_01": 0.06 * unit_idx,
                        "acg_00": 0.07 * unit_idx,
                        "acg_01": 0.08 * unit_idx,
                        "snr": float(2.5 + unit_idx),
                        "amp_mean": float(8 + unit_idx),
                        "template_norm": float(4 + unit_idx),
                    }
                )
                row_id += 1
    return pd.DataFrame(rows)


def _random_input(n_rows: int, *, with_context: bool, seed: int) -> PreparedModelInput:
    rng = np.random.default_rng(seed)
    context = rng.normal(size=(n_rows, 4)).astype(np.float32) if with_context else None
    context_mask = np.ones((n_rows, 4), dtype=bool) if with_context else None
    return PreparedModelInput(
        row_uid=np.array([f"row_{seed}_{idx}" for idx in range(n_rows)]),
        recording_key=np.array([f"rec_{idx % 4}" for idx in range(n_rows)]),
        study_set=np.array(["PAIRED_ENGLISH"] * n_rows),
        wf=rng.normal(size=(n_rows, 30)).astype(np.float32),
        acg=rng.normal(size=(n_rows, 40)).astype(np.float32),
        scalars=rng.normal(size=(n_rows, 6)).astype(np.float32),
        wf_mask=np.ones((n_rows, 30), dtype=bool),
        acg_mask=np.ones((n_rows, 40), dtype=bool),
        scalar_mask=np.ones((n_rows, 6), dtype=bool),
        context=context,
        context_mask=context_mask,
    )


def test_main_split_seals_holdout_recordings() -> None:
    frame = _synthetic_frame()
    split = make_main_split(frame, paired_final_holdout_rows=5, random_state=7)

    holdout = set(split.paired_test["recording_key"].astype(str))
    assert not (holdout & set(split.paired_finetune["recording_key"].astype(str)))
    assert not (holdout & set(split.ssl_pool["recording_key"].astype(str)))


def test_stress_splits_exclude_held_out_study() -> None:
    frame = _synthetic_frame()
    splits = build_stress_splits(frame)

    assert {s.held_out_study for s in splits} == {"PAIRED_ENGLISH", "PAIRED_KAMPFF"}
    for split in splits:
        held_out = split.held_out_study
        assert set(split.paired_test["study_set"].astype(str)) == {held_out}
        assert held_out not in set(split.paired_finetune["study_set"].astype(str))
        assert held_out not in set(split.ssl_pool["study_set"].astype(str))


def test_feature_table_keeps_context_out_of_unit_encoder() -> None:
    frame = _synthetic_frame()
    prepared = prepare_feature_table_from_frame(frame)

    assert all("_recording_" not in col for col in prepared.unit_feature_cols)
    assert all(not col.startswith("study_indicator__") for col in prepared.context_feature_cols)
    assert prepared.context_feature_cols
    assert prepared.context_feature_cols_by_target["fpos"]
    assert prepared.context_feature_cols_by_target["fmiss"]
    assert len(prepared.context_feature_cols_by_target["fpos"]) < len(prepared.context_feature_cols)
    assert len(prepared.context_feature_cols_by_target["fmiss"]) < len(prepared.context_feature_cols)
    assert all(not col.startswith("study_indicator__") for col in prepared.context_feature_cols_by_target["fpos"])
    assert all(not col.startswith("study_indicator__") for col in prepared.context_feature_cols_by_target["fmiss"])
    assert prepared.shape_feature_cols


def test_ssl_and_supervised_smoke_runs(tmp_path: Path) -> None:
    config = SSLDomainAdaptationConfig(
        output_root=tmp_path,
        ssl_pretrain_epochs=1,
        source_warmup_epochs=1,
        source_train_epochs=1,
        finetune_epochs=1,
        paired_finetune_val_splits=2,
        batch_size=8,
        verbose=False,
    )
    ssl_train = _random_input(24, with_context=False, seed=1)
    ssl_val = _random_input(12, with_context=False, seed=2)
    ssl_result = pretrain_ssl_autoencoder(
        train_data=ssl_train,
        val_data=ssl_val,
        config=config,
        checkpoint_path=tmp_path / "ssl.pt",
    )
    assert ssl_result.encoder_state

    source_train = _random_input(32, with_context=True, seed=3)
    source_val = _random_input(12, with_context=True, seed=4)
    paired_train_a = _random_input(10, with_context=True, seed=5)
    paired_val_a = _random_input(6, with_context=True, seed=6)
    paired_train_b = _random_input(10, with_context=True, seed=7)
    paired_val_b = _random_input(6, with_context=True, seed=8)
    paired_full = _random_input(16, with_context=True, seed=9)
    paired_test = _random_input(8, with_context=True, seed=10)

    def _target(n: int, seed: int) -> np.ndarray:
        return np.clip(np.random.default_rng(seed).uniform(0.05, 0.95, size=n), 0.0, 1.0)

    result = run_supervised_experiment(
        arm="contextual",
        target_name="fpos",
        source_target_name="fpos",
        source_train_data=source_train,
        source_train_target=_target(len(source_train), 11),
        source_val_data=source_val,
        source_val_target=_target(len(source_val), 12),
        paired_train_splits=[
            (paired_train_a, _target(len(paired_train_a), 13), paired_val_a, _target(len(paired_val_a), 14)),
            (paired_train_b, _target(len(paired_train_b), 15), paired_val_b, _target(len(paired_val_b), 16)),
        ],
        paired_full_train_data=paired_full,
        paired_full_train_target=_target(len(paired_full), 17),
        paired_test_data=paired_test,
        paired_test_target=_target(len(paired_test), 18),
        output_checkpoint_dir=tmp_path / "checkpoints",
        model_stem="smoke_contextual",
        config=config,
        pretrained_encoder_state=ssl_result.encoder_state,
    )

    assert np.isfinite(result.predictions["pred"]).all()
    assert len(result.predictions) == len(paired_test)
    assert result.selected_finetune_epochs >= 1
