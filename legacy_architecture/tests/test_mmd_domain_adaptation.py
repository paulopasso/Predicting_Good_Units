from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crash_tests.mmd_domain_adaptation.config import MMDDomainAdaptationConfig
from crash_tests.mmd_domain_adaptation.dataset import (
    PreparedModelInput,
    prepare_feature_table_from_frame,
)
from crash_tests.mmd_domain_adaptation.model import rbf_multi_scale_mmd
from crash_tests.mmd_domain_adaptation.train import (
    evaluate_state,
    finetune_labeled_target,
    train_source_with_mmd,
)
from crash_tests.self_supervised_domain_adaptation.split_utils import make_main_split


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
                    "num_tested": 10,
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

    paired_recordings = [f"PAIRED_ENGLISH::study::rec_{idx}" for idx in range(4)]
    for rec_idx, recording_key in enumerate(paired_recordings):
        for unit_idx in range(3):
            rows.append(
                {
                    "row_uid": f"paired_match_{row_id}",
                    "study_set": "PAIRED_ENGLISH",
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
                    "num_tested": 10,
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
                    "study_set": "PAIRED_ENGLISH",
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
                    "num_tested": 12,
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


def _random_input(n_rows: int, seed: int) -> PreparedModelInput:
    rng = np.random.default_rng(seed)
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
        context=None,
        context_mask=None,
    )


def test_main_split_seals_holdout_recordings() -> None:
    frame = _synthetic_frame()
    split = make_main_split(frame, paired_final_holdout_rows=5, random_state=7)
    holdout = set(split.paired_test["recording_key"].astype(str))
    assert not (holdout & set(split.paired_finetune["recording_key"].astype(str)))
    assert not (holdout & set(split.ssl_pool["recording_key"].astype(str)))


def test_feature_table_stays_unit_only_for_neural_path() -> None:
    prepared = prepare_feature_table_from_frame(_synthetic_frame())
    assert all("_recording_" not in col for col in prepared.unit_feature_cols)
    assert all(not col.startswith("study_indicator__") for col in prepared.unit_feature_cols)
    assert prepared.shape_feature_cols
    assert prepared.lightgbm_context_cols


def test_mmd_loss_behaviour() -> None:
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    y_same = x.clone()
    y_shifted = x + 3.0
    same_loss = float(rbf_multi_scale_mmd(x, y_same).item())
    shifted_loss = float(rbf_multi_scale_mmd(x, y_shifted).item())
    assert same_loss < 1e-5
    assert shifted_loss > same_loss


def test_mmd_training_smoke_runs(tmp_path: Path) -> None:
    config = MMDDomainAdaptationConfig(
        output_root=tmp_path,
        source_train_epochs=1,
        finetune_epochs=1,
        batch_size=8,
        patience=1,
        verbose=False,
    )
    source_train = _random_input(32, seed=1)
    source_val = _random_input(12, seed=2)
    target_unlabeled = _random_input(24, seed=3)
    target_train = _random_input(16, seed=4)
    target_test = _random_input(8, seed=5)

    def _target(n: int, seed: int) -> np.ndarray:
        return np.clip(np.random.default_rng(seed).uniform(0.05, 0.95, size=n), 0.0, 1.0)

    source_result = train_source_with_mmd(
        source_train_data=source_train,
        source_train_target=_target(len(source_train), 10),
        source_val_data=source_val,
        source_val_target=_target(len(source_val), 11),
        target_unlabeled_data=target_unlabeled,
        mmd_lambda=0.0,
        config=config,
        target_name="fpos",
        checkpoint_path=tmp_path / "source.pt",
    )
    assert source_result.model_state
    first_row = source_result.history[0]
    assert abs(first_row["train_total_loss"] - first_row["train_task_loss"]) < 1e-6

    finetune_result = finetune_labeled_target(
        source_state=source_result.model_state,
        train_data=target_train,
        train_target=_target(len(target_train), 12),
        config=config,
        target_name="fpos",
        random_state=13,
        checkpoint_path=tmp_path / "finetune.pt",
    )
    metrics, pred = evaluate_state(
        model_state=finetune_result.model_state,
        eval_data=target_test,
        eval_target=_target(len(target_test), 14),
        config=config,
    )
    assert np.isfinite(pred).all()
    assert metrics["mae"] >= 0.0
