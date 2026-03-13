from __future__ import annotations

import numpy as np
import pandas as pd

from qc_neural.transfer import SingleTargetDomainAdaptiveRegressor


def _make_frame(n_rows: int, seed: int, include_strict_fmiss: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for idx in range(30):
        data[f"wf_bin_{idx:02d}"] = rng.normal(size=n_rows).astype(float)
    for idx in range(40):
        data[f"acg_{idx:02d}"] = rng.normal(size=n_rows).astype(float)
    data["snr_z"] = rng.normal(size=n_rows).astype(float)
    data["amp_mean_recording_pct"] = rng.uniform(size=n_rows).astype(float)
    data["accuracy"] = rng.uniform(size=n_rows).astype(float)
    data["fpos"] = rng.uniform(size=n_rows).astype(float)
    data["fmiss_extended"] = rng.uniform(size=n_rows).astype(float)
    if include_strict_fmiss:
        data["fmiss"] = rng.uniform(size=n_rows).astype(float)
    return pd.DataFrame(data)


def test_single_target_transfer_regressor_predicts_bounded_values() -> None:
    source_df = _make_frame(48, seed=1, include_strict_fmiss=False)
    target_labeled_df = _make_frame(16, seed=2, include_strict_fmiss=True)
    target_unlabeled_df = _make_frame(24, seed=3, include_strict_fmiss=True).drop(columns=["accuracy", "fpos", "fmiss", "fmiss_extended"])

    feature_columns = [c for c in source_df.columns if c.startswith("wf_bin_") or c.startswith("acg_")]
    feature_columns += ["snr_z", "amp_mean_recording_pct"]

    model = SingleTargetDomainAdaptiveRegressor(
        feature_columns=feature_columns,
        target_name="fpos",
        source_target_name="fpos",
        pretrain_epochs=1,
        finetune_epochs=1,
        batch_size=8,
        random_state=7,
    )
    model.fit(source_df=source_df, target_labeled_df=target_labeled_df, target_unlabeled_df=target_unlabeled_df)

    pred = model.predict(target_labeled_df.head(5))

    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
    assert ((pred >= 0.0) & (pred <= 1.0)).all()
