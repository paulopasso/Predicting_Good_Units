from __future__ import annotations

import numpy as np

from crash_tests.ml_statistics_workbench.fpos_signal_embedding_stack import (
    FPosSignalEmbeddingStackConfig,
    _TORCH_AVAILABLE,
)


def test_signal_embedding_config_resolves_output_dir() -> None:
    config = FPosSignalEmbeddingStackConfig(run_name="unit_test_run")
    assert str(config.resolved_output_dir()).endswith("unit_test_run")


def test_signal_embedding_smoke_defaults_are_positive() -> None:
    config = FPosSignalEmbeddingStackConfig()
    assert config.conv_embed_dim > 0
    assert config.conv_hidden_channels > 0
    assert config.ae_epochs > 0
    assert config.batch_size > 0


def test_torch_flag_is_boolean() -> None:
    assert isinstance(_TORCH_AVAILABLE, bool)
    assert np.isfinite(float(int(_TORCH_AVAILABLE in (True, False))))
